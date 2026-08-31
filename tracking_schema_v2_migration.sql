-- ============================================================
-- ShieldFL — Tracking Schema v2 Migration
-- Adds the inbound_messages table and a trigger to keep
-- phone_number_registry updated from inbound-only events.
--
-- Run ONCE against an existing tracking.db:
--   sqlite3 tracking.db < tracking_schema_v2_migration.sql
--
-- Safe to re-run: all statements use IF NOT EXISTS / OR IGNORE.
-- ============================================================

-- ------------------------------------------------------------
-- NEW TABLE: inbound_messages
--
-- Sibling to suspicious_messages, but for SMS received without
-- any associated transfer.  Key differences from suspicious_messages:
--
--   - No amount / transfer_id / verdict (no payment to approve or block).
--   - Stores sms_risk_score directly (the only model output relevant here).
--   - inbound_action: what the device DID after scoring, not a financial verdict.
--     Possible values:
--       FLAGGED_SENDER  — risk >= threshold; sender added to WATCH/BLOCKED registry
--       LOGGED_ONLY     — risk below threshold; recorded for audit but no action
--   - verdict column is absent intentionally.  Adding one would require redefining
--     the CHECK on suspicious_messages, creating confusion between "transfer verdict"
--     and "inbound action".  Keeping them separate tables is the right call.
--
-- Privacy guarantee (same as suspicious_messages):
--   Raw message text is NEVER stored here.  Only the SHA-256 hash is persisted
--   in the shared tracking DB.  The raw text stays local to the edge device.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inbound_messages (
    inbound_id          TEXT PRIMARY KEY,          -- uuid4 hex
    device_id           TEXT NOT NULL REFERENCES edge_devices(device_id),
    provider_id         TEXT NOT NULL REFERENCES providers(provider_id),
    sender_phone        TEXT NOT NULL,
    message_text_hash   TEXT NOT NULL,             -- SHA-256 of raw body
    sms_risk_score      REAL NOT NULL,
    inbound_action      TEXT NOT NULL CHECK (inbound_action IN (
                            'FLAGGED_SENDER',      -- risk >= flag_threshold; registry updated
                            'LOGGED_ONLY'          -- below threshold; audit trail only
                        )),
    flag_threshold_used REAL NOT NULL,             -- threshold at decision time (auditable)
    model_round_id      INTEGER REFERENCES global_model_versions(round_id),
    is_ground_truth_spam INTEGER,                  -- nullable; known in simulation mode only
    received_at         TEXT NOT NULL,             -- ISO-8601 UTC from the edge device clock
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_inbound_sender   ON inbound_messages(sender_phone);
CREATE INDEX IF NOT EXISTS idx_inbound_device   ON inbound_messages(device_id);
CREATE INDEX IF NOT EXISTS idx_inbound_action   ON inbound_messages(inbound_action);
CREATE INDEX IF NOT EXISTS idx_inbound_risk     ON inbound_messages(sms_risk_score);

-- ------------------------------------------------------------
-- TRIGGER: keep phone_number_registry updated from inbound events.
--
-- DESIGN DECISION (explicit, not a formality):
--   A high-risk inbound SMS (sms_risk >= flag_threshold_used) flags
--   the sender in phone_number_registry even with ZERO transfer activity.
--
-- Rationale:
--   The phone number is the persistent attack vector. Phishers send
--   bulk SMS to probe victims before any transfer is attempted. If
--   detection is gated on transfer activity, the registry stays blind
--   to a number that has already sent hundreds of phishing messages
--   across the network. Once flagged from inbound-only events, a
--   subsequent transfer FROM or TO that number is intercepted at
--   assess_transfer_risk() before the ML models even run (via the
--   get_phone_reputation() pre-check).
--
-- Trigger condition: inbound_action = 'FLAGGED_SENDER' only.
--   LOGGED_ONLY rows do NOT touch the registry — they are audit
--   trail entries for sub-threshold messages with no reputation impact.
--
-- Registry status escalation:
--   First flag: WATCH
--   times_blocked contributes 0 (no transfer was blocked — inbound only)
--   current_status escalates to BLOCKED only if verdict = 'BLOCKED' on
--   a suspicious_messages row — inbound events can only reach WATCH.
-- ------------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS trg_inbound_update_registry
AFTER INSERT ON inbound_messages
WHEN NEW.inbound_action = 'FLAGGED_SENDER'
BEGIN
    INSERT INTO phone_number_registry (
        phone_number, times_flagged, times_blocked,
        cumulative_risk_score, avg_risk_score,
        current_status, first_flagged_at, last_flagged_at
    )
    VALUES (
        NEW.sender_phone,
        1,
        0,                           -- inbound events never directly block
        NEW.sms_risk_score,
        NEW.sms_risk_score,
        CASE WHEN NEW.sms_risk_score >= 0.5 THEN 'WATCH' ELSE 'CLEAR' END,
        NEW.received_at,
        NEW.received_at
    )
    ON CONFLICT(phone_number) DO UPDATE SET
        times_flagged         = times_flagged + 1,
        cumulative_risk_score = cumulative_risk_score + NEW.sms_risk_score,
        avg_risk_score        = (cumulative_risk_score + NEW.sms_risk_score) / (times_flagged + 1),
        current_status        = CASE
            WHEN current_status = 'BLOCKED' THEN 'BLOCKED'  -- never downgrade a block
            WHEN NEW.sms_risk_score >= 0.5 THEN 'WATCH'
            ELSE current_status
        END,
        last_flagged_at       = NEW.received_at;
END;

-- ------------------------------------------------------------
-- UPDATED VIEW: extend v_message_escalation_chain to include
-- inbound-only flagging events so the dashboard shows the
-- complete reputation build-up for a phone number.
-- ------------------------------------------------------------
DROP VIEW IF EXISTS v_message_escalation_chain;
CREATE VIEW v_message_escalation_chain AS
-- Transfer-linked detections (original)
SELECT
    sm.message_id       AS event_id,
    'TRANSFER'          AS event_type,
    sm.sender_phone     AS sender_phone,  -- populated below
    sm.receiver_phone   AS target_phone,
    sm.device_id,
    sm.provider_id,
    p.provider_name,
    sm.total_risk_score AS risk_score,
    sm.verdict          AS action,
    sm.detected_tier,
    sm.detected_at      AS event_at,
    te.from_tier,
    te.to_tier,
    te.reason           AS escalation_reason,
    te.escalated_at,
    pnr.times_flagged   AS phone_total_flags,
    pnr.current_status  AS phone_status
FROM suspicious_messages sm
-- suspicious_messages has no sender_phone column; use receiver_phone as target
-- (transfer direction: edge assessed the receiver, not the sender)
JOIN providers p ON p.provider_id = sm.provider_id
LEFT JOIN tier_escalations te  ON te.message_id  = sm.message_id
LEFT JOIN phone_number_registry pnr ON pnr.phone_number = sm.receiver_phone

UNION ALL

-- Inbound-only flagging events (new)
SELECT
    im.inbound_id       AS event_id,
    'INBOUND'           AS event_type,
    im.sender_phone     AS sender_phone,
    NULL                AS target_phone,
    im.device_id,
    im.provider_id,
    p.provider_name,
    im.sms_risk_score   AS risk_score,
    im.inbound_action   AS action,
    'EDGE'              AS detected_tier,
    im.received_at      AS event_at,
    NULL                AS from_tier,
    NULL                AS to_tier,
    NULL                AS escalation_reason,
    NULL                AS escalated_at,
    pnr.times_flagged   AS phone_total_flags,
    pnr.current_status  AS phone_status
FROM inbound_messages im
JOIN providers p ON p.provider_id = im.provider_id
LEFT JOIN phone_number_registry pnr ON pnr.phone_number = im.sender_phone

ORDER BY event_at DESC;
