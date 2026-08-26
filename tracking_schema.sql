-- ============================================================
-- ShieldFL — Suspicious Activity Tracking Schema
-- Persists what TRANSFERS_LOG currently only holds in memory:
-- individual detections, per-tier escalation history, and a
-- cross-tier phone-number reputation registry the FL framework
-- (NVFlare, recommended above) can train against and query.
--
-- Engine: written portable SQL; tested mentally against SQLite
-- (matches your existing users.db pattern) — swap AUTOINCREMENT
-- for SERIAL/IDENTITY if you move to Postgres later.
-- ============================================================

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- TIER 1: EDGE — registry of edge devices (S1-0, S2-1, etc.)
-- Mirrors your in-memory DEVICES dict so detections can be
-- traced back to a specific edge node.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS edge_devices (
    device_id       TEXT PRIMARY KEY,          -- e.g. 'S1-0'
    provider_id     TEXT NOT NULL,             -- e.g. 'S1'
    stakeholder_type TEXT DEFAULT 'Subscriber',
    risk_tier       TEXT CHECK (risk_tier IN ('HIGH', 'MEDIUM', 'LOW')),
    created_at      TEXT DEFAULT (datetime('now')),
    last_active_at  TEXT
);

-- ------------------------------------------------------------
-- TIER 2: PROVIDER — MTN / Telecel / AirtelTigo
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS providers (
    provider_id     TEXT PRIMARY KEY,          -- 'S1', 'S2', 'S3'
    provider_name   TEXT NOT NULL,             -- 'MTN', 'Telecel', 'AirtelTigo'
    escrow_threshold REAL DEFAULT 0.65
);

-- ------------------------------------------------------------
-- TIER 3: GLOBAL — model versions produced by each FL round.
-- Lets you trace which model version flagged which message —
-- essential for "learn from" auditability and for rolling back
-- if a round degrades accuracy.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS global_model_versions (
    round_id            INTEGER PRIMARY KEY,
    fl_algorithm        TEXT,                  -- fedprox / fedopt / fedavg
    sms_accuracy        REAL,
    call_accuracy       REAL,
    holdout_sms_accuracy REAL,
    dp_enabled          INTEGER DEFAULT 0,
    dp_noise            REAL,
    checkpoint_path     TEXT,                  -- e.g. models_checkpoint/global_sms_model.pth
    trained_at          TEXT DEFAULT (datetime('now'))
);

-- ------------------------------------------------------------
-- CORE FACT TABLE: every assessed message/transfer.
-- This is the persisted version of TRANSFERS_LOG.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS suspicious_messages (
    message_id          TEXT PRIMARY KEY,      -- reuse transfer_id (uuid4[:8])
    device_id           TEXT NOT NULL REFERENCES edge_devices(device_id),
    provider_id         TEXT NOT NULL REFERENCES providers(provider_id),
    receiver_phone      TEXT NOT NULL,
    message_text_hash   TEXT NOT NULL,         -- SHA-256 of message body — never store raw
                                                -- SMS text in this shared table; keep raw
                                                -- content local to the edge device to
                                                -- preserve the FL privacy guarantee.
    amount              REAL,
    sms_risk_score      REAL,
    contact_risk_score  REAL,
    amount_risk_score   REAL,
    total_risk_score    REAL NOT NULL,
    verdict             TEXT NOT NULL CHECK (verdict IN
                            ('APPROVED', 'HELD_IN_ESCROW', 'RELEASED_FROM_ESCROW', 'BLOCKED')),
    detected_tier       TEXT NOT NULL CHECK (detected_tier IN ('EDGE', 'PROVIDER', 'GLOBAL')),
    model_round_id      INTEGER REFERENCES global_model_versions(round_id),
    is_ground_truth_spam INTEGER,              -- nullable: only known in simulation mode
    detected_at         TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_phone ON suspicious_messages(receiver_phone);
CREATE INDEX IF NOT EXISTS idx_messages_device ON suspicious_messages(device_id);
CREATE INDEX IF NOT EXISTS idx_messages_verdict ON suspicious_messages(verdict);

-- ------------------------------------------------------------
-- ESCALATION TRAIL: how a single message's decision moved
-- between tiers (edge auto-approve -> provider escrow ->
-- global block, etc.). One row per hand-off.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tier_escalations (
    escalation_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id          TEXT NOT NULL REFERENCES suspicious_messages(message_id),
    from_tier           TEXT NOT NULL CHECK (from_tier IN ('EDGE', 'PROVIDER', 'GLOBAL')),
    to_tier             TEXT NOT NULL CHECK (to_tier IN ('PROVIDER', 'GLOBAL')),
    reason              TEXT,                  -- e.g. 'risk score exceeds provider threshold'
    escalated_at        TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_escalations_message ON tier_escalations(message_id);

-- ------------------------------------------------------------
-- PHONE NUMBER REPUTATION REGISTRY — the "affiliated numbers"
-- ledger. This is the cross-tier shared artifact: it contains
-- NO message content, only aggregated risk about a number, so
-- it can safely be shared/synced across providers and the
-- global tier without violating per-client data privacy.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS phone_number_registry (
    phone_number         TEXT PRIMARY KEY,
    times_flagged        INTEGER DEFAULT 0,
    times_blocked        INTEGER DEFAULT 0,
    cumulative_risk_score REAL DEFAULT 0.0,
    avg_risk_score        REAL DEFAULT 0.0,
    current_status        TEXT DEFAULT 'WATCH'
                            CHECK (current_status IN ('CLEAR', 'WATCH', 'BLOCKED')),
    first_flagged_at      TEXT,
    last_flagged_at       TEXT,
    escalated_to_global   INTEGER DEFAULT 0    -- 1 once seen across 2+ providers
);

CREATE INDEX IF NOT EXISTS idx_registry_status ON phone_number_registry(current_status);

-- ------------------------------------------------------------
-- TRIGGER: keep phone_number_registry in sync automatically
-- whenever a new suspicious_messages row lands. This is what
-- lets the registry feed the AI framework's training pipeline
-- without a separate ETL job.
-- ------------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS trg_update_registry
AFTER INSERT ON suspicious_messages
BEGIN
    INSERT INTO phone_number_registry (
        phone_number, times_flagged, times_blocked,
        cumulative_risk_score, avg_risk_score,
        current_status, first_flagged_at, last_flagged_at
    )
    VALUES (
        NEW.receiver_phone,
        1,
        CASE WHEN NEW.verdict = 'BLOCKED' THEN 1 ELSE 0 END,
        NEW.total_risk_score,
        NEW.total_risk_score,
        CASE WHEN NEW.verdict = 'BLOCKED' THEN 'BLOCKED'
             WHEN NEW.total_risk_score >= 0.5 THEN 'WATCH'
             ELSE 'CLEAR' END,
        NEW.detected_at,
        NEW.detected_at
    )
    ON CONFLICT(phone_number) DO UPDATE SET
        times_flagged = times_flagged + 1,
        times_blocked = times_blocked + (CASE WHEN NEW.verdict = 'BLOCKED' THEN 1 ELSE 0 END),
        cumulative_risk_score = cumulative_risk_score + NEW.total_risk_score,
        avg_risk_score = (cumulative_risk_score + NEW.total_risk_score) / (times_flagged + 1),
        current_status = CASE
            WHEN NEW.verdict = 'BLOCKED' OR current_status = 'BLOCKED' THEN 'BLOCKED'
            WHEN NEW.total_risk_score >= 0.5 OR current_status = 'WATCH' THEN 'WATCH'
            ELSE 'CLEAR'
        END,
        last_flagged_at = NEW.detected_at;
END;

-- ------------------------------------------------------------
-- VIEW: full escalation chain for a message, joined with tier
-- context — the query the frontend/API layer will use most.
-- ------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_message_escalation_chain AS
SELECT
    sm.message_id,
    sm.receiver_phone,
    sm.device_id,
    sm.provider_id,
    p.provider_name,
    sm.total_risk_score,
    sm.verdict,
    sm.detected_tier,
    sm.detected_at,
    te.from_tier,
    te.to_tier,
    te.reason AS escalation_reason,
    te.escalated_at,
    pnr.times_flagged AS phone_total_flags,
    pnr.current_status AS phone_status
FROM suspicious_messages sm
JOIN providers p ON p.provider_id = sm.provider_id
LEFT JOIN tier_escalations te ON te.message_id = sm.message_id
LEFT JOIN phone_number_registry pnr ON pnr.phone_number = sm.receiver_phone
ORDER BY sm.detected_at DESC;

-- ------------------------------------------------------------
-- Seed the three known providers (safe to re-run: INSERT OR IGNORE)
-- ------------------------------------------------------------
INSERT OR IGNORE INTO providers (provider_id, provider_name, escrow_threshold) VALUES
    ('S1', 'MTN', 0.65),
    ('S2', 'Telecel', 0.65),
    ('S3', 'AirtelTigo', 0.65);
