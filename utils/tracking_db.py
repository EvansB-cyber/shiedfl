"""
ShieldFL — Shared Tracking Database Layer (v2)

Provides a clean Python API over tracking.db.  Every function is a thin
wrapper that opens a connection, executes one or two statements, and closes.
No connection pooling is needed at this scale; SQLite handles concurrent
reads fine and writes are infrequent (one per FL round / per intercepted
message).

Schema is self-managed: _ensure_schema() runs once at import time and is
idempotent — safe to re-import after a schema update.

v2 additions (Step 4 — Inbound Message Pipeline):
  - inbound_messages table
  - trg_inbound_update_registry trigger
  - log_inbound_message() function
  - upsert_phone_reputation() direct-write helper
  - get_inbound_stats() for the test / verification path
"""
import sqlite3
import os
import hashlib

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tracking.db")

# ---------------------------------------------------------------------------
# Inbound flagging threshold — DESIGN DECISION (see receive_inbound() in
# edge_device.py for full rationale):
#   sms_risk >= INBOUND_FLAG_THRESHOLD  → inbound_action = 'FLAGGED_SENDER'
#                                          sender entered into phone_number_registry
#   sms_risk <  INBOUND_FLAG_THRESHOLD  → inbound_action = 'LOGGED_ONLY'
#                                          audit trail only, no reputation impact
#
# Set to 0.75 (high-confidence phishing) rather than 0.5 (borderline) to
# avoid false-positives poisoning the cross-device reputation ledger.
# A number that scores 0.6 once might just be aggressive marketing — it
# should not be network-wide WATCH after a single message.
INBOUND_FLAG_THRESHOLD: float = 0.75
# ---------------------------------------------------------------------------


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # allows concurrent readers
    return conn


# ---------------------------------------------------------------------------
# Self-managed schema — runs once at import, idempotent.
# ---------------------------------------------------------------------------
_SCHEMA_APPLIED = False

def _ensure_schema():
    """
    Applies the base schema + v2 migration in a single transaction.
    All statements use IF NOT EXISTS / OR IGNORE so re-running is safe.
    Called automatically on the first _connect() that needs a table.
    """
    global _SCHEMA_APPLIED
    if _SCHEMA_APPLIED:
        return

    conn = _connect()
    conn.executescript("""
    PRAGMA foreign_keys = ON;

    -- ── Base tables (originally in tracking_schema.sql) ──────────────────
    CREATE TABLE IF NOT EXISTS edge_devices (
        device_id        TEXT PRIMARY KEY,
        provider_id      TEXT NOT NULL,
        stakeholder_type TEXT DEFAULT 'Subscriber',
        risk_tier        TEXT CHECK (risk_tier IN ('HIGH', 'MEDIUM', 'LOW')),
        created_at       TEXT DEFAULT (datetime('now')),
        last_active_at   TEXT
    );

    CREATE TABLE IF NOT EXISTS providers (
        provider_id    TEXT PRIMARY KEY,
        provider_name  TEXT NOT NULL,
        escrow_threshold REAL DEFAULT 0.65
    );
    INSERT OR IGNORE INTO providers (provider_id, provider_name, escrow_threshold)
        VALUES ('S1','MTN',0.65),('S2','Telecel',0.65),('S3','AirtelTigo',0.65);

    CREATE TABLE IF NOT EXISTS global_model_versions (
        round_id             INTEGER PRIMARY KEY,
        fl_algorithm         TEXT,
        sms_accuracy         REAL,
        call_accuracy        REAL,
        holdout_sms_accuracy REAL,
        dp_enabled           INTEGER DEFAULT 0,
        dp_noise             REAL,
        checkpoint_path      TEXT,
        trained_at           TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS suspicious_messages (
        message_id          TEXT PRIMARY KEY,
        device_id           TEXT NOT NULL REFERENCES edge_devices(device_id),
        provider_id         TEXT NOT NULL REFERENCES providers(provider_id),
        receiver_phone      TEXT NOT NULL,
        message_text_hash   TEXT NOT NULL,
        amount              REAL,
        sms_risk_score      REAL,
        contact_risk_score  REAL,
        amount_risk_score   REAL,
        total_risk_score    REAL NOT NULL,
        verdict             TEXT NOT NULL CHECK (verdict IN
                                ('AUTO_APPROVE','AUTO_BLOCK','MANUAL_REVIEW',
                                 'APPROVED','HELD_IN_ESCROW','RELEASED_FROM_ESCROW','BLOCKED')),
        detected_tier       TEXT NOT NULL CHECK (detected_tier IN ('EDGE','PROVIDER','GLOBAL')),
        model_round_id      INTEGER REFERENCES global_model_versions(round_id),
        is_ground_truth_spam INTEGER,
        detected_at         TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_messages_phone   ON suspicious_messages(receiver_phone);
    CREATE INDEX IF NOT EXISTS idx_messages_device  ON suspicious_messages(device_id);
    CREATE INDEX IF NOT EXISTS idx_messages_verdict ON suspicious_messages(verdict);

    CREATE TABLE IF NOT EXISTS tier_escalations (
        escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id    TEXT NOT NULL REFERENCES suspicious_messages(message_id),
        from_tier     TEXT NOT NULL CHECK (from_tier IN ('EDGE','PROVIDER','GLOBAL')),
        to_tier       TEXT NOT NULL CHECK (to_tier   IN ('PROVIDER','GLOBAL')),
        reason        TEXT,
        escalated_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_escalations_message ON tier_escalations(message_id);

    CREATE TABLE IF NOT EXISTS phone_number_registry (
        phone_number          TEXT PRIMARY KEY,
        times_flagged         INTEGER DEFAULT 0,
        times_blocked         INTEGER DEFAULT 0,
        cumulative_risk_score REAL    DEFAULT 0.0,
        avg_risk_score        REAL    DEFAULT 0.0,
        current_status        TEXT    DEFAULT 'WATCH'
                                CHECK (current_status IN ('CLEAR','WATCH','BLOCKED')),
        first_flagged_at      TEXT,
        last_flagged_at       TEXT,
        escalated_to_global   INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_registry_status ON phone_number_registry(current_status);

    CREATE TRIGGER IF NOT EXISTS trg_update_registry
    AFTER INSERT ON suspicious_messages
    BEGIN
        INSERT INTO phone_number_registry (
            phone_number, times_flagged, times_blocked,
            cumulative_risk_score, avg_risk_score,
            current_status, first_flagged_at, last_flagged_at
        ) VALUES (
            NEW.receiver_phone, 1,
            CASE WHEN NEW.verdict = 'BLOCKED' OR NEW.verdict = 'AUTO_BLOCK' THEN 1 ELSE 0 END,
            NEW.total_risk_score, NEW.total_risk_score,
            CASE WHEN NEW.verdict IN ('BLOCKED','AUTO_BLOCK') THEN 'BLOCKED'
                 WHEN NEW.total_risk_score >= 0.5 THEN 'WATCH'
                 ELSE 'CLEAR' END,
            NEW.detected_at, NEW.detected_at
        )
        ON CONFLICT(phone_number) DO UPDATE SET
            times_flagged         = times_flagged + 1,
            times_blocked         = times_blocked + (
                CASE WHEN NEW.verdict IN ('BLOCKED','AUTO_BLOCK') THEN 1 ELSE 0 END),
            cumulative_risk_score = cumulative_risk_score + NEW.total_risk_score,
            avg_risk_score        = (cumulative_risk_score + NEW.total_risk_score)
                                    / (times_flagged + 1),
            current_status        = CASE
                WHEN NEW.verdict IN ('BLOCKED','AUTO_BLOCK') OR current_status = 'BLOCKED'
                     THEN 'BLOCKED'
                WHEN NEW.total_risk_score >= 0.5 OR current_status = 'WATCH' THEN 'WATCH'
                ELSE 'CLEAR' END,
            last_flagged_at       = NEW.detected_at;
    END;

    -- ── v2: inbound_messages (Step 4) ─────────────────────────────────────
    CREATE TABLE IF NOT EXISTS inbound_messages (
        inbound_id          TEXT PRIMARY KEY,
        device_id           TEXT NOT NULL REFERENCES edge_devices(device_id),
        provider_id         TEXT NOT NULL REFERENCES providers(provider_id),
        sender_phone        TEXT NOT NULL,
        message_text_hash   TEXT NOT NULL,
        sms_risk_score      REAL NOT NULL,
        inbound_action      TEXT NOT NULL CHECK (inbound_action IN
                                ('FLAGGED_SENDER', 'LOGGED_ONLY')),
        flag_threshold_used REAL NOT NULL,
        model_round_id      INTEGER REFERENCES global_model_versions(round_id),
        is_ground_truth_spam INTEGER,
        received_at         TEXT NOT NULL,
        created_at          TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_inbound_sender ON inbound_messages(sender_phone);
    CREATE INDEX IF NOT EXISTS idx_inbound_device ON inbound_messages(device_id);
    CREATE INDEX IF NOT EXISTS idx_inbound_action ON inbound_messages(inbound_action);
    CREATE INDEX IF NOT EXISTS idx_inbound_risk   ON inbound_messages(sms_risk_score);

    CREATE TRIGGER IF NOT EXISTS trg_inbound_update_registry
    AFTER INSERT ON inbound_messages
    WHEN NEW.inbound_action = 'FLAGGED_SENDER'
    BEGIN
        INSERT INTO phone_number_registry (
            phone_number, times_flagged, times_blocked,
            cumulative_risk_score, avg_risk_score,
            current_status, first_flagged_at, last_flagged_at
        ) VALUES (
            NEW.sender_phone, 1, 0,
            NEW.sms_risk_score, NEW.sms_risk_score,
            CASE WHEN NEW.sms_risk_score >= 0.5 THEN 'WATCH' ELSE 'CLEAR' END,
            NEW.received_at, NEW.received_at
        )
        ON CONFLICT(phone_number) DO UPDATE SET
            times_flagged         = times_flagged + 1,
            cumulative_risk_score = cumulative_risk_score + NEW.sms_risk_score,
            avg_risk_score        = (cumulative_risk_score + NEW.sms_risk_score)
                                    / (times_flagged + 1),
            current_status        = CASE
                WHEN current_status = 'BLOCKED' THEN 'BLOCKED'
                WHEN NEW.sms_risk_score >= 0.5  THEN 'WATCH'
                ELSE current_status END,
            last_flagged_at       = NEW.received_at;
    END;
    """)
    conn.close()
    _SCHEMA_APPLIED = True


_ensure_schema()


# ---------------------------------------------------------------------------
# Device registration
# ---------------------------------------------------------------------------

def ensure_device(device_id: str, provider_id: str, stakeholder_type: str = "Subscriber"):
    """Idempotent — call once per device at startup, or lazily on first sighting."""
    conn = _connect()
    conn.execute(
        """
        INSERT INTO edge_devices (device_id, provider_id, stakeholder_type, last_active_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(device_id) DO UPDATE SET last_active_at = datetime('now')
        """,
        (device_id, provider_id, stakeholder_type),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Transfer-linked detection (original)
# ---------------------------------------------------------------------------

def log_suspicious_message(
    message_id: str,
    device_id: str,
    provider_id: str,
    receiver_phone: str,
    message_text: str,
    amount: float,
    risk_report: dict,
    verdict: str,
    detected_tier: str,
    model_round_id: int = None,
    is_ground_truth_spam: bool = None,
):
    """
    Persists one assessed transfer into suspicious_messages.

    trg_update_registry fires automatically after INSERT and keeps
    phone_number_registry in sync — no separate call needed.

    message_text is SHA-256 hashed here; raw SMS text is never stored
    in the shared tracking DB (kept local to the edge device).
    """
    text_hash = hashlib.sha256(message_text.encode("utf-8")).hexdigest()
    conn = _connect()
    conn.execute(
        """
        INSERT OR IGNORE INTO suspicious_messages (
            message_id, device_id, provider_id, receiver_phone,
            message_text_hash, amount,
            sms_risk_score, contact_risk_score, amount_risk_score, total_risk_score,
            verdict, detected_tier, model_round_id, is_ground_truth_spam
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id, device_id, provider_id, receiver_phone,
            text_hash, amount,
            risk_report.get("sms_risk_score"),
            risk_report.get("contact_risk_score"),
            risk_report.get("amount_risk_score"),
            risk_report.get("total_risk_score"),
            verdict, detected_tier, model_round_id,
            int(is_ground_truth_spam) if is_ground_truth_spam is not None else None,
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tier escalation trail
# ---------------------------------------------------------------------------

def log_tier_escalation(message_id: str, from_tier: str, to_tier: str, reason: str):
    """One row per tier hand-off for a given message."""
    conn = _connect()
    conn.execute(
        "INSERT INTO tier_escalations (message_id, from_tier, to_tier, reason) VALUES (?,?,?,?)",
        (message_id, from_tier, to_tier, reason),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# FL round model version
# ---------------------------------------------------------------------------

def log_model_version(
    round_id: int,
    fl_algorithm: str,
    metrics: dict,
    checkpoint_path: str = None,
    dp_enabled: bool = False,
    dp_noise: float = None,
):
    """Call once per completed FL round so every detection is traceable."""
    conn = _connect()
    conn.execute(
        """
        INSERT OR REPLACE INTO global_model_versions (
            round_id, fl_algorithm, sms_accuracy, call_accuracy,
            holdout_sms_accuracy, dp_enabled, dp_noise, checkpoint_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            round_id, fl_algorithm,
            metrics.get("sms_accuracy"),
            metrics.get("call_accuracy"),
            metrics.get("holdout_sms_accuracy"),
            int(dp_enabled), dp_noise, checkpoint_path,
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# v2 — Inbound message path (Step 4, no transfer attached)
# ---------------------------------------------------------------------------

def log_inbound_message(
    inbound_id: str,
    device_id: str,
    provider_id: str,
    sender_phone: str,
    message_text: str,
    sms_risk_score: float,
    received_at: str,
    model_round_id: int = None,
    is_ground_truth_spam: bool = None,
    flag_threshold: float = INBOUND_FLAG_THRESHOLD,
) -> str:
    """
    Persists one inbound SMS (no transfer attached) to inbound_messages.

    Design decision enforced here:
      sms_risk >= flag_threshold  → inbound_action = 'FLAGGED_SENDER'
                                    trg_inbound_update_registry fires and
                                    upserts the sender into phone_number_registry.
      sms_risk <  flag_threshold  → inbound_action = 'LOGGED_ONLY'
                                    Row written for audit trail only; registry
                                    is NOT touched (avoiding false-positive pollution).

    The flag_threshold default (INBOUND_FLAG_THRESHOLD = 0.75) is intentionally
    higher than the transfer-path threshold (0.5 in the trigger) because:
      - Inbound-only events have no transfer amount to corroborate the risk.
      - A single ML score without amount/contact context is noisier.
      - False positives here affect the network-wide reputation ledger for ALL
        providers, not just one device — the cost of a wrong WATCH entry is higher.

    Args:
        inbound_id:           uuid4 hex (caller-generated).
        device_id:            Edge device that received the message.
        provider_id:          Provider owning that device.
        sender_phone:         MSISDN of the sender (the number being assessed).
        message_text:         Raw SMS body — SHA-256 hashed before storage.
        sms_risk_score:       Output of SMSFraudCNN (0.0–1.0).
        received_at:          ISO-8601 UTC timestamp from the edge device clock.
        model_round_id:       Current FL round, or None.
        is_ground_truth_spam: Ground-truth label (simulation mode only).
        flag_threshold:       Override the module default if needed.

    Returns:
        The inbound_action applied: 'FLAGGED_SENDER' or 'LOGGED_ONLY'.
    """
    text_hash = hashlib.sha256(message_text.encode("utf-8")).hexdigest()
    inbound_action = (
        "FLAGGED_SENDER" if sms_risk_score >= flag_threshold else "LOGGED_ONLY"
    )

    conn = _connect()
    conn.execute(
        """
        INSERT OR IGNORE INTO inbound_messages (
            inbound_id, device_id, provider_id, sender_phone,
            message_text_hash, sms_risk_score,
            inbound_action, flag_threshold_used,
            model_round_id, is_ground_truth_spam, received_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            inbound_id, device_id, provider_id, sender_phone,
            text_hash, sms_risk_score,
            inbound_action, flag_threshold,
            model_round_id,
            int(is_ground_truth_spam) if is_ground_truth_spam is not None else None,
            received_at,
        ),
    )
    conn.commit()
    conn.close()

    return inbound_action


def upsert_phone_reputation(
    phone_number: str,
    risk_score: float,
    verdict: str,
    event_at: str,
):
    """
    Direct Python upsert into phone_number_registry — used when the caller
    needs to update reputation WITHOUT inserting into suspicious_messages
    or inbound_messages (e.g. manual admin override, test fixture setup).

    Under normal operation the triggers handle this automatically.
    """
    conn = _connect()
    conn.execute(
        """
        INSERT INTO phone_number_registry (
            phone_number, times_flagged, times_blocked,
            cumulative_risk_score, avg_risk_score,
            current_status, first_flagged_at, last_flagged_at
        ) VALUES (?, 1, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(phone_number) DO UPDATE SET
            times_flagged         = times_flagged + 1,
            times_blocked         = times_blocked + excluded.times_blocked,
            cumulative_risk_score = cumulative_risk_score + excluded.cumulative_risk_score,
            avg_risk_score        = (cumulative_risk_score + excluded.cumulative_risk_score)
                                    / (times_flagged + 1),
            current_status        = CASE
                WHEN excluded.current_status = 'BLOCKED' OR current_status = 'BLOCKED'
                     THEN 'BLOCKED'
                WHEN excluded.current_status = 'WATCH'   OR current_status = 'WATCH'
                     THEN 'WATCH'
                ELSE 'CLEAR' END,
            last_flagged_at       = excluded.last_flagged_at
        """,
        (
            phone_number,
            1 if verdict in ("BLOCKED", "AUTO_BLOCK") else 0,
            risk_score,
            risk_score,
            "BLOCKED" if verdict in ("BLOCKED", "AUTO_BLOCK") else
            ("WATCH" if risk_score >= 0.5 else "CLEAR"),
            event_at,
            event_at,
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_phone_reputation(phone_number: str) -> dict | None:
    """
    Quick reputation lookup — call this BEFORE running ML models.
    A number already BLOCKED network-wide doesn't need re-scoring.
    """
    conn = _connect()
    row = conn.execute(
        "SELECT phone_number, times_flagged, times_blocked, avg_risk_score, current_status "
        "FROM phone_number_registry WHERE phone_number = ?",
        (phone_number,),
    ).fetchone()
    conn.close()
    if row:
        return {
            "phone_number":  row[0],
            "times_flagged": row[1],
            "times_blocked": row[2],
            "avg_risk_score": row[3],
            "current_status": row[4],
        }
    return None


def get_inbound_stats(sender_phone: str = None) -> dict:
    """
    Returns summary counts from inbound_messages.
    Used by the test / verification path to confirm rows were written.

    Args:
        sender_phone: Filter to a specific number, or None for totals.

    Returns:
        {
          "total":          int,
          "flagged_sender": int,
          "logged_only":    int,
          "avg_risk":       float | None,
          "rows":           list[dict]   # most recent 20
        }
    """
    conn = _connect()
    where = "WHERE sender_phone = ?" if sender_phone else ""
    params = (sender_phone,) if sender_phone else ()

    total     = conn.execute(f"SELECT COUNT(*) FROM inbound_messages {where}", params).fetchone()[0]
    flagged   = conn.execute(
        f"SELECT COUNT(*) FROM inbound_messages {where} {'AND' if sender_phone else 'WHERE'} "
        f"inbound_action='FLAGGED_SENDER'" if sender_phone else
        "SELECT COUNT(*) FROM inbound_messages WHERE inbound_action='FLAGGED_SENDER'",
        params if sender_phone else ()
    ).fetchone()[0]
    avg_risk  = conn.execute(
        f"SELECT AVG(sms_risk_score) FROM inbound_messages {where}", params
    ).fetchone()[0]

    rows_raw = conn.execute(
        f"SELECT inbound_id, sender_phone, sms_risk_score, inbound_action, received_at "
        f"FROM inbound_messages {where} ORDER BY received_at DESC LIMIT 20",
        params,
    ).fetchall()
    conn.close()

    rows = [
        {"inbound_id": r[0], "sender_phone": r[1], "sms_risk_score": r[2],
         "inbound_action": r[3], "received_at": r[4]}
        for r in rows_raw
    ]
    return {
        "total":          total,
        "flagged_sender": flagged,
        "logged_only":    total - flagged,
        "avg_risk":       round(avg_risk, 4) if avg_risk is not None else None,
        "rows":           rows,
    }