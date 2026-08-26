import sqlite3
import os
import hashlib

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tracking.db")


def _connect():
    return sqlite3.connect(DB_PATH)


def ensure_device(device_id: str, provider_id: str, stakeholder_type: str = "Subscriber"):
    """Idempotent — call this once per device at startup, or lazily on first sighting."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO edge_devices (device_id, provider_id, stakeholder_type, last_active_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(device_id) DO UPDATE SET last_active_at = datetime('now')
        """,
        (device_id, provider_id, stakeholder_type),
    )
    conn.commit()
    conn.close()


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
    The phone_number_registry table updates itself automatically
    via the trg_update_registry trigger defined in tracking_schema.sql —
    no separate call needed for that part.

    message_text is hashed here, never stored raw, so the shared
    tracking DB never holds actual SMS content (kept local to the
    edge device that received it — see schema comments).
    """
    text_hash = hashlib.sha256(message_text.encode("utf-8")).hexdigest()

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO suspicious_messages (
            message_id, device_id, provider_id, receiver_phone,
            message_text_hash, amount,
            sms_risk_score, contact_risk_score, amount_risk_score, total_risk_score,
            verdict, detected_tier, model_round_id, is_ground_truth_spam
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            device_id,
            provider_id,
            receiver_phone,
            text_hash,
            amount,
            risk_report.get("sms_risk_score"),
            risk_report.get("contact_risk_score"),
            risk_report.get("amount_risk_score"),
            risk_report.get("total_risk_score"),
            verdict,
            detected_tier,
            model_round_id,
            is_ground_truth_spam,
        ),
    )
    conn.commit()
    conn.close()


def log_tier_escalation(message_id: str, from_tier: str, to_tier: str, reason: str):
    """Call this whenever a message's status changes tier — e.g. edge auto-approved
    it but the provider later escrows it based on the dynamic RISK_THRESHOLD."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO tier_escalations (message_id, from_tier, to_tier, reason)
        VALUES (?, ?, ?, ?)
        """,
        (message_id, from_tier, to_tier, reason),
    )
    conn.commit()
    conn.close()


def log_model_version(round_id: int, fl_algorithm: str, metrics: dict, checkpoint_path: str = None,
                       dp_enabled: bool = False, dp_noise: float = None):
    """Call this once per completed FL round (from /api/federated/round) so every
    detection can be traced back to the exact model version that made the call."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO global_model_versions (
            round_id, fl_algorithm, sms_accuracy, call_accuracy,
            holdout_sms_accuracy, dp_enabled, dp_noise, checkpoint_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            round_id,
            fl_algorithm,
            metrics.get("sms_accuracy"),
            metrics.get("call_accuracy"),
            metrics.get("holdout_sms_accuracy"),
            int(dp_enabled),
            dp_noise,
            checkpoint_path,
        ),
    )
    conn.commit()
    conn.close()


def get_phone_reputation(phone_number: str):
    """Quick lookup the risk engine can call before even running the ML models —
    a number already BLOCKED elsewhere in the network doesn't need re-scoring."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT phone_number, times_flagged, times_blocked, avg_risk_score, current_status "
        "FROM phone_number_registry WHERE phone_number = ?",
        (phone_number,),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "phone_number": row[0],
            "times_flagged": row[1],
            "times_blocked": row[2],
            "avg_risk_score": row[3],
            "current_status": row[4],
        }
    return None