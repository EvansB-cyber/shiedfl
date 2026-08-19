# edge/feedback.py
def record_feedback(sms_id: str, sms_text: str, label: int, db_conn):
    sms_hash = hashlib.sha256(sms_text.encode()).hexdigest()
    db_conn.execute(
        "INSERT INTO local_sms_labels (sms_id, sms_hash, label) VALUES (?, ?, ?)",
        (sms_id, sms_hash, label)
    )
    db_conn.commit()