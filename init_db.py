CREATE TABLE IF NOT EXISTS local_sms_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sms_id TEXT NOT NULL,
    sms_hash TEXT NOT NULL,  -- hash, not raw text, if this table is ever inspected/exported
    label INTEGER NOT NULL,  -- 1 = confirmed smishing, 0 = false positive
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);