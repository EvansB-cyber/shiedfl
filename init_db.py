import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "edge_layer", "databases")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS local_sms_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sms_id TEXT NOT NULL,
    sms_hash TEXT NOT NULL,
    label INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _ensure_db_path(db_path: str) -> str:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return db_path


def create_tables(db_path: str | None = None):
    """
    Creates the local_sms_labels table in the configured database.

    If no path is supplied, it creates the table in every SQLite database already
    used by the edge layer so the feedback insert path matches the runtime DB files.
    """
    if db_path:
        db_files = [_ensure_db_path(db_path)]
    else:
        os.makedirs(DB_DIR, exist_ok=True)
        db_files = [
            os.path.join(DB_DIR, name)
            for name in sorted(os.listdir(DB_DIR))
            if name.endswith(".db")
        ]
        if not db_files:
            db_files = [_ensure_db_path(os.path.join(DB_DIR, "feedback.db"))]

    created = []
    for path in db_files:
        conn = sqlite3.connect(path)
        try:
            conn.execute(CREATE_TABLE_SQL)
            conn.commit()
            created.append(path)
        finally:
            conn.close()
    return created


if __name__ == "__main__":
    created = create_tables()
    print(f"Ensured local_sms_labels table exists in: {created}")