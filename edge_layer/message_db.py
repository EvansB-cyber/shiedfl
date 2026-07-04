import sqlite3
import os

class MessageDatabase:
    """
    SQLite wrapper for managing messages history records on a specific edge device database.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_phone TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    is_spam INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    def seed_initial_messages(self, seed_data):
        """
        Populates messages table with initial logs if empty.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages")
            if cursor.fetchone()[0] == 0:
                cursor.executemany("""
                    INSERT INTO messages (sender_phone, message_text, timestamp, is_spam)
                    VALUES (?, ?, ?, ?)
                """, seed_data)
                conn.commit()

    def get_all_messages(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, sender_phone, message_text, timestamp, is_spam FROM messages ORDER BY id DESC")
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "sender_phone": r[1],
                    "message_text": r[2],
                    "timestamp": r[3],
                    "is_spam": bool(r[4])
                }
                for r in rows
            ]

    def add_message(self, sender_phone, message_text, timestamp, is_spam=False):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (sender_phone, message_text, timestamp, is_spam)
                VALUES (?, ?, ?, ?)
            """, (sender_phone, message_text, timestamp, 1 if is_spam else 0))
            conn.commit()
            return cursor.lastrowid
            
    def get_message_counts(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM messages WHERE is_spam = 1")
            spam = cursor.fetchone()[0]
            return {"total": total, "spam": spam}
