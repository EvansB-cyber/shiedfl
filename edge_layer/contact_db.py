import sqlite3
import os

class ContactDatabase:
    """
    SQLite wrapper for managing contact records on a specific edge device database.
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
                CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    phone TEXT UNIQUE NOT NULL,
                    is_trusted INTEGER NOT NULL DEFAULT 1,
                    risk_score REAL NOT NULL DEFAULT 0.0
                )
            """)
            conn.commit()

    def seed_initial_contacts(self, seed_data):
        """
        Populates contacts table with initial list of contacts if it is currently empty.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM contacts")
            if cursor.fetchone()[0] == 0:
                cursor.executemany("""
                    INSERT INTO contacts (name, phone, is_trusted, risk_score)
                    VALUES (?, ?, ?, ?)
                """, seed_data)
                conn.commit()

    def get_all_contacts(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, phone, is_trusted, risk_score FROM contacts")
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "phone": r[2],
                    "is_trusted": bool(r[3]),
                    "risk_score": r[4]
                }
                for r in rows
            ]

    def get_contact_by_phone(self, phone):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, phone, is_trusted, risk_score FROM contacts WHERE phone = ?", (phone,))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "phone": row[2],
                    "is_trusted": bool(row[3]),
                    "risk_score": row[4]
                }
            return None

    def add_contact(self, name, phone, is_trusted=True, risk_score=0.0):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO contacts (name, phone, is_trusted, risk_score)
                    VALUES (?, ?, ?, ?)
                """, (name, phone, 1 if is_trusted else 0, risk_score))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Phone number already exists
            return False

    def delete_contact(self, contact_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
            conn.commit()
            return cursor.rowcount > 0

    def update_risk_score(self, phone, risk_score):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE contacts SET risk_score = ? WHERE phone = ?", (risk_score, phone))
            conn.commit()
