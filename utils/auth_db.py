import sqlite3
import os
from passlib.context import CryptContext

# FIXED — uses Render's /data/users.db in production, local path in dev
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "users.db")
)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pre-registered machine identities for edge nodes and providers
DEVICE_ACCOUNTS = {
    "edge-node": {"password": "edge-secret-2026", "role": "device"},
    "provider-node": {"password": "provider-secret-2026", "role": "provider"},
}

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def init_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'admin'
        )
    """)

    cursor.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]
    if "role" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'admin'")

    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        admin_user = os.environ.get("ADMIN_USERNAME", "admin")
        admin_pass = os.environ.get("ADMIN_PASSWORD", "password")
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (admin_user, get_password_hash(admin_pass), "admin")
        )

    conn.commit()
    conn.close()

def get_user(username: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "password_hash": row[2], "role": row[3]}
    return None

def authenticate(username: str, password: str):
    if username in DEVICE_ACCOUNTS:
        acct = DEVICE_ACCOUNTS[username]
        if password == acct["password"]:
            return {"username": username, "role": acct["role"]}
        return None

    user = get_user(username)
    if not user:
        return None
    if verify_password(password, user["password_hash"]):
        return {"username": user["username"], "role": user.get("role", "admin")}
    return None

def update_user_credentials(old_username: str, new_username: str, new_password: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_hash = get_password_hash(new_password)
    try:
        cursor.execute(
            "UPDATE users SET username = ?, password_hash = ? WHERE username = ?",
            (new_username, new_hash, old_username)
        )
        conn.commit()
        success = cursor.rowcount > 0
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

init_db()
