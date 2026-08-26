import sqlite3
import os
from passlib.context import CryptContext

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "users.db")
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

    # ── 2FA / TOTP support ──────────────────────────────────────
    # totp_secret: base32 secret from pyotp, set on /api/auth/2fa/setup,
    #              stays "pending" (unconfirmed) until verify() succeeds.
    # totp_enabled: 0/1 flag flipped only after a successful verify().
    if "totp_secret" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
    if "totp_enabled" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN totp_enabled INTEGER NOT NULL DEFAULT 0")

    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", get_password_hash("password"), "admin")
        )

    conn.commit()
    conn.close()

def get_user(username: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, password_hash, role, totp_secret, totp_enabled FROM users WHERE username = ?",
        (username,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "username": row[1],
            "password_hash": row[2],
            "role": row[3],
            "totp_secret": row[4],
            "totp_enabled": bool(row[5]),
        }
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

def set_totp_secret(username: str, secret: str):
    """Store a freshly generated secret. Left unconfirmed (totp_enabled=0)
    until the user verifies a code against it."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET totp_secret = ?, totp_enabled = 0 WHERE username = ?",
        (secret, username)
    )
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success

def enable_totp(username: str):
    """Flip totp_enabled on. Call only after a successful code verification."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET totp_enabled = 1 WHERE username = ?", (username,))
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success

def disable_totp(username: str):
    """Turn 2FA off and clear the stored secret so a stale one can't be reused."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET totp_enabled = 0, totp_secret = NULL WHERE username = ?",
        (username,)
    )
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success

init_db()