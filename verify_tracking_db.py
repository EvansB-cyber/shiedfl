import sqlite3

conn = sqlite3.connect("tracking.db")
rows = conn.execute(
    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
).fetchall()
conn.close()

print([r[0] for r in rows])