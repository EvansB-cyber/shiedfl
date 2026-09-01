import sqlite3
import glob

print(f"{'File':<35} {'Rows':>6} {'Fraud':>7} {'NotFraud':>9} {'FraudPct':>9}")
print("-" * 70)

total_rows = 0
total_fraud = 0

for path in sorted(glob.glob("*.db")):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT is_fraud, COUNT(*) FROM sms GROUP BY is_fraud")
    counts = dict(cur.fetchall())
    fraud = counts.get(1, 0)
    not_fraud = counts.get(0, 0)
    rows = fraud + not_fraud
    pct = (fraud / rows * 100) if rows else 0

    print(f"{path:<35} {rows:>6} {fraud:>7} {not_fraud:>9} {pct:>8.1f}%")

    total_rows += rows
    total_fraud += fraud
    conn.close()

print("-" * 70)
overall_pct = (total_fraud / total_rows * 100) if total_rows else 0
print(f"{'TOTAL':<35} {total_rows:>6} {total_fraud:>7} {total_rows - total_fraud:>9} {overall_pct:>8.1f}%")