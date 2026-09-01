"""
Fix / clean smishing_training_set_*.db files.

Usage:
    python fix_training_db.py smishing_training_set_1.db
    python fix_training_db.py smishing_training_set_1.db --apply   (actually writes changes)
    python fix_training_db.py *.db --apply                        (process many at once, in PowerShell use a loop instead — see bottom)

Without --apply, this only REPORTS what it would change (dry run).
With --apply, it actually modifies the .db file in place.
Always keep a backup before running with --apply on data you care about.
"""

import sqlite3
import sys
import argparse
import shutil


def inspect(conn):
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM sms")
    total = cur.fetchone()[0]

    cur.execute("SELECT sms_body, COUNT(*) c FROM sms GROUP BY sms_body HAVING c > 1")
    dupes = cur.fetchall()
    dupe_row_total = sum(c for _, c in dupes)

    cur.execute("SELECT is_fraud, COUNT(*) FROM sms GROUP BY is_fraud")
    label_counts = dict(cur.fetchall())

    cur.execute("SELECT COUNT(*) FROM sms WHERE sms_body IS NULL OR TRIM(sms_body) = ''")
    empty_bodies = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM sms WHERE LENGTH(recipient_number) != 10")
    bad_numbers = cur.fetchone()[0]

    print(f"Total rows:            {total}")
    print(f"Duplicate message groups: {len(dupes)}  ({dupe_row_total} rows involved)")
    print(f"Label balance:          not-fraud={label_counts.get(0, 0)}  fraud={label_counts.get(1, 0)}")
    print(f"Empty/NULL sms_body:    {empty_bodies}")
    print(f"Malformed phone numbers (not 10 digits): {bad_numbers}")

    return {
        "total": total,
        "dupes": dupes,
        "dupe_row_total": dupe_row_total,
        "empty_bodies": empty_bodies,
        "bad_numbers": bad_numbers,
    }


def dedupe(conn, apply=False):
    """Keep only the first occurrence (lowest id) of each duplicate sms_body."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id FROM sms
        WHERE id NOT IN (
            SELECT MIN(id) FROM sms GROUP BY sms_body
        )
    """)
    dupe_ids = [row[0] for row in cur.fetchall()]

    print(f"\nWould delete {len(dupe_ids)} duplicate rows (keeping first occurrence of each message).")
    if apply and dupe_ids:
        cur.executemany("DELETE FROM sms WHERE id = ?", [(i,) for i in dupe_ids])
        conn.commit()
        print(f"Deleted {len(dupe_ids)} rows.")
    elif not apply:
        print("(dry run — pass --apply to actually delete)")

    return dupe_ids


def fix_phone_numbers(conn, apply=False):
    """Report rows where recipient_number isn't exactly 10 digits."""
    cur = conn.cursor()
    cur.execute("SELECT id, recipient_number FROM sms WHERE LENGTH(recipient_number) != 10")
    bad = cur.fetchall()
    print(f"\n{len(bad)} rows with malformed phone numbers:")
    for row in bad[:20]:
        print(" ", row)
    if len(bad) > 20:
        print(f"  ... and {len(bad) - 20} more")
    # Not auto-fixed: a malformed number could mean many different things
    # (missing leading 0, extra whitespace, wrong country code). Inspect
    # the list above and decide the correct fix rather than guessing.
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default is dry run)")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a .bak copy before --apply")
    args = parser.parse_args()

    if args.apply and not args.no_backup:
        backup_path = args.db_path + ".bak"
        shutil.copy2(args.db_path, backup_path)
        print(f"Backed up to {backup_path} before modifying.\n")

    conn = sqlite3.connect(args.db_path)

    print(f"=== {args.db_path} ===")
    inspect(conn)
    dedupe(conn, apply=args.apply)
    fix_phone_numbers(conn, apply=args.apply)

    print("\n--- After changes ---")
    inspect(conn)

    conn.close()


if __name__ == "__main__":
    main()