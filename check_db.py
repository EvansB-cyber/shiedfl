import sqlite3
import os

def check_databases():
    db_dir = "C:/Users/Appau Robert Atsu/.gemini/antigravity/scratch/project/edge_layer/databases"
    
    if not os.path.exists(db_dir):
        print(f"Databases directory not found at: {db_dir}")
        return
        
    db_files = [f for f in os.listdir(db_dir) if f.endswith(".db")]
    if not db_files:
        print("No SQLite databases found.")
        return
        
    print("=" * 60)
    print("INSPECTING LOCAL EDGE DATABASES")
    print("=" * 60)
    
    for db_file in sorted(db_files):
        db_path = os.path.join(db_dir, db_file)
        print(f"\nDatabase: {db_file}")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check contacts table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='contacts';")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM contacts")
                contact_count = cursor.fetchone()[0]
                print(f"  [contacts] table: {contact_count} records")
                
                # Fetch a sample
                cursor.execute("SELECT id, name, phone, is_trusted, risk_score FROM contacts LIMIT 3")
                samples = cursor.fetchall()
                for s in samples:
                    print(f"    - ID: {s[0]} | Name: {s[1]} | Phone: {s[2]} | Trusted: {s[3]} | Risk: {s[4]}")
            else:
                print("  [contacts] table missing!")
                
            # Check messages table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages';")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM messages")
                msg_count = cursor.fetchone()[0]
                print(f"  [messages] table: {msg_count} records")
                
                # Fetch a sample
                cursor.execute("SELECT id, sender_phone, SUBSTR(message_text, 1, 35) as msg, is_spam FROM messages LIMIT 2")
                samples = cursor.fetchall()
                for s in samples:
                    print(f"    - ID: {s[0]} | Sender: {s[1]} | Text: {s[2]}... | Spam: {s[3]}")
            else:
                print("  [messages] table missing!")
                
            conn.close()
        except Exception as e:
            print(f"  Error reading database: {e}")
            
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_databases()
