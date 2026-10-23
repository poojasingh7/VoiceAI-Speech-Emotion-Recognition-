import os
import sys
import sqlite3
import json

# Force UTF-8 stdout for Windows consoles
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

db_paths = ["backend/voice_analysis.db", "voice_analysis.db"]
db_path = next((p for p in db_paths if os.path.exists(p)), None)

if not db_path:
    print("Database file voice_analysis.db not found.")
    exit(1)

con = sqlite3.connect(db_path)
cur = con.cursor()

def print_table(table_name, limit=10):
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [col[1] for col in cur.fetchall()]
    
    cur.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT {limit}")
    rows = cur.fetchall()
    
    print(f"\n{'='*75}")
    print(f" 🗄️  TABLE: {table_name.upper()} (Showing up to {limit} most recent rows)")
    print(f"{'='*75}")
    print(" | ".join([f"{c:<18}" for c in cols]))
    print("-" * 75)
    
    if not rows:
        print("  (Empty table)")
    else:
        for r in rows:
            line = []
            for val in r:
                s = str(val) if val is not None else "NULL"
                if len(s) > 22:
                    s = s[:19] + "..."
                line.append(f"{s:<18}")
            print(" | ".join(line))

def execute_query(query):
    print(f"\nExecuting SQL: {query}")
    print("-" * 50)
    try:
        cur.execute(query)
        rows = cur.fetchall()
        for r in rows:
            print(r)
    except Exception as e:
        print("SQL Error:", e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom SQL query
        query = " ".join(sys.argv[1:])
        execute_query(query)
    else:
        print(f"\n📂 Database Location: {os.path.abspath(db_path)}")
        for t in ["users", "audio_recordings", "emotion_predictions", "transcripts", "emotion_segments"]:
            print_table(t, limit=5)
        print(f"\n{'='*75}")
        print("💡 Tip: Run any SQL query like:")
        print("   python view_database.py \"SELECT id, transcribed_text, sentiment_score FROM transcripts\"")
        print(f"{'='*75}\n")
    con.close()
