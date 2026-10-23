import sqlite3

con = sqlite3.connect("backend/voice_analysis.db")
cur = con.cursor()

# Check and add columns to users
cols = [c[1] for c in cur.execute("PRAGMA table_info(users)").fetchall()]
if "gmail" not in cols:
    cur.execute("ALTER TABLE users ADD COLUMN gmail VARCHAR(150)")
cur.execute("UPDATE users SET gmail = email WHERE gmail IS NULL")

if "password" not in cols:
    cur.execute("ALTER TABLE users ADD COLUMN password VARCHAR(255)")
cur.execute("UPDATE users SET password = 'PoojaPassword123' WHERE password IS NULL")

# Check and add user_email to audio_recordings
rec_cols = [c[1] for c in cur.execute("PRAGMA table_info(audio_recordings)").fetchall()]
if "user_email" not in rec_cols:
    cur.execute("ALTER TABLE audio_recordings ADD COLUMN user_email VARCHAR(150)")
cur.execute("UPDATE audio_recordings SET user_email = (SELECT email FROM users WHERE users.id = audio_recordings.user_id)")
cur.execute("UPDATE audio_recordings SET user_email = 'poojasinghkshatriya18@gmail.com' WHERE user_email IS NULL")

con.commit()
con.close()
print("Database schema updated with visible 'gmail', 'password', and 'user_email'!")
