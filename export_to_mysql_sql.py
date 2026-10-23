import sqlite3
import os
import json

db_path = "backend/voice_analysis.db" if os.path.exists("backend/voice_analysis.db") else "voice_analysis.db"

if not os.path.exists(db_path):
    print("Database not found.")
    exit(1)

con = sqlite3.connect(db_path)
cur = con.cursor()

def esc(val):
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"

out_sql = """-- ==============================================================
-- VoiceAI Platform: MySQL Workbench Schema & Data Import
-- Database: voice_analysis
-- Compatible with MySQL 8.0 & MySQL Workbench
-- ==============================================================

CREATE DATABASE IF NOT EXISTS voice_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE voice_analysis;

-- Disable foreign key checks for clean recreation
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS emotion_segments;
DROP TABLE IF EXISTS transcripts;
DROP TABLE IF EXISTS emotion_predictions;
DROP TABLE IF EXISTS audio_recordings;
DROP TABLE IF EXISTS users;

SET FOREIGN_KEY_CHECKS = 1;

-- 1. Table: users
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    gmail VARCHAR(150) NULL,
    password VARCHAR(255) NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Table: audio_recordings
CREATE TABLE audio_recordings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL,
    user_email VARCHAR(150) NULL,
    file_path VARCHAR(500) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    duration FLOAT DEFAULT 0.0,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    audio_status VARCHAR(50) DEFAULT 'processed',
    quality_metrics JSON NULL,
    CONSTRAINT fk_recordings_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. Table: emotion_predictions
CREATE TABLE emotion_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    recording_id INT NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    predicted_emotion VARCHAR(50) NOT NULL,
    model_score FLOAT NOT NULL,
    all_probabilities JSON NULL,
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_predictions_recording FOREIGN KEY (recording_id) REFERENCES audio_recordings(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. Table: transcripts
CREATE TABLE transcripts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    recording_id INT NOT NULL,
    transcribed_text TEXT NULL,
    language VARCHAR(20) DEFAULT 'en',
    transcription_status VARCHAR(50) DEFAULT 'completed',
    sentiment_label VARCHAR(50) NULL,
    sentiment_score FLOAT DEFAULT 0.0,
    sentiment_details JSON NULL,
    discrepancy_flag VARCHAR(100) NULL,
    CONSTRAINT fk_transcripts_recording FOREIGN KEY (recording_id) REFERENCES audio_recordings(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. Table: emotion_segments
CREATE TABLE emotion_segments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    recording_id INT NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    predicted_emotion VARCHAR(50) NOT NULL,
    model_score FLOAT NOT NULL,
    pitch_mean FLOAT DEFAULT 0.0,
    energy_mean FLOAT DEFAULT 0.0,
    speaker_tag VARCHAR(50) DEFAULT 'Speaker 1',
    CONSTRAINT fk_segments_recording FOREIGN KEY (recording_id) REFERENCES audio_recordings(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ==============================================================
-- DATA INSERTS FROM YOUR VOICE AI RECORDINGS
-- ==============================================================
"""

# Users
rows = cur.execute("SELECT id, name, email, password_hash, created_at, gmail, password FROM users").fetchall()
for r in rows:
    out_sql += f"INSERT INTO users (id, name, email, password_hash, created_at, gmail, password) VALUES ({r[0]}, {esc(r[1])}, {esc(r[2])}, {esc(r[3])}, {esc(r[4])}, {esc(r[5])}, {esc(r[6])});\n"

# Audio recordings
rows = cur.execute("SELECT id, user_id, file_path, filename, duration, upload_date, audio_status, quality_metrics, user_email FROM audio_recordings").fetchall()
for r in rows:
    u_id = r[1] if r[1] is not None else "NULL"
    out_sql += f"INSERT INTO audio_recordings (id, user_id, file_path, filename, duration, upload_date, audio_status, quality_metrics, user_email) VALUES ({r[0]}, {u_id}, {esc(r[2])}, {esc(r[3])}, {r[4]}, {esc(r[5])}, {esc(r[6])}, {esc(r[7])}, {esc(r[8])});\n"

# Emotion predictions
rows = cur.execute("SELECT id, recording_id, model_name, predicted_emotion, model_score, all_probabilities, prediction_date FROM emotion_predictions").fetchall()
for r in rows:
    out_sql += f"INSERT INTO emotion_predictions (id, recording_id, model_name, predicted_emotion, model_score, all_probabilities, prediction_date) VALUES ({r[0]}, {r[1]}, {esc(r[2])}, {esc(r[3])}, {r[4]}, {esc(r[5])}, {esc(r[6])});\n"

# Transcripts
rows = cur.execute("SELECT id, recording_id, transcribed_text, language, transcription_status, sentiment_label, sentiment_score, sentiment_details, discrepancy_flag FROM transcripts").fetchall()
for r in rows:
    out_sql += f"INSERT INTO transcripts (id, recording_id, transcribed_text, language, transcription_status, sentiment_label, sentiment_score, sentiment_details, discrepancy_flag) VALUES ({r[0]}, {r[1]}, {esc(r[2])}, {esc(r[3])}, {esc(r[4])}, {esc(r[5])}, {r[6]}, {esc(r[7])}, {esc(r[8])});\n"

# Emotion segments
rows = cur.execute("SELECT id, recording_id, start_time, end_time, predicted_emotion, model_score, pitch_mean, energy_mean, speaker_tag FROM emotion_segments").fetchall()
for r in rows:
    out_sql += f"INSERT INTO emotion_segments (id, recording_id, start_time, end_time, predicted_emotion, model_score, pitch_mean, energy_mean, speaker_tag) VALUES ({r[0]}, {r[1]}, {r[2]}, {r[3]}, {esc(r[4])}, {r[5]}, {r[6]}, {r[7]}, {esc(r[8])});\n"

out_sql += """
-- ==============================================================
-- VERIFICATION QUERIES FOR MYSQL WORKBENCH
-- ==============================================================
SELECT 'users' AS `Table`, COUNT(*) AS `Total Rows` FROM users
UNION ALL
SELECT 'audio_recordings', COUNT(*) FROM audio_recordings
UNION ALL
SELECT 'emotion_predictions', COUNT(*) FROM emotion_predictions
UNION ALL
SELECT 'transcripts', COUNT(*) FROM transcripts
UNION ALL
SELECT 'emotion_segments', COUNT(*) FROM emotion_segments;

-- View all transcripts with sentiment
SELECT id, recording_id, transcribed_text, sentiment_label, sentiment_score 
FROM transcripts;
"""

with open("voice_analysis_workbench.sql", "w", encoding="utf-8") as f:
    f.write(out_sql)

print("Generated voice_analysis_workbench.sql successfully!")
con.close()
