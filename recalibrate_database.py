import os
import sys
import json
import sqlite3
import numpy as np

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from audio_processor import (
    load_audio_from_path,
    extract_features_vector,
    compute_vocal_analytics,
    segment_audio,
    assess_audio_quality,
    EMOTIONS,
)
from speech_service import transcribe_audio_file, detect_verbal_vocal_discrepancy
from ml_models import ModelManager

db_path = os.path.join(backend_dir, "voice_analysis.db")
if not os.path.exists(db_path):
    print("Database not found!")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

mm = ModelManager.get_instance()
mm.initialize_or_bootstrap()

# Fetch all recordings
cur.execute("SELECT id, file_path, filename FROM audio_recordings")
recordings = cur.fetchall()
print(f"Recalibrating {len(recordings)} recordings in database...")

for rec_id, fpath, fname in recordings:
    # Resolve file path
    target = fpath
    if not os.path.exists(target):
        cand = os.path.join(backend_dir, "uploads", fname)
        if os.path.exists(cand):
            target = cand
        else:
            cand = os.path.join(backend_dir, "demo_audio", fname)
            if os.path.exists(cand):
                target = cand
    if not os.path.exists(target):
        print(f"Skipping recording {rec_id} ({fname}): file not found")
        continue

    try:
        y, sr = load_audio_from_path(target, target_sr=22050)
        va = compute_vocal_analytics(y, sr)
        feats = extract_features_vector(y, sr)
        
        # Whisper transcription
        transcript_info = transcribe_audio_file(target)
        sent = transcript_info.get("sentiment", {"label": "neutral", "score": 0.0})
        
        # Prediction with prosodic + multimodal prior
        preds = mm.predict_all(feats, vocal_analytics=va, lexical_sentiment=sent)
        primary_emo = preds["primary"]["predicted_emotion"]
        ensemble_emo = preds["ensemble"]["predicted_emotion"]
        
        print(f"Recording #{rec_id} ({fname}): -> {ensemble_emo} (conf {preds['ensemble']['confidence']*100:.1f}%)")
        
        # 1. Update emotion_predictions
        # Delete old predictions for this recording
        cur.execute("DELETE FROM emotion_predictions WHERE recording_id = ?", (rec_id,))
        for model_name, res in preds["models"].items():
            cur.execute("""
                INSERT INTO emotion_predictions (recording_id, model_name, predicted_emotion, model_score, all_probabilities, prediction_date)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (
                rec_id,
                model_name,
                res["predicted_emotion"],
                res["confidence"],
                json.dumps(res["probabilities"])
            ))

        # 2. Update transcripts
        discrepancy = detect_verbal_vocal_discrepancy(
            text_sentiment=sent["label"],
            vocal_emotion=ensemble_emo,
        )
        cur.execute("DELETE FROM transcripts WHERE recording_id = ?", (rec_id,))
        cur.execute("""
            INSERT INTO transcripts (recording_id, transcribed_text, language, transcription_status, sentiment_label, sentiment_score, sentiment_details, discrepancy_flag)
            VALUES (?, ?, ?, 'completed', ?, ?, ?, ?)
        """, (
            rec_id,
            transcript_info.get("text", ""),
            transcript_info.get("language", "en"),
            sent["label"],
            sent.get("score", 0.0),
            json.dumps(sent.get("details", {})),
            discrepancy.get("flag", "Congruent")
        ))

        # 3. Update emotion_segments
        cur.execute("DELETE FROM emotion_segments WHERE recording_id = ?", (rec_id,))
        raw_segs = segment_audio(y, sr, window_sec=2.5, hop_sec=1.5)
        for s_start, s_end, seg_y in raw_segs:
            seg_feat = extract_features_vector(seg_y, sr)
            seg_va = compute_vocal_analytics(seg_y, sr)
            seg_pred = mm.predict_all(seg_feat, vocal_analytics=seg_va)
            cur.execute("""
                INSERT INTO emotion_segments (recording_id, start_time, end_time, predicted_emotion, model_score, pitch_mean, energy_mean, speaker_tag)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'Speaker 1')
            """, (
                rec_id,
                s_start,
                s_end,
                seg_pred["primary"]["predicted_emotion"],
                seg_pred["primary"]["confidence"],
                seg_va["pitch_mean_hz"],
                seg_va["energy_mean"]
            ))

        # 4. Update audio_recordings
        cur.execute("""
            UPDATE audio_recordings
            SET duration = ?, quality_metrics = ?
            WHERE id = ?
        """, (
            round(float(len(y) / sr), 2),
            json.dumps(assess_audio_quality(y, sr)),
            rec_id
        ))

    except Exception as e:
        print(f"Error recalibrating recording {rec_id}: {e}")

conn.commit()
conn.close()
print("All database records successfully recalibrated!")
