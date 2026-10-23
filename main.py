import os
import io
import time
import json
import uuid
import datetime
from typing import Optional, List
import numpy as np
import soundfile as sf
import librosa
from fastapi import (
    FastAPI, File, UploadFile, Form, Depends, HTTPException,
    status, WebSocket, WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

try:
    from database import engine, Base, get_db
    from models_db import User, AudioRecording, EmotionPrediction, Transcript, EmotionSegment
    from auth import (
        hash_password, verify_password, create_access_token,
        get_current_user, get_optional_current_user
    )
    from audio_processor import (
        load_audio_from_path, load_audio_from_bytes, assess_audio_quality,
        compute_vocal_analytics, extract_features_vector,
        generate_spectrogram_image, segment_audio, EMOTIONS
    )
    from ml_models import ModelManager
    from speech_service import (
        transcribe_audio_file, detect_verbal_vocal_discrepancy
    )
    from call_analyzer import analyze_customer_support_call
except ImportError:
    from backend.database import engine, Base, get_db
    from backend.models_db import User, AudioRecording, EmotionPrediction, Transcript, EmotionSegment
    from backend.auth import (
        hash_password, verify_password, create_access_token,
        get_current_user, get_optional_current_user
    )
    from backend.audio_processor import (
        load_audio_from_path, load_audio_from_bytes, assess_audio_quality,
        compute_vocal_analytics, extract_features_vector,
        generate_spectrogram_image, segment_audio, EMOTIONS
    )
    from backend.ml_models import ModelManager
    from backend.speech_service import (
        transcribe_audio_file, detect_verbal_vocal_discrepancy
    )
    from backend.call_analyzer import analyze_customer_support_call

# Initialize Database tables
Base.metadata.create_all(bind=engine)

# Initialize Model Manager
model_manager = ModelManager.get_instance()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

app = FastAPI(
    title="Speech Emotion Recognition & Voice Analysis API",
    description="Full-stack AI platform: Classical ML (SVM, MLP) + Deep Learning (CNN, CNN-LSTM) + Real-time Audio + Explainable AI + Whisper STT",
    version="2.0.0",
)

# Enable CORS for React frontend (Vite default port 5173, 3000, 8000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Pydantic Schemas -----------------

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

# ----------------- Auth Endpoints -----------------

@app.post("/api/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists."
        )
    user = User(
        name=req.name,
        email=req.email.lower(),
        gmail=req.email.lower(),
        password=req.password,
        password_hash=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"user_id": user.id, "email": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "name": user.name, "email": user.email},
    }


@app.post("/api/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email.lower()).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )
    # Ensure readable password and gmail are populated for MySQL Workbench visibility
    if not user.password:
        user.password = req.password
    if not user.gmail:
        user.gmail = user.email
    db.commit()

    token = create_access_token({"user_id": user.id, "email": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "name": user.name, "email": user.email},
    }


@app.get("/api/auth/me")
def get_me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rec_count = db.query(AudioRecording).filter(AudioRecording.user_id == user.id).count()
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "recording_count": rec_count,
    }

# ----------------- Demo Audio Endpoints -----------------

@app.get("/api/audio/demos")
def list_demos():
    """Lists available demo audio files ready for 1-click testing."""
    if not os.path.exists(DEMO_DIR):
        return []
    files = [f for f in os.listdir(DEMO_DIR) if f.endswith(".wav")]
    results = []
    for f in sorted(files):
        label = f.replace(".wav", "").replace("_", " ").title()
        results.append({
            "filename": f,
            "label": label,
            "url": f"/api/audio/demo-file/{f}",
        })
    return results


@app.get("/api/audio/demo-file/{filename}")
def stream_demo_audio(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(DEMO_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Demo file not found.")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/audio/file/{recording_id}")
def stream_recording_audio(recording_id: int, db: Session = Depends(get_db)):
    rec = db.query(AudioRecording).filter(AudioRecording.id == recording_id).first()
    if not rec or not os.path.exists(rec.file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(rec.file_path, media_type="audio/wav")

# ----------------- Full Audio Analysis Pipeline -----------------

@app.post("/api/audio/analyze")
async def analyze_audio(
    file: Optional[UploadFile] = File(None),
    demo_name: Optional[str] = Form(None),
    user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Complete analysis pipeline:
    1. Audio Loading & Format Conversion
    2. Audio Quality & Uncertainty Check
    3. Vocal Analytics (F0 pitch, RMS, ZCR, spectral centroid)
    4. 170-dim Feature Extraction
    5. ML & DL Inference (SVM, MLP, CNN, CNN-LSTM, Ensemble)
    6. Sliding Window Segmentation (Emotion Timeline)
    7. Whisper Speech-to-Text Transcription & Sentiment
    8. Verbal vs Vocal Discrepancy Detection
    9. Spectrogram Visualization Image
    10. Persist to 5 Database Tables
    """
    target_path = None
    original_filename = "recorded_audio.wav"

    if demo_name:
        safe_demo = os.path.basename(demo_name)
        demo_src = os.path.join(DEMO_DIR, safe_demo)
        if not os.path.exists(demo_src):
            raise HTTPException(status_code=404, detail="Demo sound file does not exist.")
        original_filename = safe_demo
        file_id = str(uuid.uuid4())[:8]
        target_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_demo}")
        with open(demo_src, "rb") as f_in, open(target_path, "wb") as f_out:
            f_out.write(f_in.read())
    elif file:
        original_filename = file.filename or "recording.wav"
        file_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(original_filename)[1] or ".wav"
        target_path = os.path.join(UPLOAD_DIR, f"{file_id}_{original_filename}")
        content = await file.read()
        with open(target_path, "wb") as f_out:
            f_out.write(content)
    else:
        raise HTTPException(status_code=400, detail="Provide an audio file upload or a demo_name.")

    # 1. Load audio with librosa / soundfile
    try:
        y, sr = load_audio_from_path(target_path, target_sr=22050)
    except Exception as e:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    duration = float(len(y) / sr)

    # 2. Audio Quality & Uncertainty
    quality = assess_audio_quality(y, sr)

    # 3. Vocal Analytics
    analytics = compute_vocal_analytics(y, sr)

    # 4. Speech-to-Text & Text Sentiment Analysis
    transcript_info = transcribe_audio_file(target_path)

    # 5. Feature Extraction & Multi-Model Inference (with Prosodic & Multimodal Fusion)
    features = extract_features_vector(y, sr)
    model_predictions = model_manager.predict_all(
        features,
        vocal_analytics=analytics,
        lexical_sentiment=transcript_info.get("sentiment"),
    )

    # 6. Sliding-window Emotion Timeline (2.5s segments, 1.5s hop)
    raw_segments = segment_audio(y, sr, window_sec=2.5, hop_sec=1.5)
    timeline_segments = []
    for s_start, s_end, seg_y in raw_segments:
        seg_feat = extract_features_vector(seg_y, sr)
        seg_analytics = compute_vocal_analytics(seg_y, sr)
        seg_res = model_manager.predict_all(seg_feat, vocal_analytics=seg_analytics)
        timeline_segments.append({
            "start_time": s_start,
            "end_time": s_end,
            "predicted_emotion": seg_res["primary"]["predicted_emotion"],
            "model_score": seg_res["primary"]["confidence"],
            "probabilities": seg_res["primary"]["probabilities"],
            "pitch_mean": seg_analytics["pitch_mean_hz"],
            "energy_mean": seg_analytics["energy_mean"],
        })

    # Emotion distribution counts across the recording
    emo_counts = {}
    for seg in timeline_segments:
        e = seg["predicted_emotion"]
        emo_counts[e] = emo_counts.get(e, 0) + 1
    total_segs = max(1, len(timeline_segments))
    emotion_distribution = [
        {"emotion": emo, "count": count, "percentage": round((count / total_segs) * 100, 1)}
        for emo, count in emo_counts.items()
    ]

    # 7. Verbal vs Vocal Discrepancy
    discrepancy = detect_verbal_vocal_discrepancy(
        text_sentiment=transcript_info["sentiment"]["label"],
        vocal_emotion=model_predictions["primary"]["predicted_emotion"],
    )

    # 8. Visual Spectrogram base64
    spectrogram_image = generate_spectrogram_image(y, sr)

    # 9. Save to Database (the 5 tables)
    user_id = user.id if (user and hasattr(user, "id")) else None
    user_email = user.email if (user and hasattr(user, "email")) else None
    if not user_id:
        first_user = db.query(User).first()
        if first_user:
            user_id = first_user.id
            user_email = first_user.email

    new_recording = AudioRecording(
        user_id=user_id,
        user_email=user_email,
        file_path=target_path,
        filename=original_filename,
        duration=round(duration, 2),
        audio_status="processed",
        quality_metrics=quality,
    )
    db.add(new_recording)
    db.commit()
    db.refresh(new_recording)

    # Store predictions for all models
    for m_key, m_info in model_predictions["models"].items():
        db_pred = EmotionPrediction(
            recording_id=new_recording.id,
            model_name=m_key,
            predicted_emotion=m_info["predicted_emotion"],
            model_score=m_info["confidence"],
            all_probabilities=m_info["probabilities"],
        )
        db.add(db_pred)

    # Store transcript
    db_transcript = Transcript(
        recording_id=new_recording.id,
        transcribed_text=transcript_info.get("text", ""),
        language=transcript_info.get("language", "en"),
        transcription_status=transcript_info.get("transcription_status", "completed"),
        sentiment_label=transcript_info["sentiment"]["label"],
        sentiment_score=transcript_info["sentiment"]["score"],
        sentiment_details=transcript_info["sentiment"]["details"],
        discrepancy_flag=discrepancy["flag"],
    )
    db.add(db_transcript)

    # Store segments
    for seg in timeline_segments:
        db_seg = EmotionSegment(
            recording_id=new_recording.id,
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            predicted_emotion=seg["predicted_emotion"],
            model_score=seg["model_score"],
            pitch_mean=seg["pitch_mean"],
            energy_mean=seg["energy_mean"],
        )
        db.add(db_seg)

    db.commit()

    return {
        "recording_id": new_recording.id,
        "filename": original_filename,
        "duration": round(duration, 2),
        "primary_prediction": model_predictions["primary"],
        "ensemble_prediction": model_predictions["ensemble"],
        "models_comparison": model_predictions["models"],
        "model_agreement": model_predictions["agreement"],
        "latencies_ms": model_predictions["latencies"],
        "vocal_analytics": analytics,
        "quality_metrics": quality,
        "uncertainty": model_predictions["uncertainty"],
        "explainability": model_predictions["explainability"],
        "spectrogram_image": spectrogram_image,
        "timeline_segments": timeline_segments,
        "emotion_distribution": emotion_distribution,
        "transcript": transcript_info,
        "discrepancy": discrepancy,
        "audio_stream_url": f"/api/audio/file/{new_recording.id}",
    }

# ----------------- Real-Time Live Microphone Chunk Endpoint -----------------

@app.post("/api/audio/live-chunk")
async def process_live_chunk(file: UploadFile = File(...)):
    """
    High-speed endpoint (<20ms) for real-time live browser microphone audio segments.
    Processes audio chunk buffer, returning instant emotion predictions and acoustic levels.
    """
    t0 = time.perf_counter()
    content = await file.read()
    try:
        y, sr = load_audio_from_bytes(content, target_sr=22050)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid audio chunk: {e}"})

    duration = float(len(y) / sr)
    if duration < 0.2:
        return {"status": "buffering", "duration": duration}

    # Audio loudness / RMS
    rms = float(np.mean(librosa.feature.rms(y=y)[0])) if len(y) > 0 else 0.0

    # Fast pitch estimation
    try:
        f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
        clean_f0 = f0[~np.isnan(f0)]
        pitch_hz = round(float(np.mean(clean_f0)), 1) if len(clean_f0) > 0 else 150.0
    except Exception:
        pitch_hz = 150.0

    va_chunk = {"pitch_mean_hz": pitch_hz, "energy_mean": rms}

    # Extract features & predict with prosody prior
    features = extract_features_vector(y, sr)
    results = model_manager.predict_all(features, vocal_analytics=va_chunk)

    processing_time = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "status": "success",
        "predicted_emotion": results["primary"]["predicted_emotion"],
        "confidence": results["primary"]["confidence"],
        "probabilities": results["primary"]["probabilities"],
        "models": {k: v["predicted_emotion"] for k, v in results["models"].items()},
        "rms_energy": round(rms, 4),
        "pitch_hz": pitch_hz,
        "processing_time_ms": processing_time,
    }

# ----------------- WebSocket for Real-Time Streaming -----------------

@app.websocket("/ws/live-audio")
async def websocket_live_audio(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) < 2000:
                continue
            try:
                y, sr = load_audio_from_bytes(data, target_sr=22050)
                if len(y) < 2200:
                    continue
                features = extract_features_vector(y, sr)
                results = model_manager.predict_all(features)
                rms = float(np.mean(librosa.feature.rms(y=y)[0])) if len(y) > 0 else 0.0

                await websocket.send_json({
                    "emotion": results["primary"]["predicted_emotion"],
                    "confidence": results["primary"]["confidence"],
                    "probabilities": results["primary"]["probabilities"],
                    "rms": round(rms, 4),
                    "timestamp": time.time(),
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass

# ----------------- Comparison Lab Benchmark Endpoint -----------------

@app.get("/api/models/comparison")
def get_model_comparisons():
    """
    Returns the benchmark comparison report:
    SVM vs MLP vs CNN vs CNN-LSTM metrics, per-emotion Precision/Recall/F1,
    and confusion matrices.
    """
    benchmark_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "saved_models", "benchmark_report.json"
    )
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r") as f:
            return json.load(f)

    # If missing, compute on the fly
    from train_or_bootstrap import bootstrap_all_models
    bootstrap_all_models()
    with open(benchmark_path, "r") as f:
        return json.load(f)

# ----------------- Dedicated Call Analyzer Endpoint -----------------

@app.post("/api/audio/call-analyzer")
async def call_analyzer_endpoint(
    recording_id: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),
    demo_name: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Executes the customer support conversation analyzer workflow.
    """
    # If existing recording_id is passed
    if recording_id:
        rec = db.query(AudioRecording).filter(AudioRecording.id == recording_id).first()
        if not rec:
            raise HTTPException(status_code=404, detail="Recording not found.")
        segments = [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "predicted_emotion": s.predicted_emotion,
                "model_score": s.model_score,
                "pitch_mean": s.pitch_mean,
                "energy_mean": s.energy_mean,
            }
            for s in rec.segments
        ]
        trans = rec.transcripts[0] if rec.transcripts else None
        trans_dict = {
            "text": trans.transcribed_text if trans else "",
            "segments": [{"start": s.start_time, "end": s.end_time, "text": f"Segment {idx+1}"} for idx, s in enumerate(rec.segments)]
        }
        return analyze_customer_support_call(segments, trans_dict, rec.duration)

    # Otherwise analyze the file or demo
    analysis = await analyze_audio(file=file, demo_name=demo_name or "customer_support_call.wav", user=None, db=db)
    return analyze_customer_support_call(
        timeline_segments=analysis["timeline_segments"],
        transcript_data=analysis["transcript"],
        duration=analysis["duration"]
    )

# ----------------- Recording History & Report Export -----------------

@app.get("/api/recordings")
def list_recordings(
    search: Optional[str] = None,
    emotion: Optional[str] = None,
    user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(AudioRecording)
    if user:
        query = query.filter(AudioRecording.user_id == user.id)

    if search:
        query = query.filter(AudioRecording.filename.ilike(f"%{search}%"))

    recordings = query.order_by(AudioRecording.upload_date.desc()).limit(50).all()

    results = []
    for r in recordings:
        top_pred = r.predictions[0] if r.predictions else None
        if emotion and top_pred and top_pred.predicted_emotion.lower() != emotion.lower():
            continue

        results.append({
            "id": r.id,
            "filename": r.filename,
            "duration": r.duration,
            "upload_date": r.upload_date.isoformat() if r.upload_date else None,
            "predicted_emotion": top_pred.predicted_emotion if top_pred else "unknown",
            "model_score": top_pred.model_score if top_pred else 0.0,
            "segment_count": len(r.segments),
            "quality_status": "Reliable" if (r.quality_metrics and r.quality_metrics.get("is_reliable")) else "Review Alert",
        })
    return results


@app.get("/api/recordings/{rec_id}")
def get_recording_detail(rec_id: int, db: Session = Depends(get_db)):
    r = db.query(AudioRecording).filter(AudioRecording.id == rec_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Recording not found")

    predictions = {p.model_name: {"predicted_emotion": p.predicted_emotion, "confidence": p.model_score, "probabilities": p.all_probabilities} for p in r.predictions}
    transcript = r.transcripts[0] if r.transcripts else None
    segments = [
        {
            "start_time": s.start_time,
            "end_time": s.end_time,
            "predicted_emotion": s.predicted_emotion,
            "model_score": s.model_score,
            "pitch_mean": s.pitch_mean,
            "energy_mean": s.energy_mean,
        }
        for s in r.segments
    ]

    return {
        "id": r.id,
        "filename": r.filename,
        "duration": r.duration,
        "upload_date": r.upload_date.isoformat() if r.upload_date else None,
        "quality_metrics": r.quality_metrics,
        "predictions": predictions,
        "transcript": {
            "text": transcript.transcribed_text if transcript else "",
            "language": transcript.language if transcript else "en",
            "sentiment_label": transcript.sentiment_label if transcript else "neutral",
            "sentiment_score": transcript.sentiment_score if transcript else 0.0,
            "discrepancy_flag": transcript.discrepancy_flag if transcript else "",
        },
        "segments": segments,
        "audio_url": f"/api/audio/file/{r.id}",
    }


@app.delete("/api/recordings/{rec_id}")
def delete_recording(rec_id: int, db: Session = Depends(get_db)):
    r = db.query(AudioRecording).filter(AudioRecording.id == rec_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Recording not found")
    if os.path.exists(r.file_path):
        try:
            os.remove(r.file_path)
        except Exception:
            pass
    db.delete(r)
    db.commit()
    return {"status": "success", "message": f"Recording {rec_id} deleted."}


@app.get("/api/recordings/{rec_id}/export")
def export_report_html(rec_id: int, db: Session = Depends(get_db)):
    r = db.query(AudioRecording).filter(AudioRecording.id == rec_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Recording not found")

    top_pred = r.predictions[0] if r.predictions else None
    transcript = r.transcripts[0] if r.transcripts else None

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speech Emotion Recognition Diagnostic Report - #{r.id}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; color: #1e293b; line-height: 1.6; }}
            .header {{ border-bottom: 2px solid #3b82f6; padding-bottom: 16px; margin-bottom: 24px; }}
            .title {{ font-size: 24px; font-weight: bold; color: #0f172a; margin: 0; }}
            .sub {{ color: #64748b; font-size: 14px; margin-top: 4px; }}
            .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 18px; }}
            .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; background: #dbeafe; color: #1d4ed8; text-transform: uppercase; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
            th {{ background: #f1f5f9; color: #475569; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1 class="title">Voice AI Diagnostic & Emotion Analysis Report</h1>
            <div class="sub">Recording ID: #{r.id} | File: {r.filename} | Duration: {r.duration}s | Date: {r.upload_date}</div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Primary Acoustic Emotion</h3>
                <div style="font-size: 28px; font-weight: 800; color: #2563eb; text-transform: capitalize;">
                    {top_pred.predicted_emotion if top_pred else 'N/A'}
                </div>
                <p>Model Confidence: <strong>{round(top_pred.model_score * 100, 1) if top_pred else 0}%</strong> ({top_pred.model_name if top_pred else 'N/A'})</p>
                <div class="badge">Audio Quality: {'Reliable' if r.quality_metrics and r.quality_metrics.get('is_reliable') else 'Flagged'}</div>
            </div>

            <div class="card">
                <h3>Spoken Word Sentiment (Whisper STT)</h3>
                <div style="font-size: 18px; font-weight: 600; color: #0f172a;">
                    "{transcript.transcribed_text if transcript and transcript.transcribed_text else 'No speech detected'}"
                </div>
                <p>Verbal Sentiment: <strong>{transcript.sentiment_label.upper() if transcript and transcript.sentiment_label else 'N/A'}</strong> (Score: {transcript.sentiment_score if transcript else 0})</p>
                <p style="color: #64748b; font-size: 13px;">{transcript.discrepancy_flag if transcript and transcript.discrepancy_flag else ''}</p>
            </div>
        </div>

        <div class="card">
            <h3>Multi-Model Prediction Consensus</h3>
            <table>
                <tr><th>Model Architecture</th><th>Type</th><th>Predicted Emotion</th><th>Confidence</th></tr>
                {"".join([f"<tr><td>{p.model_name}</td><td>{'Classical ML' if p.model_name in ['SVM', 'MLP'] else 'Deep Learning'}</td><td><b>{p.predicted_emotion}</b></td><td>{round(p.model_score*100, 1)}%</td></tr>" for p in r.predictions])}
            </table>
        </div>

        <div class="card">
            <h3>Emotion Journey Timeline ({len(r.segments)} Segments)</h3>
            <table>
                <tr><th>Time Window</th><th>Predicted Emotion</th><th>Confidence</th><th>Pitch (F0)</th><th>Energy (RMS)</th></tr>
                {"".join([f"<tr><td>{s.start_time:.1f}s - {s.end_time:.1f}s</td><td><b>{s.predicted_emotion}</b></td><td>{round(s.model_score*100, 1)}%</td><td>{s.pitch_mean:.1f} Hz</td><td>{s.energy_mean:.4f}</td></tr>" for s in r.segments])}
            </table>
        </div>

        <div class="footer">
            Generated by AI Voice Analysis Platform. This document is intended as decision-support analytical insight and does not constitute a clinical psychiatric diagnosis.
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Serve static frontend if built in frontend/dist
frontend_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Don't intercept API routes or docs
        if full_path.startswith("api/") or full_path.startswith("ws/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = os.path.join(frontend_dist, full_path)
        if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    @app.get("/")
    def read_root():
        return {
            "status": "online",
            "service": "Speech Emotion Recognition Platform API",
            "version": "2.0.0",
            "endpoints": {
                "analyze": "/api/audio/analyze",
                "live_chunk": "/api/audio/live-chunk",
                "comparison": "/api/models/comparison",
                "call_analyzer": "/api/audio/call-analyzer",
                "demos": "/api/audio/demos",
                "docs": "/docs",
            }
        }
