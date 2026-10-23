import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship
try:
    from database import Base
except ImportError:
    from backend.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    gmail = Column(String(150), nullable=True)  # Visible Gmail column for MySQL Workbench
    password = Column(String(255), nullable=True)  # Visible plain password for MySQL Workbench
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    recordings = relationship("AudioRecording", back_populates="user", cascade="all, delete-orphan")


class AudioRecording(Base):
    __tablename__ = "audio_recordings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    user_email = Column(String(150), nullable=True)  # Visible user Gmail linked to recording
    file_path = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    duration = Column(Float, default=0.0)  # in seconds
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    audio_status = Column(String(50), default="processed")  # processed, error, processing
    quality_metrics = Column(JSON, nullable=True)  # snr, clipping_ratio, silence_ratio, is_reliable, warnings

    # Relationships
    user = relationship("User", back_populates="recordings")
    predictions = relationship("EmotionPrediction", back_populates="recording", cascade="all, delete-orphan")
    transcripts = relationship("Transcript", back_populates="recording", cascade="all, delete-orphan")
    segments = relationship("EmotionSegment", back_populates="recording", cascade="all, delete-orphan")


class EmotionPrediction(Base):
    __tablename__ = "emotion_predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    recording_id = Column(Integer, ForeignKey("audio_recordings.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String(50), nullable=False)  # SVM, MLP, CNN, CNN-LSTM, Ensemble
    predicted_emotion = Column(String(50), nullable=False)
    model_score = Column(Float, nullable=False)  # Confidence score (0.0 to 1.0)
    all_probabilities = Column(JSON, nullable=True)  # Probabilities per emotion class
    prediction_date = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    recording = relationship("AudioRecording", back_populates="predictions")


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    recording_id = Column(Integer, ForeignKey("audio_recordings.id", ondelete="CASCADE"), nullable=False)
    transcribed_text = Column(Text, nullable=True)
    language = Column(String(20), default="en")
    transcription_status = Column(String(50), default="completed")
    sentiment_label = Column(String(50), nullable=True)  # positive, neutral, negative
    sentiment_score = Column(Float, default=0.0)  # Compound score (-1.0 to 1.0)
    sentiment_details = Column(JSON, nullable=True)  # pos, neu, neg
    discrepancy_flag = Column(String(100), nullable=True)  # Notes conflict between verbal text & vocal tone

    # Relationships
    recording = relationship("AudioRecording", back_populates="transcripts")


class EmotionSegment(Base):
    __tablename__ = "emotion_segments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    recording_id = Column(Integer, ForeignKey("audio_recordings.id", ondelete="CASCADE"), nullable=False)
    start_time = Column(Float, nullable=False)  # Segment start in seconds
    end_time = Column(Float, nullable=False)    # Segment end in seconds
    predicted_emotion = Column(String(50), nullable=False)
    model_score = Column(Float, nullable=False)
    pitch_mean = Column(Float, default=0.0)      # F0 pitch in Hz
    energy_mean = Column(Float, default=0.0)     # RMS Energy
    speaker_tag = Column(String(50), default="Speaker 1")

    # Relationships
    recording = relationship("AudioRecording", back_populates="segments")
