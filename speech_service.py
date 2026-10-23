import os
import re
import logging
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Lazy-loaded whisper model instance
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            logger.info("Loading Whisper 'tiny' model for fast CPU speech transcription...")
            _whisper_model = whisper.load_model("tiny")
        except Exception as e:
            logger.warning(f"Whisper model load failed: {e}. Transcription will use fallback.")
            _whisper_model = False
    return _whisper_model


def analyze_text_sentiment(text: str) -> dict:
    """
    Analyzes sentiment of transcribed speech using VADER.
    """
    if not text or not text.strip():
        return {
            "label": "neutral",
            "score": 0.0,
            "details": {"pos": 0.0, "neu": 1.0, "neg": 0.0},
        }

    try:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        compound = float(scores["compound"])

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": round(compound, 3),
            "details": {
                "positive": round(float(scores["pos"]), 3),
                "neutral": round(float(scores["neu"]), 3),
                "negative": round(float(scores["neg"]), 3),
            },
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {
            "label": "neutral",
            "score": 0.0,
            "details": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        }


def detect_verbal_vocal_discrepancy(text_sentiment: str, vocal_emotion: str) -> dict:
    """
    Evaluates whether spoken words and acoustic emotion match or diverge.
    Key highlight from requirements:
    'The words may sound positive while the voice sounds frustrated.
    The two outputs should remain separate rather than assuming either one reveals the speaker's true feelings.'
    """
    vocal_emotion = vocal_emotion.lower()
    text_sentiment = text_sentiment.lower()

    is_congruent = True
    discrepancy_type = "aligned"
    flag = "✅ Congruent Expression: Spoken words and vocal acoustics align."
    analysis_note = (
        f"Verbal content is '{text_sentiment}' and vocal tone is '{vocal_emotion}'. "
        "No significant cognitive dissonance or masked affect observed."
    )

    # Discrepancy cases:
    if text_sentiment == "positive" and vocal_emotion in ["angry", "sad", "fearful", "disgust"]:
        is_congruent = False
        discrepancy_type = "masked_negative_or_sarcasm"
        flag = "⚠️ Masked Frustration / Sarcasm Flag"
        analysis_note = (
            f"Speaker used positive verbal vocabulary ({text_sentiment}), yet acoustic vocal markers "
            f"exhibit tense '{vocal_emotion}' characteristics. This pattern frequently indicates "
            "polite masking of frustration, passive distress, or sarcastic nuance."
        )
    elif text_sentiment == "negative" and vocal_emotion in ["happy", "calm"]:
        is_congruent = False
        discrepancy_type = "stoic_or_composed"
        flag = "ℹ️ Controlled / Stoic Delivery"
        analysis_note = (
            f"Speaker described negative issues or distress, yet maintained a composed, "
            f"controlled vocal tone ({vocal_emotion}). Indicates professional restraint or desensitization."
        )

    return {
        "is_congruent": is_congruent,
        "discrepancy_type": discrepancy_type,
        "flag": flag,
        "analysis_note": analysis_note,
        "verbal_sentiment": text_sentiment,
        "vocal_emotion": vocal_emotion,
    }


def transcribe_audio_file(audio_input) -> dict:
    """
    Transcribes audio file using Whisper with timestamped segments and sentiment analysis.
    Supports either file path or preloaded numpy array. Loads via soundfile/librosa to 16kHz
    so external ffmpeg installation is not required.
    """
    model = get_whisper_model()

    if model:
        try:
            import soundfile as sf
            import librosa
            if isinstance(audio_input, str):
                file_path = audio_input
                if not os.path.isabs(file_path) and not os.path.exists(file_path):
                    # Try resolving relative to backend directory
                    cand = os.path.join(os.path.dirname(__file__), file_path)
                    if os.path.exists(cand):
                        file_path = cand
                from audio_processor import load_audio_from_path
                y, sr = load_audio_from_path(file_path, target_sr=16000)
            elif isinstance(audio_input, np.ndarray):
                y = audio_input.astype(np.float32)
            else:
                y = None

            if y is not None:
                result = model.transcribe(
                    y,
                    fp16=False,
                    language="en",
                    temperature=0.0,
                    verbose=False,
                )
                raw_text = result.get("text", "").strip()
                detected_lang = result.get("language", "en")
                segments = []

                for seg in result.get("segments", []):
                    seg_text = seg.get("text", "").strip()
                    seg_sent = analyze_text_sentiment(seg_text)
                    segments.append({
                        "start": round(float(seg.get("start", 0.0)), 2),
                        "end": round(float(seg.get("end", 0.0)), 2),
                        "text": seg_text,
                        "sentiment": seg_sent["label"],
                        "sentiment_score": seg_sent["score"],
                    })

                full_sentiment = analyze_text_sentiment(raw_text)

                return {
                    "transcription_status": "completed",
                    "text": raw_text if raw_text else "(Spoken speech processed)",
                    "language": detected_lang,
                    "segments": segments,
                    "sentiment": full_sentiment,
                }
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")

    # Graceful fallback if Whisper is disabled or failed
    return {
        "transcription_status": "simulated",
        "text": "Thank you for contacting customer support. I am following up on my previous inquiry.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Thank you for contacting customer support.", "sentiment": "positive", "sentiment_score": 0.36},
            {"start": 2.5, "end": 5.0, "text": "I am following up on my previous inquiry.", "sentiment": "neutral", "sentiment_score": 0.0},
        ],
        "sentiment": {"label": "positive", "score": 0.36, "details": {"positive": 0.28, "neutral": 0.72, "negative": 0.0}},
    }
