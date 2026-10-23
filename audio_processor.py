import os
import io
import base64
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Canonical emotions
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def _decode_with_av(source, target_sr: int = 22050):
    """Universal decoder using PyAV supporting webm, ogg, mp3, wav, flac, etc."""
    import av
    container = av.open(source)
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if not audio_stream:
        raise ValueError("No audio stream found in source")
    
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    frames = []
    for frame in container.decode(audio_stream):
        for rf in resampler.resample(frame):
            frames.append(rf.to_ndarray().flatten())
    if frames:
        return np.concatenate(frames).astype(np.float32), target_sr
    return np.zeros(target_sr, dtype=np.float32), target_sr


def load_audio_from_path(file_path: str, target_sr: int = 22050):
    """Safely loads an audio file, returning mono signal and sample rate."""
    try:
        data, sr = sf.read(file_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return data, sr
    except Exception:
        pass

    try:
        return _decode_with_av(file_path, target_sr=target_sr)
    except Exception:
        pass

    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        raise ValueError(f"Could not decode audio from {file_path}: {e}")


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 22050):
    """Loads audio from raw byte buffer supporting wav, webm, mp3, ogg."""
    bio = io.BytesIO(audio_bytes)
    try:
        data, sr = sf.read(bio, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return data, sr
    except Exception:
        pass

    bio.seek(0)
    try:
        return _decode_with_av(bio, target_sr=target_sr)
    except Exception as e:
        raise ValueError(f"Could not decode audio buffer: {e}")


def assess_audio_quality(y: np.ndarray, sr: int) -> dict:
    """
    Evaluates audio quality:
    - Clipping detection
    - Signal-to-Noise Ratio (SNR) estimation
    - Silence ratio
    - Length check
    - Reliability score
    """
    warnings = []
    duration = len(y) / float(sr)

    # 1. Length check
    if duration < 0.6:
        warnings.append("Audio is very short (<0.6s). Predictions may have lower reliability.")

    # 2. Clipping detection
    peak = np.max(np.abs(y)) if len(y) > 0 else 0
    clipping_samples = np.sum(np.abs(y) >= 0.985)
    clipping_pct = (clipping_samples / float(len(y))) * 100.0 if len(y) > 0 else 0.0
    is_clipping = clipping_pct > 0.05 or peak >= 0.999
    if is_clipping:
        warnings.append(f"Audio clipping detected ({clipping_pct:.2f}% of samples). Microphones was overloaded.")

    # 3. Silence ratio & Energy
    rms_frames = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    silence_threshold = 0.015
    silent_frames = np.sum(rms_frames < silence_threshold)
    silence_ratio = float(silent_frames / max(len(rms_frames), 1))
    if silence_ratio > 0.65:
        warnings.append("High background silence or weak speech volume detected.")

    # 4. Estimated SNR (Signal to Noise Ratio)
    # Estimate noise from lowest 10% energy frames
    sorted_rms = np.sort(rms_frames)
    noise_floor = np.mean(sorted_rms[:max(1, int(len(sorted_rms) * 0.15))]) + 1e-6
    signal_floor = np.mean(sorted_rms[int(len(sorted_rms) * 0.5):]) + 1e-6
    snr_estimate = 20.0 * np.log10(max(signal_floor / noise_floor, 1.0))
    if snr_estimate < 8.0:
        warnings.append(f"Low signal-to-noise ratio (~{snr_estimate:.1f} dB). Background noise present.")

    is_reliable = len(warnings) == 0 or (len(warnings) == 1 and duration >= 1.0 and not is_clipping)

    return {
        "duration": round(duration, 2),
        "snr_db": round(float(snr_estimate), 1),
        "is_clipping": bool(is_clipping),
        "clipping_pct": round(float(clipping_pct), 2),
        "silence_ratio": round(float(silence_ratio), 2),
        "mean_rms": round(float(np.mean(rms_frames)), 4),
        "peak_amplitude": round(float(peak), 3),
        "is_reliable": bool(is_reliable),
        "warnings": warnings,
    }


def compute_vocal_analytics(y: np.ndarray, sr: int) -> dict:
    """
    Computes acoustic properties:
    - Pitch / F0 in Hz (mean, min, max, std, trajectory)
    - RMS Energy & Loudness
    - Speech Rate & Pauses
    - Spectral Centroid, Rolloff, Contrast, Flatness
    - Zero-Crossing Rate
    """
    if len(y) == 0:
        y = np.zeros(sr // 2, dtype=np.float32)

    # 1. Pitch / F0 estimation using fast YIN algorithm (65 Hz to 500 Hz range for human speech)
    f0 = None
    try:
        f0 = librosa.yin(
            y,
            fmin=65,
            fmax=500,
            sr=sr,
            frame_length=1024,
            hop_length=512,
        )
        # Filter voiced frames using energy threshold and valid range
        rms_local = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
        # Match lengths
        min_len = min(len(f0), len(rms_local))
        f0 = f0[:min_len]
        rms_local = rms_local[:min_len]
        voiced_mask = (rms_local > 0.015) & (f0 >= 65) & (f0 <= 500)
        voiced_f0 = f0[voiced_mask]
    except Exception:
        voiced_f0 = np.array([])

    if len(voiced_f0) > 0 and not np.all(np.isnan(voiced_f0)):
        pitch_mean = float(np.nanmean(voiced_f0))
        pitch_std = float(np.nanstd(voiced_f0))
        pitch_min = float(np.nanmin(voiced_f0))
        pitch_max = float(np.nanmax(voiced_f0))
        voiced_ratio = float(len(voiced_f0) / max(len(f0), 1))
    else:
        pitch_mean, pitch_std, pitch_min, pitch_max, voiced_ratio = 150.0, 20.0, 100.0, 220.0, 0.4

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    energy_mean = float(np.mean(rms))
    energy_max = float(np.max(rms)) if len(rms) > 0 else 0.0
    energy_std = float(np.std(rms))

    # 3. Spectral Features
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid = float(np.mean(sc))
    s_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    spectral_rolloff = float(np.mean(s_rolloff))
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = float(np.mean(zcr))

    # 4. Tempo / Speech rhythm (estimated on first 4s for speed)
    try:
        y_tempo = y[:min(len(y), sr * 4)]
        onset_env = librosa.onset.onset_strength(y=y_tempo, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 115.0
        else:
            tempo = float(tempo)
    except Exception:
        tempo = 115.0

    # 5. Pitch trajectory downsampled for charts (30 points)
    if f0 is not None and len(f0) > 0:
        clean_f0 = np.where(np.isnan(f0), 0.0, f0)
        # downsample to ~30 points
        step = max(1, len(clean_f0) // 30)
        pitch_contour = [round(float(val), 1) for val in clean_f0[::step][:30]]
    else:
        pitch_contour = [150.0] * 20

    return {
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 1),
        "pitch_min_hz": round(pitch_min, 1),
        "pitch_max_hz": round(pitch_max, 1),
        "voiced_ratio": round(voiced_ratio, 2),
        "energy_mean": round(energy_mean, 4),
        "energy_max": round(energy_max, 4),
        "energy_std": round(energy_std, 4),
        "spectral_centroid": round(spectral_centroid, 1),
        "spectral_rolloff": round(spectral_rolloff, 1),
        "zero_crossing_rate": round(zcr_mean, 4),
        "estimated_tempo_bpm": round(tempo, 1),
        "pitch_contour": pitch_contour,
    }


def extract_features_vector(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extracts a fixed 170-dimensional feature vector for ML/DL models:
    - 40 MFCC mean + 40 MFCC std (80)
    - 12 Chroma mean + 12 Chroma std (24)
    - 40 Mel Spectrogram pooled mean (40)
    - 7 Spectral Contrast mean + 7 std (14)
    - ZCR mean + std (2)
    - RMS mean + std (2)
    - Spectral Rolloff mean + std (2)
    - Spectral Centroid mean + std (2)
    - Pitch mean, std, max, voiced ratio (4)
    Total = 170 dimensions
    """
    if len(y) == 0:
        y = np.zeros(sr, dtype=np.float32)

    features = []

    # MFCC (40 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Chroma STFT (12 bins)
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
    except Exception:
        features.extend([0.0] * 24)

    # Mel Spectrogram (40 bands pooled)
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        features.extend(np.mean(mel, axis=1))
    except Exception:
        features.extend([0.0] * 40)

    # Spectral Contrast (7 bands)
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))
    except Exception:
        features.extend([0.0] * 14)

    # Zero Crossing Rate (mean, std)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    # RMS Energy (mean, std)
    rms = librosa.feature.rms(y=y)[0]
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    # Spectral Rolloff
    try:
        s_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(float(np.mean(s_roll)))
        features.append(float(np.std(s_roll)))
    except Exception:
        features.extend([0.0, 0.0])

    # Spectral Centroid
    try:
        s_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(float(np.mean(s_cent)))
        features.append(float(np.std(s_cent)))
    except Exception:
        features.extend([0.0, 0.0])

    # Pitch proxy (autocorrelation based on first 2 seconds)
    try:
        y_sub = y[:min(len(y), sr * 2)]
        autocorr = librosa.autocorrelate(y_sub, max_size=min(len(y_sub), sr // 50))
        features.append(float(np.mean(autocorr)))
        features.append(float(np.std(autocorr)))
        features.append(float(np.max(autocorr)))
        features.append(float(np.sum(rms > 0.02) / max(len(rms), 1)))
    except Exception:
        features.extend([0.0, 0.0, 0.0, 0.0])

    feat_arr = np.array(features, dtype=np.float32)
    # Replace any NaNs or Infs
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return feat_arr


def generate_spectrogram_image(y: np.ndarray, sr: int = 22050) -> str:
    """
    Generates a base64 encoded PNG of the Log-Mel Spectrogram for Explainable AI visual inspection.
    """
    try:
        plt.figure(figsize=(7, 2.5), dpi=90)
        plt.style.use("dark_background")
        s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
        s_db = librosa.power_to_db(s, ref=np.max)
        librosa.display.specshow(s_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Acoustic Mel-Spectrogram (Frequency vs Time)", fontsize=10, color="#94a3b8")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f172a", edgecolor="none")
        plt.close()
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        plt.close("all")
        return ""


def segment_audio(y: np.ndarray, sr: int, window_sec: float = 3.0, hop_sec: float = 2.0):
    """
    Segments audio into sliding windows for the Emotion Timeline.
    Returns list of (start_sec, end_sec, segment_audio_array)
    """
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    total_samples = len(y)

    segments = []
    if total_samples <= window_samples:
        segments.append((0.0, round(total_samples / float(sr), 2), y))
        return segments

    start = 0
    while start < total_samples:
        end = min(start + window_samples, total_samples)
        segment_y = y[start:end]
        start_sec = round(start / float(sr), 2)
        end_sec = round(end / float(sr), 2)
        segments.append((start_sec, end_sec, segment_y))
        if end == total_samples:
            break
        start += hop_samples

    return segments
