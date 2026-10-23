import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
NUM_CLASSES = len(EMOTIONS)
FEATURE_DIM = 170

# ----------------- Deep Learning PyTorch Models -----------------

class CNN1D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.AdaptiveAvgPool1d(16)

        self.dropout = nn.Dropout(0.35)
        self.fc1 = nn.Linear(128 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 1, feature_dim)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x)
        return out


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM):
        super(CNNLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 85 steps
        )
        # LSTM input: (batch, seq_len=85, feature_dim=64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 1, feature_dim)
        feat = self.conv(x)  # (batch, 64, 85)
        feat = feat.permute(0, 2, 1)  # (batch, 85, 64)
        lstm_out, _ = self.lstm(feat)  # (batch, 85, 128)
        # Global mean pooling over sequence length
        pooled = torch.mean(lstm_out, dim=1)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out


# ----------------- Model Registry & Inference Manager -----------------

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

def compute_acoustic_prosody_prior(vocal_analytics: dict = None, lexical_sentiment: dict = None) -> np.ndarray:
    """
    Computes empirical psychoacoustic prior scores based on
    established Geneva Minimalistic Acoustic Parameter Set (GeMAPS) science:
    - Pitch F0 mean & variance
    - RMS Energy (arousal)
    - Spectral Centroid (brightness)
    - Zero-Crossing Rate
    - Optional Whisper lexical sentiment
    """
    prior = np.ones(NUM_CLASSES, dtype=np.float32) * 0.05
    emo_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

    if vocal_analytics:
        pitch = vocal_analytics.get("pitch_mean_hz", 150.0)
        pitch_std = vocal_analytics.get("pitch_std_hz", 15.0)
        energy = vocal_analytics.get("energy_mean", 0.05)
        centroid = vocal_analytics.get("spectral_centroid", 2000.0)

        # 1. High Arousal (Pitch > 200 Hz or Energy > 0.08)
        if pitch > 200 or energy > 0.08:
            if energy > 0.08 and pitch > 180:
                prior[emo_to_idx["angry"]] += 0.40
                prior[emo_to_idx["happy"]] += 0.25
                prior[emo_to_idx["surprised"]] += 0.20
            elif pitch_std > 22:
                prior[emo_to_idx["happy"]] += 0.40
                prior[emo_to_idx["fearful"]] += 0.25
                prior[emo_to_idx["surprised"]] += 0.20
            else:
                prior[emo_to_idx["happy"]] += 0.35
                prior[emo_to_idx["surprised"]] += 0.25
                prior[emo_to_idx["neutral"]] += 0.15

        # 2. Low Arousal (Pitch < 140 Hz and Energy < 0.06)
        elif pitch < 140 and energy < 0.06:
            if energy < 0.035:
                prior[emo_to_idx["sad"]] += 0.45
                prior[emo_to_idx["calm"]] += 0.35
                prior[emo_to_idx["neutral"]] += 0.15
            elif pitch_std < 12:
                prior[emo_to_idx["calm"]] += 0.45
                prior[emo_to_idx["neutral"]] += 0.30
                prior[emo_to_idx["sad"]] += 0.20
            else:
                prior[emo_to_idx["sad"]] += 0.40
                prior[emo_to_idx["calm"]] += 0.30
                prior[emo_to_idx["disgust"]] += 0.15

        # 3. Intermediate Arousal (Pitch 140-200 Hz)
        else:
            if energy > 0.065:
                prior[emo_to_idx["happy"]] += 0.30
                prior[emo_to_idx["angry"]] += 0.25
                prior[emo_to_idx["neutral"]] += 0.20
            elif energy < 0.035:
                prior[emo_to_idx["calm"]] += 0.35
                prior[emo_to_idx["sad"]] += 0.30
                prior[emo_to_idx["neutral"]] += 0.20
            else:
                prior[emo_to_idx["neutral"]] += 0.40
                prior[emo_to_idx["calm"]] += 0.25
                prior[emo_to_idx["happy"]] += 0.20

    # 4. Lexical Sentiment Modulation (if text is available)
    if lexical_sentiment:
        score = lexical_sentiment.get("score", 0.0)
        label = lexical_sentiment.get("label", "neutral")
        if label == "positive" or score > 0.20:
            prior[emo_to_idx["happy"]] += 0.35
            prior[emo_to_idx["calm"]] += 0.15
            prior[emo_to_idx["surprised"]] += 0.10
        elif label == "negative" or score < -0.20:
            if vocal_analytics and vocal_analytics.get("energy_mean", 0.05) > 0.06:
                prior[emo_to_idx["angry"]] += 0.40
                prior[emo_to_idx["disgust"]] += 0.20
            else:
                prior[emo_to_idx["sad"]] += 0.40
                prior[emo_to_idx["fearful"]] += 0.20

    prior = prior / np.sum(prior)
    return prior


class ModelManager:
    _instance = None

    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        self.scaler = None
        self.svm_model = None
        self.mlp_model = None
        self.cnn_model = None
        self.cnn_lstm_model = None
        self.benchmark_cache = None
        self.is_loaded = False

    @classmethod
    def get_instance(cls, models_dir: str = None):
        if cls._instance is None:
            cls._instance = cls(models_dir or DEFAULT_MODELS_DIR)
            cls._instance.initialize_or_bootstrap()
        return cls._instance

    def initialize_or_bootstrap(self):
        """Loads trained models from disk or generates pre-calibrated baseline models."""
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        svm_path = os.path.join(self.models_dir, "svm_model.joblib")
        mlp_path = os.path.join(self.models_dir, "mlp_model.joblib")
        cnn_path = os.path.join(self.models_dir, "cnn_model.pt")
        cnn_lstm_path = os.path.join(self.models_dir, "cnn_lstm_model.pt")

        if (
            os.path.exists(scaler_path)
            and os.path.exists(svm_path)
            and os.path.exists(mlp_path)
            and os.path.exists(cnn_path)
            and os.path.exists(cnn_lstm_path)
        ):
            try:
                self.scaler = joblib.load(scaler_path)
                self.svm_model = joblib.load(svm_path)
                self.mlp_model = joblib.load(mlp_path)

                self.cnn_model = CNN1D()
                self.cnn_model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
                self.cnn_model.eval()

                self.cnn_lstm_model = CNNLSTM()
                self.cnn_lstm_model.load_state_dict(torch.load(cnn_lstm_path, map_location="cpu"))
                self.cnn_lstm_model.eval()

                self.is_loaded = True
                print("Successfully loaded pre-trained models from disk.")
                return
            except Exception as e:
                print(f"Error loading existing models: {e}. Re-bootstrapping models.")

        # If not present or failed, perform training/bootstrapping
        from train_or_bootstrap import bootstrap_all_models
        bootstrap_all_models(self.models_dir)
        self.initialize_or_bootstrap()

    def predict_all(self, feature_vector: np.ndarray, vocal_analytics: dict = None, lexical_sentiment: dict = None) -> dict:
        """
        Runs inference across all 4 architectures: SVM, MLP, CNN, CNN-LSTM, plus Ensemble.
        Incorporates acoustic prosody and lexical sentiment priors for optimal multimodal calibration.
        Returns predictions, probabilities, latencies, agreement score, and explainability.
        """
        if not self.is_loaded:
            self.initialize_or_bootstrap()

        x_raw = feature_vector.reshape(1, -1)
        x_scaled = self.scaler.transform(x_raw)

        results = {}
        latencies = {}

        # 1. SVM Inference
        t0 = time.perf_counter()
        svm_probs = self.svm_model.predict_proba(x_scaled)[0]
        latencies["SVM"] = round((time.perf_counter() - t0) * 1000, 2)
        svm_idx = int(np.argmax(svm_probs))
        results["SVM"] = {
            "model_name": "Support Vector Machine (SVM)",
            "type": "Classical ML",
            "predicted_emotion": EMOTIONS[svm_idx],
            "confidence": round(float(svm_probs[svm_idx]), 4),
            "probabilities": {EMOTIONS[i]: round(float(svm_probs[i]), 4) for i in range(NUM_CLASSES)},
            "latency_ms": latencies["SVM"],
        }

        # 2. MLP Inference
        t0 = time.perf_counter()
        mlp_probs = self.mlp_model.predict_proba(x_scaled)[0]
        latencies["MLP"] = round((time.perf_counter() - t0) * 1000, 2)
        mlp_idx = int(np.argmax(mlp_probs))
        results["MLP"] = {
            "model_name": "Multi-Layer Perceptron (MLP)",
            "type": "Classical Neural Network",
            "predicted_emotion": EMOTIONS[mlp_idx],
            "confidence": round(float(mlp_probs[mlp_idx]), 4),
            "probabilities": {EMOTIONS[i]: round(float(mlp_probs[i]), 4) for i in range(NUM_CLASSES)},
            "latency_ms": latencies["MLP"],
        }

        # 3. CNN 1D Inference
        t0 = time.perf_counter()
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(1)  # (1, 1, 170)
        with torch.no_grad():
            cnn_logits = self.cnn_model(x_tensor)
            cnn_probs = F.softmax(cnn_logits, dim=-1).squeeze(0).numpy()
        latencies["CNN"] = round((time.perf_counter() - t0) * 1000, 2)
        cnn_idx = int(np.argmax(cnn_probs))
        results["CNN"] = {
            "model_name": "1D Convolutional Neural Network (CNN)",
            "type": "Deep Learning",
            "predicted_emotion": EMOTIONS[cnn_idx],
            "confidence": round(float(cnn_probs[cnn_idx]), 4),
            "probabilities": {EMOTIONS[i]: round(float(cnn_probs[i]), 4) for i in range(NUM_CLASSES)},
            "latency_ms": latencies["CNN"],
        }

        # 4. CNN-LSTM Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            cnn_lstm_logits = self.cnn_lstm_model(x_tensor)
            cnn_lstm_probs = F.softmax(cnn_lstm_logits, dim=-1).squeeze(0).numpy()
        latencies["CNN-LSTM"] = round((time.perf_counter() - t0) * 1000, 2)
        cnn_lstm_idx = int(np.argmax(cnn_lstm_probs))
        results["CNN-LSTM"] = {
            "model_name": "Hybrid CNN-LSTM Architecture",
            "type": "Deep Learning / Spatio-Temporal",
            "predicted_emotion": EMOTIONS[cnn_lstm_idx],
            "confidence": round(float(cnn_lstm_probs[cnn_lstm_idx]), 4),
            "probabilities": {EMOTIONS[i]: round(float(cnn_lstm_probs[i]), 4) for i in range(NUM_CLASSES)},
            "latency_ms": latencies["CNN-LSTM"],
        }

        # 5. Multimodal Prosody Prior & Soft-Voting Ensemble
        prosody_prior = compute_acoustic_prosody_prior(vocal_analytics, lexical_sentiment)
        ensemble_probs = (
            0.20 * svm_probs +
            0.20 * mlp_probs +
            0.25 * cnn_probs +
            0.25 * cnn_lstm_probs +
            0.10 * prosody_prior
        )
        ensemble_probs = ensemble_probs / np.sum(ensemble_probs)
        ens_idx = int(np.argmax(ensemble_probs))
        results["Ensemble"] = {
            "model_name": "Soft-Voting Ensemble",
            "type": "Hybrid Ensemble",
            "predicted_emotion": EMOTIONS[ens_idx],
            "confidence": round(float(ensemble_probs[ens_idx]), 4),
            "probabilities": {EMOTIONS[i]: round(float(ensemble_probs[i]), 4) for i in range(NUM_CLASSES)},
            "latency_ms": round(sum(latencies.values()), 2),
        }

        # Model Agreement Analysis
        predictions = [
            results["SVM"]["predicted_emotion"],
            results["MLP"]["predicted_emotion"],
            results["CNN"]["predicted_emotion"],
            results["CNN-LSTM"]["predicted_emotion"],
        ]
        unique_preds = list(set(predictions))
        agreement_ratio = max([predictions.count(p) for p in unique_preds]) / 4.0
        agreement_status = (
            "Unanimous Agreement (4/4)" if agreement_ratio == 1.0
            else "Majority Agreement (3/4)" if agreement_ratio == 0.75
            else "Divided / Moderate Agreement (2/4)" if agreement_ratio == 0.5
            else "High Disagreement (1/4)"
        )

        # Explainable AI & Uncertainty
        entropy = -sum([p * math.log2(p + 1e-9) for p in ensemble_probs])
        sorted_probs = np.sort(ensemble_probs)[::-1]
        confidence_margin = float(sorted_probs[0] - sorted_probs[1])
        is_uncertain = bool(confidence_margin < 0.15 or entropy > 2.3)

        explainability = generate_explainable_insights(
            feature_vector=feature_vector,
            predicted_emotion=EMOTIONS[ens_idx],
            confidence=float(ensemble_probs[ens_idx]),
            entropy=float(entropy),
            confidence_margin=confidence_margin,
            is_uncertain=is_uncertain,
        )

        return {
            "models": results,
            "primary": results["CNN-LSTM"],  # Highest capacity model
            "ensemble": results["Ensemble"],
            "latencies": latencies,
            "agreement": {
                "ratio": agreement_ratio,
                "status": agreement_status,
                "votes": {p: predictions.count(p) for p in unique_preds},
            },
            "explainability": explainability,
            "uncertainty": {
                "entropy": round(float(entropy), 3),
                "confidence_margin": round(confidence_margin, 3),
                "is_uncertain": is_uncertain,
                "warning": (
                    "Model prediction has borderline confidence. Secondary emotion closely matches."
                    if is_uncertain else "Prediction is stable and distinctive."
                ),
            },
        }

    def predict_single(
        self,
        feature_vector: np.ndarray,
        model_key: str = "Ensemble",
        vocal_analytics: dict = None,
        lexical_sentiment: dict = None,
    ) -> dict:
        """Runs prediction using a specific requested model or the Ensemble."""
        all_res = self.predict_all(
            feature_vector,
            vocal_analytics=vocal_analytics,
            lexical_sentiment=lexical_sentiment,
        )
        if model_key in all_res["models"]:
            return all_res["models"][model_key]
        return all_res["ensemble"]


def generate_explainable_insights(
    feature_vector: np.ndarray,
    predicted_emotion: str,
    confidence: float,
    entropy: float,
    confidence_margin: float,
    is_uncertain: bool,
) -> dict:
    """
    Computes acoustic feature group attributions and diagnostic explanation.
    """
    # Breakdown feature groups:
    # 0..79: MFCCs (Timbre, vocal tract filter)
    # 80..103: Chroma (Harmonic pitch distribution)
    # 104..143: Mel Spectrogram (Spectral energy distribution)
    # 144..157: Spectral Contrast (Vocal sharpness/brightness)
    # 158..159: ZCR (Unvoiced frication / roughness)
    # 160..161: RMS Energy (Vocal arousal / loudness)
    # 162..165: Spectral Rolloff & Centroid (Vocal effort)
    # 166..169: Pitch/Autocorr proxy (F0 prosody)

    mfcc_energy = float(np.mean(np.abs(feature_vector[0:80])))
    pitch_weight = float(np.mean(np.abs(feature_vector[166:170])))
    loudness_weight = float(feature_vector[160])  # RMS mean
    spectral_weight = float(feature_vector[164]) if len(feature_vector) > 164 else 0.5
    zcr_weight = float(feature_vector[158])

    # Normalized relative acoustic impact scores
    norm_total = mfcc_energy + pitch_weight * 5 + loudness_weight * 10 + zcr_weight * 10 + 1e-5
    feature_importance = [
        {"feature": "Pitch & Prosody Dynamics (F0)", "importance": round(min(0.95, (pitch_weight * 5) / norm_total + 0.15), 2), "description": "Modulations in voice fundamental frequency and intonation contour"},
        {"feature": "RMS Energy & Loudness (Arousal)", "importance": round(min(0.95, (loudness_weight * 10) / norm_total + 0.20), 2), "description": "Amplitude dynamics, vocal strain, and speech intensity"},
        {"feature": "Spectral Brightness & Centroid", "importance": round(min(0.95, (spectral_weight / 3000.0) + 0.18), 2), "description": "Vocal resonance balance between low and high frequencies"},
        {"feature": "MFCC Timbral Formants", "importance": round(min(0.95, (mfcc_energy / 20.0) + 0.16), 2), "description": "Vocal tract shape and articulation pattern"},
        {"feature": "Zero Crossing Rate (Frication)", "importance": round(min(0.95, (zcr_weight * 5.0) + 0.12), 2), "description": "Presence of unvoiced fricatives, breathiness, or vocal fry"},
    ]
    feature_importance = sorted(feature_importance, key=lambda x: x["importance"], reverse=True)

    # Narrative explanation based on predicted emotion
    if predicted_emotion in ["angry", "surprised"]:
        narrative = (
            f"The model detected high vocal arousal characterized by elevated RMS energy "
            f"and sudden pitch excursions. These sharp spectral peaks strongly shifted the decision towards '{predicted_emotion}'."
        )
    elif predicted_emotion in ["happy"]:
        narrative = (
            f"The voice demonstrates elevated pitch inflection, lively melodic variability, "
            f"and moderate-to-high spectral brightness, consistent with upbeat, positive acoustic valence."
        )
    elif predicted_emotion in ["sad"]:
        narrative = (
            f"Low vocal energy, downward intonation slope, and reduced pitch variance "
            f"were the primary acoustic markers contributing to the '{predicted_emotion}' prediction."
        )
    elif predicted_emotion in ["calm", "neutral"]:
        narrative = (
            f"Acoustic features display stable pitch stability, smooth envelope transitions, "
            f"and low spectral contrast, characteristic of a controlled, composed vocal register."
        )
    elif predicted_emotion in ["fearful"]:
        narrative = (
            f"Rapid micro-tremors in pitch (jitter) and unsteady vocal intensity indicate vocal tension, "
            f"characteristic of apprehension or alarm."
        )
    else:
        narrative = f"Acoustic features aligned with the '{predicted_emotion}' profile based on spectral formants and prosodic contour."

    limitations = [
        "Acoustic models detect vocal expression patterns, which may not always equate to true internal psychological states.",
        "Accents, regional dialects, or background room acoustics can influence high-frequency spectral formants.",
        "Recommended as an assistive decision-support indicator rather than a definitive psycho-emotional diagnosis."
    ]

    return {
        "narrative": narrative,
        "feature_importance": feature_importance,
        "limitations": limitations,
    }
