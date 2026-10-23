import os
import sys

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from audio_processor import load_audio_from_path, extract_features_vector, compute_vocal_analytics
from ml_models import ModelManager

mm = ModelManager.get_instance()
mm.initialize_or_bootstrap()

test_files = [
    "demo_audio/angry_speech.wav",
    "demo_audio/happy_speech.wav",
    "demo_audio/sad_speech.wav",
    "demo_audio/calm_speech.wav",
    "demo_audio/fearful_speech.wav",
    "uploads/76e6c30a_sad.wav",
    "uploads/831b03a8_happy.wav",
    "uploads/92e96410_angry.wav",
]

print("\n" + "=" * 90)
print(" VERIFYING MODEL MANAGER PREDICTIONS ON REAL AUDIO FILES")
print("=" * 90)

for rel in test_files:
    fpath = os.path.join(backend_dir, rel)
    if not os.path.exists(fpath):
        continue
    y, sr = load_audio_from_path(fpath)
    va = compute_vocal_analytics(y, sr)
    feat = extract_features_vector(y, sr)
    res = mm.predict_all(feat, vocal_analytics=va)
    
    fname = os.path.basename(rel)
    models_summary = " | ".join([f"{k}: {v['predicted_emotion']}" for k, v in res['models'].items() if k != 'Ensemble'])
    print(f"{fname:<25} -> Ensemble: {res['ensemble']['predicted_emotion']:<8} ({res['ensemble']['confidence']*100:.1f}%) | {res['agreement']['status']}")
    print(f"   Individual Models: {models_summary}")
    print("-" * 90)
