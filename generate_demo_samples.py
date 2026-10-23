import os
import numpy as np
import soundfile as sf

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_audio")

def generate_formant_voice(
    duration: float = 4.0,
    sr: int = 22050,
    base_f0: float = 160.0,
    pitch_mod_freq: float = 2.0,
    pitch_mod_depth: float = 15.0,
    rms_target: float = 0.08,
    brightness: float = 1.0,
    jitter: float = 0.0,
) -> np.ndarray:
    """
    Synthesizes speech-like harmonic vocal tone with formants, prosody, and breath noise.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Intonation contour (F0)
    pitch_contour = base_f0 + pitch_mod_depth * np.sin(2 * np.pi * pitch_mod_freq * t)
    if jitter > 0:
        pitch_contour += np.random.normal(0, jitter, len(t))

    # Phase accumulation
    phase = 2 * np.pi * np.cumsum(pitch_contour) / sr

    # Harmonic series (voice source)
    signal = np.zeros_like(t)
    harmonics = [1.0, 0.7 * brightness, 0.5 * brightness, 0.3 * (brightness**1.5), 0.2 * (brightness**2)]
    for idx, amp in enumerate(harmonics, start=1):
        signal += amp * np.sin(phase * idx)

    # Vocal tract formants (F1 ~ 500Hz, F2 ~ 1500Hz, F3 ~ 2500Hz)
    f1 = 0.4 * np.sin(2 * np.pi * 500 * t)
    f2 = 0.25 * np.sin(2 * np.pi * 1500 * t)
    signal = signal * (1.0 + f1 + f2)

    # Speech envelope (syllabic rhythm at ~3.5 Hz)
    envelope = (
        0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t - np.pi / 2)) *
        np.exp(-((t % 0.35) / 0.15))
    )
    envelope = np.clip(envelope, 0.05, 1.0)
    # Smooth start/end
    fade_samples = int(sr * 0.1)
    if len(t) > 2 * fade_samples:
        envelope[:fade_samples] *= np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    speech = signal * envelope

    # Normalize to target RMS
    cur_rms = np.sqrt(np.mean(speech**2)) + 1e-6
    speech = speech * (rms_target / cur_rms)

    # Add light ambient breath noise
    speech += np.random.normal(0, 0.003, len(t))

    return np.clip(speech, -0.95, 0.95).astype(np.float32)


def create_all_demo_files():
    os.makedirs(DEMO_DIR, exist_ok=True)
    sr = 22050

    demos = {
        "happy_speech.wav": generate_formant_voice(
            duration=5.0, sr=sr, base_f0=220.0, pitch_mod_freq=2.8, pitch_mod_depth=35.0,
            rms_target=0.10, brightness=1.4, jitter=1.5
        ),
        "calm_speech.wav": generate_formant_voice(
            duration=5.0, sr=sr, base_f0=130.0, pitch_mod_freq=0.8, pitch_mod_depth=8.0,
            rms_target=0.04, brightness=0.7, jitter=0.2
        ),
        "angry_speech.wav": generate_formant_voice(
            duration=4.5, sr=sr, base_f0=260.0, pitch_mod_freq=3.5, pitch_mod_depth=45.0,
            rms_target=0.15, brightness=1.8, jitter=4.0
        ),
        "fearful_speech.wav": generate_formant_voice(
            duration=4.5, sr=sr, base_f0=240.0, pitch_mod_freq=5.0, pitch_mod_depth=25.0,
            rms_target=0.06, brightness=1.1, jitter=6.5
        ),
        "sad_speech.wav": generate_formant_voice(
            duration=5.5, sr=sr, base_f0=115.0, pitch_mod_freq=0.5, pitch_mod_depth=5.0,
            rms_target=0.03, brightness=0.5, jitter=0.8
        ),
    }

    # Generate customer support multi-turn call (~15 seconds)
    turn1_cust_angry = generate_formant_voice(duration=3.5, sr=sr, base_f0=250.0, pitch_mod_depth=40.0, rms_target=0.13, brightness=1.6)
    turn2_agent_calm = generate_formant_voice(duration=3.5, sr=sr, base_f0=140.0, pitch_mod_depth=10.0, rms_target=0.05, brightness=0.8)
    turn3_cust_tense = generate_formant_voice(duration=3.5, sr=sr, base_f0=190.0, pitch_mod_depth=20.0, rms_target=0.08, brightness=1.1)
    turn4_agent_reassure = generate_formant_voice(duration=3.5, sr=sr, base_f0=135.0, pitch_mod_depth=8.0, rms_target=0.05, brightness=0.8)
    call_audio = np.concatenate([turn1_cust_angry, turn2_agent_calm, turn3_cust_tense, turn4_agent_reassure])
    demos["customer_support_call.wav"] = call_audio

    for filename, audio in demos.items():
        out_path = os.path.join(DEMO_DIR, filename)
        sf.write(out_path, audio, sr)
        print(f"Created demo audio: {out_path} ({len(audio)/sr:.1f}s)")

if __name__ == "__main__":
    create_all_demo_files()
