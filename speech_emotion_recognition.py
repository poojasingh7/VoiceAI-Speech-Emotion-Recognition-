"""
Speech Emotion Recognition with librosa + sklearn (MLPClassifier) on RAVDESS.

Setup:
    pip install librosa soundfile numpy scikit-learn

Dataset:
    Download RAVDESS (Audio_Speech_Actors_01-24) and extract so the layout is:
        <DATA_DIR>/Actor_01/03-01-01-01-01-01-01.wav
        <DATA_DIR>/Actor_02/...
    Then set DATA_DIR below.
"""

import glob
import os

import librosa
import numpy as np
import soundfile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ---- Configuration ---------------------------------------------------------
DATA_DIR = r"C:\path\to\ravdess data"   # <- change this

# RAVDESS filename field 3 -> emotion
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Emotions we want to classify
OBSERVED_EMOTIONS = ["calm", "happy", "fearful", "disgust"]


# ---- Feature extraction ----------------------------------------------------
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    """Extract MFCC, Chroma and Mel features from a sound file."""
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        # Convert stereo to mono if needed
        if X.ndim > 1:
            X = X.mean(axis=1)

        result = np.array([])

        if chroma:
            stft = np.abs(librosa.stft(X))

        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0
            )
            result = np.hstack((result, mfccs))

        if chroma:
            chroma_feat = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, chroma_feat))

        if mel:
            mel_feat = np.mean(
                librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, mel_feat))

    return result


# ---- Data loading ----------------------------------------------------------
def load_data(test_size=0.25):
    x, y = [], []
    pattern = os.path.join(DATA_DIR, "Actor_*", "*.wav")

    for file in glob.glob(pattern):
        file_name = os.path.basename(file)
        emotion = EMOTIONS[file_name.split("-")[2]]

        if emotion not in OBSERVED_EMOTIONS:
            continue

        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


def main():
    # 1. Load data and split into train/test (75% / 25%)
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)

    print(f"Samples -> train: {x_train.shape[0]}, test: {x_test.shape[0]}")
    print(f"Features extracted: {x_train.shape[1]}")

    # 2. Initialize the Multi Layer Perceptron classifier
    model = MLPClassifier(
        alpha=0.01,
        batch_size=256,
        epsilon=1e-08,
        hidden_layer_sizes=(300,),
        learning_rate="adaptive",
        max_iter=500,
    )

    # 3. Train
    model.fit(x_train, y_train)

    # 4. Predict and evaluate
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
