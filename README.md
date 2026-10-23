# Speech Emotion Recognition

## Files
- `speech_emotion_recognition.py` - trains an MLPClassifier on RAVDESS using librosa features (MFCC, Chroma, Mel).
- `speech_emotion_ui.html` - animated browser demo (open it in Chrome, Edge or Firefox). It uses a lightweight in-browser estimate, not the trained model.
- `requirements.txt` - Python dependencies.

## Run the model
1. `pip install -r requirements.txt`
2. Download RAVDESS speech audio and extract it so you have `Actor_01`, `Actor_02`, and so on.
3. Set `DATA_DIR` at the top of `speech_emotion_recognition.py`.
4. `python speech_emotion_recognition.py`

Expect roughly 70% accuracy on calm, happy, fearful and disgust.

## Open the UI
Double-click `speech_emotion_ui.html`. Use the microphone, upload audio, or play the demo voice.
