import librosa
import numpy as np
from keras.models import load_model
import sounddevice as sd
import wavio

# Load your trained model
model = load_model("emotion_voice_project.keras")

# Emotions (same order as during training)
EMOTIONS = ["happy", "sad", "angry", "neutral"]

def extract_features(file_path):
    data, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    return np.expand_dims(mfccs, axis=0)

def record_audio(duration=3, fs=44100):
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write("test.wav", audio, fs, sampwidth=2)
    print("✅ Recording complete. Saved as test.wav")

def predict_emotion():
    features = extract_features("test.wav")
    prediction = model.predict(features)
    emotion = EMOTIONS[np.argmax(prediction)]
    print(f"🎯 Detected Emotion: {emotion}")

if __name__ == "__main__":
    record_audio()
    predict_emotion()
