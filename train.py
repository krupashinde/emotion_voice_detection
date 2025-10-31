import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# --- Paths ---
DATA_DIR = "data"
emotions = ["happy", "sad", "angry", "neutral"]

# --- Extract features ---
def extract_features(file_path):
    try:
        data, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print("❌ Error extracting", file_path, ":", e)
        return None

# --- Load data ---
X, y = [], []
for emotion in emotions:
    folder = os.path.join(DATA_DIR, emotion)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(folder, file))
            if features is not None:
                X.append(features)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples.")

# --- Encode labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# --- Model ---
model = Sequential([
    Dense(256, input_shape=(40,), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(emotions), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0005), metrics=['accuracy'])

# --- Train ---
model.fit(X, y_categorical, epochs=150, batch_size=32, validation_split=0.2)

# --- Save ---
model.save("emotion_voice_project.keras")
print("🎯 Model trained & saved successfully!")
