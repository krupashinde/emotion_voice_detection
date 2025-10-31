# 🎙️ Emotion Voice Detection using Deep Learning

This project detects human emotions (Happy, Sad, Angry, Neutral) from voice recordings using **Python**, **Librosa**, and **TensorFlow/Keras**.

---

## 🚀 Features
-  Trains a deep learning model (CNN) to classify emotions.  
-  Real-time emotion prediction using your voice.  
-  Supports multiple datasets (custom `.wav` files).  

---

## 📁 Project Structure

emotion_voice_project/
│
├── 📂 data/
│   ├── 📂 happy/
│   │   ├── OAF_happy1.wav
│   │   ├── OAF_happy2.wav
│   │   └── ... (more happy samples)
│   │
│   ├── 📂 sad/
│   │   ├── OAF_sad1.wav
│   │   ├── OAF_sad2.wav
│   │   └── ... (more sad samples)
│   │
│   ├── 📂 angry/
│   │   ├── OAF_angry1.wav
│   │   ├── OAF_angry2.wav
│   │   └── ... (more angry samples)
│   │
│   └── 📂 neutral/
│       ├── OAF_neutral1.wav
│       ├── OAF_neutral2.wav
│       └── ... (more neutral samples)
│
├── train.py                 # Script to extract features & train CNN model
├── app.py                   # Script for real-time emotion prediction
├── requirements.txt         # List of Python dependencies
├── emotion_model_small.h5   # Trained model file
├── emotion_voice_project.keras  # Optional: Saved Keras model format
└── README.md                # Project documentation

 




