import os
import numpy as np
import librosa
from keras.models import load_model
import pickle

MODEL_PATH = '. ./models/emotion_model.h5'
model = load_model(MODEL_PATH)

# Load label encoder from training
import pandas as pd
df = pd.read_pickle('../features/mfcc/mfcc.pkl')
labels = df['label'].unique()

# Rebuild label encoder for prediction
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(labels)

# Input audio file path
audio_file = input("Enter path to .wav file to predict: ")

if not os.path.exists(audio_file):
    print("File not found.")
    exit()

y, sr = librosa.load(audio_file, duration=3, offset=0.5)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc_scaled = np.mean(mfcc.T, axis=0)

prediction = model.predict(np.array([mfcc_scaled]))
predicted_label = le.inverse_transform([np.argmax(prediction)])
print("Predicted Emotion:", predicted_label[0])
