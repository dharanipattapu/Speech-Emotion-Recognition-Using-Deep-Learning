import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Title and UI setup
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` audio file and the app will predict the speaker's emotion.")

# Load trained model
MODEL_PATH = 'task2/models/emotion_model.h5'
model = load_model(MODEL_PATH)

# Load label encoder
df = pd.read_pickle('task2/features/mfcc/mfcc.pkl')
labels = df['label'].unique()
le = LabelEncoder()
le.fit(labels)

# Upload .wav file
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Process audio and predict emotion
    try:
        y, sr = librosa.load(uploaded_file, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)

        prediction = model.predict(np.array([mfcc_scaled]))
        predicted_label = le.inverse_transform([np.argmax(prediction)])

        st.success(f"🧠 Predicted Emotion: **{predicted_label[0]}**")

    except Exception as e:
        st.error(f"⚠️ Error processing audio file: {e}")
