import os
import numpy as np
import pandas as pd
import librosa
import pickle

DATA_PATH = 'task2/data'
FEATURE_PATH = 'task2/features/mfcc'
os.makedirs(FEATURE_PATH, exist_ok=True)

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features():
    data = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.wav'):
                try:
                    path = os.path.join(root, file)
                    y, sr = librosa.load(path, duration=3, offset=0.5)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    mfcc_scaled = np.mean(mfcc.T, axis=0)
                    emotion_code = file.split('-')[2]
                    emotion = emotion_map.get(emotion_code)
                    if emotion:
                        data.append([mfcc_scaled, emotion])
                    else:
                        print(f"Unknown emotion code: {emotion_code}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    df = pd.DataFrame(data, columns=['feature', 'label'])
    print(f"Extracted features from {len(df)} files.")
    df.to_pickle(os.path.join(FEATURE_PATH, 'mfcc.pkl'))

if __name__ == "__main__":
    extract_features()
    print("Features extracted and saved.")
