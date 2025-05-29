import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import os

FEATURE_PATH = 'task2/features/mfcc/mfcc.pkl'
MODEL_PATH = 'task2/models/emotion_model.h5'
os.makedirs('task2/models', exist_ok=True)

df = pd.read_pickle(FEATURE_PATH)
if df.empty:
    raise ValueError("No features found. Run preprocess.py first.")

X = np.array(df['feature'].tolist())
y = np.array(df['label'].tolist())

le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

model.save(MODEL_PATH)
print("Model trained and saved.")
