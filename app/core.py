import os
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@dataclass
class SignLanguageConfig:
    """Configuration for the Sign Language Detection System."""
    data_dir: str = "data_collection"
    model_file: str = "model.h5"
    weights_file: str = "model_weights.h5"
    sequence_length: int = 30
    sequences_per_sign: int = 10
    epochs: int = 2000
    feature_length: int = 1662  # MediaPipe Holistic full feature vector

class FeatureExtractor:
    """Handles MediaPipe Holistic detection and feature extraction."""
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def get_model(self):
        return self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame, model):
        """Process a single frame and return image + results."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        """Draws landmarks with a custom style."""
        # Face
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
            self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        # Pose
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
        # Left Hand
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
        # Right Hand
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    def extract_keypoints(self, results) -> np.ndarray:
        """Extracts flattened keypoints from results."""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

class DataCollector:
    """Manages data collection and storage."""
    def __init__(self, config: SignLanguageConfig):
        self.config = config
        self.extractor = FeatureExtractor()

    def setup_directories(self, actions: List[str]):
        """Creates necessary directories for data collection."""
        for action in actions:
            for sequence in range(self.config.sequences_per_sign):
                try:
                    os.makedirs(os.path.join(self.config.data_dir, action, str(sequence)))
                except:
                    pass

    def save_frame(self, action: str, sequence: int, frame_num: int, keypoints: np.ndarray):
        """Saves a single frame's keypoints."""
        npy_path = os.path.join(self.config.data_dir, action, str(sequence), str(frame_num))
        np.save(npy_path, keypoints)

    def load_data(self, actions: List[str]):
        """Loads collected data for training."""
        sequences, labels = [], []
        label_map = {label:num for num, label in enumerate(actions)}

        for action in actions:
            for sequence in range(self.config.sequences_per_sign):
                window = []
                for frame_num in range(self.config.sequence_length):
                    res = np.load(os.path.join(self.config.data_dir, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])

        return np.array(sequences), to_categorical(labels).astype(int)

class SignModel:
    """Manages the LSTM model."""
    def __init__(self, config: SignLanguageConfig):
        self.config = config
        self.model = None

    def build(self, num_actions: int):
        """Builds the LSTM architecture."""
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.config.sequence_length, self.config.feature_length)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_actions, activation='softmax'))
        
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model = model
        return model

    def train(self, X, y):
        """Trains the model."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        
        # Create a callback or return history if needed, but for now just fit
        history = self.model.fit(X_train, y_train, epochs=self.config.epochs, validation_data=(X_test, y_test))
        return history

    def save(self):
        """Saves the model weights."""
        if self.model:
            self.model.save(self.config.model_file)
            self.model.save_weights(self.config.weights_file)
            # Actually, let's just use model.save which saves everything
            self.model.save(self.config.model_file)

    def load(self, model_path: str = None):
        """Loads the model."""
        path = model_path if model_path else self.config.model_file
        if os.path.exists(path):
            self.model = load_model(path)
            return True
        return False

    def predict(self, sequence: np.ndarray):
        """Predicts the sign from a sequence."""
        if self.model:
            res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
            return res
        return None
