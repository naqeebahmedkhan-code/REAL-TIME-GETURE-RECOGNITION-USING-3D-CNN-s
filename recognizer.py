import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

class GestureRecognizer:
    def __init__(self):
        with open("models/gesture-model.json", "r") as f:
            self.model = model_from_json(f.read())
        self.model.load_weights("models/gesture-model.weights.h5")
        self.labels = ['move-right', 'move-left', 'fist', 'no-gesture']

    def predict(self, frame):
        processed = cv2.resize(frame, (120, 120))
        processed = np.expand_dims(processed, axis=[0, -1]) / 255.0
        prediction = self.model.predict(processed, verbose=0)
        return self.labels[np.argmax(prediction)]
