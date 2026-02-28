import cv2
import numpy as np
from pathlib import Path


class EyeDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(cascade_path)

        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect_eyes(self, image: np.ndarray) -> bool:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            return False

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=8,
                minSize=(20, 20)
            )

            if len(eyes) >= 2:
                return True

        return False

    def get_eye_quality_score(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return 0.0

        max_eyes = 0
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
            )
            max_eyes = max(max_eyes, len(eyes))

        return min(max_eyes / 2.0, 1.0)


def get_hemoglobin_status(value: float) -> dict:
    if value < 12.0:
        return {
            "status": "LOW",
            "color": "#FF5252",
            "message": "⚠️ Low hemoglobin level - Consult a doctor",
            "severity": "warning"
        }
    elif value < 13.5:
        return {
            "status": "BORDERLINE",
            "color": "#FFC107",
            "message": "⚡ Borderline hemoglobin level - Monitor your health",
            "severity": "caution"
        }
    elif value <= 17.5:
        return {
            "status": "SAFE",
            "color": "#4CAF50",
            "message": "✓ Hemoglobin level is healthy",
            "severity": "safe"
        }
    else:
        return {
            "status": "HIGH",
            "color": "#FF9800",
            "message": "⚠️ High hemoglobin level - Consult a doctor",
            "severity": "warning"
        }
