import cv2
import numpy as np
from pathlib import Path


class EyeDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(cascade_path)

        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def _eyes_look_plausible(self, face_box, eye_boxes) -> bool:
        if face_box is None or len(eye_boxes) < 2:
            return False

        x, y, w, h = face_box
        sorted_eyes = sorted(eye_boxes, key=lambda box: box[0])[:2]
        (x1, y1, w1, h1), (x2, y2, w2, h2) = sorted_eyes

        c1x = x1 + w1 / 2.0
        c1y = y1 + h1 / 2.0
        c2x = x2 + w2 / 2.0
        c2y = y2 + h2 / 2.0

        face_w = float(w)
        face_h = float(h)
        face_center_x = x + face_w / 2.0
        face_center_y = y + face_h / 2.0

        eye_span = abs(c2x - c1x)
        eye_mid_y = (c1y + c2y) / 2.0

        if not (0.20 * face_w <= eye_span <= 0.88 * face_w):
            return False
        if abs(c1y - c2y) > 0.18 * face_h:
            return False
        if eye_mid_y > y + 0.68 * face_h:
            return False
        if abs((c1x + c2x) / 2.0 - face_center_x) > 0.24 * face_w:
            return False
        if abs(eye_mid_y - face_center_y) > 0.22 * face_h:
            return False

        return True

    def _best_face_and_eyes(self, gray: np.ndarray):
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=10,
            minSize=(70, 70),
        )

        best_face = None
        best_eyes = []
        for (x, y, w, h) in faces:
            face_area = float(w * h)
            image_area = float(gray.shape[0] * gray.shape[1] + 1e-6)
            face_area_ratio = face_area / image_area
            if face_area_ratio < 0.04 or face_area_ratio > 0.55:
                continue

            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.08,
                minNeighbors=12,
                minSize=(22, 22),
            )
            if self._eyes_look_plausible((x, y, w, h), eyes):
                if best_face is None or face_area > best_face[2] * best_face[3]:
                    best_face = (x, y, w, h)
                    best_eyes = eyes

        return best_face, best_eyes

    def detect_eyes(self, image: np.ndarray) -> bool:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        face_box, eyes = self._best_face_and_eyes(enhanced)
        return face_box is not None and len(eyes) >= 2

    def get_eye_quality_score(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        face_box, eyes = self._best_face_and_eyes(enhanced)
        if face_box is None or len(eyes) < 2:
            return 0.0

        face_w = float(face_box[2])
        face_h = float(face_box[3])
        image_area = float(gray.shape[0] * gray.shape[1] + 1e-6)
        face_area_ratio = float(face_box[2] * face_box[3]) / image_area
        eye_strength = min(len(eyes) / 2.0, 1.0)
        size_strength = min(face_area_ratio / 0.22, 1.0)
        aspect_strength = 1.0 - min(abs((face_w / (face_h + 1e-6)) - 1.0), 0.6)

        return float(max(0.0, min(1.0, 0.45 * eye_strength + 0.35 * size_strength + 0.20 * aspect_strength)))


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
