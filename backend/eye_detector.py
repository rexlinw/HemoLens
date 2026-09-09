import cv2
import numpy as np
from pathlib import Path


class EyeDetector:
    def __init__(self):
        self.supports_cascade = bool(
            hasattr(cv2, 'CascadeClassifier')
            and hasattr(cv2, 'data')
            and getattr(cv2.data, 'haarcascades', None)
        )

        self.eye_cascade = None
        self.face_cascade = None

        if self.supports_cascade:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                self.eye_cascade = cv2.CascadeClassifier(cascade_path)

                face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

                self.supports_cascade = not self.eye_cascade.empty() and not self.face_cascade.empty()
            except Exception:
                self.supports_cascade = False
                self.eye_cascade = None
                self.face_cascade = None

    def _fallback_quality_score(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rgb = image.astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-8)))

        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_fraction = float((skin_mask > 0).mean())

        brightness_score = 1.0 - min(abs(brightness - 120.0) / 120.0, 1.0)
        blur_score = min(blur / 120.0, 1.0)
        entropy_score = min(entropy / 6.0, 1.0)
        skin_score = min(max(skin_fraction, 0.02) / 0.18, 1.0)

        return float(max(0.0, min(1.0, 0.30 * brightness_score + 0.25 * blur_score + 0.20 * entropy_score + 0.25 * skin_score)))

    def _fallback_detect_eyes(self, image: np.ndarray) -> bool:
        return self._fallback_quality_score(image) >= 0.40

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
        if not self.supports_cascade or self.eye_cascade is None or self.face_cascade is None:
            return self._fallback_detect_eyes(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        face_box, eyes = self._best_face_and_eyes(enhanced)
        if face_box is not None and len(eyes) >= 2:
            return True
        return self._fallback_detect_eyes(image)

    def get_eye_quality_score(self, image: np.ndarray) -> float:
        if not self.supports_cascade or self.eye_cascade is None or self.face_cascade is None:
            return self._fallback_quality_score(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        face_box, eyes = self._best_face_and_eyes(enhanced)
        if face_box is None or len(eyes) < 2:
            return self._fallback_quality_score(image)

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
