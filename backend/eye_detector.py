import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


class EyeDetector:
    """Detect whether an image contains a human eye.

    Uses Haar cascades as the primary detector and a multi-signal
    eye-feature analyser as the fallback.  The fallback checks for
    sclera, iris/pupil circles, conjunctiva colour and eyelid edges
    so that non-eye images (landscapes, objects, hands …) are rejected.
    """

    # Minimum number of secondary signals required to accept an image
    # when the cascade detector does not fire.
    MIN_EYE_SIGNALS = 2

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

    # ------------------------------------------------------------------
    # Secondary eye-feature signals (used when cascades don't fire)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_unnatural_colors(rgb: np.ndarray, threshold: float = 0.035) -> bool:
        """Detect non-biological colors that cannot appear in an eye/skin capture:
        - Vivid green / lime: H in [35, 85], S > 55, V > 55
        - Vivid cyan / sky blue: H in [85, 110], S > 65, V > 65
        - Vivid neon yellow: H in [22, 34], S > 120, V > 90
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        green = (h >= 35) & (h <= 85) & (s > 55) & (v > 55)
        cyan = (h > 85) & (h <= 110) & (s > 65) & (v > 65)
        yellow = (h >= 22) & (h <= 34) & (s > 120) & (v > 90)
        frac = float((green | cyan | yellow).mean())
        return frac > threshold

    @staticmethod
    def _detect_sclera(hsv: np.ndarray, image_area: float) -> bool:
        """Look for a white/near-white region (sclera) via HSV thresholds.

        Sclera is characterised by low saturation and high value.
        We expect it to occupy roughly 2–38 % of the image in a
        close-up eye photo, and it must form a coherent region
        (not random scattered pixels or an all-encompassing white table).
        """
        mask = cv2.inRange(hsv, (0, 0, 160), (180, 50, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        total_white = float((mask > 0).sum())
        white_fraction = total_white / image_area
        if not (0.02 <= white_fraction <= 0.38):
            return False

        # Coherence: largest connected blob must be ≥ 35% of total white area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(cv2.contourArea(c) for c in contours)
        if largest < total_white * 0.35:
            return False

        # Border check: a white tabletop or canvas backdrop spans the full boundary
        top = float((mask[0, :] > 0).mean())
        bottom = float((mask[-1, :] > 0).mean())
        left = float((mask[:, 0] > 0).mean())
        right = float((mask[:, -1] > 0).mean())
        if sum([top > 0.45, bottom > 0.45, left > 0.45, right > 0.45]) >= 3:
            return False

        return True

    @staticmethod
    def _detect_iris_pupil(gray: np.ndarray, hsv: Optional[np.ndarray] = None, rgb: Optional[np.ndarray] = None) -> bool:
        """Detect circular iris/pupil shapes with HoughCircles.
        
        Verifies that detected circles are genuine ocular features rather than
        uniform bright circular discs (mugs, plates, watches, lights) or red fruit (apples, tomatoes).
        """
        h, w = gray.shape[:2]
        min_dim = min(h, w)
        min_r = max(int(min_dim * 0.04), 5)
        max_r = max(int(min_dim * 0.35), min_r + 5)

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.4,
            minDist=min_dim // 4,
            param1=60,
            param2=35,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            return False

        if hsv is None:
            white_mask = gray > 200
        else:
            white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 45, 255)) > 0

        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), max(2, int(r * 0.85)), 255, -1)
            pts = mask > 0
            if pts.sum() == 0:
                continue
            white_frac = float((white_mask[pts] > 0).mean())
            mean_gray = float(gray[pts].mean())
            # Human irises are pigmented and have pupil aperture; they are NOT solid white discs
            if white_frac > 0.35 or mean_gray > 215:
                continue
            # Circle interior must have texture / contrast (not flat solid disc)
            if float(gray[pts].std()) < 6.0:
                continue

            # Reject red fruit/objects (apples, tomatoes, red balls)
            if rgb is not None:
                r_c = float(rgb[pts, 0].mean())
                g_c = float(rgb[pts, 1].mean())
                b_c = float(rgb[pts, 2].mean())
                if r_c - g_c > 35 and r_c - b_c > 35:
                    continue

            # Check pupil aperture in center: must be dark
            pupil_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(pupil_mask, (cx, cy), max(2, int(r * 0.35)), 255, -1)
            p_pts = pupil_mask > 0
            if p_pts.sum() > 0:
                pupil_gray = float(gray[p_pts].mean())
                if pupil_gray > 85.0:
                    continue

            return True

        return False

    @staticmethod
    def _check_periorbital_skin(rgb: np.ndarray) -> bool:
        """Check for human facial/periorbital skin tissue around the eye."""
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 175, 127))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask_hsv = cv2.inRange(hsv, (0, 15, 30), (25, 200, 255))
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        rgb_rule = (r >= g) & (g >= b) & (r - b > 10)
        mask = mask_ycrcb & mask_hsv & rgb_rule
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        total = int((mask > 0).sum())
        frac = float(total) / float(rgb.shape[0] * rgb.shape[1] + 1e-6)
        if frac < 0.03:
            return False
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(cv2.contourArea(c) for c in contours)
        return largest >= total * 0.35

    @staticmethod
    def _check_conjunctiva_color(rgb: np.ndarray) -> bool:
        """Check whether the image contains pinkish-red tissue (conjunctiva).

        Conjunctiva tissue sits in a specific hue range (roughly 0–20 or
        155–180 in OpenCV HSV) and has moderate-to-high saturation with
        a reddish R-channel dominance. It forms an elongated crescent
        bordering the lower eyelid, NOT a spherical fruit.
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask_low = cv2.inRange(hsv, (0, 35, 70), (18, 200, 240))
        mask_high = cv2.inRange(hsv, (155, 35, 70), (180, 200, 240))
        mask = cv2.bitwise_or(mask_low, mask_high)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        total_pink = float((mask > 0).sum())
        image_area = float(rgb.shape[0] * rgb.shape[1])
        pink_fraction = total_pink / image_area

        if not (0.03 <= pink_fraction <= 0.65):
            return False

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        # Reject round spherical fruit (apples, tomatoes, balls)
        valid_mucosa = False
        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area < total_pink * 0.25:
                continue
            peri = cv2.arcLength(c, True)
            circ = 4 * np.pi * c_area / (peri * peri + 1e-6)
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = float(bw) / float(bh + 1e-6)

            # Spherical fruit has high circularity and ~1:1 aspect ratio
            if circ > 0.68 and 0.80 <= aspect <= 1.25 and c_area > 0.12 * image_area:
                continue

            valid_mucosa = True
            break

        return valid_mucosa

    @staticmethod
    def _detect_eyelid_edges(gray: np.ndarray) -> bool:
        """Look for strong horizontal curved edges (eyelid margins).

        Eyelids produce dominant horizontal edges in the image.
        We check that horizontal Sobel energy clearly exceeds vertical
        Sobel energy and that edge density is in a realistic range
        (too high = noise, too low = no edges at all).
        """
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        h_energy = float(np.abs(sobel_h).mean())
        v_energy = float(np.abs(sobel_v).mean())
        edge_density = float((np.abs(sobel_h) > 30).mean())
        return (
            h_energy > 12.0
            and 0.05 <= edge_density <= 0.55
            and h_energy > v_energy * 1.2
        )

    # ------------------------------------------------------------------
    # Orchestrator — combine signals with anatomical rules
    # ------------------------------------------------------------------

    def _detect_eye_features(self, image: np.ndarray) -> tuple:
        """Run secondary eye-feature checks.

        Returns (is_eye: bool, signal_count: int, signal_names: list[str]).
        An image is accepted only when it exhibits genuine ocular tissue
        and matches either the palpebral conjunctiva pattern or an
        open-eye anatomical structure.
        """
        rgb = image.astype(np.uint8)
        if len(rgb.shape) == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

        # 1. Unnatural non-biological color rejection (limes, yellow banana, neon annotations)
        if self._has_unnatural_colors(rgb):
            return False, 0, []

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        image_area = float(gray.shape[0] * gray.shape[1])

        signals: list[str] = []

        has_sclera = self._detect_sclera(hsv, image_area)
        if has_sclera:
            signals.append("sclera")

        has_iris = self._detect_iris_pupil(gray, hsv, rgb)
        if has_iris:
            signals.append("iris_pupil")

        has_conjunctiva = self._check_conjunctiva_color(rgb)
        if has_conjunctiva:
            signals.append("conjunctiva_color")

        has_skin = self._check_periorbital_skin(rgb)
        if has_skin:
            signals.append("periorbital_skin")

        has_eyelids = self._detect_eyelid_edges(gray)
        if has_eyelids and (has_skin or has_conjunctiva):
            signals.append("eyelid_edges")

        # Eye structural patterns:
        # Route A: Palpebral conjunctiva close-up (clinical capture)
        # Sclera + mucosal conjunctiva
        route_a = has_sclera and has_conjunctiva

        # Route B: Open eye close-up
        # Iris/pupil + flanking sclera
        route_b = has_iris and has_sclera

        is_eye = (route_a or route_b)
        return is_eye, len(signals), signals

    # ------------------------------------------------------------------
    # Quality score (meaningful only when eye features are present)
    # ------------------------------------------------------------------

    def _compute_eye_quality(self, image: np.ndarray) -> float:
        """Return a quality score in [0, 1].

        Returns 0.0 when no eye features are detected.  When eye
        features *are* detected the score captures brightness, focus
        and signal strength.
        """
        is_eye, signal_count, _ = self._detect_eye_features(image)
        if not is_eye:
            return 0.0

        rgb = image.astype(np.uint8)
        if len(rgb.shape) == 2:
            gray = rgb
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        brightness_score = 1.0 - min(abs(brightness - 120.0) / 120.0, 1.0)
        blur_score = min(blur / 120.0, 1.0)
        signal_score = min(signal_count / 4.0, 1.0)

        return float(max(0.0, min(1.0,
            0.30 * brightness_score + 0.30 * blur_score + 0.40 * signal_score
        )))

    # ------------------------------------------------------------------
    # Cascade helpers (unchanged from original)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_eyes(self, image: np.ndarray) -> bool:
        """Return True only if the image plausibly contains a human eye."""
        # Try cascade first
        if self.supports_cascade and self.eye_cascade is not None and self.face_cascade is not None:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            face_box, eyes = self._best_face_and_eyes(enhanced)
            if face_box is not None and len(eyes) >= 2:
                return True

        # Cascade failed — use multi-signal fallback (NOT generic quality)
        is_eye, _, _ = self._detect_eye_features(image)
        return is_eye

    def get_eye_quality_score(self, image: np.ndarray) -> float:
        """Return eye quality score.  0.0 when no eye features detected."""
        if self.supports_cascade and self.eye_cascade is not None and self.face_cascade is not None:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            face_box, eyes = self._best_face_and_eyes(enhanced)
            if face_box is not None and len(eyes) >= 2:
                face_w = float(face_box[2])
                face_h = float(face_box[3])
                image_area = float(gray.shape[0] * gray.shape[1] + 1e-6)
                face_area_ratio = float(face_box[2] * face_box[3]) / image_area
                eye_strength = min(len(eyes) / 2.0, 1.0)
                size_strength = min(face_area_ratio / 0.22, 1.0)
                aspect_strength = 1.0 - min(abs((face_w / (face_h + 1e-6)) - 1.0), 0.6)

                return float(max(0.0, min(1.0,
                    0.45 * eye_strength + 0.35 * size_strength + 0.20 * aspect_strength
                )))

        # No cascade hit — use secondary feature quality
        return self._compute_eye_quality(image)


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
