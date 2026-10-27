"""
Reject non-clinical images before hemoglobin prediction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from eye_detector import EyeDetector
from nail_feature_extraction import NailFeatureExtractor
from palm_feature_extraction import PalmFeatureExtractor


@dataclass
class ValidationResult:
    valid: bool
    score: float
    message: str


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image.astype(np.uint8)


def _basic_image_checks(rgb: np.ndarray, modality: str = "general") -> Optional[str]:
    h, w = rgb.shape[:2]
    min_dim = 24 if modality == "nail" else 60
    if min(h, w) < min_dim:
        return "Image resolution is too low."

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 20:
        return "Image is too dark."
    if mean_brightness > 250:
        return "Image is overexposed."

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var > 20000.0:
        return "Image is too noisy or corrupted."

    min_blur = 2.0 if modality == "palm" else (8.0 if modality == "nail" else 25.0)
    if lap_var < min_blur:
        return "Image is too blurry. Use a sharper, well-focused photo."

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-8)))
    min_entropy = 2.0 if modality in ("nail", "palm") else 3.2
    if entropy < min_entropy:
        return "Image lacks enough detail for analysis."

    return None


def _skin_fraction(rgb: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return float((mask > 0).mean())


def _skin_blob_fraction(rgb: np.ndarray) -> Tuple[float, float]:
    """Return (total_skin_fraction, largest_coherent_skin_blob_fraction)."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    total_skin = float((mask > 0).mean())
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return total_skin, 0.0
    largest_blob = float(stats[1:, cv2.CC_STAT_AREA].max() / float(rgb.shape[0] * rgb.shape[1] + 1e-6))
    return total_skin, largest_blob


def _largest_box(boxes):
    return max(boxes, key=lambda box: box[2] * box[3]) if len(boxes) else None


def _eyes_look_plausible(face_box, eye_boxes) -> bool:
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

    if not (0.18 * face_w <= eye_span <= 0.92 * face_w):
        return False

    if abs(c1y - c2y) > 0.22 * face_h:
        return False

    if eye_mid_y > y + 0.72 * face_h:
        return False

    if abs((c1x + c2x) / 2.0 - face_center_x) > 0.30 * face_w:
        return False

    if abs(eye_mid_y - face_center_y) > 0.28 * face_h:
        return False

    return True


def _has_eye_features(image: np.ndarray, eye_detector: EyeDetector) -> bool:
    """Return True only when the image has concrete eye-like structures.

    Uses the multi-signal detector inside EyeDetector (sclera, iris,
    conjunctiva colour, eyelid edges).  This is the gate that prevents
    generic "nice looking" images from being accepted.
    """
    is_eye, _, _ = eye_detector._detect_eye_features(image)
    return is_eye


def validate_eye(image: np.ndarray, eye_detector: EyeDetector) -> ValidationResult:
    rgb = _to_rgb(image)
    basic = _basic_image_checks(rgb, modality="eye")
    if basic:
        return ValidationResult(False, 0.0, basic)

    # Reject non-biological color pollution (fruit, foliage, cyan/neon graphics)
    if hasattr(eye_detector, "_has_unnatural_colors") and eye_detector._has_unnatural_colors(rgb):
        return ValidationResult(
            False,
            0.0,
            "This image contains non-biological colors (fruit, foliage, or graphic overlay) and is not an eye. Please upload a clear eye or conjunctiva capture."
        )

    # ---- Step 1: check for eye features (multi-signal) ----
    has_features = _has_eye_features(rgb, eye_detector)

    # Reject non-eye skin images (e.g. a palm or hand dropped into the eye slot)
    skin_frac = _skin_fraction(rgb)
    if skin_frac > 0.35 and not has_features:
        return ValidationResult(
            False,
            0.0,
            "This looks like a hand, palm or plain-skin photo, not an eye. "
            "Capture a close-up of the eye, or pull down the lower eyelid so the "
            "white of the eye or the pink inner lid is clearly visible."
        )

    if not has_features:
        # Last resort: try cascade path
        detected_by_cascade = False
        if getattr(eye_detector, "supports_cascade", False):
            detected_by_cascade = eye_detector.detect_eyes(rgb)

        if not detected_by_cascade:
            return ValidationResult(
                False,
                0.0,
                "This does not appear to be an eye image. "
                "Please upload a clear, close-up photo of the eye or palpebral conjunctiva.",
            )

    # ---- Step 2: cascade-based refinements (face + eye geometry) ----
    quality = float(eye_detector.get_eye_quality_score(rgb))
    if quality <= 0.0:
        return ValidationResult(
            False,
            0.0,
            "This does not appear to be an eye image. "
            "Please upload a clear, close-up photo of the eye or palpebral conjunctiva.",
        )

    if getattr(eye_detector, "supports_cascade", False) and eye_detector.face_cascade is not None and not has_features:
        # Only apply strict face-geometry checks when the multi-signal
        # detector did NOT fire.  When has_features is True the image
        # already passed sclera/iris/conjunctiva checks so we trust it.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        faces = eye_detector.face_cascade.detectMultiScale(
            enhanced,
            scaleFactor=1.08,
            minNeighbors=8,
            minSize=(60, 60),
        )
        face_box = _largest_box(faces)

        if face_box is not None:
            face_x, face_y, face_w, face_h = face_box
            face_area_ratio = float(
                (face_w * face_h) / float(rgb.shape[0] * rgb.shape[1] + 1e-6)
            )

            if face_area_ratio < 0.04 or face_area_ratio > 0.55:
                return ValidationResult(
                    False,
                    quality,
                    "Image does not look like a close-up eye photo. "
                    "Fill more of the frame with the eye area.",
                )

            face_roi = rgb[face_y:face_y + face_h, face_x:face_x + face_w]
            face_skin_fraction = _skin_fraction(face_roi)
            if face_skin_fraction < 0.05:
                return ValidationResult(
                    False,
                    quality,
                    "Image does not look like a human eye photo. "
                    "Include visible facial skin around the eye.",
                )

            face_aspect = float(face_w) / float(face_h + 1e-6)
            if face_aspect < 0.75 or face_aspect > 1.45:
                return ValidationResult(
                    False,
                    quality,
                    "Detected face region looks implausible for an eye photo.",
                )

            roi_gray = enhanced[face_y:face_y + face_h, face_x:face_x + face_w]
            eye_boxes = eye_detector.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.08,
                minNeighbors=10,
                minSize=(24, 24),
            )
            if not _eyes_look_plausible(face_box, eye_boxes):
                return ValidationResult(
                    False,
                    quality,
                    "Image does not look like a true eye/conjunctiva capture.",
                )

    # ---- Step 3: centre-region saturation sanity check ----
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    center = hsv[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    if center.size > 0:
        sat_mean = float(center[:, :, 1].mean())
        if sat_mean < 12 or sat_mean > 190:
            return ValidationResult(
                False,
                quality,
                "Image does not look like an eye region. "
                "Please capture the conjunctiva or eye area.",
            )

    # ---- Accepted ----
    detected = eye_detector.detect_eyes(rgb)
    score = min(1.0, 0.5 * quality + 0.5 * (1.0 if detected else 0.0))
    return ValidationResult(True, max(score, 0.40), "Eye image accepted.")


def validate_nail(
    image: np.ndarray,
    eye_detector: Optional[EyeDetector] = None,
) -> ValidationResult:
    rgb = _to_rgb(image)
    basic = _basic_image_checks(rgb, modality="nail")
    if basic:
        return ValidationResult(False, 0.0, basic)

    skin_frac, max_blob = _skin_blob_fraction(rgb)
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    r_dom = float(((r > g - 15) & (r > b - 5)).mean())
    r_b_diff = float(r.mean() - b.mean())

    # Biological tissue presence: requires either coherent skin blob or warm/pink nail bed tissue
    if (skin_frac < 0.02 and max_blob < 0.01) and (r_dom < 0.20 or r_b_diff < 5.0):
        return ValidationResult(
            False,
            skin_frac,
            "No fingernail or skin tissue detected. Use a clear photo of a fingernail.",
        )

    # Texture/contrast check: reject uniform/flat non-organic artifacts
    if float(np.std(rgb)) < 10.0:
        return ValidationResult(
            False,
            0.0,
            "Image lacks detail or texture. Use a clear photo of a fingernail.",
        )

    score = float(min(1.0, 0.40 + max_blob * 0.4 + max(0.0, r_dom) * 0.2))
    return ValidationResult(True, max(score, 0.50), "Nail image accepted.")


def validate_palm(
    image: np.ndarray,
    eye_detector: Optional[EyeDetector] = None,
) -> ValidationResult:
    rgb = _to_rgb(image)
    basic = _basic_image_checks(rgb, modality="palm")
    if basic:
        return ValidationResult(False, 0.0, basic)

    skin_frac, max_blob = _skin_blob_fraction(rgb)
    if skin_frac < 0.10 or max_blob < 0.08:
        return ValidationResult(
            False,
            skin_frac,
            "No palm or hand skin detected. Photograph an open palm facing the camera.",
        )

    # Sanity check: hand/palm skin is not green or blue
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    if float((r - g).mean()) < -15.0:
        return ValidationResult(
            False,
            skin_frac,
            "Image color does not match expected palm skin tones.",
        )

    score = float(min(1.0, 0.35 + skin_frac * 0.5 + max_blob * 0.2))
    return ValidationResult(True, max(score, 0.50), "Palm image accepted.")


def validate_multimodal_inputs(
    eye_detector: EyeDetector,
    eye: Optional[np.ndarray] = None,
    nail: Optional[np.ndarray] = None,
    palm: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, ValidationResult], List[str], Optional[str]]:
    """
    Validate each provided modality. A request is accepted only when every
    supplied modality passes validation; otherwise random or wrong-category
    images could be silently dropped while the model still returns a number.
    """
    results: Dict[str, ValidationResult] = {}
    accepted: List[str] = []
    errors: List[str] = []

    if eye is not None:
        r = validate_eye(eye, eye_detector)
        results["eye"] = r
        if r.valid:
            accepted.append("eye")
        else:
            errors.append(f"Eye: {r.message}")

    if nail is not None:
        r = validate_nail(nail, eye_detector)
        results["nail"] = r
        if r.valid:
            accepted.append("nail")
        else:
            errors.append(f"Nail: {r.message}")

    if palm is not None:
        r = validate_palm(palm, eye_detector)
        results["palm"] = r
        if r.valid:
            accepted.append("palm")
        else:
            errors.append(f"Palm: {r.message}")

    if not accepted:
        return results, [], "No valid clinical images provided."

    if errors:
        msg = " ".join(errors)
        return results, accepted, msg

    return results, accepted, None
