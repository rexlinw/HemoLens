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


def _basic_image_checks(rgb: np.ndarray) -> Optional[str]:
    h, w = rgb.shape[:2]
    if min(h, w) < 80:
        return "Image resolution is too low."

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 25:
        return "Image is too dark."
    if mean_brightness > 245:
        return "Image is overexposed."

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 35.0:
        return "Image is too blurry. Use a sharper, well-focused photo."

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-8)))
    if entropy < 3.2:
        return "Image lacks enough detail for analysis."

    return None


def _skin_fraction(rgb: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return float((mask > 0).mean())


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


def validate_eye(image: np.ndarray, eye_detector: EyeDetector) -> ValidationResult:
    rgb = _to_rgb(image)
    basic = _basic_image_checks(rgb)
    if basic:
        return ValidationResult(False, 0.0, basic)

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

    detected = eye_detector.detect_eyes(rgb)
    quality = float(eye_detector.get_eye_quality_score(rgb))

    if face_box is None or not detected:
        return ValidationResult(
            False,
            quality,
            "No clear eyes detected. Use a close-up, well-lit eye or conjunctiva photo.",
        )

    face_x, face_y, face_w, face_h = face_box
    face_area_ratio = float((face_w * face_h) / float(rgb.shape[0] * rgb.shape[1] + 1e-6))
    if face_area_ratio < 0.02 or face_area_ratio > 0.70:
        return ValidationResult(
            False,
            quality,
            "Image does not look like a close-up eye photo. Fill more of the frame with the eye area.",
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

    if quality < 0.75:
        return ValidationResult(
            False,
            quality,
            "Eye image quality is too low. Move closer, improve lighting, and keep eyes in focus.",
        )

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    center = hsv[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    if center.size > 0:
        sat_mean = float(center[:, :, 1].mean())
        if sat_mean < 20 or sat_mean > 170:
            return ValidationResult(
                False,
                quality,
                "Image does not look like an eye region. Please capture the conjunctiva or eye area.",
            )

    score = min(1.0, 0.5 * quality + 0.5 * (1.0 if detected else 0.0))
    return ValidationResult(True, score, "Eye image accepted.")


def validate_nail(image: np.ndarray) -> ValidationResult:
    rgb = _to_rgb(image)
    basic = _basic_image_checks(rgb)
    if basic:
        return ValidationResult(False, 0.0, basic)

    skin_frac = _skin_fraction(rgb)
    if skin_frac < 0.08:
        return ValidationResult(
            False,
            skin_frac,
            "No fingernail or skin region detected. Use a clear nail-bed photo.",
        )

    morph = NailFeatureExtractor.extract_morphological_features(rgb)
    detected_area = morph.get("nail_detected_area", 0.0)
    largest_pct = morph.get("nail_largest_area_pct", 0.0)
    num_contours = morph.get("nail_num_contours", 0.0)
    circularity = morph.get("nail_circularity", 0.0)

    if largest_pct < 0.02 or detected_area < 0.03:
        return ValidationResult(
            False,
            detected_area,
            "Fingernail not detected. Center the nail bed in frame with good lighting.",
        )

    if detected_area > 0.42 or num_contours > 8:
        return ValidationResult(
            False,
            detected_area,
            "Could not isolate a fingernail. Use a single nail on plain background.",
        )

    if largest_pct < detected_area * 0.25:
        return ValidationResult(
            False,
            largest_pct,
            "Fingernail region unclear. Fill the frame with one nail bed.",
        )

    color = NailFeatureExtractor.extract_color_features(rgb)
    rg_diff = color.get("nail_rg_diff", 0.0)
    sat = color.get("nail_saturation_mean", 0.0)
    if rg_diff < 2.0 or sat < 25:
        return ValidationResult(
            False,
            detected_area,
            "Image does not match expected nail color patterns.",
        )

    if circularity < 0.05 and largest_pct < 0.08:
        return ValidationResult(
            False,
            circularity,
            "Fingernail shape not recognized.",
        )

    score = float(min(1.0, largest_pct * 8 + skin_frac * 0.5 + min(circularity, 1.0) * 0.2))
    return ValidationResult(True, score, "Nail image accepted.")


def validate_palm(image: np.ndarray) -> ValidationResult:
    rgb = _to_rgb(image)
    basic = _basic_image_checks(rgb)
    if basic:
        return ValidationResult(False, 0.0, basic)

    skin_frac = _skin_fraction(rgb)
    if skin_frac < 0.12:
        return ValidationResult(
            False,
            skin_frac,
            "No palm or hand skin detected. Photograph an open palm facing the camera.",
        )

    extractor = PalmFeatureExtractor()
    feats = extractor.extract_all_features(rgb)
    h, w = rgb.shape[:2]
    area_ratio = feats.get("palm_area", 0.0) / float(h * w + 1e-6)
    solidity = feats.get("palm_solidity", 0.0)

    if area_ratio < 0.12:
        return ValidationResult(
            False,
            area_ratio,
            "Palm region too small. Fill the frame with your open palm.",
        )

    if solidity < 0.55:
        return ValidationResult(
            False,
            solidity,
            "Could not identify a palm shape. Use a clear photo of an open palm.",
        )

    rg_diff = feats.get("palm_r_g_diff", 0.0)
    if rg_diff < -5:
        return ValidationResult(
            False,
            area_ratio,
            "Image does not match expected palm color patterns.",
        )

    score = float(min(1.0, area_ratio * 2 + skin_frac + solidity * 0.3))
    return ValidationResult(True, score, "Palm image accepted.")


def validate_multimodal_inputs(
    eye_detector: EyeDetector,
    eye: Optional[np.ndarray] = None,
    nail: Optional[np.ndarray] = None,
    palm: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, ValidationResult], List[str], Optional[str]]:
    """
    Validate each provided modality. Returns per-modality results, list of
    accepted modality keys, and an error message if none are valid.
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
        r = validate_nail(nail)
        results["nail"] = r
        if r.valid:
            accepted.append("nail")
        else:
            errors.append(f"Nail: {r.message}")

    if palm is not None:
        r = validate_palm(palm)
        results["palm"] = r
        if r.valid:
            accepted.append("palm")
        else:
            errors.append(f"Palm: {r.message}")

    if not accepted:
        msg = " ".join(errors) if errors else "No valid clinical images provided."
        return results, [], msg

    return results, accepted, None
