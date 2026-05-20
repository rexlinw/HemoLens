import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from feature_extraction import FeatureExtractor
from nail_feature_extraction import NailFeatureExtractor
from palm_feature_extraction import PalmFeatureExtractor
from preprocessing import ImagePreprocessor

MODALITIES = ("eye", "nail", "palm")

EYE_COUNT = len(FeatureExtractor.get_feature_names())
NAIL_COUNT = len(NailFeatureExtractor.get_feature_names())
PALM_COUNT = len(PalmFeatureExtractor().get_feature_names())
TOTAL_FEATURES = EYE_COUNT + NAIL_COUNT + PALM_COUNT

EYE_SLICE = slice(0, EYE_COUNT)
NAIL_SLICE = slice(EYE_COUNT, EYE_COUNT + NAIL_COUNT)
PALM_SLICE = slice(EYE_COUNT + NAIL_COUNT, TOTAL_FEATURES)


def load_multimodal_artifacts(model_dir: Path) -> Tuple[Optional[object], Optional[object], Optional[dict]]:
    model_path = model_dir / "hemolens_multimodal.pkl"
    scaler_path = model_dir / "multimodal_scaler.pkl"
    config_path = model_dir / "multimodal_config.json"

    if not model_path.exists() or not scaler_path.exists():
        return None, None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    config = None
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    return model, scaler, config


def _rgb_from_array(image_array: np.ndarray) -> np.ndarray:
    if len(image_array.shape) == 2:
        return cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    if image_array.shape[2] == 3:
        return image_array.astype(np.uint8)
    return image_array


def extract_eye_features(image_array: np.ndarray) -> np.ndarray:
    rgb = _rgb_from_array(image_array)
    preprocessed = ImagePreprocessor.preprocess(
        rgb,
        resize=True,
        normalize=False,
        denoise=True,
        enhance_contrast=True,
    )
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(preprocessed)
    names = FeatureExtractor.get_feature_names()
    return np.array([features[n] for n in names], dtype=np.float64)


def extract_nail_features(image_array: np.ndarray) -> np.ndarray:
    rgb = _rgb_from_array(image_array)
    rgb = cv2.resize(rgb, (256, 256))
    extractor = NailFeatureExtractor()
    features = extractor.extract_all_features(rgb)
    names = NailFeatureExtractor.get_feature_names()
    return np.array([features[n] for n in names], dtype=np.float64)


def extract_palm_features(image_array: np.ndarray) -> np.ndarray:
    rgb = _rgb_from_array(image_array)
    rgb = cv2.resize(rgb, (256, 256))
    extractor = PalmFeatureExtractor()
    features = extractor.extract_all_features(rgb)
    names = extractor.get_feature_names()
    return np.array([features[n] for n in names], dtype=np.float64)


def build_feature_vector(
    eye: Optional[np.ndarray] = None,
    nail: Optional[np.ndarray] = None,
    palm: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    vector = np.zeros(TOTAL_FEATURES, dtype=np.float64)
    present: List[str] = []

    if eye is not None:
        vector[EYE_SLICE] = extract_eye_features(eye)
        present.append("eye")
    if nail is not None:
        vector[NAIL_SLICE] = extract_nail_features(nail)
        present.append("nail")
    if palm is not None:
        vector[PALM_SLICE] = extract_palm_features(palm)
        present.append("palm")

    return vector, present


def predict_multimodal(
    model,
    scaler,
    eye: Optional[np.ndarray] = None,
    nail: Optional[np.ndarray] = None,
    palm: Optional[np.ndarray] = None,
) -> Dict:
    vector, modalities = build_feature_vector(eye=eye, nail=nail, palm=palm)

    if not modalities:
        raise ValueError("At least one modality image is required")

    scaled = scaler.transform(vector.reshape(1, -1))
    estimate = float(model.predict(scaled)[0])
    estimate = float(np.clip(estimate, 6.0, 18.0))

    return {
        "hemoglobin_estimate": estimate,
        "modalities_used": modalities,
        "feature_count": TOTAL_FEATURES,
        "active_features": sum(
            [
                EYE_COUNT if "eye" in modalities else 0,
                NAIL_COUNT if "nail" in modalities else 0,
                PALM_COUNT if "palm" in modalities else 0,
            ]
        ),
    }
