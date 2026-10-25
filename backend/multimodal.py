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

# Baseline clinical importance weights grounded in diagnostic accuracy literature:
# - Eye (55%): Thin non-keratinized mucosal microvasculature with zero melanin (Kalantri et al. 2010; Sheth et al. 1972)
# - Finger/Nail (25%): Subungual capillary loops visible under keratin bed (Mannino et al. 2018)
# - Palm (20%): Palmar crease pallor, systemic screening site (WHO IMCI; Stoltzfus et al. 1999)
CLINICAL_MODALITY_WEIGHTS: Dict[str, float] = {
    "eye": 0.55,
    "nail": 0.25,
    "palm": 0.20,
}

CLINICAL_LITERATURE_EVIDENCE: Dict[str, Dict[str, str]] = {
    "eye": {
        "weight_percent": "55%",
        "anatomical_site": "Palpebral Conjunctiva",
        "citation": "Kalantri et al. (PLOS ONE 2010), Sheth et al. (Lancet 1972), Suner et al. (Ann Emerg Med 2007)",
        "rationale": "Direct non-keratinized microvascular window with zero melanin interference. Demonstrates highest sensitivity (73%), specificity (93%), and positive likelihood ratio (LR+ 4.49) of all anatomical sites.",
    },
    "nail": {
        "weight_percent": "25%",
        "anatomical_site": "Subungual Fingernail Bed",
        "citation": "Mannino et al. (Nature Communications 2018)",
        "rationale": "High capillary loop density beneath translucent nail plate (correlation r = 0.82), subject to ambient temperature vasoconstriction and blanching pressure.",
    },
    "palm": {
        "weight_percent": "20%",
        "anatomical_site": "Palmar Skin / Creases",
        "citation": "WHO IMCI Guidelines, Stoltzfus et al. (Am J Clin Nutr 1999)",
        "rationale": "Established systemic pallor screening site. Effective for severe anemia (Hb < 7 g/dL) but exhibits optical attenuation and variance from stratum corneum thickness and manual labor.",
    },
}


def compute_modality_weights(
    modalities: List[str],
    sqi_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute normalized weights for active modalities based on clinical literature,
    optionally adjusted by Signal Quality Index (SQI).
    """
    if not modalities:
        return {}

    raw_weights = {}
    for mod in modalities:
        base_w = CLINICAL_MODALITY_WEIGHTS.get(mod, 0.20)
        if sqi_scores and mod in sqi_scores and sqi_scores[mod] > 0:
            quality_factor = 0.5 + 0.5 * (min(100.0, max(0.0, sqi_scores[mod])) / 100.0)
            raw_weights[mod] = base_w * quality_factor
        else:
            raw_weights[mod] = base_w

    total = sum(raw_weights.values())
    if total <= 0:
        equal_w = round(1.0 / len(modalities), 4)
        return {mod: equal_w for mod in modalities}

    return {mod: round(w / total, 4) for mod, w in raw_weights.items()}


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


def predict_single_modality(model, scaler, modality: str, image: np.ndarray) -> float:
    kwargs = {modality: image}
    vec, _ = build_feature_vector(**kwargs)
    scaled = scaler.transform(vec.reshape(1, -1))
    return float(np.clip(float(model.predict(scaled)[0]), 6.0, 18.0))


def predict_multimodal(
    model,
    scaler,
    eye: Optional[np.ndarray] = None,
    nail: Optional[np.ndarray] = None,
    palm: Optional[np.ndarray] = None,
    modalities_filter: Optional[List[str]] = None,
    sqi_scores: Optional[Dict[str, float]] = None,
) -> Dict:
    use_eye = eye if modalities_filter is None or "eye" in modalities_filter else None
    use_nail = nail if modalities_filter is None or "nail" in modalities_filter else None
    use_palm = palm if modalities_filter is None or "palm" in modalities_filter else None

    active_images = {}
    if use_eye is not None:
        active_images["eye"] = use_eye
    if use_nail is not None:
        active_images["nail"] = use_nail
    if use_palm is not None:
        active_images["palm"] = use_palm

    if not active_images:
        raise ValueError("At least one validated modality image is required")

    modalities = list(active_images.keys())

    # 1. Compute individual modality estimates
    modality_estimates: Dict[str, float] = {}
    for mod, img in active_images.items():
        modality_estimates[mod] = predict_single_modality(model, scaler, mod, img)

    # 2. Compute literature-derived normalized weights
    modality_weights = compute_modality_weights(modalities, sqi_scores=sqi_scores)

    # 3. Calculate clinical weighted average
    if len(modalities) == 1:
        mod = modalities[0]
        final_estimate = modality_estimates[mod]
    else:
        final_estimate = float(
            sum(modality_weights[mod] * modality_estimates[mod] for mod in modalities)
        )
    final_estimate = float(np.clip(final_estimate, 6.0, 18.0))

    # 4. Modality contributions
    modality_contributions = {
        mod: {
            "estimate": round(modality_estimates[mod], 2),
            "weight": round(modality_weights[mod], 4),
            "weighted_contribution": round(modality_weights[mod] * modality_estimates[mod], 2),
        }
        for mod in modalities
    }

    return {
        "hemoglobin_estimate": round(final_estimate, 2),
        "modalities_used": modalities,
        "modality_weights": modality_weights,
        "modality_estimates": {k: round(v, 2) for k, v in modality_estimates.items()},
        "modality_contributions": modality_contributions,
        "clinical_evidence": {mod: CLINICAL_LITERATURE_EVIDENCE[mod] for mod in modalities},
        "feature_count": TOTAL_FEATURES,
        "active_features": sum(
            [
                EYE_COUNT if "eye" in modalities else 0,
                NAIL_COUNT if "nail" in modalities else 0,
                PALM_COUNT if "palm" in modalities else 0,
            ]
        ),
    }
