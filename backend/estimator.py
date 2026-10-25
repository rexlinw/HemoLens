from __future__ import annotations

import io
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from eye_detector import EyeDetector, get_hemoglobin_status
from feature_extraction import FeatureExtractor
from image_validator import validate_eye, validate_multimodal_inputs
from multimodal import load_multimodal_artifacts, predict_multimodal
from preprocessing import ImagePreprocessor
from uncertainty import compute_signal_quality_index, estimate_confidence_interval

MODEL_DIR = Path(__file__).resolve().parent / "models"


def load_rgb_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def decode_image_bytes(contents: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return np.array(image)
    except Exception:
        arr = np.frombuffer(contents, dtype=np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("Could not decode image")
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def _feature_vector_from_image(image: np.ndarray) -> np.ndarray:
    preprocessed = ImagePreprocessor.preprocess(
        image,
        resize=True,
        normalize=False,
        denoise=True,
        enhance_contrast=True,
    )
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(preprocessed)
    feature_names = extractor.get_feature_names()
    return np.array([[features[name] for name in feature_names]], dtype=np.float64)


@dataclass
class HemoglobinEstimator:
    model_dir: Path = MODEL_DIR
    ridge_model: Any = field(init=False, default=None)
    scaler: Any = field(init=False, default=None)
    eye_detector: Optional[EyeDetector] = field(init=False, default=None)
    multimodal_model: Any = field(init=False, default=None)
    multimodal_scaler: Any = field(init=False, default=None)
    multimodal_config: Optional[dict] = field(init=False, default=None)
    loaded: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.load()

    def load(self) -> None:
        self.ridge_model = None
        self.scaler = None
        self.eye_detector = EyeDetector()
        self.multimodal_model = None
        self.multimodal_scaler = None
        self.multimodal_config = None

        ridge_path = self.model_dir / "hemolens_ridge_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"

        try:
            with open(ridge_path, "rb") as handle:
                self.ridge_model = pickle.load(handle)
            with open(scaler_path, "rb") as handle:
                self.scaler = pickle.load(handle)
        except FileNotFoundError:
            self.ridge_model = None
            self.scaler = None

        self.multimodal_model, self.multimodal_scaler, self.multimodal_config = load_multimodal_artifacts(
            self.model_dir
        )
        self.loaded = any(
            [
                self.ridge_model is not None and self.scaler is not None,
                self.multimodal_model is not None and self.multimodal_scaler is not None,
            ]
        )
        if not self.loaded:
            raise RuntimeError(f"No estimator artifacts found in {self.model_dir}")

    def status(self) -> dict:
        multimodal_config = self.multimodal_config or {}
        return {
            "ready": self.loaded,
            "model_dir": str(self.model_dir),
            "eye_model_loaded": self.ridge_model is not None and self.scaler is not None,
            "multimodal_model_loaded": self.multimodal_model is not None and self.multimodal_scaler is not None,
            "multimodal_metrics": {
                "model_type": multimodal_config.get("model_type"),
                "modalities": multimodal_config.get("modalities"),
                "r2": multimodal_config.get("r2"),
                "mae": multimodal_config.get("mae"),
                "training_samples": multimodal_config.get("training_samples"),
                "test_samples": multimodal_config.get("test_samples"),
            },
        }

    def health(self) -> dict:
        status = self.status()
        healthy = status["eye_model_loaded"] or status["multimodal_model_loaded"]
        return {
            "status": "healthy" if healthy else "unhealthy",
            "models_loaded": healthy,
            "eye_model_loaded": status["eye_model_loaded"],
            "multimodal_model_loaded": status["multimodal_model_loaded"],
            "multimodal_metrics": status["multimodal_metrics"],
        }

    def _build_prediction(
        self,
        estimate: float,
        model_name: str,
        validation: dict,
        warning: str | None = None,
        sqi: dict | None = None,
        modalities_count: int = 1,
    ) -> dict:
        clipped = float(np.clip(estimate, 6.0, 18.0))
        health = get_hemoglobin_status(clipped)
        sqi_score = sqi["score"] if sqi else 75
        conf_interval = estimate_confidence_interval(
            clipped,
            sqi_score=sqi_score,
            modalities_count=modalities_count,
        )
        result = {
            "status": "success",
            "model": model_name,
            "hemoglobin_estimate": clipped,
            "confidence_interval": conf_interval,
            "signal_quality_index": sqi or {"score": sqi_score, "level": "HIGH"},
            "unit": "g/dL",
            "health_status": health["status"],
            "health_message": health["message"],
            "health_color": health["color"],
            "validation": validation,
        }
        if warning:
            result["warning"] = warning
        return result

    def predict_eye_only(self, eye: np.ndarray) -> dict:
        if self.ridge_model is None or self.scaler is None:
            raise RuntimeError("Eye-only artifacts are not available")

        validation = validate_eye(eye, self.eye_detector)
        validation_payload = {
            "eye": {
                "valid": validation.valid,
                "score": float(validation.score),
                "message": validation.message,
            }
        }
        if not validation.valid:
            return {
                "status": "invalid_image",
                "model": "ridge_eye",
                "message": validation.message,
                "validation": validation_payload,
            }

        features_array = _feature_vector_from_image(eye)
        estimate = float(self.ridge_model.predict(self.scaler.transform(features_array))[0])
        sqi = compute_signal_quality_index(eye, modality="eye")
        result = self._build_prediction(
            estimate, "ridge_eye", validation_payload, sqi=sqi, modalities_count=1
        )
        result["eye_quality_score"] = float(validation.score)
        return result

    def predict_multimodal(self, eye: np.ndarray | None = None, nail: np.ndarray | None = None, palm: np.ndarray | None = None) -> dict:
        if self.multimodal_model is None or self.multimodal_scaler is None:
            raise RuntimeError("Multimodal artifacts are not available")

        validation_results, accepted, error_msg = validate_multimodal_inputs(
            self.eye_detector,
            eye=eye,
            nail=nail,
            palm=palm,
        )
        validation_payload = {
            key: {
                "valid": result.valid,
                "score": float(result.score),
                "message": result.message,
            }
            for key, result in validation_results.items()
        }
        if not accepted or error_msg:
            return {
                "status": "invalid_image",
                "model": "multimodal",
                "message": error_msg or "No valid clinical images provided.",
                "validation": validation_payload,
            }

        sqi_scores = {}
        if eye is not None and "eye" in accepted:
            sqi_scores["eye"] = compute_signal_quality_index(eye, modality="eye")["score"]
        if nail is not None and "nail" in accepted:
            sqi_scores["nail"] = compute_signal_quality_index(nail, modality="nail")["score"]
        if palm is not None and "palm" in accepted:
            sqi_scores["palm"] = compute_signal_quality_index(palm, modality="palm")["score"]

        prediction = predict_multimodal(
            self.multimodal_model,
            self.multimodal_scaler,
            eye=eye,
            nail=nail,
            palm=palm,
            modalities_filter=accepted,
            sqi_scores=sqi_scores,
        )
        primary = eye if eye is not None else (nail if nail is not None else palm)
        sqi = compute_signal_quality_index(primary, modality="eye" if eye is not None else "nail")
        result = self._build_prediction(
            prediction["hemoglobin_estimate"],
            "multimodal",
            validation_payload,
            warning=error_msg,
            sqi=sqi,
            modalities_count=len(prediction["modalities_used"]),
        )
        result["modalities_used"] = prediction["modalities_used"]
        result["modality_weights"] = prediction["modality_weights"]
        result["modality_estimates"] = prediction["modality_estimates"]
        result["modality_contributions"] = prediction["modality_contributions"]
        result["clinical_evidence"] = prediction["clinical_evidence"]
        result["active_features"] = prediction["active_features"]
        if "eye" in accepted and eye is not None:
            result["eye_quality_score"] = float(validation_results["eye"].score)
        return result

    def predict(
        self,
        eye: np.ndarray | None = None,
        nail: np.ndarray | None = None,
        palm: np.ndarray | None = None,
        prefer_multimodal: bool = True,
    ) -> dict:
        if not any([eye is not None, nail is not None, palm is not None]):
            raise ValueError("Provide at least one image")

        if nail is not None or palm is not None:
            return {
                "status": "invalid_image",
                "model": "eye_only",
                "message": "This estimator only accepts eye images. Remove nail or palm images.",
                "validation": {},
            }

        if eye is not None and self.ridge_model is not None and self.scaler is not None:
            return self.predict_eye_only(eye)

        if prefer_multimodal and self.multimodal_model is not None and self.multimodal_scaler is not None:
            return self.predict_multimodal(eye=eye)

        raise RuntimeError("No suitable model is available for the provided image")

    def predict_from_paths(
        self,
        eye_path: str | Path | None = None,
        nail_path: str | Path | None = None,
        palm_path: str | Path | None = None,
        prefer_multimodal: bool = True,
    ) -> dict:
        eye = load_rgb_image(eye_path) if eye_path else None
        nail = load_rgb_image(nail_path) if nail_path else None
        palm = load_rgb_image(palm_path) if palm_path else None
        return self.predict(eye=eye, nail=nail, palm=palm, prefer_multimodal=prefer_multimodal)
