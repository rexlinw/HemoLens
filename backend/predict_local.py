#!/usr/bin/env python3
"""Run HemoLens predictions locally without the API server or Expo app."""

from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from eye_detector import EyeDetector, get_hemoglobin_status
from feature_extraction import FeatureExtractor
from image_validator import validate_eye, validate_multimodal_inputs
from multimodal import load_multimodal_artifacts, predict_multimodal
from preprocessing import ImagePreprocessor
from uncertainty import compute_signal_quality_index, estimate_confidence_interval

MODEL_DIR = Path(__file__).parent / "models"

_ridge_model = None
_eye_scaler = None
_eye_detector = None
_multimodal_model = None
_multimodal_scaler = None
_multimodal_config = None
_loaded = False


def load_rgb_from_path(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_rgb_from_bytes(data: bytes) -> np.ndarray:
    try:
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        arr = np.frombuffer(data, dtype=np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("Could not decode image")
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def load_models() -> dict:
    global _ridge_model, _eye_scaler, _eye_detector
    global _multimodal_model, _multimodal_scaler, _multimodal_config, _loaded

    if _loaded:
        return get_model_status()

    eye_ok = False
    try:
        with open(MODEL_DIR / "hemolens_ridge_model.pkl", "rb") as f:
            _ridge_model = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            _eye_scaler = pickle.load(f)
        _eye_detector = EyeDetector()
        eye_ok = True
    except Exception as exc:
        print(f"Warning: eye-only model unavailable: {exc}", file=sys.stderr)

    _multimodal_model, _multimodal_scaler, _multimodal_config = load_multimodal_artifacts(MODEL_DIR)
    _loaded = True

    status = get_model_status()
    if not status["eye_loaded"] and not status["multimodal_loaded"]:
        raise RuntimeError(f"No models found in {MODEL_DIR}")

    return status


def get_model_status() -> dict:
    return {
        "eye_loaded": _ridge_model is not None and _eye_scaler is not None,
        "multimodal_loaded": _multimodal_model is not None and _multimodal_scaler is not None,
        "multimodal_config": _multimodal_config or {},
    }


def _format_health(hb: float, sqi: dict | None = None, modalities_count: int = 1) -> dict:
    health = get_hemoglobin_status(hb)
    sqi_score = sqi["score"] if sqi else 75
    conf = estimate_confidence_interval(hb, sqi_score=sqi_score, modalities_count=modalities_count)
    return {
        "hemoglobin_estimate": round(hb, 2),
        "confidence_interval": conf,
        "signal_quality_index": sqi or {"score": sqi_score, "level": "HIGH"},
        "unit": "g/dL",
        "health_status": health["status"],
        "health_message": health["message"],
        "health_color": health["color"],
    }


def predict_eye_only(image: np.ndarray) -> dict:
    if _ridge_model is None or _eye_scaler is None:
        raise RuntimeError("Eye-only model not loaded")

    validation = validate_eye(image, _eye_detector)
    if not validation.valid:
        return {
            "status": "invalid_image",
            "model": "ridge_eye",
            "message": validation.message,
            "validation": {
                "eye": {
                    "valid": validation.valid,
                    "score": float(validation.score),
                    "message": validation.message,
                }
            },
        }

    preprocessed = ImagePreprocessor.preprocess(
        image,
        resize=True,
        normalize=False,
        denoise=True,
        enhance_contrast=True,
    )
    features = FeatureExtractor.extract_all_features(preprocessed)
    names = FeatureExtractor.get_feature_names()
    features_array = np.array([[features[name] for name in names]])
    hb = float(np.clip(_ridge_model.predict(_eye_scaler.transform(features_array))[0], 6.0, 18.0))
    sqi = compute_signal_quality_index(image, modality="eye")

    result = _format_health(hb, sqi=sqi, modalities_count=1)
    result.update(
        {
            "status": "success",
            "model": "ridge_eye",
            "eye_quality_score": float(validation.score),
            "validation": {
                "eye": {
                    "valid": True,
                    "score": float(validation.score),
                    "message": validation.message,
                }
            },
        }
    )
    return result


def predict_multimodal_local(
    eye: Optional[np.ndarray] = None,
    nail: Optional[np.ndarray] = None,
    palm: Optional[np.ndarray] = None,
) -> dict:
    if _multimodal_model is None or _multimodal_scaler is None:
        raise RuntimeError("Multimodal model not loaded")

    validation_results, accepted, error_msg = validate_multimodal_inputs(
        _eye_detector,
        eye=eye,
        nail=nail,
        palm=palm,
    )

    if not accepted or error_msg:
        return {
            "status": "invalid_image",
            "model": "multimodal",
            "message": error_msg or "No valid clinical images provided.",
            "validation": {
                key: {
                    "valid": result.valid,
                    "score": float(result.score),
                    "message": result.message,
                }
                for key, result in validation_results.items()
            },
        }

    sqi_scores = {}
    if eye is not None and "eye" in accepted:
        sqi_scores["eye"] = compute_signal_quality_index(eye, modality="eye")["score"]
    if nail is not None and "nail" in accepted:
        sqi_scores["nail"] = compute_signal_quality_index(nail, modality="nail")["score"]
    if palm is not None and "palm" in accepted:
        sqi_scores["palm"] = compute_signal_quality_index(palm, modality="palm")["score"]

    prediction = predict_multimodal(
        _multimodal_model,
        _multimodal_scaler,
        eye=eye,
        nail=nail,
        palm=palm,
        modalities_filter=accepted,
        sqi_scores=sqi_scores,
    )

    primary = eye if eye is not None else (nail if nail is not None else palm)
    sqi = compute_signal_quality_index(primary, modality="eye" if eye is not None else "nail")
    result = _format_health(
        prediction["hemoglobin_estimate"],
        sqi=sqi,
        modalities_count=len(prediction["modalities_used"]),
    )
    result.update(
        {
            "status": "success",
            "model": "multimodal",
            "modalities_used": prediction["modalities_used"],
            "modality_weights": prediction["modality_weights"],
            "modality_estimates": prediction["modality_estimates"],
            "modality_contributions": prediction["modality_contributions"],
            "clinical_evidence": prediction["clinical_evidence"],
            "active_features": prediction["active_features"],
            "warning": error_msg,
            "validation": {
                key: {
                    "valid": result_obj.valid,
                    "score": float(result_obj.score),
                    "message": result_obj.message,
                }
                for key, result_obj in validation_results.items()
            },
        }
    )

    if "eye" in accepted and eye is not None:
        result["eye_quality_score"] = float(validation_results["eye"].score)

    return result


def predict_from_images(
    eye: Optional[np.ndarray] = None,
    nail: Optional[np.ndarray] = None,
    palm: Optional[np.ndarray] = None,
    prefer_multimodal: bool = True,
) -> dict:
    load_models()

    if not any([eye is not None, nail is not None, palm is not None]):
        raise ValueError("Provide at least one image (eye, nail, or palm)")

    status = get_model_status()
    use_multimodal = prefer_multimodal and status["multimodal_loaded"]

    if nail is not None or palm is not None:
        if not status["multimodal_loaded"]:
            raise RuntimeError("Nail/palm prediction requires the multimodal model")
        return predict_multimodal_local(eye=eye, nail=nail, palm=palm)

    if use_multimodal:
        return predict_multimodal_local(eye=eye)

    if eye is not None and status["eye_loaded"]:
        return predict_eye_only(eye)

    raise RuntimeError("No suitable model available for the provided inputs")


def print_result(result: dict) -> None:
    print()
    if result.get("status") == "invalid_image":
        print("Result: INVALID IMAGE")
        print(f"Reason: {result.get('message')}")
        for key, info in (result.get("validation") or {}).items():
            print(f"  {key}: valid={info['valid']} score={info['score']:.2f} — {info['message']}")
        return

    print("Result: SUCCESS")
    print(f"  Hemoglobin: {result['hemoglobin_estimate']} {result['unit']}")
    print(f"  Status: {result['health_status']} — {result['health_message']}")
    print(f"  Model: {result.get('model')}")
    if result.get("modalities_used"):
        print(f"  Modalities: {', '.join(result['modalities_used'])}")
    if result.get("eye_quality_score") is not None:
        print(f"  Eye quality: {result['eye_quality_score']:.2f}")
    if result.get("warning"):
        print(f"  Note: {result['warning']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HemoLens predictions locally")
    parser.add_argument("--eye", type=Path, help="Path to eye image")
    parser.add_argument("--nail", type=Path, help="Path to nail/finger image")
    parser.add_argument("--palm", type=Path, help="Path to palm image")
    parser.add_argument(
        "--ridge-only",
        action="store_true",
        help="Use eye-only Ridge model instead of multimodal (eye images only)",
    )
    args = parser.parse_args()

    if not any([args.eye, args.nail, args.palm]):
        parser.error("Provide at least one of --eye, --nail, or --palm")

    eye = load_rgb_from_path(args.eye) if args.eye else None
    nail = load_rgb_from_path(args.nail) if args.nail else None
    palm = load_rgb_from_path(args.palm) if args.palm else None

    status = load_models()
    print("Models loaded:")
    print(f"  Eye (Ridge): {'yes' if status['eye_loaded'] else 'no'}")
    print(f"  Multimodal: {'yes' if status['multimodal_loaded'] else 'no'}")

    result = predict_from_images(
        eye=eye,
        nail=nail,
        palm=palm,
        prefer_multimodal=not args.ridge_only,
    )
    print_result(result)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
