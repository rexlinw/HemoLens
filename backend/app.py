from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np
import pickle
import io
import time
from PIL import Image
from pathlib import Path

import cv2
from preprocessing import ImagePreprocessor
from feature_extraction import FeatureExtractor
from eye_detector import EyeDetector, get_hemoglobin_status
from multimodal import load_multimodal_artifacts, predict_multimodal

app = FastAPI(
    title="HemoLens API v3.0",
    description="Multimodal non-invasive hemoglobin estimation (eye, nail, palm)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

MODEL_DIR = Path(__file__).parent / "models"
RIDGE_MODEL_PATH = MODEL_DIR / "hemolens_ridge_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
GB_MODEL_PATH = MODEL_DIR / "hemolens_gb_model.pkl"
SVR_MODEL_PATH = MODEL_DIR / "hemolens_svr_model.pkl"

ridge_model = None
gb_model = None
scaler = None
eye_detector = None
multimodal_model = None
multimodal_scaler = None
multimodal_config = None
models_loaded = False
multimodal_loaded = False
EYE_QUALITY_THRESHOLD = 0.5


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


def load_models():
    global ridge_model, gb_model, scaler, eye_detector, models_loaded
    global multimodal_model, multimodal_scaler, multimodal_config, multimodal_loaded

    eye_ok = False
    try:
        with open(RIDGE_MODEL_PATH, 'rb') as f:
            ridge_model = pickle.load(f)
        print(f"✓ Ridge model loaded: {RIDGE_MODEL_PATH}")

        if GB_MODEL_PATH.exists():
            with open(GB_MODEL_PATH, 'rb') as f:
                gb_model = pickle.load(f)
            print("✓ Gradient Boosting model loaded (backup)")

        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler loaded: {SCALER_PATH}")

        eye_detector = EyeDetector()
        print("✓ Eye detector initialized")
        models_loaded = True
        eye_ok = True
    except FileNotFoundError as e:
        print(f"✗ Error loading eye models: {e}")

    multimodal_model, multimodal_scaler, multimodal_config = load_multimodal_artifacts(MODEL_DIR)
    if multimodal_model is not None:
        multimodal_loaded = True
        cfg = multimodal_config or {}
        print(f"✓ Multimodal model loaded (R²={cfg.get('r2', 'n/a')}, MAE={cfg.get('mae', 'n/a')} g/dL)")
    else:
        print("⚠ Multimodal model not found — /predict/multimodal unavailable")

    return eye_ok or multimodal_loaded


@app.on_event("startup")
async def startup_event():
    print("\n" + "="*70)
    print("HemoLens API v3.0 - Starting")
    print("="*70)
    success = load_models()
    if success:
        print("\n✓ API ready")
        if models_loaded:
            print("  Eye-only: Ridge (46 features)")
        if multimodal_loaded:
            print("  Multimodal: eye + nail + palm (97 features)")
    else:
        print("\n✗ Failed to load models!")
    print("="*70 + "\n")


@app.get("/")
async def root():
    return {
        "name": "HemoLens API v3.0",
        "description": "Multimodal non-invasive hemoglobin estimation",
        "endpoints": {
            "GET /health": "Check API health and model status",
            "GET /info": "Get model information",
            "POST /predict": "Predict from single eye image",
            "POST /predict/multimodal": "Predict from eye, nail, and/or palm images",
            "POST /predict/batch": "Batch eye predictions",
        },
        "multimodal_available": multimodal_loaded,
    }


@app.get("/health")
async def health_check():
    healthy = models_loaded or multimodal_loaded
    return {
        "status": "healthy" if healthy else "unhealthy",
        "models_loaded": models_loaded,
        "multimodal_loaded": multimodal_loaded,
        "model_type": "Multimodal (eye+nail+palm)" if multimodal_loaded else "Ridge (eye only)",
        "multimodal_metrics": multimodal_config if multimodal_config else None,
        "eye_metrics": {"r2": 0.6267, "mae": 0.96, "unit": "g/dL"} if models_loaded else None,
    }


@app.get("/info")
async def get_model_info():
    feature_names = list(FeatureExtractor.get_feature_names())

    return {
        "model_name": "HemoLens Multimodal",
        "version": "3.0",
        "multimodal": multimodal_config,
        "multimodal_loaded": multimodal_loaded,
        "features": {
            "count": len(feature_names),
            "names": feature_names,
            "description": [
                "RGB (4): Color means and R/G ratio",
                "LAB (6): Perceptual color space features",
                "HSV (6): Hue, saturation, value",
                "YCrCb (3): Skin tone features",
                "Statistical (12): Mean, std, percentiles, skewness",
                "Edge (4): Sobel edge detection",
                "Contrast (3): RMS contrast, brightness, dynamic range",
                "Histogram (8): Entropy, energy, mean, std, skewness, kurtosis"
            ]
        },
        "performance": {
            "r2_score": 0.6267,
            "mae": 0.96,
            "rmse": 1.3745,
            "mape": 7.49,
            "unit": "g/dL"
        },
        "training_data": {
            "total_samples": 145,
            "train_samples": 116,
            "test_samples": 29,
            "regions": ["India (63)", "Italy (82)"],
            "hemoglobin_range": [7.0, 17.4],
            "hemoglobin_mean": 12.61,
            "hemoglobin_std": 2.38
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )

    if file.content_type.startswith("image/") is False:
        raise HTTPException(status_code=400, detail="File must be an image")

    start_time = time.time()

    try:
        contents = await file.read()
        image_array = decode_image_bytes(contents)

        # Require both detection and minimum quality score to avoid false positives
        eye_quality = eye_detector.get_eye_quality_score(image_array)
        detected = eye_detector.detect_eyes(image_array)

        # Log detection info for debugging
        print(f"[PREDICT] file={file.filename} detected={detected} eye_quality={eye_quality:.3f}")

        # Allow images that meet the eye-quality threshold even if detect_eyes() failed
        if eye_quality < EYE_QUALITY_THRESHOLD and not detected:
            print(f"[PREDICT] Rejected: low-quality or no eye (threshold={EYE_QUALITY_THRESHOLD})")
            return {
                "status": "no_eyes_detected",
                "message": "❌ No clear eyes detected in image. Please provide a close-up, well-lit eye image.",
                "hemoglobin_estimate": None,
                "health_status": None,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "filename": file.filename,
                "eye_quality_score": float(eye_quality)
            }

        preprocessed = ImagePreprocessor.preprocess(
            image_array,
            resize=True,
            normalize=False,
            denoise=True,
            enhance_contrast=True
        )

        feature_extractor = FeatureExtractor()
        features_dict = feature_extractor.extract_all_features(preprocessed)

        feature_names = feature_extractor.get_feature_names()
        features_array = np.array([[features_dict[name] for name in feature_names]])

        features_scaled = scaler.transform(features_array)

        hemoglobin_estimate = ridge_model.predict(features_scaled)[0]

        hemoglobin_estimate = np.clip(hemoglobin_estimate, 6.0, 18.0)

        health_status = get_hemoglobin_status(hemoglobin_estimate)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "hemoglobin_estimate": float(hemoglobin_estimate),
            "eye_quality_score": float(eye_quality),
            "unit": "g/dL",
            "health_status": health_status["status"],
            "health_message": health_status["message"],
            "health_color": health_status["color"],
            "processing_time_ms": processing_time_ms,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict-multimodal")
@app.post("/predict/multimodal")
async def predict_multimodal_endpoint(
    eye_file: Optional[UploadFile] = File(None),
    nail_file: Optional[UploadFile] = File(None),
    palm_file: Optional[UploadFile] = File(None),
):
    if not multimodal_loaded:
        raise HTTPException(
            status_code=503,
            detail="Multimodal model not loaded. Run train_multimodal_production.py first.",
        )

    if not any([eye_file, nail_file, palm_file]):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one image: eye_file, nail_file, or palm_file",
        )

    start_time = time.time()
    eye_array = nail_array = palm_array = None
    eye_quality = None

    try:
        if eye_file and eye_file.filename:
            contents = await eye_file.read()
            eye_array = decode_image_bytes(contents)
            eye_quality = float(eye_detector.get_eye_quality_score(eye_array))
            detected = eye_detector.detect_eyes(eye_array)
            if eye_quality < EYE_QUALITY_THRESHOLD and not detected:
                return {
                    "status": "no_eyes_detected",
                    "message": "No clear eyes detected. Add nail/palm images or use a clearer eye photo.",
                    "hemoglobin_estimate": None,
                    "health_status": None,
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "eye_quality_score": eye_quality,
                }

        if nail_file and nail_file.filename:
            contents = await nail_file.read()
            nail_array = decode_image_bytes(contents)

        if palm_file and palm_file.filename:
            contents = await palm_file.read()
            palm_array = decode_image_bytes(contents)

        prediction = predict_multimodal(
            multimodal_model,
            multimodal_scaler,
            eye=eye_array,
            nail=nail_array,
            palm=palm_array,
        )

        hemoglobin_estimate = prediction["hemoglobin_estimate"]
        health_status = get_hemoglobin_status(hemoglobin_estimate)

        return {
            "status": "success",
            "hemoglobin_estimate": hemoglobin_estimate,
            "unit": "g/dL",
            "health_status": health_status["status"],
            "health_message": health_status["message"],
            "health_color": health_status["color"],
            "modalities_used": prediction["modalities_used"],
            "active_features": prediction["active_features"],
            "model": "multimodal",
            "eye_quality_score": eye_quality,
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )

    results = []
    feature_extractor = FeatureExtractor()
    feature_names = feature_extractor.get_feature_names()

    for file in files:
        try:
            contents = await file.read()
            try:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                image_array = np.array(image)
            except Exception:
                arr = np.frombuffer(contents, dtype=np.uint8)
                cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if cv_img is None:
                    raise
                image_array = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # Skip images that don't contain clear eyes to avoid spurious predictions
            eye_quality = eye_detector.get_eye_quality_score(image_array)
            detected = eye_detector.detect_eyes(image_array)
            print(f"[BATCH] file={file.filename} detected={detected} eye_quality={eye_quality:.3f}")
            if not detected or eye_quality < EYE_QUALITY_THRESHOLD:
                results.append({
                    "filename": file.filename,
                    "status": "no_eyes_detected",
                    "hemoglobin_estimate": None,
                    "unit": "g/dL",
                    "eye_quality_score": float(eye_quality)
                })
                continue

            preprocessed = ImagePreprocessor.preprocess(
                image_array,
                resize=True,
                normalize=False,
                denoise=True,
                enhance_contrast=True
            )

            features_dict = feature_extractor.extract_all_features(preprocessed)
            features_array = np.array([[features_dict[name] for name in feature_names]])

            features_scaled = scaler.transform(features_array)
            hemoglobin = float(np.clip(ridge_model.predict(features_scaled)[0], 6.0, 18.0))

            results.append({
                "filename": file.filename,
                "status": "success",
                "hemoglobin_estimate": hemoglobin,
                "unit": "g/dL",
                "eye_quality_score": float(eye_quality)
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

    return {
        "total": len(files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }


@app.get("/models/status")
async def models_status():
    return {
        "ridge_model": {
            "loaded": ridge_model is not None,
            "path": str(RIDGE_MODEL_PATH),
            "r2": 0.6267,
            "mae": 0.96,
            "status": "EYE_ONLY",
        },
        "multimodal_model": {
            "loaded": multimodal_loaded,
            "config": multimodal_config,
            "status": "PRIMARY" if multimodal_loaded else "NOT_LOADED",
        },
        "overall_status": "Ready" if (models_loaded or multimodal_loaded) else "Not Ready",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
