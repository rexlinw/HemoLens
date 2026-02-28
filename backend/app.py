from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pickle
import json
import io
import time
from PIL import Image
from pathlib import Path

from preprocessing import ImagePreprocessor
from feature_extraction import FeatureExtractor
from eye_detector import EyeDetector, get_hemoglobin_status

app = FastAPI(
    title="HemoLens API v2.0",
    description="Non-invasive Hemoglobin Estimation from Eye Images (46-Feature Ridge Model)",
    version="2.0.0"
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
models_loaded = False


def load_models():
    global ridge_model, gb_model, scaler, eye_detector, models_loaded

    try:
        with open(RIDGE_MODEL_PATH, 'rb') as f:
            ridge_model = pickle.load(f)
        print(f"✓ Ridge model loaded: {RIDGE_MODEL_PATH}")

        if GB_MODEL_PATH.exists():
            with open(GB_MODEL_PATH, 'rb') as f:
                gb_model = pickle.load(f)
            print(f"✓ Gradient Boosting model loaded (backup)")

        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler loaded: {SCALER_PATH}")

        eye_detector = EyeDetector()
        print(f"✓ Eye detector initialized")

        models_loaded = True
        return True

    except FileNotFoundError as e:
        print(f"✗ Error loading models: {e}")
        print(f"  Available files in {MODEL_DIR}:")
        if MODEL_DIR.exists():
            for f in MODEL_DIR.glob("*.pkl"):
                print(f"    - {f.name}")
        return False


@app.on_event("startup")
async def startup_event():
    print("\n" + "="*70)
    print("HemoLens API v2.0 - Starting")
    print("="*70)
    success = load_models()
    if success:
        print("\n✓ API Ready for hemoglobin predictions")
        print("  Model: Ridge Regression (46 features)")
        print("  Accuracy: R² = 0.6267, MAE = 0.96 g/dL")
    else:
        print("\n✗ Failed to load models!")
    print("="*70 + "\n")


@app.get("/")
async def root():
    return {
        "name": "HemoLens API v2.0",
        "description": "Non-invasive Hemoglobin Estimation from Eye Images",
        "endpoints": {
            "GET /health": "Check API health and model status",
            "GET /info": "Get model information and features",
            "POST /predict": "Predict hemoglobin from image",
            "POST /predict/batch": "Batch predictions from multiple images"
        },
        "model": {
            "type": "Ridge Regression (46-Feature)",
            "accuracy_r2": 0.6267,
            "accuracy_mae": 0.96,
            "features": 46
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "model_type": "Ridge Regression (46 features)",
        "accuracy": {
            "r2": 0.6267,
            "mae": 0.96,
            "unit": "g/dL"
        }
    }


@app.get("/info")
async def get_model_info():
    feature_names = list(FeatureExtractor.get_feature_names())

    return {
        "model_name": "Ridge Regression Ensemble",
        "version": "2.0",
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
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)

        if not eye_detector.detect_eyes(image_array):
            return {
                "status": "no_eyes_detected",
                "message": "❌ No eyes detected in image. Please provide a clear image of your eye.",
                "hemoglobin_estimate": None,
                "health_status": None,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "filename": file.filename
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
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_array = np.array(image)

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
                "unit": "g/dL"
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
            "rmse": 1.3745,
            "status": "PRIMARY"
        },
        "gb_model": {
            "loaded": gb_model is not None,
            "r2": 0.6242,
            "mae": 1.1529,
            "status": "BACKUP"
        },
        "scaler": {
            "loaded": scaler is not None,
            "features": 46
        },
        "overall_status": "Ready" if models_loaded else "Not Ready"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
