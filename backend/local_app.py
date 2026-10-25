from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from estimator import HemoglobinEstimator, decode_image_bytes

app = FastAPI(
    title="HemoLens Local Estimator",
    description="Local hemoglobin estimation from eye, nail, and palm images.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

estimator = HemoglobinEstimator()


@app.get("/")
async def root():
    return {
        "name": "HemoLens Local Estimator",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Model and runtime status",
            "POST /validate": "Validate any supplied image(s)",
            "POST /predict": "Predict hemoglobin from eye, nail, and/or palm images",
        },
        "ready": estimator.status()["ready"],
    }


@app.get("/health")
async def health():
    return estimator.health()


@app.post("/validate")
async def validate(
    eye_file: Optional[UploadFile] = File(None),
    nail_file: Optional[UploadFile] = File(None),
    palm_file: Optional[UploadFile] = File(None),
):
    if not any([eye_file, nail_file, palm_file]):
        raise HTTPException(status_code=400, detail="Provide at least one of eye_file, nail_file, or palm_file")

    eye = nail = palm = None
    try:
        if eye_file and eye_file.filename:
            eye = decode_image_bytes(await eye_file.read())
        if nail_file and nail_file.filename:
            nail = decode_image_bytes(await nail_file.read())
        if palm_file and palm_file.filename:
            palm = decode_image_bytes(await palm_file.read())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    try:
        result = estimator.predict(eye=eye, nail=nail, palm=palm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": result["status"],
        "model": result.get("model"),
        "validation": result.get("validation", {}),
        "message": result.get("message"),
        "warning": result.get("warning"),
    }


@app.post("/predict")
async def predict(
    eye_file: Optional[UploadFile] = File(None),
    nail_file: Optional[UploadFile] = File(None),
    palm_file: Optional[UploadFile] = File(None),
):
    if not any([eye_file, nail_file, palm_file]):
        raise HTTPException(status_code=400, detail="Provide at least one of eye_file, nail_file, or palm_file")

    start_time = time.time()
    eye = nail = palm = None

    try:
        if eye_file and eye_file.filename:
            eye = decode_image_bytes(await eye_file.read())
        if nail_file and nail_file.filename:
            nail = decode_image_bytes(await nail_file.read())
        if palm_file and palm_file.filename:
            palm = decode_image_bytes(await palm_file.read())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    try:
        result = estimator.predict(eye=eye, nail=nail, palm=palm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result["processing_time_ms"] = int((time.time() - start_time) * 1000)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
