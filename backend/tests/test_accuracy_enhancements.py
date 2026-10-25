import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing import ImagePreprocessor
from uncertainty import compute_signal_quality_index, estimate_confidence_interval
from estimator import HemoglobinEstimator


def test_color_constancy_balances_warm_cast():
    # Create image with strong yellow/warm tint
    img = np.full((100, 100, 3), [220, 180, 80], dtype=np.uint8)
    balanced = ImagePreprocessor.apply_color_constancy(img, method="shades_of_gray")
    
    # Blue channel should be increased relative to Red
    original_rg_diff = int(img[:, :, 0].mean()) - int(img[:, :, 2].mean())
    balanced_rg_diff = int(balanced[:, :, 0].mean()) - int(balanced[:, :, 2].mean())
    assert balanced_rg_diff < original_rg_diff


def test_normalize_illumination():
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    norm = ImagePreprocessor.normalize_illumination(img)
    assert norm.shape == img.shape
    assert norm.dtype == np.uint8


def test_conjunctiva_segmentation():
    # Synthetic eye patch: white sclera, dark border, and vascular pink tissue
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :30] = [240, 240, 240]  # Sclera (white)
    img[:, 30:70] = [190, 80, 90]   # Vascular conjunctiva (pinkish-red)
    img[:, 70:] = [20, 20, 20]      # Dark border / lashes

    mask, masked = ImagePreprocessor.segment_conjunctiva_tissue(img)
    assert mask.shape == (100, 100)
    # The middle section should have the highest mask coverage
    assert mask[:, 35:65].mean() > mask[:, :20].mean()


def test_signal_quality_index():
    # Sharp image with good contrast
    sharp_img = np.zeros((150, 150, 3), dtype=np.uint8)
    sharp_img[::2, ::2] = [200, 120, 130]
    sharp_img[1::2, 1::2] = [80, 50, 60]

    sqi_sharp = compute_signal_quality_index(sharp_img, modality="eye")
    assert "score" in sqi_sharp
    assert "level" in sqi_sharp
    assert 0 <= sqi_sharp["score"] <= 100

    # Flat solid gray image (no focus/detail)
    flat_img = np.full((150, 150, 3), 120, dtype=np.uint8)
    sqi_flat = compute_signal_quality_index(flat_img, modality="eye")
    assert sqi_flat["sharpness"] < sqi_sharp["sharpness"]


def test_confidence_interval():
    ci = estimate_confidence_interval(13.2, sqi_score=85, modalities_count=3)
    assert ci["lower"] < 13.2 < ci["upper"]
    assert ci["confidence_level"] == "90%"
    assert ci["unit"] == "g/dL"

    # Low SQI should produce a wider interval
    ci_low = estimate_confidence_interval(13.2, sqi_score=40, modalities_count=1)
    assert ci_low["margin"] > ci["margin"]


def test_estimator_includes_uncertainty():
    estimator = HemoglobinEstimator()
    sample_path = Path(__file__).resolve().parents[2] / "data/eyes/india/1/20200118_164733.jpg"
    if not sample_path.exists():
        pytest.skip("Dataset sample not found")

    import cv2
    img = cv2.cvtColor(cv2.imread(str(sample_path)), cv2.COLOR_BGR2RGB)
    res = estimator.predict_eye_only(img)

    assert res["status"] == "success"
    assert "confidence_interval" in res
    assert "signal_quality_index" in res
    assert res["confidence_interval"]["lower"] <= res["hemoglobin_estimate"] <= res["confidence_interval"]["upper"]
