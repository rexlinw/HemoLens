from pathlib import Path
import cv2
import numpy as np
import pytest

from multimodal import (
    CLINICAL_MODALITY_WEIGHTS,
    CLINICAL_LITERATURE_EVIDENCE,
    compute_modality_weights,
    predict_multimodal,
    load_multimodal_artifacts,
)
from estimator import HemoglobinEstimator
from image_validator import validate_eye

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def test_clinical_modality_weights_sum_to_one():
    assert abs(sum(CLINICAL_MODALITY_WEIGHTS.values()) - 1.0) < 1e-6
    assert CLINICAL_MODALITY_WEIGHTS["eye"] == 0.55
    assert CLINICAL_MODALITY_WEIGHTS["nail"] == 0.25
    assert CLINICAL_MODALITY_WEIGHTS["palm"] == 0.20


def test_clinical_evidence_structure():
    for mod in ("eye", "nail", "palm"):
        assert mod in CLINICAL_LITERATURE_EVIDENCE
        ev = CLINICAL_LITERATURE_EVIDENCE[mod]
        assert "citation" in ev
        assert "rationale" in ev
        assert "anatomical_site" in ev
        assert "weight_percent" in ev


def test_compute_modality_weights_all_modalities():
    w = compute_modality_weights(["eye", "nail", "palm"])
    assert w["eye"] == 0.55
    assert w["nail"] == 0.25
    assert w["palm"] == 0.20
    assert abs(sum(w.values()) - 1.0) < 1e-3


def test_compute_modality_weights_subsets():
    # Eye + Nail
    w_en = compute_modality_weights(["eye", "nail"])
    expected_eye = round(0.55 / 0.80, 4)
    expected_nail = round(0.25 / 0.80, 4)
    assert abs(w_en["eye"] - expected_eye) < 1e-3
    assert abs(w_en["nail"] - expected_nail) < 1e-3
    assert abs(sum(w_en.values()) - 1.0) < 1e-3

    # Eye + Palm
    w_ep = compute_modality_weights(["eye", "palm"])
    expected_eye_p = round(0.55 / 0.75, 4)
    expected_palm = round(0.20 / 0.75, 4)
    assert abs(w_ep["eye"] - expected_eye_p) < 1e-3
    assert abs(w_ep["palm"] - expected_palm) < 1e-3

    # Nail + Palm
    w_np = compute_modality_weights(["nail", "palm"])
    assert abs(w_np["nail"] - round(0.25 / 0.45, 4)) < 1e-3
    assert abs(w_np["palm"] - round(0.20 / 0.45, 4)) < 1e-3

    # Single modality
    assert compute_modality_weights(["eye"])["eye"] == 1.0
    assert compute_modality_weights(["nail"])["nail"] == 1.0
    assert compute_modality_weights(["palm"])["palm"] == 1.0


def test_compute_modality_weights_sqi_quality_adjustment():
    # Equal baseline, but nail has higher SQI
    w_base = compute_modality_weights(["eye", "nail"], sqi_scores={"eye": 50.0, "nail": 50.0})
    w_boosted = compute_modality_weights(["eye", "nail"], sqi_scores={"eye": 50.0, "nail": 100.0})
    # Nail weight should increase when nail SQI is higher
    assert w_boosted["nail"] > w_base["nail"]
    assert abs(sum(w_boosted.values()) - 1.0) < 1e-3


def test_predict_multimodal_weighted_fusion():
    model, scaler, _ = load_multimodal_artifacts(MODEL_DIR)
    if model is None or scaler is None:
        pytest.skip("Multimodal model artifacts not found")

    nail_path = next((DATA_DIR / "nails").rglob("*.png"), None)
    palm_path = next((DATA_DIR / "palms").rglob("*.png"), None)
    if not nail_path or not palm_path:
        pytest.skip("Dataset images not found")

    nail_img = cv2.cvtColor(cv2.imread(str(nail_path)), cv2.COLOR_BGR2RGB)
    palm_img = cv2.cvtColor(cv2.imread(str(palm_path)), cv2.COLOR_BGR2RGB)

    pred = predict_multimodal(model, scaler, nail=nail_img, palm=palm_img)

    assert "hemoglobin_estimate" in pred
    assert "modality_weights" in pred
    assert "modality_estimates" in pred
    assert "modality_contributions" in pred
    assert "clinical_evidence" in pred

    # Verify weighted formula consistency
    weights = pred["modality_weights"]
    estimates = pred["modality_estimates"]
    expected = sum(weights[m] * estimates[m] for m in pred["modalities_used"])
    assert abs(pred["hemoglobin_estimate"] - round(expected, 2)) < 0.05


def test_estimator_predict_multimodal_end_to_end():
    estimator = HemoglobinEstimator()
    if not estimator.loaded:
        pytest.skip("Estimator not loaded")

    nail_path = next((DATA_DIR / "nails").rglob("*.png"), None)
    palm_path = next((DATA_DIR / "palms").rglob("*.png"), None)
    if not nail_path or not palm_path:
        pytest.skip("Dataset images not found")

    nail_img = cv2.cvtColor(cv2.imread(str(nail_path)), cv2.COLOR_BGR2RGB)
    palm_img = cv2.cvtColor(cv2.imread(str(palm_path)), cv2.COLOR_BGR2RGB)

    res = estimator.predict_multimodal(nail=nail_img, palm=palm_img)
    assert res["status"] == "success"
    assert "modality_weights" in res
    assert "modality_contributions" in res
    assert "clinical_evidence" in res
    assert res["modality_weights"]["nail"] > 0
    assert res["modality_weights"]["palm"] > 0
