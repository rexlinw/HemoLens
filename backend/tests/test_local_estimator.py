from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimator import HemoglobinEstimator


def test_local_estimator_accepts_real_eye_image():
    image_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "eyes"
        / "india"
        / "1"
        / "20200118_164733.jpg"
    )
    assert image_path.exists(), f"Missing test image: {image_path}"

    estimator = HemoglobinEstimator()
    result = estimator.predict_from_paths(eye_path=image_path)

    assert result["status"] == "success", result
    assert 6.0 <= result["hemoglobin_estimate"] <= 18.0
    assert result["model"] in {"multimodal", "ridge_eye"}
