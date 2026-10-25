from pathlib import Path
import cv2
import numpy as np
import pytest

from eye_detector import EyeDetector
from image_validator import (
    validate_nail,
    validate_palm,
    validate_multimodal_inputs,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@pytest.fixture(scope="module")
def detector():
    return EyeDetector()


class TestNailValidation:
    def test_accepts_real_nail_dataset_images(self, detector):
        nails_dir = DATA_DIR / "nails"
        if not nails_dir.exists():
            pytest.skip("Nails data directory not found")

        nail_files = sorted(list(nails_dir.rglob("*.png")) + list(nails_dir.rglob("*.jpg")))[:20]
        if not nail_files:
            pytest.skip("No nail dataset files found")

        accepted = 0
        for f in nail_files:
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            result = validate_nail(img, detector)
            if result.valid:
                accepted += 1

        assert accepted >= 18, f"Expected >= 90% nail acceptance, got {accepted}/{len(nail_files)}"

    @pytest.mark.parametrize(
        "image_factory",
        [
            lambda: np.ones((150, 150, 3), dtype=np.uint8) * 255,  # white
            lambda: np.zeros((150, 150, 3), dtype=np.uint8),  # black
            lambda: np.ones((150, 150, 3), dtype=np.uint8) * 128,  # gray
            lambda: np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8),  # noise
        ],
    )
    def test_rejects_synthetic_nail(self, image_factory, detector):
        img = image_factory()
        res = validate_nail(img, detector)
        assert res.valid is False


class TestPalmValidation:
    def test_accepts_real_palm_dataset_images(self, detector):
        palms_dir = DATA_DIR / "palms"
        if not palms_dir.exists():
            pytest.skip("Palms data directory not found")

        palm_files = sorted(list(palms_dir.rglob("*.png")) + list(palms_dir.rglob("*.jpg")))[:20]
        if not palm_files:
            pytest.skip("No palm dataset files found")

        accepted = 0
        for f in palm_files:
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            result = validate_palm(img, detector)
            if result.valid:
                accepted += 1

        assert accepted >= 18, f"Expected >= 90% palm acceptance, got {accepted}/{len(palm_files)}"

    @pytest.mark.parametrize(
        "image_factory",
        [
            lambda: np.ones((150, 150, 3), dtype=np.uint8) * 255,  # white
            lambda: np.zeros((150, 150, 3), dtype=np.uint8),  # black
            lambda: np.ones((150, 150, 3), dtype=np.uint8) * 128,  # gray
            lambda: np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8),  # noise
        ],
    )
    def test_rejects_synthetic_palm(self, image_factory, detector):
        img = image_factory()
        res = validate_palm(img, detector)
        assert res.valid is False


def test_validate_multimodal_accepts_valid_modalities(detector):
    nails_dir = DATA_DIR / "nails"
    palms_dir = DATA_DIR / "palms"
    if not nails_dir.exists() or not palms_dir.exists():
        pytest.skip("Data directory missing")

    nail_path = next(nails_dir.rglob("*.png"), None)
    palm_path = next(palms_dir.rglob("*.png"), None)
    if not nail_path or not palm_path:
        pytest.skip("Images not found")

    nail_img = cv2.cvtColor(cv2.imread(str(nail_path)), cv2.COLOR_BGR2RGB)
    palm_img = cv2.cvtColor(cv2.imread(str(palm_path)), cv2.COLOR_BGR2RGB)

    results, accepted, err = validate_multimodal_inputs(detector, nail=nail_img, palm=palm_img)
    assert err is None
    assert "nail" in accepted
    assert "palm" in accepted
