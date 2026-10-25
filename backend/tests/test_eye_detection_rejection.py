"""Tests to verify that the eye detector rejects non-eye images and accepts real eye images."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eye_detector import EyeDetector
from image_validator import validate_eye


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# Helpers — synthetic non-eye images
# ---------------------------------------------------------------------------

def _solid_color(r, g, b, size=256):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = (r, g, b)
    return img


def _random_noise(size=256):
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)


def _gradient(size=256):
    row = np.linspace(0, 255, size, dtype=np.uint8)
    gray = np.tile(row, (size, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _green_landscape():
    """Green grass + blue sky — nothing eye-like."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128, :] = (135, 206, 235)   # sky blue (RGB)
    img[128:, :] = (34, 139, 34)     # forest green
    return img


def _text_like():
    """Black text on white background."""
    img = np.full((256, 256, 3), 245, dtype=np.uint8)
    cv2.putText(img, "Hello World", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
    return img


def _add_texture(img, sigma=8):
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _coffee_cup():
    """White ceramic mug with circular rim and coffee inside."""
    img = np.full((300, 300, 3), 160, dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (240, 240, 240), -1)
    cv2.circle(img, (150, 150), 60, (50, 30, 20), -1)
    return _add_texture(img)


def _watch():
    """Wristwatch with white dial and circular bezel."""
    img = np.full((300, 300, 3), 120, dtype=np.uint8)
    cv2.circle(img, (150, 150), 75, (230, 230, 230), -1)
    cv2.circle(img, (150, 150), 10, (20, 20, 20), -1)
    cv2.line(img, (150, 150), (150, 100), (20, 20, 20), 3)
    return _add_texture(img)


def _watch_on_arm():
    """Watch worn on wrist (human skin background)."""
    img = np.full((300, 300, 3), (190, 140, 120), dtype=np.uint8)
    cv2.rectangle(img, (120, 0), (180, 300), (30, 30, 30), -1)
    cv2.circle(img, (150, 150), 70, (240, 240, 240), -1)
    cv2.line(img, (150, 150), (150, 105), (20, 20, 20), 3)
    return _add_texture(img)


def _plate():
    """Ceramic plate with circular rim."""
    img = np.full((300, 300, 3), 100, dtype=np.uint8)
    cv2.circle(img, (150, 150), 90, (245, 245, 245), -1)
    cv2.circle(img, (150, 150), 40, (180, 50, 30), -1)
    return _add_texture(img)


def _car_headlight():
    """Car headlight with reflector housing."""
    img = np.full((300, 300, 3), 80, dtype=np.uint8)
    cv2.circle(img, (150, 150), 85, (235, 235, 235), -1)
    cv2.circle(img, (150, 150), 35, (255, 240, 180), -1)
    return _add_texture(img)


def _notebook_paper():
    """Ruled horizontal lines on white paper."""
    img = np.full((300, 300, 3), 245, dtype=np.uint8)
    for y in range(30, 300, 30):
        cv2.line(img, (0, y), (300, y), (50, 50, 200), 2)
    return _add_texture(img)


# ---------------------------------------------------------------------------
# Test: reject synthetic non-eye images
# ---------------------------------------------------------------------------

class TestRejectsNonEyeImages:
    detector = EyeDetector()

    @pytest.mark.parametrize("image,description", [
        (_solid_color(180, 180, 180), "solid gray"),
        (_solid_color(255, 0, 0), "solid red"),
        (_solid_color(0, 128, 0), "solid green"),
        (_solid_color(0, 0, 255), "solid blue"),
        (_random_noise(), "random noise"),
        (_gradient(), "grayscale gradient"),
        (_green_landscape(), "green landscape"),
        (_text_like(), "text on white"),
        (_coffee_cup(), "coffee cup"),
        (_watch(), "wristwatch"),
        (_watch_on_arm(), "watch on arm"),
        (_plate(), "dinner plate"),
        (_car_headlight(), "car headlight"),
        (_notebook_paper(), "ruled notebook paper"),
    ])
    def test_detect_eyes_returns_false(self, image, description):
        result = self.detector.detect_eyes(image)
        assert result is False, f"detect_eyes should reject '{description}'"

    @pytest.mark.parametrize("image,description", [
        (_solid_color(180, 180, 180), "solid gray"),
        (_random_noise(), "random noise"),
        (_gradient(), "grayscale gradient"),
        (_green_landscape(), "green landscape"),
        (_text_like(), "text on white"),
        (_coffee_cup(), "coffee cup"),
        (_watch(), "wristwatch"),
        (_watch_on_arm(), "watch on arm"),
        (_plate(), "dinner plate"),
        (_car_headlight(), "car headlight"),
        (_notebook_paper(), "ruled notebook paper"),
    ])
    def test_validate_eye_rejects(self, image, description):
        result = validate_eye(image, self.detector)
        assert result.valid is False, (
            f"validate_eye should reject '{description}', got: {result.message}"
        )


# ---------------------------------------------------------------------------
# Test: reject nail / palm images submitted as eye
# ---------------------------------------------------------------------------

class TestRejectsCrossModality:
    detector = EyeDetector()

    def _first_image_in(self, subdir: str):
        d = DATA_DIR / subdir
        if not d.exists():
            pytest.skip(f"Data directory missing: {d}")
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files = sorted(d.rglob(ext))
            if files:
                return files[0]
        pytest.skip(f"No images found in {d}")

    def test_rejects_nail_as_eye(self):
        path = self._first_image_in("nails")
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        result = validate_eye(image, self.detector)
        assert result.valid is False, (
            f"Nail image '{path.name}' should be rejected as eye: {result.message}"
        )

    def test_rejects_palm_as_eye(self):
        path = self._first_image_in("palms")
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        result = validate_eye(image, self.detector)
        assert result.valid is False, (
            f"Palm image '{path.name}' should be rejected as eye: {result.message}"
        )


# ---------------------------------------------------------------------------
# Test: accept real eye images from dataset
# ---------------------------------------------------------------------------

class TestAcceptsRealEyeImages:
    detector = EyeDetector()

    def _eye_images(self):
        eyes_dir = DATA_DIR / "eyes"
        if not eyes_dir.exists():
            pytest.skip(f"Eye data directory missing: {eyes_dir}")
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(sorted(eyes_dir.rglob(ext))[:5])
        if not images:
            pytest.skip("No eye images found")
        return images[:5]

    def test_detect_eyes_returns_true(self):
        for path in self._eye_images():
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            result = self.detector.detect_eyes(image)
            assert result is True, f"detect_eyes should accept eye image '{path.name}'"

    def test_validate_eye_accepts(self):
        for path in self._eye_images():
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            result = validate_eye(image, self.detector)
            assert result.valid is True, (
                f"validate_eye should accept eye image '{path.name}': {result.message}"
            )
