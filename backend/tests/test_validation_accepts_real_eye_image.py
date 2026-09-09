import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eye_detector import EyeDetector
from image_validator import validate_eye


def test_real_eye_dataset_photo_is_accepted():
    image_path = (
        Path(__file__).resolve().parents[2]
        / 'data'
        / 'eyes'
        / 'india'
        / '1'
        / '20200118_164733.jpg'
    )
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None, f'Missing test image: {image_path}'

    result = validate_eye(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), EyeDetector())
    assert result.valid is True, result.message
