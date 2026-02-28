import numpy as np
import cv2
from PIL import Image
from typing import Tuple


class ImagePreprocessor:
    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        return np.array(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image = image / 255.0
        return image

    @staticmethod
    def remove_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            return cv2.bilateralFilter(image, kernel_size, 75, 75)

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            lab[:, :, 0] = l_channel
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def preprocess(image: np.ndarray, resize: bool = True, normalize: bool = True,
                   denoise: bool = True, enhance_contrast: bool = True) -> np.ndarray:
        processed = image.copy()

        if denoise:
            processed = ImagePreprocessor.remove_noise(processed)

        if enhance_contrast:
            processed = ImagePreprocessor.enhance_contrast(processed)

        if resize:
            processed = ImagePreprocessor.resize_image(processed)

        if normalize:
            processed = ImagePreprocessor.normalize_image(processed)

        return processed
