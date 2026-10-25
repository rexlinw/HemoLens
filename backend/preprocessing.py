import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple


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
        return cv2.bilateralFilter(image, kernel_size, 75, 75)

    @staticmethod
    def apply_color_constancy(image: np.ndarray, method: str = "shades_of_gray", p: int = 6) -> np.ndarray:
        """Apply Shades-of-Gray or Gray-World chromatic normalization.

        Estimates the illuminant using the Minkowski p-norm and adapts
        color channels to eliminate camera white-balance and ambient lighting casts.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image

        img_float = image.astype(np.float64)

        if method == "gray_world":
            p = 1

        # Minkowski p-norm per channel
        # Power sum with numerical stability
        p_sums = np.power(img_float, p).sum(axis=(0, 1))
        illuminant = np.power(p_sums / (image.shape[0] * image.shape[1] + 1e-6), 1.0 / p)

        # Target norm (neutral gray)
        norm_target = np.sqrt(np.sum(illuminant ** 2)) / np.sqrt(3.0)
        scale = norm_target / (illuminant + 1e-6)

        # Apply gain with gentle blending to avoid over-amplifying noise
        # 80% color constancy + 20% original retains natural appearance
        gain = 0.80 * scale + 0.20 * np.ones_like(scale)
        normalized = np.clip(img_float * gain, 0, 255).astype(np.uint8)

        return normalized

    @staticmethod
    def normalize_illumination(image: np.ndarray) -> np.ndarray:
        """Adaptive luminance normalization in LAB color space.

        Separates chrominance from luminance so blood/tissue hue ratios
        are preserved while dim/uneven lighting is equalized.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(l_channel)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        return ImagePreprocessor.normalize_illumination(image)

    @staticmethod
    def segment_conjunctiva_tissue(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment the vascular conjunctiva tissue.

        Filters out:
          - Sclera (high brightness, low saturation)
          - Pupil / Eyelashes / Shadows (very low brightness)
          - Outer facial skin (non-vascular low redness)

        Returns (mask, masked_image) where mask is uint8 {0, 255}.
        """
        rgb = image.astype(np.uint8)
        if len(rgb.shape) != 3 or rgb.shape[2] != 3:
            mask = np.ones(rgb.shape[:2], dtype=np.uint8) * 255
            return mask, rgb

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        a_channel = lab[:, :, 1]  # Green to Red (neutral is 128)

        # 1. Non-sclera mask (sclera is bright with low saturation)
        sclera_mask = (v_channel > 185) & (s_channel < 42)

        # 2. Non-dark mask (eyelashes, pupil, deep shadows)
        dark_mask = (v_channel < 40)

        # 3. Vascular tissue signature (pinkish-red hue and positive a* channel)
        # Red/pink hue wraps around 0 in HSV (0-22 or 160-180)
        hue_vascular = ((h_channel <= 24) | (h_channel >= 158)) & (s_channel >= 25)
        # Elevated a* in LAB indicates blood/vascular redness
        lab_vascular = (a_channel >= 133)

        # Combine
        tissue_mask = (hue_vascular | lab_vascular) & ~sclera_mask & ~dark_mask
        mask_uint8 = tissue_mask.astype(np.uint8) * 255

        # Morphological cleanup: remove small noisy specks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # If tissue was isolated, extract the largest coherent component(s)
        total_pixels = mask_clean.size
        tissue_pixels = np.count_nonzero(mask_clean)

        # Fallback: if segmentation is too sparse (less than 2% of frame),
        # return full non-sclera/non-dark region to prevent losing data
        if tissue_pixels < 0.02 * total_pixels:
            fallback = (~sclera_mask & ~dark_mask).astype(np.uint8) * 255
            masked = cv2.bitwise_and(rgb, rgb, mask=fallback)
            return fallback, masked

        masked = cv2.bitwise_and(rgb, rgb, mask=mask_clean)
        return mask_clean, masked

    @staticmethod
    def preprocess(image: np.ndarray, resize: bool = True, normalize: bool = False,
                   denoise: bool = True, enhance_contrast: bool = True,
                   color_constancy: bool = True) -> np.ndarray:
        """Full clinical preprocessing pipeline with lighting normalization."""
        processed = image.copy()

        if color_constancy and len(processed.shape) == 3:
            processed = ImagePreprocessor.apply_color_constancy(processed, method="shades_of_gray")

        if denoise:
            processed = ImagePreprocessor.remove_noise(processed)

        if enhance_contrast:
            processed = ImagePreprocessor.normalize_illumination(processed)

        if resize:
            processed = ImagePreprocessor.resize_image(processed)

        if normalize:
            processed = ImagePreprocessor.normalize_image(processed)

        return processed
