#!/usr/bin/env python3
"""
Fingernail-specific feature extraction for anemia detection.
Extracts color, texture, and morphological features from nail images.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple


class NailFeatureExtractor:
    """Extract features from fingernail images for anemia detection."""

    @staticmethod
    def extract_color_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract color-based features from nail bed (pink area).
        
        Features:
        - Red/pink intensity (R channel dominant)
        - Pallor (low saturation, low value)
        - Nail bed hue distribution
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]

        # Use center 60% of image (likely nail bed, avoid edges/cuticle)
        y_start = int(h * 0.2)
        y_end = int(h * 0.8)
        x_start = int(w * 0.15)
        x_end = int(w * 0.85)
        nail_bed = image[y_start:y_end, x_start:x_end]

        # Convert to HSV and LAB for better color analysis
        hsv = cv2.cvtColor(nail_bed, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(nail_bed, cv2.COLOR_RGB2LAB).astype(np.float32)

        rgb_f = nail_bed.astype(np.float32)

        features = {
            # RGB channel means
            "nail_r_mean": float(rgb_f[:, :, 0].mean()),
            "nail_g_mean": float(rgb_f[:, :, 1].mean()),
            "nail_b_mean": float(rgb_f[:, :, 2].mean()),

            # R-G difference (redness indicator; high = more anemic typically shows lower)
            "nail_rg_diff": float((rgb_f[:, :, 0] - rgb_f[:, :, 1]).mean()),

            # HSV features
            "nail_hue_mean": float(hsv[:, :, 0].mean()),
            "nail_saturation_mean": float(hsv[:, :, 1].mean()),
            "nail_value_mean": float(hsv[:, :, 2].mean()),
            "nail_saturation_std": float(hsv[:, :, 1].std()),

            # LAB features (perceptual color)
            "nail_l_mean": float(lab[:, :, 0].mean()),  # Lightness
            "nail_a_mean": float(lab[:, :, 1].mean()),  # Green-Red axis
            "nail_b_mean": float(lab[:, :, 2].mean()),  # Blue-Yellow axis

            # Pallor metric: low saturation + low hue (pale/white appearance)
            "nail_pallor_score": float(
                (255 - hsv[:, :, 1].mean()) / 255 + (1 - hsv[:, :, 2].mean() / 255) / 2
            ),
        }

        return features

    @staticmethod
    def extract_texture_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features from nail surface.
        
        Features:
        - Edge density
        - Entropy
        - Texture roughness
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Local Binary Pattern-like texture
        h, w = gray.shape
        y_start = int(h * 0.2)
        y_end = int(h * 0.8)
        x_start = int(w * 0.15)
        x_end = int(w * 0.85)
        roi = gray[y_start:y_end, x_start:x_end]

        # Entropy (texture complexity)
        hist, _ = np.histogram(roi, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

        # Laplacian for edge-ness
        laplacian = cv2.Laplacian(roi, cv2.CV_64F)

        features = {
            "nail_edge_density": float(np.mean(edge_magnitude > 10)),
            "nail_edge_mean": float(edge_magnitude.mean()),
            "nail_edge_std": float(edge_magnitude.std()),
            "nail_entropy": float(entropy),
            "nail_laplacian_mean": float(np.abs(laplacian).mean()),
            "nail_laplacian_std": float(np.abs(laplacian).std()),
            "nail_contrast_std": float(gray.std()),
        }

        return features

    @staticmethod
    def extract_morphological_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features (size, orientation, etc.).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        h, w = gray.shape

        # Detect nail region using color thresholding
        # Nail is typically pinkish (R > G, G > B)
        if len(image.shape) == 3:
            rgb = image.astype(np.float32)
            nail_mask = (rgb[:, :, 0] > rgb[:, :, 1]) & (rgb[:, :, 1] > rgb[:, :, 2])
            nail_mask = nail_mask.astype(np.uint8) * 255

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_CLOSE, kernel)
            nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(nail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            features = {
                "nail_detected_area": float((nail_mask > 0).sum() / nail_mask.size),
                "nail_num_contours": float(len(contours)),
            }

            if contours:
                # Largest contour
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                perimeter = cv2.arcLength(largest, True)
                circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

                features["nail_largest_area_pct"] = float(area / nail_mask.size)
                features["nail_circularity"] = float(circularity)

                # Fit ellipse if enough points
                if len(largest) >= 5:
                    ellipse = cv2.fitEllipse(largest)
                    (cx, cy), (ma, mi), angle = ellipse
                    features["nail_ellipse_ratio"] = float(ma / (mi + 1e-6))
                else:
                    features["nail_ellipse_ratio"] = 1.0

            else:
                features["nail_largest_area_pct"] = 0.0
                features["nail_circularity"] = 0.0
                features["nail_ellipse_ratio"] = 0.0
        else:
            features = {
                "nail_detected_area": 0.0,
                "nail_num_contours": 0.0,
                "nail_largest_area_pct": 0.0,
                "nail_circularity": 0.0,
                "nail_ellipse_ratio": 0.0,
            }

        features["nail_aspect_ratio"] = float(h / (w + 1e-6))

        return features

    @classmethod
    def extract_all_features(cls, image: np.ndarray) -> Dict[str, float]:
        """
        Extract all nail features.
        
        Returns a dictionary with ~35 features covering:
        - Color (RGB, HSV, LAB)
        - Texture (edges, entropy, contrast)
        - Morphology (size, shape, detected area)
        """
        features = {}
        features.update(cls.extract_color_features(image))
        features.update(cls.extract_texture_features(image))
        features.update(cls.extract_morphological_features(image))
        return features

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return list of all feature names."""
        return [
            # Color
            "nail_r_mean", "nail_g_mean", "nail_b_mean",
            "nail_rg_diff", "nail_hue_mean", "nail_saturation_mean",
            "nail_value_mean", "nail_saturation_std", "nail_l_mean",
            "nail_a_mean", "nail_b_mean", "nail_pallor_score",
            # Texture
            "nail_edge_density", "nail_edge_mean", "nail_edge_std",
            "nail_entropy", "nail_laplacian_mean", "nail_laplacian_std",
            "nail_contrast_std",
            # Morphology
            "nail_detected_area", "nail_num_contours", "nail_largest_area_pct",
            "nail_circularity", "nail_ellipse_ratio", "nail_aspect_ratio",
        ]
