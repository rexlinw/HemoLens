#!/usr/bin/env python3
"""
Palm Feature Extractor - Extract features from palm images for anemia detection.

Features:
- Palmar color indicators (pallor detection)
- Vascular patterns
- Texture and creases
- Morphological characteristics

Reference:
- Palmar pallor (pale appearance) is a key clinical sign of anemia
- Healthy palms show prominent pink/red coloration
- Anemic palms show whitening, especially at fingertips and margins
"""

import cv2
import numpy as np
from typing import Dict


class PalmFeatureExtractor:
    """Extract anemia indicators from palm images."""

    def extract_all_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract all palm features from image.

        Args:
            image: RGB image array (preferably preprocessed/resized)

        Returns:
            Dictionary with feature names and values
        """
        features = {}

        # Color features (pallor detection)
        color_features = self._extract_color_features(image)
        features.update(color_features)

        # Vascular features
        vascular_features = self._extract_vascular_features(image)
        features.update(vascular_features)

        # Texture features (creases, wrinkles)
        texture_features = self._extract_texture_features(image)
        features.update(texture_features)

        # Morphological features
        morph_features = self._extract_morphological_features(image)
        features.update(morph_features)

        return features

    def _extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color-based features for pallor detection.

        Key indicators:
        - Low red channel → pallor
        - High red-green difference → healthy (vascularity)
        - Low saturation → whitening
        """
        features = {}

        # Convert to different color spaces
        rgb = image if len(image.shape) == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        # RGB means
        features['palm_r_mean'] = float(np.mean(rgb[:, :, 0]))
        features['palm_g_mean'] = float(np.mean(rgb[:, :, 1]))
        features['palm_b_mean'] = float(np.mean(rgb[:, :, 2]))

        # Red dominance (healthy indicator)
        features['palm_r_g_diff'] = float(np.mean(rgb[:, :, 0]) - np.mean(rgb[:, :, 1]))
        features['palm_r_b_diff'] = float(np.mean(rgb[:, :, 0]) - np.mean(rgb[:, :, 2]))

        # Pallor score: inverse of redness
        # High score = pallid, Low score = healthy
        total_color = np.mean(rgb)
        redness = np.mean(rgb[:, :, 0])
        features['palm_pallor_score'] = float(max(0, 1.0 - (redness / max(1, total_color))))

        # HSV features
        features['palm_h_mean'] = float(np.mean(hsv[:, :, 0]))
        features['palm_s_mean'] = float(np.mean(hsv[:, :, 1]))  # Saturation (low in pallor)
        features['palm_v_mean'] = float(np.mean(hsv[:, :, 2]))  # Value/brightness

        # LAB features (perceptually uniform)
        features['palm_l_mean'] = float(np.mean(lab[:, :, 0]))   # Lightness
        features['palm_a_mean'] = float(np.mean(lab[:, :, 1]))   # Green-red axis
        features['palm_b_mean'] = float(np.mean(lab[:, :, 2]))   # Blue-yellow axis

        # Whiteness indicator (high L, low saturation)
        features['palm_whiteness'] = float(np.mean(lab[:, :, 0]) / 100.0)

        return features

    def _extract_vascular_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract vascular pattern features.

        Healthy palms show:
        - Prominent vascular lines (darker threads)
        - Strong color contrast
        Anemic palms show:
        - Faded vascular patterns
        - Reduced contrast
        """
        features = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Edge detection for vascular patterns
        edges = cv2.Canny(gray, 50, 150)
        features['palm_edge_density'] = float(np.sum(edges > 0) / edges.size)

        # Vascular prominence (local contrast)
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        features['palm_vascular_prominence'] = float(np.mean(morph))

        # Color contrast (healthy palms have more color variation)
        rgb = image if len(image.shape) == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        r_std = float(np.std(rgb[:, :, 0]))
        g_std = float(np.std(rgb[:, :, 1]))
        b_std = float(np.std(rgb[:, :, 2]))
        features['palm_color_contrast'] = float((r_std + g_std + b_std) / 3.0)

        return features

    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features (skin roughness, creases).

        Indicators:
        - High texture → healthy (skin elasticity)
        - Low texture → anemic (less visible creases)
        """
        features = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Laplacian (edge/texture detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['palm_laplacian_mean'] = float(np.mean(np.abs(laplacian)))
        features['palm_laplacian_std'] = float(np.std(laplacian))

        # Entropy (texture complexity)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features['palm_texture_entropy'] = float(entropy)

        # Local binary pattern-like feature
        # Compute standard deviation as texture metric
        features['palm_texture_std'] = float(np.std(gray))

        return features

    def _extract_morphological_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features from palm shape and structure.
        """
        features = {}

        # Convert to grayscale and binary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Threshold to get palm region (skin is typically lighter)
        _, binary = cv2.threshold(gray, gray.mean(), 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Largest contour = palm
            largest_contour = max(contours, key=cv2.contourArea)
            palm_area = cv2.contourArea(largest_contour)
            features['palm_area'] = float(palm_area)

            # Perimeter
            perimeter = cv2.arcLength(largest_contour, True)
            features['palm_perimeter'] = float(perimeter)

            # Circularity
            if perimeter > 0:
                circularity = 4 * np.pi * palm_area / (perimeter ** 2)
                features['palm_circularity'] = float(circularity)
            else:
                features['palm_circularity'] = 0.0

            # Convexity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = palm_area / hull_area
                features['palm_solidity'] = float(solidity)
            else:
                features['palm_solidity'] = 0.0

            # Fit ellipse if enough points
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                (center, (major_axis, minor_axis), angle) = ellipse
                if minor_axis > 0:
                    aspect_ratio = major_axis / minor_axis
                    features['palm_aspect_ratio'] = float(aspect_ratio)
                else:
                    features['palm_aspect_ratio'] = 1.0
            else:
                features['palm_aspect_ratio'] = 1.0

            # Number of defects (palm fingers)
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            defects = cv2.convexityDefects(largest_contour, hull)
            features['palm_num_defects'] = float(len(defects) if defects is not None else 0)

        else:
            # No contour found
            features['palm_area'] = 0.0
            features['palm_perimeter'] = 0.0
            features['palm_circularity'] = 0.0
            features['palm_solidity'] = 0.0
            features['palm_aspect_ratio'] = 0.0
            features['palm_num_defects'] = 0.0

        return features

    def get_feature_names(self):
        """Return list of all feature names in extraction order."""
        return [
            # Color features (11)
            'palm_r_mean', 'palm_g_mean', 'palm_b_mean',
            'palm_r_g_diff', 'palm_r_b_diff',
            'palm_pallor_score',
            'palm_h_mean', 'palm_s_mean', 'palm_v_mean',
            'palm_l_mean', 'palm_a_mean', 'palm_b_mean',
            'palm_whiteness',
            # Vascular features (3)
            'palm_edge_density', 'palm_vascular_prominence', 'palm_color_contrast',
            # Texture features (4)
            'palm_laplacian_mean', 'palm_laplacian_std',
            'palm_texture_entropy', 'palm_texture_std',
            # Morphological features (6)
            'palm_area', 'palm_perimeter', 'palm_circularity',
            'palm_solidity', 'palm_aspect_ratio', 'palm_num_defects',
        ]

    def get_feature_count(self):
        """Return total number of features extracted."""
        return len(self.get_feature_names())
