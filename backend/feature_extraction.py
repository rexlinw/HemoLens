import numpy as np
import cv2
from typing import Dict, Tuple
from scipy import stats


class FeatureExtractor:
    @staticmethod
    def extract_rgb_features(image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return {"R_mean": 0, "G_mean": 0, "B_mean": 0, "RG_ratio": 0}

        r_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        b_mean = np.mean(image[:, :, 2])
        rg_ratio = r_mean / (g_mean + 1e-6)

        return {
            "R_mean": r_mean,
            "G_mean": g_mean,
            "B_mean": b_mean,
            "RG_ratio": rg_ratio,
        }

    @staticmethod
    def extract_lab_features(image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return {"L_mean": 0, "a_mean": 0, "b_mean": 0,
                   "L_std": 0, "a_std": 0, "b_std": 0}

        lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)

        return {
            "L_mean": np.mean(lab[:, :, 0]),
            "a_mean": np.mean(lab[:, :, 1]),
            "b_mean": np.mean(lab[:, :, 2]),
            "L_std": np.std(lab[:, :, 0]),
            "a_std": np.std(lab[:, :, 1]),
            "b_std": np.std(lab[:, :, 2]),
        }

    @staticmethod
    def extract_hsv_features(image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return {"H_mean": 0, "S_mean": 0, "V_mean": 0,
                   "H_std": 0, "S_std": 0, "V_std": 0}

        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)

        return {
            "H_mean": np.mean(hsv[:, :, 0]),
            "S_mean": np.mean(hsv[:, :, 1]),
            "V_mean": np.mean(hsv[:, :, 2]),
            "H_std": np.std(hsv[:, :, 0]),
            "S_std": np.std(hsv[:, :, 1]),
            "V_std": np.std(hsv[:, :, 2]),
        }

    @staticmethod
    def extract_ycrcb_features(image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return {"Y_mean": 0, "Cr_mean": 0, "Cb_mean": 0}

        ycrcb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2YCrCb)

        return {
            "Y_mean": np.mean(ycrcb[:, :, 0]),
            "Cr_mean": np.mean(ycrcb[:, :, 1]),
            "Cb_mean": np.mean(ycrcb[:, :, 2]),
        }

    @staticmethod
    def extract_statistical_features(image: np.ndarray) -> Dict[str, float]:
        features = {}

        if len(image.shape) == 3:
            for i, channel_name in enumerate(['R', 'G', 'B']):
                channel = image[:, :, i].flatten()
                features[f"{channel_name}_std"] = np.std(channel)
                features[f"{channel_name}_q25"] = np.percentile(channel, 25)
                features[f"{channel_name}_q75"] = np.percentile(channel, 75)
                features[f"{channel_name}_skewness"] = stats.skew(channel)

        return features

    @staticmethod
    def extract_edge_features(image: np.ndarray) -> Dict[str, float]:
        features = {}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.hypot(sobelx, sobely)

        features["edge_mean"] = np.mean(edges)
        features["edge_std"] = np.std(edges)
        features["edge_max"] = np.max(edges)
        features["edge_density"] = np.sum(edges > 30) / edges.size

        return features

    @staticmethod
    def extract_contrast_features(image: np.ndarray) -> Dict[str, float]:
        features = {}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        features["contrast_rms"] = np.sqrt(np.mean((gray - np.mean(gray))**2))
        features["brightness"] = np.mean(gray)
        features["dynamic_range"] = np.max(gray) - np.min(gray)

        return features

    @staticmethod
    def extract_histogram_features(image: np.ndarray) -> Dict[str, float]:
        features = {}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        features["hist_entropy"] = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        features["hist_energy"] = np.sum(hist ** 2)
        features["hist_mean"] = np.mean(np.arange(256) * hist)
        features["hist_std"] = np.std(np.arange(256) * hist)
        features["hist_skewness"] = stats.skew(np.arange(256) * hist)
        features["hist_kurtosis"] = stats.kurtosis(np.arange(256) * hist)
        features["hist_uniformity"] = np.sum(hist**2)
        features["hist_peak"] = np.max(hist)

        return features

    @staticmethod
    def extract_all_features(image: np.ndarray) -> Dict[str, float]:
        features = {}

        features.update(FeatureExtractor.extract_rgb_features(image))
        features.update(FeatureExtractor.extract_lab_features(image))
        features.update(FeatureExtractor.extract_hsv_features(image))
        features.update(FeatureExtractor.extract_ycrcb_features(image))
        features.update(FeatureExtractor.extract_statistical_features(image))
        features.update(FeatureExtractor.extract_edge_features(image))
        features.update(FeatureExtractor.extract_contrast_features(image))
        features.update(FeatureExtractor.extract_histogram_features(image))

        return features

    @staticmethod
    def extract_features_batch(images: list) -> Tuple[np.ndarray, list]:
        all_features = []
        feature_names = None

        for image in images:
            features_dict = FeatureExtractor.extract_all_features(image)

            if feature_names is None:
                feature_names = list(features_dict.keys())

            feature_vector = [features_dict[name] for name in feature_names]
            all_features.append(feature_vector)

        return np.array(all_features), feature_names

    @staticmethod
    def get_feature_names() -> list:
        return [
            "R_mean", "G_mean", "B_mean", "RG_ratio",
            "L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std",
            "H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std",
            "Y_mean", "Cr_mean", "Cb_mean",
            "R_std", "R_q25", "R_q75", "R_skewness",
            "G_std", "G_q25", "G_q75", "G_skewness",
            "B_std", "B_q25", "B_q75", "B_skewness",
            "edge_mean", "edge_std", "edge_max", "edge_density",
            "contrast_rms", "brightness", "dynamic_range",
            "hist_entropy", "hist_energy", "hist_mean", "hist_std",
            "hist_skewness", "hist_kurtosis", "hist_uniformity", "hist_peak"
        ]
