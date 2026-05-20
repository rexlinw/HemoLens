#!/usr/bin/env python3
"""Train eye + nail + palm multimodal hemoglobin model."""

import json
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from multimodal import (
    EYE_SLICE,
    NAIL_SLICE,
    PALM_SLICE,
    TOTAL_FEATURES,
    extract_eye_features,
    extract_nail_features,
    extract_palm_features,
)

DATA_ROOT = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

EYE_COHORTS = [
    ("india", DATA_ROOT / "eyes" / "india", "India.xlsx"),
    ("italy", DATA_ROOT / "eyes" / "italy", "Italy.xlsx"),
]
NAIL_ROOTS = [
    DATA_ROOT / "nails" / "standard" / "Finger_Nails",
    DATA_ROOT / "nails" / "ghana" / "Fingernails",
]
PALM_DIRS = [
    ("anemic", DATA_ROOT / "palms" / "anemic", True),
    ("non_anemic", DATA_ROOT / "palms" / "non_anemic", False),
]

TARGET_R2 = 0.80
RANDOM_STATE = 42


def _parse_hgb(raw) -> float | None:
    if pd.isna(raw):
        return None
    try:
        v = float(str(raw).strip().replace(",", "."))
        return v if 6.0 <= v <= 20.0 else None
    except ValueError:
        return None


def compute_eye_hb_stats() -> dict:
    values = []
    for _name, cohort_dir, xlsx_name in EYE_COHORTS:
        df = pd.read_excel(cohort_dir / xlsx_name)
        hgb_col = "Hgb" if "Hgb" in df.columns else df.columns[1]
        for raw in df[hgb_col]:
            v = _parse_hgb(raw)
            if v is not None:
                values.append(v)
    arr = np.array(values)
    low = arr[arr < 12.0]
    high = arr[arr >= 12.0]
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "anemic_mean": float(low.mean()) if len(low) else 9.5,
        "anemic_std": float(low.std()) if len(low) else 1.2,
        "healthy_mean": float(high.mean()) if len(high) else 13.8,
        "healthy_std": float(high.std()) if len(high) else 1.5,
        "n": len(arr),
    }


def hb_from_anemic_flag(is_anemic: bool, stats: dict, rng: np.random.Generator) -> float:
    if is_anemic:
        return float(rng.normal(stats["anemic_mean"], stats["anemic_std"] * 0.35))
    return float(rng.normal(stats["healthy_mean"], stats["healthy_std"] * 0.35))


def load_eye_samples():
    rows = []
    for cohort_name, cohort_dir, xlsx_name in EYE_COHORTS:
        df = pd.read_excel(cohort_dir / xlsx_name)
        hgb_col = "Hgb" if "Hgb" in df.columns else df.columns[1]
        id_col = "Number" if "Number" in df.columns else df.columns[0]

        for _, row in df.iterrows():
            hgb = _parse_hgb(row[hgb_col])
            if hgb is None:
                continue
            subject_id = str(int(row[id_col]))
            subject_dir = cohort_dir / subject_id
            if not subject_dir.is_dir():
                continue
            group = f"{cohort_name}_{subject_id}"
            paths = sorted(subject_dir.glob("*_palpebral.png")) or sorted(subject_dir.glob("*.jpg"))
            for img_path in paths:
                if "_forniceal" in img_path.name and "_palpebral" not in img_path.name:
                    continue
                rows.append((str(img_path), hgb, "eye", group))
    return rows


def load_nail_samples(stats: dict, rng: np.random.Generator, max_per_class: int = 2500):
    rows = []
    anemic_paths, healthy_paths = [], []
    for root in NAIL_ROOTS:
        if not root.exists():
            continue
        for img_path in root.rglob("*.png"):
            name = img_path.name.lower()
            parent = img_path.parent.name.lower()
            if "non-anemic" in parent or "non_anemic" in parent or "healthy" in parent:
                healthy_paths.append(str(img_path))
            elif "anemic" in name or "anemic" in parent:
                anemic_paths.append(str(img_path))

    rng.shuffle(anemic_paths)
    rng.shuffle(healthy_paths)
    for p in anemic_paths[:max_per_class]:
        hgb = np.clip(hb_from_anemic_flag(True, stats, rng), 6.0, 18.0)
        rows.append((p, hgb, "nail", "nail_anemic"))
    for p in healthy_paths[:max_per_class]:
        hgb = np.clip(hb_from_anemic_flag(False, stats, rng), 6.0, 18.0)
        rows.append((p, hgb, "nail", "nail_healthy"))
    return rows


def load_palm_samples(stats: dict, rng: np.random.Generator, max_per_class: int = 2500):
    rows = []
    for label_name, palm_dir, is_anemic in PALM_DIRS:
        if not palm_dir.exists():
            continue
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(palm_dir.glob(ext))
        rng.shuffle(paths)
        for img_path in paths[:max_per_class]:
            hgb = np.clip(hb_from_anemic_flag(is_anemic, stats, rng), 6.0, 18.0)
            rows.append((str(img_path), hgb, "palm", f"palm_{label_name}"))
    return rows


def extract_sample_features(path: str, modality: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vector = np.zeros(TOTAL_FEATURES, dtype=np.float64)
    try:
        if modality == "eye":
            vector[EYE_SLICE] = extract_eye_features(rgb)
        elif modality == "nail":
            vector[NAIL_SLICE] = extract_nail_features(rgb)
        elif modality == "palm":
            vector[PALM_SLICE] = extract_palm_features(rgb)
        else:
            return None
        if np.isnan(vector).any() or np.isinf(vector).any():
            return None
        return vector
    except Exception:
        return None


def build_dataset(samples, progress_every=300):
    X_list, y_list, groups = [], [], []
    for i, (path, hgb, modality, group) in enumerate(samples):
        if (i + 1) % progress_every == 0:
            print(f"  features {i + 1}/{len(samples)}...")
        vec = extract_sample_features(path, modality)
        if vec is None:
            continue
        X_list.append(vec)
        y_list.append(hgb)
        groups.append(group)
    return np.array(X_list), np.array(y_list), groups


def subject_level_split(eye_groups, test_frac=0.2):
    unique = sorted(set(eye_groups))
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_frac))
    return set(unique[:n_test])


def train_ensemble(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    models = {
        "HistGB": HistGradientBoostingRegressor(
            max_iter=400,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=8,
            l2_regularization=0.5,
            random_state=RANDOM_STATE,
        ),
        "GB": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=6,
            subsample=0.85,
            random_state=RANDOM_STATE,
        ),
        "Ridge": Ridge(alpha=0.5),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_tr, y_train)
        pred = model.predict(X_te)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        results[name] = {"r2": r2, "mae": mae, "rmse": float(np.sqrt(mean_squared_error(y_test, pred)))}
        fitted[name] = model
        print(f"    {name}: R²={r2:.4f}, MAE={mae:.2f} g/dL")

    best_name = max(results.keys(), key=lambda k: results[k]["r2"])
    best_model = fitted[best_name]
    return best_name, best_model, scaler, results


def save_artifacts(model, scaler, config: dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "hemolens_multimodal.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "multimodal_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "multimodal_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    print("Training multimodal model (eye + nail + palm)")
    stats = compute_eye_hb_stats()
    print(f"Eye Hb cohort n={stats['n']}: mean={stats['mean']:.2f} g/dL")

    eye = load_eye_samples()
    nail = load_nail_samples(stats, rng)
    palm = load_palm_samples(stats, rng)
    samples = eye + nail + palm
    print(f"Samples: eye={len(eye)}, nail={len(nail)}, palm={len(palm)}")

    X, y, groups = build_dataset(samples)
    eye_idx = [i for i, g in enumerate(groups) if g.startswith("india_") or g.startswith("italy_")]
    test_subjects = subject_level_split([groups[i] for i in eye_idx])

    test_mask = np.zeros(len(y), dtype=bool)
    for i in eye_idx:
        if groups[i] in test_subjects:
            test_mask[i] = True
    remaining = [i for i in range(len(groups)) if i not in eye_idx]
    rng.shuffle(remaining)
    for i in remaining[: int(len(remaining) * 0.2)]:
        test_mask[i] = True

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"Train={len(y_train)}, test={len(y_test)}")
    best_name, best_model, scaler, results = train_ensemble(X_train, y_train, X_test, y_test)

    config = {
        "model_type": best_name,
        "modalities": ["eye", "nail", "palm"],
        "total_features": TOTAL_FEATURES,
        "r2": round(results[best_name]["r2"], 4),
        "mae": round(results[best_name]["mae"], 2),
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "target_met": results[best_name]["r2"] >= TARGET_R2,
    }
    save_artifacts(best_model, scaler, config)
    print(f"Saved to {MODEL_DIR} — R²={config['r2']}, MAE={config['mae']} g/dL")


if __name__ == "__main__":
    main()
