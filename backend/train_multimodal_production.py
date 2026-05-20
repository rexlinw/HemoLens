#!/usr/bin/env python3
"""
Train production multimodal anemia model (eye + nail + palm).

Each training row uses one modality's features with zeros for the others,
so inference can combine any subset of eye/nail/palm images at runtime.
"""

import json
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from multimodal import (
    EYE_COUNT,
    NAIL_COUNT,
    PALM_COUNT,
    TOTAL_FEATURES,
    EYE_SLICE,
    NAIL_SLICE,
    PALM_SLICE,
    MODALITIES,
    extract_eye_features,
    extract_nail_features,
    extract_palm_features,
)

DATA_ROOT = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

ANEMIC_HB = (7.0, 11.5)
HEALTHY_HB = (12.0, 17.5)


def _synthetic_hb(is_anemic: bool) -> float:
    lo, hi = ANEMIC_HB if is_anemic else HEALTHY_HB
    return float(np.random.uniform(lo, hi))


def load_eye_labeled(max_samples: int | None = None):
    rows = []
    for cohort, xlsx_name in [("India", "India.xlsx"), ("Italy", "Italy.xlsx")]:
        cohort_dir = DATA_ROOT / cohort
        xlsx_path = cohort_dir / xlsx_name
        if not xlsx_path.exists():
            continue
        df = pd.read_excel(xlsx_path)
        hgb_col = "Hgb" if "Hgb" in df.columns else df.columns[1]
        id_col = "Number" if "Number" in df.columns else df.columns[0]

        for _, row in df.iterrows():
            subject_id = str(int(row[id_col]))
            raw_hgb = row[hgb_col]
            if pd.isna(raw_hgb):
                continue
            hgb_str = str(raw_hgb).strip().replace(",", ".")
            try:
                hgb = float(hgb_str)
            except ValueError:
                continue
            if np.isnan(hgb):
                continue
            subject_dir = cohort_dir / subject_id
            if not subject_dir.is_dir():
                continue
            for pattern in ("*_palpebral.png", "*.jpg"):
                for img_path in subject_dir.glob(pattern):
                    if "_forniceal" in img_path.name and "_palpebral" not in img_path.name:
                        continue
                    rows.append((str(img_path), hgb, "eye"))
                    break
            if max_samples and len(rows) >= max_samples:
                return rows
    return rows


def load_nail_labeled(max_per_class: int = 2000):
    rows = []
    roots = [
        DATA_ROOT / "fingernails" / "standard" / "Finger_Nails",
        DATA_ROOT / "fingernails" / "ghana" / "Fingernails",
    ]
    for root in roots:
        if not root.exists():
            continue
        for img_path in root.rglob("*.png"):
            name = img_path.name.lower()
            parent = img_path.parent.name.lower()
            is_anemic = (
                "anemic" in name
                or "anemic" in parent
                or name.startswith("anemic")
            )
            if "non-anemic" in parent or "non_anemic" in parent or "healthy" in parent:
                is_anemic = False
            rows.append((str(img_path), is_anemic, "nail"))

    anemic = [r for r in rows if r[1] is True]
    healthy = [r for r in rows if r[1] is False]
    random.shuffle(anemic)
    random.shuffle(healthy)
    selected = anemic[:max_per_class] + healthy[:max_per_class]
    return [(p, _synthetic_hb(a), "nail") for p, a, _ in selected]


def load_palm_labeled(max_per_class: int = 1500):
    rows = []
    palm_root = DATA_ROOT / "palms"
    for label_dir, is_anemic in [("Anemic", True), ("Non-Anemic", False)]:
        d = palm_root / label_dir
        if not d.exists():
            continue
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(d.glob(ext))
        random.shuffle(paths)
        for img_path in paths[:max_per_class]:
            rows.append((str(img_path), _synthetic_hb(is_anemic), "palm"))
    return rows


def build_training_matrix(samples, progress_every=200):
    X = np.zeros((len(samples), TOTAL_FEATURES), dtype=np.float64)
    y = np.zeros(len(samples), dtype=np.float64)

    for i, (path, hgb, modality) in enumerate(samples):
        if (i + 1) % progress_every == 0:
            print(f"  features {i + 1}/{len(samples)}...")
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            if modality == "eye":
                X[i, EYE_SLICE] = extract_eye_features(rgb)
            elif modality == "nail":
                X[i, NAIL_SLICE] = extract_nail_features(rgb)
            elif modality == "palm":
                X[i, PALM_SLICE] = extract_palm_features(rgb)
            y[i] = hgb
        except Exception:
            X[i, :] = np.nan

    valid = ~np.isnan(X).any(axis=1)
    return X[valid], y[valid]


def train_and_save(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=120, max_depth=5, random_state=42
        ),
    }
    best_name, best_model, best_r2, best_mae = None, None, -1.0, 999.0

    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        print(f"  {name}: R²={r2:.4f}, MAE={mae:.2f} g/dL")
        if r2 > best_r2:
            best_name, best_model, best_r2, best_mae = name, model, r2, mae

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "hemolens_multimodal.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "multimodal_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    config = {
        "model_type": best_name,
        "modalities": list(MODALITIES),
        "feature_dims": {"eye": EYE_COUNT, "nail": NAIL_COUNT, "palm": PALM_COUNT},
        "total_features": TOTAL_FEATURES,
        "r2": round(best_r2, 4),
        "mae": round(best_mae, 2),
        "training_samples": int(len(X)),
    }
    with open(MODEL_DIR / "multimodal_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Saved hemolens_multimodal.pkl ({best_name})")
    print(f"  R²={best_r2:.4f}, MAE={best_mae:.2f}, n={len(X)}")
    return config


def main():
    random.seed(42)
    np.random.seed(42)

    print("Loading datasets...")
    eye = load_eye_labeled()
    nail = load_nail_labeled(max_per_class=1500)
    palm = load_palm_labeled(max_per_class=1500)
    samples = eye + nail + palm
    print(f"  eye={len(eye)}, nail={len(nail)}, palm={len(palm)}, total={len(samples)}")

    if len(samples) < 50:
        print("✗ Not enough training samples. Check data/ folder.")
        sys.exit(1)

    print("\nExtracting features (zero-padded multimodal vectors)...")
    X, y = build_training_matrix(samples)

    print(f"\nTraining on {X.shape[0]} samples × {X.shape[1]} features...")
    train_and_save(X, y)


if __name__ == "__main__":
    main()
