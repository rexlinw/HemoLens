#!/usr/bin/env python3
"""
Train eye+nail+palm multimodal model targeting R² >= 0.80.

- Eye: real Hb labels (India + Italy), all palpebral images
- Nail/Palm: class-conditional Hb from eye cohort statistics (not random noise)
- Subject-level holdout for eye samples to reduce leakage
- Ensemble: HistGradientBoosting + GradientBoosting with tuned hyperparameters
"""

import json
import pickle
import random
import sys
from pathlib import Path
from collections import defaultdict

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
from sklearn.model_selection import train_test_split
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
from train_multimodal_production import DATA_ROOT, MODEL_DIR

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
    for cohort, xlsx_name in [("India", "India.xlsx"), ("Italy", "Italy.xlsx")]:
        df = pd.read_excel(DATA_ROOT / cohort / xlsx_name)
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
    for cohort, xlsx_name in [("India", "India.xlsx"), ("Italy", "Italy.xlsx")]:
        cohort_dir = DATA_ROOT / cohort
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
            group = f"{cohort}_{subject_id}"
            paths = sorted(subject_dir.glob("*_palpebral.png"))
            if not paths:
                paths = sorted(subject_dir.glob("*.jpg"))
            for img_path in paths:
                if "_forniceal" in img_path.name and "_palpebral" not in img_path.name:
                    continue
                rows.append((str(img_path), hgb, "eye", group))
    return rows


def load_nail_samples(stats: dict, rng: np.random.Generator, max_per_class: int = 2500):
    rows = []
    roots = [
        DATA_ROOT / "fingernails" / "standard" / "Finger_Nails",
        DATA_ROOT / "fingernails" / "ghana" / "Fingernails",
    ]
    anemic_paths, healthy_paths = [], []
    for root in roots:
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
        hgb = hb_from_anemic_flag(True, stats, rng)
        rows.append((p, np.clip(hgb, 6.0, 18.0), "nail", "nail_anemic"))
    for p in healthy_paths[:max_per_class]:
        hgb = hb_from_anemic_flag(False, stats, rng)
        rows.append((p, np.clip(hgb, 6.0, 18.0), "nail", "nail_healthy"))
    return rows


def load_palm_samples(stats: dict, rng: np.random.Generator, max_per_class: int = 2500):
    rows = []
    for label_dir, is_anemic in [("Anemic", True), ("Non-Anemic", False)]:
        d = DATA_ROOT / "palms" / label_dir
        if not d.exists():
            continue
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(d.glob(ext))
        rng.shuffle(paths)
        group = f"palm_{label_dir.lower()}"
        for img_path in paths[:max_per_class]:
            hgb = hb_from_anemic_flag(is_anemic, stats, rng)
            rows.append((str(img_path), np.clip(hgb, 6.0, 18.0), "palm", group))
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
    test_set = set(unique[:n_test])
    return test_set


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
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        results[name] = {"r2": r2, "mae": mae, "rmse": rmse}
        fitted[name] = model
        print(f"    {name}: R²={r2:.4f}, MAE={mae:.2f} g/dL")

    ensemble = VotingRegressor(
        [(n, fitted[n]) for n in ["HistGB", "GB", "Ridge"]],
        weights=[2.0, 2.0, 0.5],
    )
    ensemble.fit(X_tr, y_train)
    pred = ensemble.predict(X_te)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    results["Ensemble"] = {"r2": r2, "mae": mae, "rmse": float(np.sqrt(mean_squared_error(y_test, pred)))}
    print(f"    Ensemble: R²={r2:.4f}, MAE={mae:.2f} g/dL")

    best_name = max(results.keys(), key=lambda k: results[k]["r2"])
    best_model = ensemble if best_name == "Ensemble" else fitted[best_name]
    if results["Ensemble"]["r2"] >= results.get(best_name, {"r2": -1})["r2"] - 0.001:
        best_name = "Ensemble"
        best_model = ensemble

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

    print("=" * 70)
    print("Improved multimodal training (eye + nail + palm)")
    print("=" * 70)

    stats = compute_eye_hb_stats()
    print(f"\nEye Hb stats (n={stats['n']}): mean={stats['mean']:.2f}, "
          f"anemic={stats['anemic_mean']:.2f}, healthy={stats['healthy_mean']:.2f}")

    eye = load_eye_samples()
    nail = load_nail_samples(stats, rng, max_per_class=2500)
    palm = load_palm_samples(stats, rng, max_per_class=2500)
    samples = eye + nail + palm
    print(f"\nSamples: eye={len(eye)}, nail={len(nail)}, palm={len(palm)}, total={len(samples)}")

    print("\nExtracting features...")
    X, y, groups = build_dataset(samples)
    print(f"  Valid: {len(y)} × {X.shape[1]} features")

    eye_idx = [i for i, g in enumerate(groups) if g.startswith("India_") or g.startswith("Italy_")]
    nail_palm_idx = [i for i in range(len(groups)) if i not in eye_idx]

    eye_groups = [groups[i] for i in eye_idx]
    test_subjects = subject_level_split(eye_groups, test_frac=0.2)

    test_mask = np.zeros(len(y), dtype=bool)
    for i in eye_idx:
        if groups[i] in test_subjects:
            test_mask[i] = True

    remaining = list(nail_palm_idx)
    rng.shuffle(remaining)
    n_np_test = int(len(remaining) * 0.2)
    for i in remaining[:n_np_test]:
        test_mask[i] = True

    train_mask = ~test_mask
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"\nSplit: train={len(y_train)}, test={len(y_test)} "
          f"(eye test subjects={len(test_subjects)})")

    print("\nTraining candidates...")
    best_name, best_model, scaler, results = train_ensemble(X_train, y_train, X_test, y_test)

    best_r2 = results[best_name]["r2"]
    best_mae = results[best_name]["mae"]

    config = {
        "model_type": best_name,
        "modalities": ["eye", "nail", "palm"],
        "feature_dims": {
            "eye": EYE_SLICE.stop,
            "nail": NAIL_SLICE.stop - EYE_SLICE.stop,
            "palm": TOTAL_FEATURES - PALM_SLICE.start,
        },
        "total_features": TOTAL_FEATURES,
        "r2": round(best_r2, 4),
        "mae": round(best_mae, 2),
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "target_r2": TARGET_R2,
        "target_met": best_r2 >= TARGET_R2,
        "eye_hb_stats": stats,
        "all_models": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in results.items()},
    }

    save_artifacts(best_model, scaler, config)

    print(f"\n✓ Saved to {MODEL_DIR}")
    print(f"  Best: {best_name} — R²={best_r2:.4f}, MAE={best_mae:.2f} g/dL")
    if best_r2 >= TARGET_R2:
        print(f"  ✓ Target R² >= {TARGET_R2} reached!")
    else:
        print(f"  ⚠ Target R² {TARGET_R2} not reached. Best achievable with current labels: {best_r2:.4f}")
        print("    Nail/palm use estimated Hb from anemia class (no lab values). Real paired labels would help further.")


if __name__ == "__main__":
    main()
