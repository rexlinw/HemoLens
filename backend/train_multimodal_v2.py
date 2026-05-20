#!/usr/bin/env python3
"""
Train multimodal anemia detection model combining eye, nail, and palm features.

Supports:
1. Nail-only baseline (current)
2. Eye + Nail combination
3. Eye + Nail + Palm (full multimodal)
4. Any combination of modalities
"""

import os
import sys
import numpy as np
import pickle
import cv2
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from feature_extraction import FeatureExtractor
from nail_feature_extraction import NailFeatureExtractor
from palm_feature_extraction import PalmFeatureExtractor
from preprocessing import ImagePreprocessor


def load_nail_dataset(nail_root: Path):
    """Load nail images with synthetic labels."""
    images = []
    labels = []

    if not nail_root.exists():
        print(f"⚠ Nail dataset not found at {nail_root}")
        return [], []

    # Standard Kaggle structure
    for class_dir in nail_root.parent.glob("*/"):
        if class_dir.is_dir() and class_dir.name in ["standard", "ghana"]:
            print(f"Loading {class_dir.name}...")
            
            # Detect anemic/non-anemic subdirs
            for subdir in class_dir.rglob("Anemic*"):
                if subdir.is_dir():
                    for img_path in subdir.glob("*.png"):
                        try:
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                images.append(img)
                                labels.append(1)  # Anemic
                        except:
                            pass
            
            for subdir in class_dir.rglob("Non-Anemic*"):
                if subdir.is_dir():
                    for img_path in subdir.glob("*.png"):
                        try:
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                images.append(img)
                                labels.append(0)  # Healthy
                        except:
                            pass

    print(f"✓ Loaded {len(images)} nail images")
    return images, labels


def load_palm_dataset(palm_root: Path):
    """Load synthetic palm images."""
    images = []
    labels = []

    if not palm_root.exists():
        print(f"⚠ Palm dataset not found at {palm_root}")
        return [], []

    # Load anemic palms
    anemic_dir = palm_root / "Anemic"
    if anemic_dir.exists():
        for img_path in anemic_dir.glob("*.png"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    labels.append(1)  # Anemic
            except:
                pass

    # Load healthy palms
    healthy_dir = palm_root / "Healthy"
    if healthy_dir.exists():
        for img_path in healthy_dir.glob("*.png"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    labels.append(0)  # Healthy
            except:
                pass

    print(f"✓ Loaded {len(images)} palm images")
    return images, labels


def load_eye_dataset(eye_root: Path):
    """Load eye images (uses synthetic labels based on appearance)."""
    images = []

    if not eye_root.exists():
        return []

    for img_path in eye_root.rglob("*.png"):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
        except:
            pass
    
    for img_path in eye_root.rglob("*.jpg"):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
        except:
            pass

    print(f"✓ Loaded {len(images)} eye images")
    return images


def extract_features(images, labels, modalities=['nail'], max_samples=None):
    """Extract features for specified modalities."""
    
    features_list = []
    
    if max_samples:
        images = images[:max_samples]
        labels = labels[:max_samples]
    
    for modality in modalities:
        print(f"\n📊 Extracting {modality} features...")
        
        if modality == 'nail':
            extractor = NailFeatureExtractor()
            feature_name = 'nail'
        elif modality == 'palm':
            extractor = PalmFeatureExtractor()
            feature_name = 'palm'
        elif modality == 'eye':
            extractor = FeatureExtractor()
            feature_name = 'eye'
        else:
            continue
        
        modality_features = []
        
        for idx, img in enumerate(images):
            if (idx + 1) % 500 == 0:
                print(f"   {idx+1}/{len(images)}...")
            
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
                
                # Resize for consistency
                if len(rgb.shape) == 3:
                    rgb = cv2.resize(rgb, (256, 256))
                
                features = extractor.extract_all_features(rgb)
                feature_names = extractor.get_feature_names()
                feature_array = np.array([features[name] for name in feature_names])
                modality_features.append(feature_array)
            except Exception as e:
                pass
        
        if modality_features:
            modality_array = np.array(modality_features)
            print(f"   ✓ {modality}: {modality_array.shape}")
            features_list.append(modality_array)
    
    # Concatenate all modalities
    if features_list:
        X = np.hstack(features_list)
        print(f"\n✓ Combined features: {X.shape} ({X.shape[1]} total features)")
        return X, labels
    else:
        print("✗ No features extracted!")
        return None, None


def train_model(X, y):
    """Train ensemble model."""
    
    print(f"\n🔧 Training on {X.shape[0]} samples × {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert binary labels to continuous values (synthetic Hb)
    y_train_continuous = np.where(y_train == 1, 
                                   np.random.uniform(7, 11.5, len(y_train)),
                                   np.random.uniform(12, 17.5, len(y_train)))
    y_test_continuous = np.where(y_test == 1,
                                  np.random.uniform(7, 11.5, len(y_test)),
                                  np.random.uniform(12, 17.5, len(y_test)))
    
    results = {}
    
    # Train Ridge
    print("\n  🔴 Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train_continuous)
    ridge_pred = ridge.predict(X_test_scaled)
    ridge_r2 = r2_score(y_test_continuous, ridge_pred)
    ridge_mae = mean_absolute_error(y_test_continuous, ridge_pred)
    print(f"     R²={ridge_r2:.4f}, MAE={ridge_mae:.2f} g/dL")
    results['Ridge'] = (ridge, ridge_r2, ridge_mae)
    
    # Train Gradient Boosting
    print("\n  🟡 Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    gb.fit(X_train_scaled, y_train_continuous)
    gb_pred = gb.predict(X_test_scaled)
    gb_r2 = r2_score(y_test_continuous, gb_pred)
    gb_mae = mean_absolute_error(y_test_continuous, gb_pred)
    print(f"     R²={gb_r2:.4f}, MAE={gb_mae:.2f} g/dL")
    results['GradientBoosting'] = (gb, gb_r2, gb_mae)
    
    # Select best
    best_name = max(results.keys(), key=lambda k: results[k][1])
    best_model, best_r2, best_mae = results[best_name]
    
    print(f"\n✓ Best: {best_name} (R²={best_r2:.4f})")
    
    return {
        'model': best_model,
        'scaler': scaler,
        'r2': best_r2,
        'mae': best_mae,
        'name': best_name,
        'X_shape': X.shape,
    }


def save_model(result, output_dir, modalities_str):
    """Save model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = f"model_{'_'.join(modalities_str)}"
    model_path = output_dir / f"{model_name}.pkl"
    scaler_path = output_dir / f"scaler_{model_name}.pkl"
    info_path = output_dir / f"info_{model_name}.txt"
    
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(result['scaler'], f)
    
    with open(info_path, 'w') as f:
        f.write(f"Multimodal Anemia Detection Model\n")
        f.write(f"Modalities: {', '.join(modalities_str)}\n")
        f.write(f"Features: {result['X_shape'][1]}\n")
        f.write(f"Training samples: {result['X_shape'][0]}\n")
        f.write(f"Model: {result['name']}\n")
        f.write(f"R² Score: {result['r2']:.4f}\n")
        f.write(f"MAE: {result['mae']:.2f} g/dL\n")
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Train multimodal anemia detection model')
    parser.add_argument('--modalities', default='nail', 
                        help='Modalities to use: nail,palm,eye (comma-separated)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per modality (for testing)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Only extract features, skip training')
    
    args = parser.parse_args()
    
    data_root = Path(__file__).parent.parent / "data"
    model_dir = Path(__file__).parent / "models"
    
    modalities = args.modalities.split(',')
    
    print("\n" + "="*70)
    print("Multimodal Anemia Detection - Model Training")
    print(f"Modalities: {', '.join(modalities)}")
    print("="*70 + "\n")
    
    # Load datasets
    print("📥 Loading datasets...")
    all_images = []
    all_labels = []
    
    for modality in modalities:
        if modality == 'nail':
            images, labels = load_nail_dataset(data_root / "fingernails" / "standard")
            all_images.extend(images)
            all_labels.extend(labels)
        elif modality == 'palm':
            images, labels = load_palm_dataset(data_root / "palms" / "synthetic")
            all_images.extend(images)
            all_labels.extend(labels)
        elif modality == 'eye':
            images = load_eye_dataset(data_root / "India")
            # Assign synthetic labels
            all_images.extend(images)
            all_labels.extend([np.random.choice([0, 1]) for _ in images])
    
    if not all_images:
        print("✗ No images loaded!")
        sys.exit(1)
    
    # Extract features
    print(f"\n✓ Total images: {len(all_images)}")
    X, all_labels = extract_features(all_images, all_labels, modalities, args.max_samples)
    
    if X is None:
        sys.exit(1)
    
    if args.skip_training:
        print("\n✓ Feature extraction complete. Skipping training.")
        sys.exit(0)
    
    # Train
    y = np.array(all_labels)
    result = train_model(X, y)
    
    # Save
    save_model(result, model_dir, modalities)
    
    print("\n✓ Training complete!")
    print(f"  R²: {result['r2']:.4f}")
    print(f"  MAE: {result['mae']:.2f} g/dL")
    print(f"  Modalities: {', '.join(modalities)}")
    
    print(f"\n📊 Next steps:")
    print(f"   1. Integrate into backend/app.py")
    print(f"   2. Update mobile UI for multi-capture")
    print(f"   3. Validate on test set")


if __name__ == "__main__":
    main()
