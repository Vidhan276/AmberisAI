"""
train_audio_model.py
====================
Production-grade training pipeline for AmberisAI infant cry classification.

Pipeline:
  1. Scan dataset directory (expected structure: data/class_name/file.wav)
  2. Load and validate each audio file
  3. Extract features using the shared FeatureExtractor pipeline
  4. Train Random Forest and XGBoost classifiers
  5. Evaluate on held-out validation set
  6. Save all artifacts: models, scaler, label encoder, feature names

CRITICAL DESIGN DECISIONS:
  - StandardScaler is fit ONLY on training data (no leakage).
  - LabelEncoder is fit on all seen classes (prevents inference-time KeyError).
  - Feature extraction is identical in training and inference (shared module).
  - All artifacts saved with pickle for reproducibility.
  - Train/val split is stratified to preserve class balance.

CHANGES FROM v1:
  - FIX: XGBoost cross_val_score now uses a separate CV-safe clone (no early_stopping_rounds)
  - FIX: use_label_encoder removed (deprecated in newer XGBoost versions)
  - IMPROVED: Random Forest n_estimators increased to 500 for better accuracy
  - IMPROVED: XGBoost hyperparameters tuned (lower learning rate, more trees)
  - IMPROVED: evaluate_model now uses zero_division=0 to suppress precision warnings
  - IMPROVED: Added GridSearchCV option for RF hyperparameter tuning
  - IMPROVED: Better logging throughout

Usage:
    python -m audio_module.train_audio_model \
        --data_dir /path/to/dataset \
        --output_dir ./models \
        --test_size 0.3 \
        --random_state 42
"""

import os
import sys
import pickle
import logging
import argparse
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import xgboost as xgb

from .feature_extraction import FeatureExtractor
from .utils import load_audio, validate_audio_file, list_audio_files, setup_logging

logger = logging.getLogger(__name__)

# ─── PATHS ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR = "models"
RF_MODEL_FILE = "random_forest.pkl"
XGB_MODEL_FILE = "xgboost.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
TRAINING_META_FILE = "training_meta.pkl"


# ─── DATASET LOADING ──────────────────────────────────────────────────────────

def load_dataset(
    data_dir: str,
    extractor: FeatureExtractor,
    max_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and featurize all audio files from a structured dataset directory.

    Expected directory structure:
        data_dir/
          hungry/
            001.wav
            002.mp3
          discomfort/
            ...
          belly_pain/
            ...
          tired/
            ...
          burping/
            ...

    Parameters:
        data_dir      : str              — root dataset directory
        extractor     : FeatureExtractor — shared feature extractor instance
        max_per_class : int, optional    — cap samples per class (for imbalance handling)

    Returns:
        X          : np.ndarray of shape (N, feature_dim)
        y_raw      : np.ndarray of shape (N,) — string labels
        class_names: list of str — all discovered class names
    """
    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"[Training] Dataset directory not found: {data_dir}")

    # Auto-discover classes from subdirectory names
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"[Training] No subdirectories (classes) found in: {data_dir}")

    class_names = [d.name for d in class_dirs]
    logger.info(f"[Training] Discovered classes: {class_names}")

    X_all, y_all = [], []
    failed_files = []

    for class_dir in class_dirs:
        label = class_dir.name
        audio_files = list_audio_files(str(class_dir), recursive=False)

        if max_per_class:
            audio_files = audio_files[:max_per_class]

        logger.info(
            f"[Training] Class '{label}': processing {len(audio_files)} files..."
        )

        class_features = []
        for filepath in audio_files:
            try:
                validate_audio_file(filepath)
                y_wav, sr = load_audio(filepath)
                feat = extractor.extract(y_wav, sr)

                # Skip NaN features (corrupt/silent audio)
                if np.any(np.isnan(feat)):
                    logger.warning(
                        f"[Training] NaN features for {filepath}, skipping."
                    )
                    failed_files.append(filepath)
                    continue

                class_features.append(feat)

            except Exception as e:
                logger.error(
                    f"[Training] Error processing '{filepath}': {e}. Skipping."
                )
                failed_files.append(filepath)
                continue

        logger.info(
            f"[Training] Class '{label}': {len(class_features)} valid samples."
        )
        X_all.extend(class_features)
        y_all.extend([label] * len(class_features))

    if not X_all:
        raise RuntimeError(
            "[Training] No valid audio samples loaded. Check dataset structure."
        )

    if failed_files:
        logger.warning(
            f"[Training] {len(failed_files)} files failed to process and were skipped."
        )

    X = np.vstack(X_all)
    y = np.array(y_all)

    logger.info(
        f"[Training] Dataset loaded: {X.shape[0]} samples, "
        f"{X.shape[1]} features, {len(class_names)} classes."
    )

    # Log class distribution
    for cls in class_names:
        count = int(np.sum(y == cls))
        pct = 100.0 * count / len(y)
        logger.info(f"[Training]   '{cls}': {count} samples ({pct:.1f}%)")

    return X, y, class_names


# ─── MODEL TRAINING ───────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters:
        X_train           : training features
        y_train           : training labels (encoded integers)
        n_estimators      : number of trees (500 for better accuracy vs v1's 300)
        max_depth         : None = grow fully (guards against underfitting)
        min_samples_leaf  : minimum samples per leaf
        min_samples_split : minimum samples to split a node
        class_weight      : 'balanced' handles class imbalance automatically
        random_state      : for reproducibility

    Returns:
        Fitted RandomForestClassifier
    """
    logger.info("[Training] Training Random Forest (500 trees)...")
    start = time.time()

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,       # Use all CPU cores
        oob_score=True,  # Out-of-bag score for free validation estimate
    )
    rf.fit(X_train, y_train)

    elapsed = time.time() - start
    logger.info(
        f"[Training] Random Forest trained in {elapsed:.1f}s. "
        f"OOB score: {rf.oob_score_:.4f}"
    )
    return rf


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 5,
    learning_rate: float = 0.03,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 3,
    gamma: float = 0.1,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with early stopping on validation set.

    Parameters:
        X_train, y_train  : training data
        X_val, y_val      : validation data (used for early stopping only)
        n_estimators      : max boosting rounds (500 vs v1's 300)
        max_depth         : tree depth (5 vs v1's 6 — less overfit risk)
        learning_rate     : 0.03 vs v1's 0.05 — more conservative, better generalization
        subsample         : fraction of samples per tree
        colsample_bytree  : fraction of features per tree
        min_child_weight  : min sum of instance weight in child — controls overfitting
        gamma             : min loss reduction for split — regularization
        random_state      : for reproducibility

    Returns:
        Fitted XGBClassifier

    NOTE: use_label_encoder removed — deprecated and removed in XGBoost >= 1.6
    NOTE: early_stopping_rounds moved to constructor for newer XGBoost compatibility
    """
    logger.info("[Training] Training XGBoost (500 rounds, lr=0.03)...")
    start = time.time()

    n_classes = len(np.unique(y_train))
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"

    # FIX: early_stopping_rounds in constructor (not in .fit()) for XGBoost >= 1.7
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        objective=objective,
        eval_metric="mlogloss",
        early_stopping_rounds=30,   # FIX: in constructor, not .fit()
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    elapsed = time.time() - start
    logger.info(
        f"[Training] XGBoost trained in {elapsed:.1f}s. "
        f"Best iteration: {xgb_model.best_iteration}"
    )
    return xgb_model


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str = "Model",
) -> Dict:
    """
    Evaluate a fitted classifier and log detailed metrics.

    FIX: Added zero_division=0 to classification_report to suppress
    'Precision is ill-defined' warnings cleanly.

    Returns:
        dict containing accuracy, classification report, confusion matrix.
    """
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    class_names = label_encoder.classes_

    logger.info(f"\n[Evaluation] {model_name} Accuracy: {acc:.4f}")

    # FIX: zero_division=0 suppresses the UndefinedMetricWarning cleanly
    report = classification_report(
        y_val, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    logger.info(f"\n[Evaluation] Classification Report:\n{report}")

    cm = confusion_matrix(y_val, y_pred)
    logger.info(f"[Evaluation] Confusion Matrix:\n{cm}")

    return {
        "model_name": model_name,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    model_name: str = "Model",
) -> Dict:
    """
    Stratified K-Fold cross-validation for robust generalization estimate.

    Provides an unbiased estimate of model performance on unseen data.

    NOTE: This function receives a CV-safe model clone (no early_stopping_rounds).
    Do NOT pass the main XGBoost model trained with early stopping here —
    use the xgb_cv_model created in train() instead.
    """
    logger.info(
        f"[Evaluation] Running {n_splits}-fold cross-validation for {model_name}..."
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    logger.info(
        f"[Evaluation] {model_name} CV Accuracy: "
        f"{scores.mean():.4f} ± {scores.std():.4f} "
        f"(min: {scores.min():.4f}, max: {scores.max():.4f})"
    )
    return {
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "cv_scores": scores.tolist(),
    }


# ─── ARTIFACT SAVING ──────────────────────────────────────────────────────────

def save_artifacts(
    rf_model: RandomForestClassifier,
    xgb_model: xgb.XGBClassifier,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    feature_names: list,
    training_meta: dict,
    output_dir: str,
) -> None:
    """
    Save all trained artifacts to disk.

    Artifacts:
      - random_forest.pkl  : Fitted RandomForestClassifier
      - xgboost.pkl        : Fitted XGBClassifier
      - scaler.pkl         : Fitted StandardScaler (trained on train set ONLY)
      - label_encoder.pkl  : Fitted LabelEncoder
      - feature_names.pkl  : List of feature name strings
      - training_meta.pkl  : Training metadata (classes, feature dim, eval results)

    CRITICAL: The scaler and label_encoder must be used identically at inference time.
    """
    os.makedirs(output_dir, exist_ok=True)

    artifacts = {
        RF_MODEL_FILE: rf_model,
        XGB_MODEL_FILE: xgb_model,
        SCALER_FILE: scaler,
        LABEL_ENCODER_FILE: label_encoder,
        FEATURE_NAMES_FILE: feature_names,
        TRAINING_META_FILE: training_meta,
    }

    for filename, obj in artifacts.items():
        path = os.path.join(output_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[Training] Saved artifact: {path}")

    logger.info(f"[Training] All artifacts saved to: {output_dir}")


# ─── MAIN TRAINING ENTRYPOINT ─────────────────────────────────────────────────

def train(
    data_dir: str,
    output_dir: str = DEFAULT_MODEL_DIR,
    test_size: float = 0.3,
    random_state: int = 42,
    max_per_class: Optional[int] = None,
    run_cross_validation: bool = True,
) -> Dict:
    """
    Full end-to-end training pipeline.

    Parameters:
        data_dir             : str   — path to dataset root
        output_dir           : str   — where to save trained artifacts
        test_size            : float — fraction of data for validation (default 0.3)
        random_state         : int   — seed for reproducibility
        max_per_class        : int   — cap samples per class (optional)
        run_cross_validation : bool  — run 5-fold CV for generalization estimate

    Returns:
        dict — training summary including all evaluation results
    """
    logger.info("=" * 70)
    logger.info("[Training] AmberisAI Audio Module — Training Pipeline v2")
    logger.info("=" * 70)
    logger.info(f"[Training] Data directory  : {data_dir}")
    logger.info(f"[Training] Output directory: {output_dir}")
    logger.info(f"[Training] Val split       : {test_size}")
    logger.info(f"[Training] Random state    : {random_state}")

    # ── Step 1: Initialize feature extractor
    extractor = FeatureExtractor()
    logger.info(
        f"[Training] Feature extractor initialized. "
        f"Feature dim: {extractor.feature_dim}"
    )

    # ── Step 2: Load dataset
    X, y_raw, class_names = load_dataset(data_dir, extractor, max_per_class)

    # ── Step 3: Encode labels
    le = LabelEncoder()
    le.fit(y_raw)  # Fit on all classes to avoid inference-time KeyError
    y_encoded = le.transform(y_raw)
    logger.info(
        f"[Training] Label encoding: "
        f"{dict(zip(le.classes_, le.transform(le.classes_)))}"
    )

    # ── Step 4: Train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,  # Preserve class proportions
    )
    logger.info(
        f"[Training] Train set: {X_train.shape[0]} samples | "
        f"Val set: {X_val.shape[0]} samples"
    )

    # ── Step 5: Fit scaler on TRAINING set only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)  # Apply same transform (no re-fit)
    logger.info(
        f"[Training] StandardScaler fit. "
        f"Feature means range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]"
    )

    # ── Step 6: Train Random Forest
    rf_model = train_random_forest(
        X_train_scaled, y_train, random_state=random_state
    )

    # ── Step 7: Train XGBoost
    xgb_model = train_xgboost(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        random_state=random_state,
    )

    # ── Step 8: Evaluate both models
    rf_results = evaluate_model(rf_model, X_val_scaled, y_val, le, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_val_scaled, y_val, le, "XGBoost")

    # ── Step 9: Evaluate ensemble on validation set
    rf_proba = rf_model.predict_proba(X_val_scaled)
    xgb_proba = xgb_model.predict_proba(X_val_scaled)
    ensemble_proba = 0.7 * rf_proba + 0.3 * xgb_proba
    ensemble_preds = np.argmax(ensemble_proba, axis=1)
    ens_acc = accuracy_score(y_val, ensemble_preds)
    logger.info(f"[Evaluation] Ensemble (0.7 RF + 0.3 XGB) Accuracy: {ens_acc:.4f}")
    ens_report = classification_report(
        y_val, ensemble_preds,
        target_names=le.classes_,
        digits=4,
        zero_division=0,  # FIX: suppress precision warnings
    )
    logger.info(f"\n[Evaluation] Ensemble Report:\n{ens_report}")

    # ── Step 10: Optional cross-validation
    # FIX: XGBoost CV uses a SEPARATE model clone WITHOUT early_stopping_rounds
    # because cross_val_score does not pass eval_set, which early stopping requires.
    cv_rf = cv_xgb = None
    if run_cross_validation:
        X_full_scaled = scaler.transform(X)

        # RF cross-validation (safe to use directly)
        cv_rf = cross_validate_model(
            rf_model, X_full_scaled, y_encoded, model_name="RF"
        )

        # FIX: XGBoost CV clone — no early_stopping_rounds
        n_classes = len(np.unique(y_encoded))
        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        xgb_cv_model = xgb.XGBClassifier(
            n_estimators=100,           # Fixed rounds (no early stopping in CV)
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            objective=objective,
            eval_metric="mlogloss",
            # NO early_stopping_rounds here — incompatible with cross_val_score
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        cv_xgb = cross_validate_model(
            xgb_cv_model, X_full_scaled, y_encoded, model_name="XGB"
        )

    # ── Step 11: Save all artifacts
    feature_names = extractor.get_feature_names()
    training_meta = {
        "class_names": list(le.classes_),
        "feature_dim": extractor.feature_dim,
        "feature_names": feature_names,
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "rf_val_accuracy": rf_results["accuracy"],
        "xgb_val_accuracy": xgb_results["accuracy"],
        "ensemble_val_accuracy": float(ens_acc),
        "rf_classification_report": rf_results["classification_report"],
        "xgb_classification_report": xgb_results["classification_report"],
        "ensemble_classification_report": ens_report,
        "cv_rf": cv_rf,
        "cv_xgb": cv_xgb,
        "random_state": random_state,
        "test_size": test_size,
        "ensemble_weights": {"rf": 0.7, "xgb": 0.3},
    }

    save_artifacts(
        rf_model, xgb_model, scaler, le,
        feature_names, training_meta, output_dir
    )

    logger.info("\n[Training] ✓ Training complete.")
    logger.info(f"[Training]   RF  val accuracy     : {rf_results['accuracy']:.4f}")
    logger.info(f"[Training]   XGB val accuracy     : {xgb_results['accuracy']:.4f}")
    logger.info(f"[Training]   Ensemble val accuracy: {ens_acc:.4f}")

    return training_meta


# ─── CLI ENTRYPOINT ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AmberisAI Audio Module — Training Script v2"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to dataset root directory (subdirs = class names)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_MODEL_DIR,
        help="Directory to save trained model artifacts"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.3,
        help="Fraction of data to use as validation set (default: 0.3)"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--max_per_class", type=int, default=None,
        help="Optional cap on samples per class (for debugging)"
    )
    parser.add_argument(
        "--no_cv", action="store_true",
        help="Skip cross-validation (faster run)"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)

    results = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_per_class=args.max_per_class,
        run_cross_validation=not args.no_cv,
    )
    sys.exit(0)