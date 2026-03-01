"""
audio_predictor.py
Production inference engine for AmberisAI infant cry classification.
Render-safe version with absolute model path resolution.
"""

import os
import pickle
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from .feature_extraction import FeatureExtractor
from .utils import load_audio_with_checks

logger = logging.getLogger(__name__)

# ─── ARTIFACT FILENAMES ───────────────────────────────────────────────────────
RF_MODEL_FILE = "random_forest.pkl"
XGB_MODEL_FILE = "xgboost.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
TRAINING_META_FILE = "training_meta.pkl"

RF_WEIGHT = 0.7
XGB_WEIGHT = 0.3
LOW_CONFIDENCE_THRESHOLD = 0.40


class AudioPredictor:

    _instance: Optional["AudioPredictor"] = None
    _lock = threading.Lock()

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._rf_model = None
        self._xgb_model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = None
        self._training_meta = None
        self._extractor = FeatureExtractor()
        self._loaded = False
        self._load_models()

    # ───────────────────────────────────────────────────────────────────────────
    # SINGLETON FACTORY (FIXED FOR RENDER)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, model_dir: str = None) -> "AudioPredictor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if model_dir is None:
                        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                        model_dir = os.path.join(BASE_DIR, "models")

                    logger.info(f"[AudioPredictor] Using model dir: {model_dir}")
                    cls._instance = cls(model_dir)
        return cls._instance

    # ───────────────────────────────────────────────────────────────────────────
    # MODEL LOADING
    # ───────────────────────────────────────────────────────────────────────────
    def _load_models(self) -> None:

        model_dir = Path(self._model_dir)

        logger.info(f"[AudioPredictor] Current working dir: {os.getcwd()}")
        logger.info(f"[AudioPredictor] Resolving models from: {model_dir}")

        if not model_dir.exists():
            raise FileNotFoundError(
                f"[AudioPredictor] Model directory not found: {model_dir}"
            )

        required_files = {
            "rf_model": RF_MODEL_FILE,
            "xgb_model": XGB_MODEL_FILE,
            "scaler": SCALER_FILE,
            "label_encoder": LABEL_ENCODER_FILE,
        }

        loaded = {}

        for key, filename in required_files.items():
            path = model_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"[AudioPredictor] Required artifact missing: {path}"
                )

            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)

            logger.info(f"[AudioPredictor] Loaded: {filename}")

        self._rf_model = loaded["rf_model"]
        self._xgb_model = loaded["xgb_model"]
        self._scaler = loaded["scaler"]
        self._label_encoder = loaded["label_encoder"]

        # Optional files
        for attr, filename in [
            ("_feature_names", FEATURE_NAMES_FILE),
            ("_training_meta", TRAINING_META_FILE),
        ]:
            path = model_dir / filename
            if path.exists():
                with open(path, "rb") as f:
                    setattr(self, attr, pickle.load(f))

        self._loaded = True

        logger.info(
            f"[AudioPredictor] ✓ Models loaded successfully | "
            f"Classes: {list(self._label_encoder.classes_)}"
        )

    # ───────────────────────────────────────────────────────────────────────────
    # VALIDATION
    # ───────────────────────────────────────────────────────────────────────────
    def _validate_feature_vector(self, features: np.ndarray) -> None:
        if np.any(np.isnan(features)):
            raise ValueError("Feature vector contains NaN values.")
        if np.any(np.isinf(features)):
            raise ValueError("Feature vector contains Inf values.")

    # ───────────────────────────────────────────────────────────────────────────
    # ENSEMBLE
    # ───────────────────────────────────────────────────────────────────────────
    def _run_ensemble(self, features_scaled: np.ndarray) -> Dict[str, float]:

        rf_proba = self._rf_model.predict_proba(features_scaled)[0]
        xgb_proba = self._xgb_model.predict_proba(features_scaled)[0]

        ensemble_proba = RF_WEIGHT * rf_proba + XGB_WEIGHT * xgb_proba
        ensemble_proba /= ensemble_proba.sum()

        classes = self._label_encoder.classes_

        return {
            cls: float(p)
            for cls, p in zip(classes, ensemble_proba)
        }

    # ───────────────────────────────────────────────────────────────────────────
    # PREDICTION
    # ───────────────────────────────────────────────────────────────────────────
    def predict_from_array(self, y: np.ndarray, sr: int, audio_meta=None):

        if not self._loaded:
            raise RuntimeError("Models not loaded.")

        features = self._extractor.extract(y, sr)
        self._validate_feature_vector(features)

        features_scaled = self._scaler.transform(features.reshape(1, -1))
        proba_dict = self._run_ensemble(features_scaled)

        sorted_probs = sorted(
            proba_dict.items(), key=lambda x: x[1], reverse=True
        )

        detected_condition = sorted_probs[0][0]
        confidence = sorted_probs[0][1]
        secondary_condition = sorted_probs[1][0]

        result = {
            "module": "audio",
            "detected_condition": detected_condition,
            "confidence": round(confidence, 4),
            "secondary_condition": secondary_condition,
            "all_probabilities": {
                k: round(v, 4) for k, v in proba_dict.items()
            },
            "low_confidence_warning": confidence < LOW_CONFIDENCE_THRESHOLD,
            "meta": {
                "model": "ensemble_rf_xgb",
                "feature_dim": int(self._extractor.feature_dim),
                "audio_duration_seconds": round(len(y) / sr, 3),
            },
        }

        return result

    def predict(self, filepath: str):
        logger.info(f"[AudioPredictor] Predicting file: {filepath}")

        y, sr, audio_meta = load_audio_with_checks(filepath)
        return self.predict_from_array(y, sr, audio_meta=audio_meta)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
