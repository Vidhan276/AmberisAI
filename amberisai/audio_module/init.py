"""
AmberisAI - Audio Analysis Module
==================================
Production-grade, research-oriented infant cry classification system.
Classifies infant cries into: hungry, discomfort, belly_pain, tired, burping

Author: AmberisAI Research Team
"""

from .feature_extraction import FeatureExtractor
from .audio_predictor import AudioPredictor
from .utils import load_audio, validate_audio_file

__version__ = "1.0.0"
__all__ = ["FeatureExtractor", "AudioPredictor", "load_audio", "validate_audio_file"]