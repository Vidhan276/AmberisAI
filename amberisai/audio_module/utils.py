"""
utils.py
========
Utility functions for the AmberisAI audio module.

Responsibilities:
  - Load audio files (.wav and .mp3) robustly
  - Validate audio files before processing
  - Normalize and preprocess raw waveforms
  - Consistent error handling and logging

Design notes:
  - All loading functions return (y: np.ndarray, sr: int) to match librosa convention.
  - Validation is strict: empty files, corrupt files, and unsupported formats all raise.
  - Real-world noise is handled at the feature extraction layer (librosa internals).
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import librosa

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_EXTENSIONS = {".wav", ".mp3"}

# Audio quality thresholds
MIN_DURATION_SECONDS = 0.5   # Clips shorter than 0.5s are likely artifacts
MAX_DURATION_SECONDS = 30.0  # Clips longer than 30s are unusual for infant cries
MIN_SAMPLE_RATE = 8000        # Below this is likely degraded quality
MAX_AMPLITUDE_THRESHOLD = 1e-5  # Below this RMS → likely silent/empty


def validate_audio_file(filepath: str) -> None:
    """
    Validate an audio file before attempting to load it.

    Checks:
      - File exists on disk
      - File is non-empty (> 0 bytes)
      - Extension is .wav or .mp3

    Parameters:
        filepath : str — path to the audio file

    Raises:
        FileNotFoundError  : file does not exist
        ValueError         : file is empty or unsupported format
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"[utils] Audio file not found: {filepath}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"[utils] Audio file is empty (0 bytes): {filepath}"
        )

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"[utils] Unsupported file format '{ext}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    logger.debug(f"[utils] File validation passed: {filepath}")


def load_audio(
    filepath: str,
    target_sr: int = 22050,
    mono: bool = True,
    validate: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file (.wav or .mp3) as a numpy array.

    Parameters:
        filepath  : str  — path to audio file
        target_sr : int  — target sample rate (resamples if needed)
        mono      : bool — convert to mono if True (required for feature extraction)
        validate  : bool — run file validation before loading

    Returns:
        Tuple[np.ndarray, int] — (waveform float32 array, sample rate)

    Raises:
        FileNotFoundError : if file doesn't exist
        ValueError        : if file is invalid or corrupt
        RuntimeError      : if librosa fails to load the file
    """
    if validate:
        validate_audio_file(filepath)

    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=mono)
        logger.info(
            f"[utils] Loaded '{Path(filepath).name}' | "
            f"SR: {sr} Hz | Duration: {len(y)/sr:.2f}s | Shape: {y.shape}"
        )
    except Exception as e:
        raise RuntimeError(
            f"[utils] Failed to load audio file '{filepath}': {e}"
        ) from e

    if len(y) == 0:
        raise ValueError(
            f"[utils] Loaded audio is empty: {filepath}"
        )

    return y, sr


def load_audio_with_checks(
    filepath: str,
    target_sr: int = 22050,
    warn_on_quality: bool = True,
) -> Tuple[np.ndarray, int, dict]:
    """
    Load audio with extended quality checks.

    Returns additional metadata about audio quality for logging/debugging.

    Returns:
        Tuple of:
            y         : np.ndarray — waveform
            sr        : int        — sample rate
            meta      : dict       — quality metadata (duration, rms, clipping, etc.)
    """
    y, sr = load_audio(filepath, target_sr=target_sr)

    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    clipping_ratio = float(np.mean(np.abs(y) > 0.99))  # Fraction of clipped samples

    meta = {
        "filepath": filepath,
        "duration_seconds": round(duration, 3),
        "sample_rate": sr,
        "rms_energy": round(rms, 6),
        "peak_amplitude": round(peak, 6),
        "clipping_ratio": round(clipping_ratio, 6),
        "num_samples": len(y),
        "warnings": [],
    }

    if warn_on_quality:
        if duration < MIN_DURATION_SECONDS:
            msg = f"Very short clip ({duration:.2f}s < {MIN_DURATION_SECONDS}s). Features may be unreliable."
            logger.warning(f"[utils] {msg}")
            meta["warnings"].append(msg)

        if duration > MAX_DURATION_SECONDS:
            msg = f"Unusually long clip ({duration:.2f}s). Only first {MAX_DURATION_SECONDS}s used."
            logger.warning(f"[utils] {msg}")
            meta["warnings"].append(msg)

        if rms < MAX_AMPLITUDE_THRESHOLD:
            msg = f"Very low RMS energy ({rms:.2e}). Audio may be silent or near-silent."
            logger.warning(f"[utils] {msg}")
            meta["warnings"].append(msg)

        if clipping_ratio > 0.01:
            msg = f"Audio clipping detected ({clipping_ratio*100:.1f}% of samples). Quality may be degraded."
            logger.warning(f"[utils] {msg}")
            meta["warnings"].append(msg)

    return y, sr, meta


def preprocess_waveform(
    y: np.ndarray,
    sr: int,
    apply_preemphasis: bool = True,
    trim_silence: bool = True,
    top_db: float = 30.0,
) -> np.ndarray:
    """
    Optional preprocessing pipeline for waveforms before feature extraction.

    Parameters:
        y                : np.ndarray — raw waveform
        sr               : int        — sample rate
        apply_preemphasis: bool       — apply pre-emphasis filter (boosts high frequencies)
        trim_silence     : bool       — trim leading/trailing silence
        top_db           : float      — silence threshold in dB for trimming

    Returns:
        np.ndarray — preprocessed waveform

    Notes:
        Pre-emphasis: a common speech processing step that compensates for
        the natural roll-off of high-frequency energy in voiced sounds.
        Formula: y[n] = y[n] - 0.97 * y[n-1]
    """
    if len(y) == 0:
        logger.warning("[utils] Empty waveform passed to preprocess_waveform")
        return y

    # Trim silence from start and end
    if trim_silence:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if len(y_trimmed) > 0:
            y = y_trimmed
            logger.debug(f"[utils] After silence trim: {len(y)/sr:.2f}s")

    # Pre-emphasis filter
    if apply_preemphasis:
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    return y.astype(np.float32)


def list_audio_files(directory: str, recursive: bool = True) -> list:
    """
    Recursively list all supported audio files in a directory.

    Parameters:
        directory : str  — root directory to search
        recursive : bool — if True, search subdirectories

    Returns:
        list of str — sorted list of absolute file paths
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"[utils] Directory not found: {directory}")

    files = []
    if recursive:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(root.rglob(f"*{ext}"))
    else:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(root.glob(f"*{ext}"))

    sorted_files = sorted([str(f.resolve()) for f in files])
    logger.info(f"[utils] Found {len(sorted_files)} audio files in '{directory}'")
    return sorted_files


def setup_logging(level: str = "INFO", logfile: Optional[str] = None) -> None:
    """
    Configure logging for the audio module.

    Parameters:
        level   : str           — logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        logfile : str, optional — path to write log file (in addition to stdout)
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler()]

    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
    )
    logging.getLogger("numba").setLevel(logging.WARNING)  # Suppress numba noise
    logging.getLogger("librosa").setLevel(logging.WARNING)