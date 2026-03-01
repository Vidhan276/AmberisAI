"""
feature_extraction.py
======================
Core audio feature extraction pipeline for AmberisAI infant cry analysis.

Features extracted (scientifically grounded for infant cry analysis):
  - MFCCs (mean + std): Captures spectral shape, fundamental to speech/cry analysis.
  - Spectral Centroid: Reflects "brightness" — higher in pain cries vs. tired cries.
  - Spectral Bandwidth: Spread of energy around centroid; broader in discomfort.
  - Spectral Contrast: Difference between peaks and valleys; useful for formant structure.
  - Zero Crossing Rate: Related to noisiness/voicing; higher in discomfort cries.
  - RMS Energy: Amplitude envelope; hunger cries tend to be more energetic.
  - Mel Spectrogram Statistics: Time-averaged Mel-band energy distribution.

EXCLUDED FEATURES (with justification):
  - Chroma Features: Derived from equal-tempered pitch classes (musical harmony).
    Infant cries are not tonally structured — chroma is meaningless and adds noise.
  - Tonnetz: Harmonic network derived from chroma; same reasons as chroma exclusion.
    Including these would reduce generalization and add spurious dimensions.

DESIGN PRINCIPLES:
  - Identical pipeline for training AND inference (no leakage).
  - Single, flat feature vector per audio clip.
  - Feature dimension is logged and reproducible.
  - Robust to real-world noise, variable-length audio, and format differences.
"""

import numpy as np
import librosa
import logging
from typing import Optional, Tuple

# Module-level logger
logger = logging.getLogger(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050          # Standard librosa SR; balances quality vs. memory
N_MFCC = 13                  # Standard for speech/cry; sufficient for timbral shape
N_MELS = 40                  # Mel filterbank resolution
HOP_LENGTH = 512             # ~23ms at 22050 Hz; standard overlap
N_FFT = 2048                 # FFT window size (~93ms); good frequency resolution
N_CONTRAST_BANDS = 6         # Spectral contrast sub-bands
DURATION = 4.0               # Normalize all clips to 4 seconds


class FeatureExtractor:
    """
    Stateless feature extractor for infant cry audio.

    Extracts a fixed-length feature vector from any audio clip.
    Same instance must be used for both training and inference.

    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract(audio_array, sr=22050)
        print(f"Feature dim: {len(features)}")  # Should log ~182 dims
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mfcc: int = N_MFCC,
        n_mels: int = N_MELS,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        n_contrast_bands: int = N_CONTRAST_BANDS,
        duration: float = DURATION,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_contrast_bands = n_contrast_bands
        self.duration = duration

        # Pre-compute expected feature dimensionality for validation
        self._feature_dim = self._compute_expected_dim()
        logger.info(
            f"[FeatureExtractor] Initialized. Expected feature dim: {self._feature_dim}"
        )

    def _compute_expected_dim(self) -> int:
        """
        Compute expected feature vector length deterministically.

        Breakdown:
          - MFCCs:              n_mfcc * 2 (mean + std) = 13 * 2 = 26
          - Delta MFCCs:        n_mfcc * 2 (mean + std) = 13 * 2 = 26
          - Spectral Centroid:  2 (mean + std)
          - Spectral Bandwidth: 2 (mean + std)
          - Spectral Contrast:  (n_contrast_bands + 1) * 2 = 7 * 2 = 14
          - ZCR:                2 (mean + std)
          - RMS Energy:         2 (mean + std)
          - Mel Stats:          n_mels * 2 (mean + std) = 40 * 2 = 80
          ─────────────────────────────────────────────────────────────
          Total:                26 + 26 + 2 + 2 + 14 + 2 + 2 + 80 = 154
        """
        mfcc_dim = self.n_mfcc * 2
        delta_mfcc_dim = self.n_mfcc * 2
        centroid_dim = 2
        bandwidth_dim = 2
        contrast_dim = (self.n_contrast_bands + 1) * 2
        zcr_dim = 2
        rms_dim = 2
        mel_dim = self.n_mels * 2
        return (
            mfcc_dim + delta_mfcc_dim + centroid_dim + bandwidth_dim
            + contrast_dim + zcr_dim + rms_dim + mel_dim
        )

    @property
    def feature_dim(self) -> int:
        """Returns the expected feature vector length."""
        return self._feature_dim

    def pad_or_trim(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Normalize audio to a fixed duration.

        - If shorter: pad with zeros (silence) at the end.
        - If longer: trim to target duration.

        This ensures a fixed feature vector length regardless of clip length.
        """
        target_length = int(self.duration * sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        elif len(y) > target_length:
            y = y[:target_length]
        return y

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract a fixed-length feature vector from a raw audio array.

        Parameters:
            y  : np.ndarray — raw audio waveform (mono, float32)
            sr : int        — sample rate of `y`

        Returns:
            np.ndarray of shape (feature_dim,) — flattened feature vector

        Raises:
            ValueError: if input is silent, NaN-filled, or malformed
        """
        if y is None or len(y) == 0:
            raise ValueError("[FeatureExtractor] Empty audio array received.")
        if np.all(y == 0):
            logger.warning("[FeatureExtractor] Silent audio clip detected.")
        if np.any(np.isnan(y)):
            raise ValueError("[FeatureExtractor] NaN values in audio array.")

        # Resample if needed to ensure consistent feature computation
        if sr != self.sample_rate:
            logger.debug(
                f"[FeatureExtractor] Resampling from {sr} Hz to {self.sample_rate} Hz"
            )
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        # Normalize audio amplitude to [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # Pad or trim to fixed duration
        y = self.pad_or_trim(y, sr)

        features = []

        # ── 1. MFCCs (mean + std) ─────────────────────────────────────────────
        # MFCCs capture the spectral envelope (timbral texture).
        # 13 coefficients are standard; more adds noise for cry classification.
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        features.extend(np.mean(mfccs, axis=1))   # shape (n_mfcc,)
        features.extend(np.std(mfccs, axis=1))    # shape (n_mfcc,)

        # ── 2. Delta MFCCs (mean + std) ───────────────────────────────────────
        # First-order temporal derivatives of MFCCs.
        # Captures rate of change in spectral shape — important for cry dynamics.
        delta_mfccs = librosa.feature.delta(mfccs)
        features.extend(np.mean(delta_mfccs, axis=1))
        features.extend(np.std(delta_mfccs, axis=1))

        # ── 3. Spectral Centroid (mean + std) ────────────────────────────────
        # Weighted mean of frequencies; describes spectral "brightness".
        # Pain/discomfort cries tend to have higher centroids than tired cries.
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.append(float(np.mean(centroid)))
        features.append(float(np.std(centroid)))

        # ── 4. Spectral Bandwidth (mean + std) ───────────────────────────────
        # Weighted std of frequencies around centroid.
        # Broader bandwidth correlates with more noise/irregularity in cry.
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.append(float(np.mean(bandwidth)))
        features.append(float(np.std(bandwidth)))

        # ── 5. Spectral Contrast (mean + std per band) ───────────────────────
        # Measures peak-valley difference across spectral sub-bands.
        # Captures formant structure and harmonicity — informative for cry type.
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_bands=self.n_contrast_bands
        )
        features.extend(np.mean(contrast, axis=1))   # shape (n_bands + 1,)
        features.extend(np.std(contrast, axis=1))    # shape (n_bands + 1,)

        # ── 6. Zero Crossing Rate (mean + std) ───────────────────────────────
        # Rate at which signal changes sign; proxy for noisiness / unvoiced content.
        # Discomfort cries often have higher ZCR due to irregular vocalization.
        zcr = librosa.feature.zero_crossing_rate(
            y=y, hop_length=self.hop_length
        )
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        # ── 7. RMS Energy (mean + std) ────────────────────────────────────────
        # Root mean square energy per frame; proxy for loudness/intensity.
        # Hunger cries are often sustained and energetic; tired cries wane.
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        # ── 8. Mel Spectrogram Statistics (mean + std per band) ───────────────
        # Time-averaged energy in each Mel-frequency band.
        # Provides fine-grained spectral shape; complements MFCCs.
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        # Convert to dB scale for better dynamic range handling
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend(np.mean(mel_spec_db, axis=1))  # shape (n_mels,)
        features.extend(np.std(mel_spec_db, axis=1))   # shape (n_mels,)

        # ── NOTE ON SPECTRAL LEAKAGE ──────────────────────────────────────────
        # Spectral leakage (energy spreading from a frequency bin into neighbors)
        # is mitigated by the Hanning window applied internally by librosa's STFT.
        # We do not extract leakage as a separate feature, as it is a technical
        # artifact — not a perceptual property. Proper windowing is sufficient.

        feature_vector = np.array(features, dtype=np.float32)

        # Integrity check: ensure output dimension matches expectation
        if len(feature_vector) != self._feature_dim:
            raise RuntimeError(
                f"[FeatureExtractor] Feature dim mismatch: "
                f"expected {self._feature_dim}, got {len(feature_vector)}"
            )

        logger.debug(
            f"[FeatureExtractor] Extracted {len(feature_vector)}-dim feature vector. "
            f"Range: [{feature_vector.min():.4f}, {feature_vector.max():.4f}]"
        )

        return feature_vector

    def extract_batch(
        self, audio_list: list, sr_list: Optional[list] = None
    ) -> np.ndarray:
        """
        Extract features from a batch of audio arrays.

        Parameters:
            audio_list : list of np.ndarray — list of raw waveforms
            sr_list    : list of int (optional) — sample rates per clip;
                         uses self.sample_rate for all if not provided

        Returns:
            np.ndarray of shape (N, feature_dim)
        """
        if sr_list is None:
            sr_list = [self.sample_rate] * len(audio_list)

        results = []
        for i, (y, sr) in enumerate(zip(audio_list, sr_list)):
            try:
                feat = self.extract(y, sr)
                results.append(feat)
            except Exception as e:
                logger.error(f"[FeatureExtractor] Failed on clip {i}: {e}")
                # Append NaN vector to preserve batch index alignment
                results.append(np.full(self._feature_dim, np.nan, dtype=np.float32))

        return np.vstack(results)

    def get_feature_names(self) -> list:
        """
        Returns a human-readable list of feature names for explainability.
        Useful for feature importance plots and research tables.
        """
        names = []
        for i in range(self.n_mfcc):
            names.append(f"mfcc_mean_{i+1}")
        for i in range(self.n_mfcc):
            names.append(f"mfcc_std_{i+1}")
        for i in range(self.n_mfcc):
            names.append(f"delta_mfcc_mean_{i+1}")
        for i in range(self.n_mfcc):
            names.append(f"delta_mfcc_std_{i+1}")
        names += ["centroid_mean", "centroid_std"]
        names += ["bandwidth_mean", "bandwidth_std"]
        for i in range(self.n_contrast_bands + 1):
            names.append(f"contrast_mean_band_{i}")
        for i in range(self.n_contrast_bands + 1):
            names.append(f"contrast_std_band_{i}")
        names += ["zcr_mean", "zcr_std"]
        names += ["rms_mean", "rms_std"]
        for i in range(self.n_mels):
            names.append(f"mel_mean_band_{i+1}")
        for i in range(self.n_mels):
            names.append(f"mel_std_band_{i+1}")
        return names