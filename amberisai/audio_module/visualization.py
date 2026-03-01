"""
visualization.py
================
Visualization utilities for AmberisAI audio module.

Provides:
  - Raw waveform plots
  - Mel spectrogram plots
  - MFCC heatmaps
  - Random Forest feature importance
  - Prediction summary visualization

Usage contexts:
  - Debugging model inputs/outputs
  - Research paper figures (publication-quality)
  - Explainability reporting for non-technical stakeholders

All functions:
  - Accept optional `ax` parameter for embedding in larger figures.
  - Return (fig, ax) or (fig, axes) tuples for caller control.
  - Save to disk if `save_path` is provided.
  - Use consistent styling via `apply_style()`.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import librosa
import librosa.display

logger = logging.getLogger(__name__)

# ─── STYLE CONSTANTS ──────────────────────────────────────────────────────────
FIGURE_DPI = 150
CMAP_SPEC = "magma"         # Perceptually uniform; good for spectrograms
CMAP_MFCC = "coolwarm"      # Diverging; intuitive for MFCC sign
CMAP_IMPORTANCE = "viridis"
TITLE_FONTSIZE = 13
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9


def apply_style() -> None:
    """Apply consistent matplotlib style for all visualizations."""
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "figure.dpi": FIGURE_DPI,
    })


def _save_if_requested(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to disk if a path is specified."""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"[Visualization] Saved figure: {save_path}")


# ─── 1. RAW WAVEFORM ──────────────────────────────────────────────────────────

def plot_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "Waveform",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    color: str = "#2E86AB",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the raw time-domain waveform of an audio signal.

    Parameters:
        y         : np.ndarray — audio waveform
        sr        : int        — sample rate
        title     : str        — plot title
        label     : str        — optional condition label (e.g., "hungry")
        ax        : plt.Axes   — existing axes to plot on (creates new if None)
        save_path : str        — save figure to this path if provided
        color     : str        — waveform color

    Returns:
        (fig, ax)
    """
    apply_style()
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure

    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color=color, linewidth=0.6, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    full_title = f"{title}" + (f" — {label}" if label else "")
    ax.set_title(full_title, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(y) / sr)
    ax.set_ylim(-1.1, 1.1)

    # Annotate duration
    duration = len(y) / sr
    ax.text(
        0.98, 0.95, f"Duration: {duration:.2f}s",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="gray"
    )

    if created_fig:
        fig.tight_layout()
        _save_if_requested(fig, save_path)

    return fig, ax


# ─── 2. MEL SPECTROGRAM ───────────────────────────────────────────────────────

def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "Mel Spectrogram",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the Mel-frequency spectrogram (dB scale) of an audio signal.

    The Mel spectrogram shows how energy is distributed across frequency
    bands over time. It uses a perceptually motivated frequency scale.

    Parameters:
        y, sr      : audio data
        n_mels     : number of Mel bands (must match feature_extraction.py)
        n_fft      : FFT window size (must match feature_extraction.py)
        hop_length : hop between frames (must match feature_extraction.py)
        title      : plot title
        label      : optional condition label
        ax         : optional existing axes
        save_path  : optional save path

    Returns:
        (fig, ax)
    """
    apply_style()
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel",
        ax=ax, cmap=CMAP_SPEC
    )

    full_title = f"{title}" + (f" — {label}" if label else "")
    ax.set_title(full_title, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel Frequency (Hz)")

    if created_fig:
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
        cbar.set_label("Power (dB)", fontsize=10)
        fig.tight_layout()
        _save_if_requested(fig, save_path)

    return fig, ax


# ─── 3. MFCC HEATMAP ─────────────────────────────────────────────────────────

def plot_mfcc_heatmap(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "MFCC Heatmap",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of MFCCs over time.

    Rows = MFCC coefficients (1–n_mfcc).
    Columns = time frames.
    Color = coefficient value.

    The MFCC heatmap reveals how spectral texture evolves over the cry duration.
    Different cry types show characteristic MFCC temporal patterns.

    Returns:
        (fig, ax)
    """
    apply_style()
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length
    )

    img = librosa.display.specshow(
        mfccs, sr=sr, hop_length=hop_length,
        x_axis="time", ax=ax,
        cmap=CMAP_MFCC
    )

    full_title = f"{title}" + (f" — {label}" if label else "")
    ax.set_title(full_title, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MFCC Coefficient")
    ax.set_yticks(range(n_mfcc))
    ax.set_yticklabels([f"C{i+1}" for i in range(n_mfcc)], fontsize=8)

    if created_fig:
        cbar = fig.colorbar(img, ax=ax, pad=0.02)
        cbar.set_label("Coefficient Value", fontsize=10)
        fig.tight_layout()
        _save_if_requested(fig, save_path)

    return fig, ax


# ─── 4. RANDOM FOREST FEATURE IMPORTANCE ─────────────────────────────────────

def plot_feature_importance(
    rf_model,
    feature_names: List[str],
    top_n: int = 30,
    title: str = "Random Forest Feature Importance (Top Features)",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot top-N most important features from a trained Random Forest.

    Uses `feature_importances_` (Mean Decrease Impurity).
    Error bars show standard deviation across trees.

    Parameters:
        rf_model      : fitted RandomForestClassifier
        feature_names : list of str — names from FeatureExtractor.get_feature_names()
        top_n         : int — number of top features to show
        title         : str — plot title
        ax            : optional axes
        save_path     : optional save path

    Returns:
        (fig, ax)
    """
    apply_style()

    importances = rf_model.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in rf_model.estimators_], axis=0
    )

    # Rank by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_std = std[indices]
    top_names = [
        feature_names[i] if i < len(feature_names) else f"feature_{i}"
        for i in indices
    ]

    created_fig = ax is None
    if ax is None:
        height = max(5, top_n * 0.28)
        fig, ax = plt.subplots(figsize=(9, height))
    else:
        fig = ax.figure

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_importances)))
    y_pos = np.arange(len(top_importances))

    bars = ax.barh(
        y_pos, top_importances[::-1],
        xerr=top_std[::-1],
        align="center",
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "ecolor": "gray", "capsize": 2}
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("_", " ") for n in top_names[::-1]], fontsize=8)
    ax.set_xlabel("Mean Decrease Impurity (Importance)", fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, max(top_importances) * 1.15)

    # Annotate top feature
    ax.annotate(
        f"Top: {top_names[0]}",
        xy=(top_importances[0], len(y_pos) - 1),
        xytext=(top_importances[0] * 0.5, len(y_pos) * 0.6),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=8, color="dimgray"
    )

    if created_fig:
        fig.tight_layout()
        _save_if_requested(fig, save_path)

    return fig, ax


# ─── 5. PREDICTION SUMMARY ────────────────────────────────────────────────────

def plot_prediction_summary(
    prediction_result: dict,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Composite figure showing prediction results + optional waveform + probabilities.

    Panels (when audio is provided):
      [Left]  : Waveform
      [Right] : Probability bar chart

    Without audio:
      [Full]  : Probability bar chart only

    Parameters:
        prediction_result : dict — output from AudioPredictor.predict()
        y, sr             : optional audio array for waveform panel
        save_path         : optional save path

    Returns:
        (fig, axes)
    """
    apply_style()

    probs = prediction_result.get("all_probabilities", {})
    detected = prediction_result.get("detected_condition", "unknown")
    confidence = prediction_result.get("confidence", 0.0)
    secondary = prediction_result.get("secondary_condition", "N/A")
    low_conf = prediction_result.get("low_confidence_warning", False)

    classes = sorted(probs.keys())
    values = [probs[c] for c in classes]
    colors = [
        "#2E86AB" if c == detected else
        "#A8DADC" if c == secondary else
        "#E9C46A"
        for c in classes
    ]

    has_audio = y is not None and sr is not None

    if has_audio:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plot_waveform(y, sr, title="Input Waveform", label=detected, ax=axes[0])
        prob_ax = axes[1]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        prob_ax = axes

    # Probability bar chart
    bars = prob_ax.bar(
        classes, values,
        color=colors, edgecolor="white", linewidth=0.8
    )
    prob_ax.set_ylim(0, 1.05)
    prob_ax.set_ylabel("Probability")
    prob_ax.set_xlabel("Condition")

    # Annotate bars with values
    for bar, val in zip(bars, values):
        if val > 0.02:
            prob_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9
            )

    title = f"Prediction: {detected.upper()} ({confidence:.1%} confidence)"
    if low_conf:
        title += " ⚠ LOW CONFIDENCE"
    prob_ax.set_title(title, fontweight="bold", color="#E76F51" if low_conf else "black")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", label=f"Primary: {detected}"),
        Patch(facecolor="#A8DADC", label=f"Secondary: {secondary}"),
        Patch(facecolor="#E9C46A", label="Other"),
    ]
    prob_ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    _save_if_requested(fig, save_path)

    return fig, axes


# ─── 6. FULL DIAGNOSTIC PANEL ────────────────────────────────────────────────

def plot_diagnostic_panel(
    y: np.ndarray,
    sr: int,
    label: Optional[str] = None,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Full 3-panel diagnostic figure: waveform + Mel spectrogram + MFCC heatmap.

    Intended for research figures and publication.

    Parameters:
        y, sr      : audio data
        label      : optional class label for titles
        n_mfcc     : must match FeatureExtractor settings
        n_mels     : must match FeatureExtractor settings
        n_fft      : must match FeatureExtractor settings
        hop_length : must match FeatureExtractor settings
        save_path  : optional save path

    Returns:
        (fig, axes) — 3-row subplot figure
    """
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    plot_waveform(y, sr, title="Waveform", label=label, ax=axes[0])
    plot_mel_spectrogram(
        y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        title="Mel Spectrogram", label=label, ax=axes[1]
    )
    plot_mfcc_heatmap(
        y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        title="MFCC Heatmap", label=label, ax=axes[2]
    )

    fig.suptitle(
        f"Audio Diagnostic Panel" + (f" — '{label}'" if label else ""),
        fontsize=15, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig, axes


def close_all() -> None:
    """Close all open matplotlib figures (call at end of batch runs)."""
    plt.close("all")
    logger.debug("[Visualization] Closed all figures.")