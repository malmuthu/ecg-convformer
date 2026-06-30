"""
visualize.py
Visualization utilities for Integrated Gradient attributions.

Produces:
    - Single beat attribution overlay 
    - Grid of attribution examples across all 5 AAMU classes
    - Clinical region validation plot

Usage:
    from src.interpretability.visualize import plot_attribution
"""

import matplotlib.pyplot as plt
import numpy as np
from mathplotlib.colors import TwoSlopeNorm
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJ_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "Normal", 1: "Supraventricular", 2: "Ventricular", 3: "Fusion", 4: "Unknown"}

CLINICAL_REGIONS = {
    "P wave": (40, 75),
    "QRS": (75, 105),
    "T wave": (115, 175),
}

def plot_attribution(beat: np.ndarray, attribution: np.ndarray, true_label: int, pred_label: int, save_path: str = None, show_clinical_regions: bool = True) -> plt.Figure:
    """
    Plot a single beat with Integrated Gradients attribution overlay.

    Args:
        beat                    : shape (187,) raw normalized beat
        attribution             : shape (187,) IG attribution scores
        true_label              : true AAMI class index
        pred_label              : predicted AAMI class index
        save_path               : save figure to this path
        show_clinical_regions   : overlay P/QRS/T region shading

    Returns:
        matplotlib Figure
    """

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(beat))

    ax.plot(x, beat, color="black", linewidth=1.5, zorder=3)

    abs_max = np.abs(attribution).max()
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    ax.imshow(
        attribution[np.newaxis, :], cmap="coolwarm", norm=norm,
        aspect="auto", extent=[0, len(beat), beat.min(), beat.max()],
        alpha=0.5, zorder=1,
    )

    if show_clinical_regions:
        for region_name, (start, end) in CLINICAL_REGIONS.items():
            ax.axvspan(start, end, alpha=0.08, color="green", zorder=0)
            ax.text((start+end)/2, beat.max()*1.05, region_name,
                    ha="center", fontsize=8, color="darkgreen")
            
    correct = "✓" if true_label == pred_label else "x"
    ax.set_title(f"True: {CLASS_NAMES[true_label]} | Predicted: {CLASS_NAMES[pred_label]} {correct}", fontsize=11)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Normalized Amplitude")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig

def plot_class_grid(beats: dict, attributions: dict, labels: dict, save_path: str = None) -> plt.Figure:
    """
    Plot a grid of attribution examples, one column per AAMI class.

    Args:
        beats       : dict {class idx: list of beat arrays}
        attributions: dict {class idx: list of attribution arrays}
        labels      : dict {class idx: list of (true, pred) tuples}
        save_path   : save figure to this path

    Returns:
        matplotlib Figure
    """

    n_classes = len(beats)
    n_examples = len(next(iter(beats.values())))

    fig, axes = plt.subplots(n_examples, n_classes, figsize=(3 * n_classes, 2.2 * n_examples), sharex=True, sharey=False)

    for col, class_idx in enumerate(sorted(beats.keys())):
        for row in range(n_examples):
            ax = axes[row, col]
            beat = beats[class_idx][row]
            attr = attributions[class_idx][row]
            x = np.arange(len(beat))

            ax.plot(x, beat, color="black", linewidth=1)
            abs_max = np.abs(attr).max() + 1e-8
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            ax.imshow(
                attr[np.newaxis, :], cmap="coolwarm", norm=norm,
                aspect="auto", extent=[0, len(beat), beat.min(), beat.max()],
                alpha=0.5)
            
            if row == 0:
                ax.set_title(CLASS_NAMES[class_idx], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig

def compute_region_attribution_score(attribution: np.ndarray) -> dict:
    """
    Compute the proportion of total attribution falling within each clinical region (P wave, QRS, T wave).

    Args:
        attribution : shape (187,) IG attribution scores

    Returns:
        dict mapping region name -> fraction of total attribution mass
    """

    abs_attr = np.abs(attribution)
    total_mass = abs_attr.sum() + 1e-8

    scores = {}

    for region_name, (start, end) in CLINICAL_REGIONS.items():
        region_mass = abs_attr[start:end].sum()
        scores[region_name] = float(region_mass / total_mass)
        covered_mass += region_mass
        
    scores["other"] = float((total_mass - covered_mass) / total_mass)

    return scores