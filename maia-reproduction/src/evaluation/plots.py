"""Plotting utilities for evaluation results.

Produces:
- Accuracy curves (per family, combined)
- Model maxima tables
- Agreement matrix heatmaps
- Accuracy vs. position complexity
- Accuracy vs. move quality
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)

# Consistent color palettes
MAIA_COLORS = sns.color_palette("Blues", 9)[2:]
STOCKFISH_COLORS = sns.color_palette("Reds", 9)[2:]
LC0_COLORS = sns.color_palette("Greens", 9)[2:]

RATING_BINS = list(range(1100, 1900, 100))
RATING_LABELS = [f"{b}-{b+99}" for b in RATING_BINS]
EXTENDED_BINS = [1000] + RATING_BINS + [2500]
EXTENDED_LABELS = ["1000"] + RATING_LABELS + ["2500"]

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def plot_accuracy_curves(
    results: dict[str, dict[int, float]],
    family: str = "Maia",
    test_bins: list[int] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
    colors: list | None = None,
) -> matplotlib.axes.Axes:
    """Plot accuracy curves for a family of models.

    Args:
        results: Dict mapping model_name -> {test_bin: accuracy}.
        family: "Maia", "Stockfish", or "Leela".
        test_bins: List of test bin values to include.
        ax: Matplotlib axis to plot on.
        title: Plot title (auto-generated if None).
        colors: Color palette.

    Returns matplotlib axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if test_bins is None:
        test_bins = sorted(set().union(*[r.keys() for r in results.values()]))

    # Sort models by name
    model_names = sorted(results.keys())
    if colors is None:
        colors = sns.color_palette("husl", len(model_names))

    for idx, model_name in enumerate(model_names):
        accs = results[model_name]
        x = [test_bins.index(b) for b in accs.keys() if b in test_bins]
        y = [accs[b] for b in test_bins if b in accs]

        if x and y:
            # Ensure sorted by x
            pairs = sorted(zip(x, y))
            xs, ys = zip(*pairs) if pairs else ([], [])
            ax.plot(
                xs, ys,
                marker="o", linewidth=2, markersize=5,
                color=colors[idx % len(colors)],
                label=f"{model_name}",
            )

    if title is None:
        title = f"Move-Matching Accuracy: {family} Models"

    bin_labels = [
        f"{b}-{b+99}" if b < 2000 else str(b)
        for b in test_bins
    ]

    ax.set_xticks(range(len(test_bins)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_xlabel("Human Rating Bin")
    ax.set_ylabel("Move-Matching Accuracy")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 0.65)

    return ax


def plot_combined_accuracy(
    maia_results: dict[str, dict[int, float]],
    stockfish_results: dict[str, dict[int, float]],
    lc0_results: dict[str, dict[int, float]],
    test_bins: list[int] | None = None,
    save_path: str | Path | None = None,
):
    """Create combined overlay plot with all three families."""
    if test_bins is None:
        all_bins: set[int] = set()
        for r in [maia_results, stockfish_results, lc0_results]:
            for v in r.values():
                all_bins.update(v.keys())
        test_bins = sorted(all_bins)

    _, ax = plt.subplots(figsize=(14, 8))

    # Representative subset: 5 Maia, 3 Stockfish, 3 Leela
    maia_subset = {k: v for k, v in sorted(maia_results.items())[::2]}
    sf_subset = {k: v for k, v in sorted(stockfish_results.items())[:5]}
    lc_subset = {k: v for k, v in sorted(lc0_results.items())[::2]}

    plot_accuracy_curves(
        maia_subset, family="Maia", test_bins=test_bins, ax=ax,
        colors=sns.color_palette("Blues_d", len(maia_subset)),
    )
    plot_accuracy_curves(
        sf_subset, family="Stockfish", test_bins=test_bins, ax=ax,
        colors=sns.color_palette("Reds_d", len(sf_subset)),
    )
    plot_accuracy_curves(
        lc_subset, family="Leela", test_bins=test_bins, ax=ax,
        colors=sns.color_palette("Greens_d", len(lc_subset)),
    )

    ax.set_title("Move-Matching Accuracy: All Engine Families")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved combined plot to {save_path}")

    return ax


def plot_agreement_heatmap(
    matrix: np.ndarray,
    model_names: list[str],
    save_path: str | Path | None = None,
):
    """Plot agreement matrix heatmap.

    Args:
        matrix: NxN agreement matrix.
        model_names: List of N model names.
        save_path: Optional path to save figure.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=model_names,
        yticklabels=model_names,
        vmin=0, vmax=1,
        ax=ax,
    )
    ax.set_title("Model Agreement Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved agreement heatmap to {save_path}")

    return fig


def plot_accuracy_vs_complexity(
    results: dict[str, list[float]],
    complexity_bins: list[float],
    title: str = "Accuracy vs. Position Complexity",
    save_path: str | Path | None = None,
):
    """Plot accuracy vs. position complexity (win-prob gap between top 2 moves)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, accuracies in results.items():
        if len(accuracies) == len(complexity_bins):
            ax.plot(
                complexity_bins, accuracies,
                marker="o", linewidth=2,
                label=model_name,
            )

    ax.set_xlabel("Position Complexity (Win-Prob Gap)")
    ax.set_ylabel("Move-Matching Accuracy")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_results_table(
    maia_results: dict[str, dict[int, float]],
    stockfish_results: dict[str, dict[int, float]],
    lc0_results: dict[str, dict[int, float]],
    test_bins: list[int],
) -> str:
    """Generate a markdown results table."""
    lines = [
        "# Move-Matching Accuracy Results\n",
        "| Model | " + " | ".join([f"Bin {b}" for b in test_bins]) + " | Peak |",
        "|-------|" + "|".join(["---"] * (len(test_bins) + 1)) + "|",
    ]

    for name, results in sorted(maia_results.items()):
        row = [f"Maia-{name}"]
        for b in test_bins:
            row.append(f"{results.get(b, 0):.3f}")
        peak_bin = max(results, key=results.get)
        row.append(f"{peak_bin} ({results[peak_bin]:.3f})")
        lines.append("| " + " | ".join(row) + " |")

    for name, results in sorted(stockfish_results.items()):
        row = [f"SF-d{name}"]
        for b in test_bins:
            row.append(f"{results.get(b, 0):.3f}")
        peak_bin = max(results, key=results.get)
        row.append(f"{peak_bin} ({results[peak_bin]:.3f})")
        lines.append("| " + " | ".join(row) + " |")

    for name, results in sorted(lc0_results.items()):
        row = [f"lc0-{name}"]
        for b in test_bins:
            row.append(f"{results.get(b, 0):.3f}")
        peak_bin = max(results, key=results.get)
        row.append(f"{peak_bin} ({results[peak_bin]:.3f})")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
