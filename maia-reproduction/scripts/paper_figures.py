"""Generate paper-style figures: 3 paper-arch Maia models + Stockfish baselines."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BINS = [1100, 1500, 1900]
BIN_LABELS = ["1100-1199", "1500-1599", "1900-1999"]


def load_data():
    with open("reports/stockfish_results.json") as f:
        sf = json.load(f)
    with open("reports/full_model_results2.json") as f:
        maia = json.load(f)

    models = []
    for d in [1, 7, 15]:
        accs = [sf.get(str(b), {}).get(str(d), {}).get("accuracy", 0) * 100 for b in BINS]
        models.append({"name": f"Stockfish depth {d}", "accs": accs})

    for b in BINS:
        sb = str(b)
        accs = []
        for tb in BINS:
            key = "full_maia" if tb == b else f"full_maia_on_{tb}"
            accs.append(maia.get(sb, {}).get(key, {}).get("accuracy", 0) * 100)
        models.append({"name": f"Maia-{b}", "accs": accs})

    return models


def fig2_accuracy_curves(models):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "Stockfish depth 1": "#1f77b4", "Stockfish depth 7": "#aec7e8",
        "Stockfish depth 15": "#7f7f7f",
    }
    markers = {"Stockfish depth 1": "s", "Stockfish depth 7": "D", "Stockfish depth 15": "v"}
    maia_colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    maia_markers = ["o", "^", "P"]

    for m in models:
        if "Stockfish" in m["name"]:
            ax.plot(BINS, m["accs"], label=m["name"],
                    color=colors.get(m["name"], "gray"),
                    marker=markers.get(m["name"], "o"),
                    linewidth=2, markersize=6)

    maia_idx = 0
    for m in models:
        if "Maia" in m["name"]:
            ax.plot(BINS, m["accs"], label=m["name"],
                    color=maia_colors[maia_idx], marker=maia_markers[maia_idx],
                    linewidth=2, markersize=6, linestyle="--")
            maia_idx += 1

    ax.set_xlabel("Rating bin of human opponent", fontsize=12)
    ax.set_ylabel("Move-matching accuracy (%)", fontsize=12)
    ax.set_title("Move-Matching Accuracy by Rating Bin\n(Our Reproduction — Paper Architecture)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(BINS)
    ax.set_xticklabels(BIN_LABELS, fontsize=10)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    plt.tight_layout()
    plt.savefig("reports/fig2_accuracy_curves.png", dpi=150)
    print("Saved: reports/fig2_accuracy_curves.png")


def fig6_agreement_matrix():
    with open("reports/stockfish_results.json") as f:
        sf = json.load(f)
    with open("reports/full_model_results2.json") as f:
        maia = json.load(f)

    labels = ["SF d=1", "SF d=7", "SF d=15",
              "Maia-1100", "Maia-1500", "Maia-1900"]
    n = len(labels)
    matrix = np.zeros((n, 3))

    for j, b in enumerate(BINS):
        for i, d in enumerate([1, 7, 15]):
            val = sf.get(str(b), {}).get(str(d), {}).get("accuracy", 0)
            matrix[i][j] = val * 100
    for i, b in enumerate(BINS):
        sb = str(b)
        for j, tb in enumerate(BINS):
            key = "full_maia" if tb == b else f"full_maia_on_{tb}"
            val = maia.get(sb, {}).get(key, {}).get("accuracy", 0)
            matrix[3 + i][j] = val * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=50, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(n))
    ax.set_xticklabels(BIN_LABELS, fontsize=10)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i, j] > 20 else "white")
    ax.set_xlabel("Rating bin of human opponent", fontsize=11)
    ax.set_title("Model Agreement Matrix (%), Our Reproduction", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("reports/fig6_agreement_matrix.png", dpi=150)
    print("Saved: reports/fig6_agreement_matrix.png")


def plot_paper_comparison():
    with open("reports/full_model_results2.json") as f:
        maia = json.load(f)
    self_bin_accs = [maia.get(str(b), {}).get("full_maia", {}).get("accuracy", 0) * 100
                     for b in BINS]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    width = 0.35
    ax.bar(x, self_bin_accs, width, label="Our Maia (paper arch)", color="#ff7f0e")
    ax.bar(x + width, [32, 35, 35], width, label="Paper (approx peak)", color="#1f77b4", alpha=0.7)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(BIN_LABELS, fontsize=10)
    ax.set_ylabel("Top-1 Move-Matching Accuracy (%)", fontsize=11)
    ax.set_title("Our Results vs. Maia Paper", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(self_bin_accs):
        ax.annotate(f"{v:.1f}%", (i, v + 1), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/paper_comparison.png", dpi=150)
    print("Saved: reports/paper_comparison.png")


def generate_table():
    with open("reports/stockfish_results.json") as f:
        sf = json.load(f)
    with open("reports/full_model_results2.json") as f:
        maia = json.load(f)
    rows = []
    for d in [1, 7, 15]:
        accs = [sf.get(str(b), {}).get(str(d), {}).get("accuracy", 0) * 100 for b in BINS]
        rows.append(f"| Stockfish depth {d} | {' | '.join(f'{a:.1f}%' for a in accs)} | 36-42% |")
    for b in BINS:
        acc = maia.get(str(b), {}).get("full_maia", {}).get("accuracy", 0) * 100
        rows.append(f"| Maia-{b} (paper arch) | {' | '.join(['—'] * 3)} | ~30-35% |")
    return "\n".join(rows)


if __name__ == "__main__":
    models = load_data()
    fig2_accuracy_curves(models)
    fig6_agreement_matrix()
    plot_paper_comparison()
    print("\nTable:\n", generate_table())
