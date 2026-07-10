"""Generate paper-style figures comparing our results to the Maia paper."""

import json, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BINS = [1100, 1500, 1900]
BIN_LABELS = ["1100-1199", "1500-1599", "1900-1999"]


def load_data():
    """Load all evaluation results."""
    with open("reports/all_baselines.json") as f:
        r = json.load(f)
    with open("reports/full_model_results2.json") as f:
        f2 = json.load(f)

    # Build accuracy matrix: rows=models, cols=bins
    # Order: SF d1, SF d7, SF d15, Maia-Reduced-1100,1500,1900, Maia-Full-1100,1500,1900
    sf_depths = [1, 7, 15]
    models = []
    for d in sf_depths:
        models.append({"name": f"Stockfish d={d}", "accs": []})
    for b in BINS:
        sb = str(b)
        acc = r.get(sb, {}).get("maia", {}).get("accuracy", 0)
        models.append({"name": f"Maia-Reduced-{b}", "accs": [acc]})
    for b in BINS:
        sb = str(b)
        acc = f2.get(sb, {}).get("full_maia", {}).get("accuracy", 0)
        models.append({"name": f"Maia-Full-{b}", "accs": [acc]})

    # Add self-bin accuracy (the paper's key metric)
    for b in BINS:
        sb = str(b)
        for m in models:
            tag = m["name"]
            if f"Full-{b}" in tag and len(m["accs"]) == 1:
                # Need cross-bin for full model too
                for tb in BINS:
                    key = f"full_maia_on_{tb}" if tb != b else "full_maia"
                    acc = f2.get(sb, {}).get(key, {}).get("accuracy", 0)
                    m["accs"].append(acc)

    # For reduced models, add cross-bin accs
    for b in BINS:
        sb = str(b)
        for m in models:
            if f"Reduced-{b}" in m["name"] and len(m["accs"]) == 1:
                for tb in BINS:
                    key = f"maia_on_bin_{tb}" if tb != b else "maia"
                    acc = r.get(sb, {}).get(key, {}).get("accuracy", 0)
                    m["accs"].append(acc)

    return models


def fig2_accuracy_curves(models):
    """Figure 2 equivalent: Move-matching accuracy across rating bins."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "Stockfish d=1": "#1f77b4",
        "Stockfish d=7": "#aec7e8",
        "Stockfish d=15": "#7f7f7f",
    }
    markers = {
        "Stockfish d=1": "s",
        "Stockfish d=7": "D",
        "Stockfish d=15": "v",
    }
    for m in models:
        if "Stockfish" in m["name"]:
            ax.plot(BINS, m["accs"], label=m["name"], color=colors.get(m["name"], "gray"),
                    marker=markers.get(m["name"], "o"), linewidth=2, markersize=6)

    # Maia lines: dashed with markers for self-bin emphasis
    maia_colors = {"Reduced": ["#ff7f0e", "#2ca02c", "#d62728"],
                   "Full": ["#ffbb78", "#98df8a", "#ff9898"]}
    for m in models:
        if "Reduced" in m["name"]:
            idx = [i for i, b in enumerate(BINS) if f"Reduced-{b}" in m["name"]][0]
            c = maia_colors["Reduced"][idx]
            label_short = m["name"].replace("Maia-Reduced-", "Maia-R-")
            ax.plot(BINS, m["accs"], label=label_short, color=c,
                    marker="o", linewidth=2, markersize=6, linestyle="--")
    for m in models:
        if "Full" in m["name"]:
            idx = [i for i, b in enumerate(BINS) if f"Full-{b}" in m["name"]][0]
            c = maia_colors["Full"][idx]
            label_short = m["name"].replace("Maia-Full-", "Maia-F-")
            ax.plot(BINS, m["accs"], label=label_short, color=c,
                    marker="^", linewidth=2, markersize=6, linestyle="-.")

    ax.set_xlabel("Rating bin of human opponent", fontsize=12)
    ax.set_ylabel("Move-matching accuracy (%)", fontsize=12)
    ax.set_title("Move-Matching Accuracy by Rating Bin\n(Our Reproduction)", fontsize=13, fontweight="bold")
    ax.set_xticks(BINS)
    ax.set_xticklabels(BIN_LABELS, fontsize=10)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    plt.tight_layout()
    plt.savefig("reports/fig2_accuracy_curves.png", dpi=150)
    print("Saved: reports/fig2_accuracy_curves.png")


def fig6_agreement_matrix():
    """Figure 6 equivalent: Agreement matrix heatmap."""
    with open("reports/all_baselines.json") as f:
        r = json.load(f)
    with open("reports/full_model_results2.json") as f:
        f2 = json.load(f)

    labels = ["SF d=1", "SF d=7", "SF d=15", "Maia-Reduced", "Maia-Full"]
    matrix = np.zeros((5, 3))

    for i, b in enumerate(BINS):
        sb = str(b)
        for j, d in enumerate([1, 7, 15]):
            key = f"sf_depth_{d}"
            if sb in r and key in r[sb]:
                matrix[j][i] = r[sb][key]["accuracy"] * 100
        if sb in r and "maia" in r[sb]:
            matrix[3][i] = r[sb]["maia"]["accuracy"] * 100
        if sb in f2 and "full_maia" in f2[sb]:
            matrix[4][i] = f2[sb]["full_maia"]["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=50, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_yticks(range(5))
    ax.set_xticklabels(BIN_LABELS, fontsize=10)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(5):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i,j] > 20 else "white")

    ax.set_xlabel("Rating bin of human opponent", fontsize=11)
    ax.set_title("Model Agreement Matrix (%), Our Reproduction", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("reports/fig6_agreement_matrix.png", dpi=150)
    print("Saved: reports/fig6_agreement_matrix.png")


def generate_table():
    """Generate accuracy comparison table in markdown."""
    with open("reports/all_baselines.json") as f:
        r = json.load(f)
    with open("reports/full_model_results2.json") as f:
        f2 = json.load(f)

    rows = []
    for d in [1, 7, 15]:
        accs = []
        for b in BINS:
            sb = str(b)
            acc = r.get(sb, {}).get(f"sf_depth_{d}", {}).get("accuracy", 0) * 100
            accs.append(f"{acc:.1f}")
        rows.append(f"| Stockfish depth {d} | {' | '.join(accs)} |")
    rows.append("|---|")
    for b in BINS:
        sb = str(b)
        acc_r = r.get(sb, {}).get("maia", {}).get("accuracy", 0) * 100
        acc_f = f2.get(sb, {}).get("full_maia", {}).get("accuracy", 0) * 100
        paper_r = "28-32"  # approx from paper
        paper_f = "30-35"
        rows.append(f"| Maia-Reduced-{b} (32ch) | {acc_r:.1f} | | {paper_r} |")
        rows.append(f"| Maia-Full-{b} (256ch) | {acc_f:.1f} | | {paper_f} |")

    return "\n".join(rows)


def plot_paper_comparison():
    """Bar chart comparing our best results to paper's stated ranges."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Our best per-bin accuracy (full-scale model at self-bin)
    ours = [19.6, 22.8, 28.2]
    # Paper approx ranges
    paper_low = [28, 30, 30]
    paper_high = [32, 35, 35]

    x = np.arange(3)
    width = 0.3

    ax.bar(x - width/2, ours, width, label="Our Reproduction", color="#ff7f0e")
    ax.bar(x + width/2, [32, 35, 35], width, label="Paper (approx peak)", color="#1f77b4", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=10)
    ax.set_ylabel("Top-1 Move-Matching Accuracy (%)", fontsize=11)
    ax.set_title("Our Results vs. Maia Paper", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate
    for i, (o, pl, ph) in enumerate(zip(ours, paper_low, paper_high)):
        ax.annotate(f"Ours: {o}%", (i - width/2, o + 1), ha="center", fontsize=8, fontweight="bold")
        ax.annotate(f"Paper: {pl}-{ph}%", (i + width/2, ph + 1), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("reports/paper_comparison.png", dpi=150)
    print("Saved: reports/paper_comparison.png")


if __name__ == "__main__":
    models = load_data()
    fig2_accuracy_curves(models)
    fig6_agreement_matrix()
    plot_paper_comparison()
    print("\nTable:\n", generate_table())
