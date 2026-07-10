"""Compute Stockfish move-matching accuracy at depths 1, 7, 15."""

import chess
import chess.engine
import json
import random
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.maia_net import MaiaNet
from src.encoding.board import board_to_tensor
from src.encoding.move import move_to_index

STOCKFISH_PATH = "stockfish.exe"
DEPTHS = [1, 7, 15]
NUM_POSITIONS = 500
BINS = [1100, 1500, 1900]
RANDOM_SEED = 42
OUTPUT = "reports/all_baselines.json"


def load_positions(bin_lower: int, n: int):
    """Load random positions from a bin's records."""
    trimmed = Path(f"data/parquet/records_{bin_lower}_trimmed.json")
    default = Path(f"data/parquet/records_{bin_lower}.json")
    path = trimmed if trimmed.exists() else default
    with open(path) as f:
        records = json.load(f)
    random.seed(RANDOM_SEED + bin_lower)
    sampled = random.sample(records, min(n, len(records)))
    return [(rec["fen"], rec["move_uci"]) for rec in sampled]


def evaluate_sf(engine, fens_and_moves: list, depth: int) -> dict:
    """Compute Stockfish accuracy at given depth."""
    correct = 0
    for fen, actual_uci in fens_and_moves:
        board = chess.Board(fen)
        try:
            result = engine.play(board, chess.engine.Limit(depth=depth))
            sf_uci = result.move.uci() if result.move else ""
        except Exception:
            sf_uci = ""
        if sf_uci == actual_uci:
            correct += 1
    return {"depth": depth, "correct": correct, "total": len(fens_and_moves),
            "accuracy": round(correct / len(fens_and_moves), 4)}


def evaluate_maia(model_path: str, fens_and_moves: list, device: str) -> dict:
    """Compute Maia accuracy on positions."""
    model = MaiaNet(in_channels=17, channels=32, blocks=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    correct = 0
    with torch.no_grad():
        for fen, actual_uci in fens_and_moves:
            board = chess.Board(fen)
            tensor = board_to_tensor(board)
            x = torch.from_numpy(tensor).float().permute(2, 0, 1).unsqueeze(0).to(device)
            policy, _ = model(x)
            probs = torch.softmax(policy, dim=1).squeeze(0)

            legal_mask = torch.zeros(64 * 73, dtype=torch.bool, device=device)
            for move in board.legal_moves:
                try:
                    legal_mask[move_to_index(move)] = True
                except ValueError:
                    continue
            probs_masked = probs.clone()
            probs_masked[~legal_mask] = 0
            best_idx = torch.argmax(probs_masked).item()

            for move in board.legal_moves:
                try:
                    if move_to_index(move) == best_idx:
                        if move.uci() == actual_uci:
                            correct += 1
                        break
                except ValueError:
                    continue

    return {"correct": correct, "total": len(fens_and_moves),
            "accuracy": round(correct / len(fens_and_moves), 4)}


def main():
    print(f"Loading positions ({NUM_POSITIONS} per bin)...")
    positions = {}
    for b in BINS:
        positions[b] = load_positions(b, NUM_POSITIONS)
        print(f"  Bin {b}: {len(positions[b])} positions")

    print(f"\nStarting Stockfish ({STOCKFISH_PATH})...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    all_results = {}
    try:
        for b in BINS:
            all_results[b] = {}
            for d in DEPTHS:
                t0 = time.time()
                print(f"\nSF bin {b} @ depth {d}...", end=" ", flush=True)
                res = evaluate_sf(engine, positions[b], d)
                elapsed = time.time() - t0
                print(f"{res['accuracy']*100:.1f}% ({res['correct']}/{res['total']}, {elapsed:.0f}s)")
                all_results[b][f"sf_depth_{d}"] = res
    finally:
        engine.quit()

    # Maia evaluations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for b in BINS:
        ckpt = f"checkpoints/maia_{b}_best.pt"
        if not Path(ckpt).exists():
            print(f"\nMaia {b}: no checkpoint found, skipping")
            continue
        t0 = time.time()
        print(f"\nMaia {b}...", end=" ", flush=True)
        res = evaluate_maia(ckpt, positions[b], device)
        elapsed = time.time() - t0
        print(f"{res['accuracy']*100:.1f}% ({res['correct']}/{res['total']}, {elapsed:.0f}s)")
        all_results[b]["maia"] = res

    # Also compute cross-bin Maia accuracies
    for b in BINS:
        ckpt = f"checkpoints/maia_{b}_best.pt"
        if not Path(ckpt).exists():
            continue
        for other_b in BINS:
            if other_b == b:
                continue
            t0 = time.time()
            print(f"\nMaia {b} on bin {other_b}...", end=" ", flush=True)
            res = evaluate_maia(ckpt, positions[other_b], device)
            elapsed = time.time() - t0
            print(f"{res['accuracy']*100:.1f}% ({res['correct']}/{res['total']}, {elapsed:.0f}s)")
            all_results[b][f"maia_on_bin_{other_b}"] = res

    # Save
    with open(OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")

    # Summary table
    print("\n" + "=" * 70)
    print("MOVE-MATCHING ACCURACY COMPARISON")
    print("=" * 70)
    models = ["sf_depth_1", "sf_depth_7", "sf_depth_15", "maia"]
    col_w = max(len(m) for m in models) + 2
    header = f"{'Eval on >':>10} |" + "".join(f" {b:>{col_w-1}}" for b in BINS)
    print(header)
    print("-" * len(header))
    for b in BINS:
        for m in models:
            if m in all_results.get(b, {}):
                r = all_results[b][m]
                row = f"{m:>10} | {r['accuracy']*100:5.1f}%"
                # Cross-bin rows
                if m == "maia":
                    for other_b in BINS:
                        key = f"maia_on_bin_{other_b}"
                        if key in all_results.get(b, {}):
                            r2 = all_results[b][key]
                            row += f" {r2['accuracy']*100:5.1f}%"
                print(row)
    print("=" * 70)

    # Generate the agreement plot
    print("\nGenerating agreement matrix plot...")
    _generate_plot(all_results)


def _generate_plot(results: dict):
    """Create the 3x3 agreement matrix plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["SF d=1", "SF d=7", "SF d=15", "Maia-1100", "Maia-1500", "Maia-1900"]
    n = len(labels)
    matrix = np.zeros((n, n))

    # Fill Stockfish rows
    for i, b in enumerate(BINS):
        for j, d in enumerate(DEPTHS):
            val = results[b].get(f"sf_depth_{d}", {}).get("accuracy", 0)
            matrix[j][i] = val * 100

    # Fill Maia rows (diagonal = same bin, off-diagonal = cross)
    for i, b in enumerate(BINS):
        for j, target_b in enumerate(BINS):
            if b == target_b:
                val = results[b].get("maia", {}).get("accuracy", 0)
            else:
                val = results[b].get(f"maia_on_bin_{target_b}", {}).get("accuracy", 0)
            matrix[3 + i][j] = val * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=50, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_yticks(range(6))
    ax.set_xticklabels([f"{b}-{b+99}" for b in [1100, 1500, 1900]], fontsize=10)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="black" if matrix[i, j] > 20 else "white")

    ax.set_xlabel("Rating bin of human opponent", fontsize=11)
    ax.set_title("Move-Matching Accuracy (%)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    out = "reports/agreement_matrix.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
