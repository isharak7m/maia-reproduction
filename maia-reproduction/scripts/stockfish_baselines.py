"""Compute Stockfish move-matching accuracy at depths 1, 7, 15."""

import chess
import chess.engine
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

STOCKFISH_PATH = "stockfish.exe"
DEPTHS = [1, 7, 15]
NUM_POSITIONS = 500
BINS = [1100, 1500, 1900]
RANDOM_SEED = 42
OUTPUT = "reports/stockfish_results.json"


def load_positions(bin_lower: int, n: int):
    trimmed = Path(f"data/parquet/records_{bin_lower}_trimmed.json")
    default = Path(f"data/parquet/records_{bin_lower}.json")
    path = trimmed if trimmed.exists() else default
    with open(path) as f:
        records = json.load(f)
    random.seed(RANDOM_SEED + bin_lower)
    sampled = random.sample(records, min(n, len(records)))
    return [(rec["fen"], rec["move_uci"]) for rec in sampled]


def evaluate_sf(engine, fens_and_moves: list, depth: int) -> dict:
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
                all_results[b][str(d)] = res
    finally:
        engine.quit()

    with open(OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")

    print("\n" + "=" * 50)
    print("STOCKFISH MOVE-MATCHING ACCURACY")
    print("=" * 50)
    print(f"{'Bin':>6} | {'d=1':>8} {'d=7':>8} {'d=15':>8}")
    print("-" * 35)
    for b in BINS:
        vals = [all_results[b].get(str(d), {}).get("accuracy", 0) * 100 for d in DEPTHS]
        print(f"{b:>6} | {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
