"""Evaluate full-scale models on same 500-position sets as Stockfish."""
import json, sys, torch, chess, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.maia_net import MaiaNet
from src.encoding.board import board_to_tensor_with_history
from src.encoding.move import move_to_index

BINS = [1100, 1500, 1900]
HISTORY = 8
IN_CH = 17 + 12 * HISTORY
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_POS = 500
OUTPUT = "reports/full_model_results.json"


def load_test_positions(bin_lower: int, n=500):
    """Load random positions with game context for history."""
    trimmed = Path(f"data/parquet/records_{bin_lower}_trimmed.json")
    default = Path(f"data/parquet/records_{bin_lower}.json")
    path = trimmed if trimmed.exists() else default
    with open(path) as f:
        records = json.load(f)
    random.seed(42 + bin_lower)
    return random.sample(records, min(n, len(records)))


def build_history_lookup(sampled_records):
    """Build a lookup from game_id -> list of (ply, fen)."""
    games = defaultdict(list)
    for rec in sampled_records:
        games[rec["game_id"]].append((rec["ply"], rec["fen"]))
    for gid in games:
        games[gid].sort(key=lambda x: x[0])
    return games


def compute_accuracy(model, sampled_records, history_lookup):
    """Top-1 move-matching accuracy with history context."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for rec in sampled_records:
            gid = rec["game_id"]
            ply = rec["ply"]
            fen = rec["fen"]
            actual_uci = rec["move_uci"]

            # Get history from same game (previous plies)
            game_moves = history_lookup[gid]
            hist_fens = []
            for p, f in game_moves:
                if p >= ply:
                    break
                hist_fens.append(f)
            # Most recent first
            hist_fens = list(reversed(hist_fens))[:HISTORY]

            board = chess.Board(fen)
            hist_boards = [chess.Board(f) for f in hist_fens]
            tensor = board_to_tensor_with_history(board, hist_boards, num_history=HISTORY)

            x = torch.from_numpy(tensor).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            policy, _ = model(x)
            probs = torch.softmax(policy, dim=1).squeeze(0)

            legal_mask = torch.zeros(64 * 73, dtype=torch.bool, device=DEVICE)
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
                    if move_to_index(move) == best_idx and move.uci() == actual_uci:
                        correct += 1
                    break
                except ValueError:
                    continue
            total += 1

    return correct / total if total > 0 else 0


def main():
    results = {}
    for b in BINS:
        samples = load_test_positions(b, NUM_POS)
        history = build_history_lookup(samples)
        print(f"Bin {b}: {len(samples)} positions, {len(history)} games")

        ckpt = f"checkpoints/maia_full_{b}_best.pt"
        if not Path(ckpt).exists():
            print(f"  No checkpoint for {b}, skipping")
            continue

        model = MaiaNet(in_channels=IN_CH, channels=256, blocks=15)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model = model.to(DEVICE)

        acc = compute_accuracy(model, samples, history)
        print(f"  Full Maia-{b}: {acc*100:.1f}%")
        results[str(b)] = {"full_maia": {"accuracy": round(acc, 4)}}

        # Cross-bin evaluation
        for target_b in BINS:
            if target_b == b:
                continue
            target_samples = load_test_positions(target_b, NUM_POS)
            target_history = build_history_lookup(target_samples)
            acc = compute_accuracy(model, target_samples, target_history)
            print(f"  Full Maia-{b} on bin {target_b}: {acc*100:.1f}%")
            results[str(b)][f"full_maia_on_{target_b}"] = {"accuracy": round(acc, 4)}

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")

    # Combined summary
    print("\n=== FULL-SCALE MOVE-MATCHING ACCURACY ===")
    for b in BINS:
        sb = str(b)
        if sb in results and "full_maia" in results[sb]:
            print(f"  Maia-{b}: {results[sb]['full_maia']['accuracy']*100:.1f}%")
            for tb in BINS:
                key = f"full_maia_on_{tb}"
                if key in results[sb]:
                    print(f"    on {tb}: {results[sb][key]['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
