"""Evaluate full-scale models using consecutive game positions (proper history)."""
import json, sys, torch, chess
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


def load_consecutive_games(bin_lower: int, max_positions=500):
    """Load consecutive positions from the start of the JSON (same game context preserved)."""
    trimmed = Path(f"data/parquet/records_{bin_lower}_trimmed.json")
    default = Path(f"data/parquet/records_{bin_lower}.json")
    path = trimmed if trimmed.exists() else default
    with open(path) as f:
        records = json.load(f)

    # Group by game_id, sort by ply
    games = defaultdict(list)
    for r in records[:5000]:  # scan first 5000 records to find games
        games[r["game_id"]].append(r)
    for gid in games:
        games[gid].sort(key=lambda x: x["ply"])

    # Build test items with history from consecutive game moves
    items = []
    for gid, game_moves in games.items():
        for i, rec in enumerate(game_moves):
            if len(items) >= max_positions:
                break
            start = max(0, i - HISTORY)
            hist_fens = [game_moves[j]["fen"] for j in range(start, i)]
            items.append({
                "fen": rec["fen"],
                "hist_fens": hist_fens,
                "actual_uci": rec["move_uci"],
                "ply": rec["ply"],
            })
        if len(items) >= max_positions:
            break

    return items[:max_positions]


def compute_accuracy(model, items):
    """Top-1 move-matching accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for item in items:
            board = chess.Board(item["fen"])
            hist_boards = [chess.Board(f) for f in reversed(item["hist_fens"])]
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
                    if move_to_index(move) == best_idx:
                        if move.uci() == item["actual_uci"]:
                            correct += 1
                        break
                except ValueError:
                    continue

    return correct


def main():
    results = {}
    for b in BINS:
        items = load_consecutive_games(b, 500)
        n = len(items)
        print(f"Bin {b}: {n} positions with history context")

        ckpt = f"checkpoints/maia_full_{b}_best.pt"
        model = MaiaNet(in_channels=IN_CH, channels=256, blocks=15)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model = model.to(DEVICE)

        correct = compute_accuracy(model, items)
        acc = correct / n * 100
        print(f"  Full Maia-{b}: {correct}/{n} = {acc:.1f}%")
        results[str(b)] = {"full_maia": {"correct": correct, "total": n, "accuracy": round(correct/n, 4)}}

        # Cross-bin
        for tb in BINS:
            if tb == b:
                continue
            t_items = load_consecutive_games(tb, 500)
            correct = compute_accuracy(model, t_items)
            acc = correct / len(t_items) * 100
            print(f"  Full Maia-{b} on bin {tb}: {correct}/{len(t_items)} = {acc:.1f}%")
            results[str(b)][f"full_maia_on_{tb}"] = {"correct": correct, "total": len(t_items), "accuracy": round(correct/len(t_items), 4)}

    with open("reports/full_model_results2.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FULL-SCALE MODELS (PROPER HISTORY) ===")
    for b in BINS:
        sb = str(b)
        if sb in results:
            print(f"  Maia-{b}: {results[sb].get('full_maia',{}).get('accuracy',0)*100:.1f}%")
            for tb in BINS:
                key = f"full_maia_on_{tb}"
                if key in results[sb]:
                    print(f"    on {tb}: {results[sb][key]['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
