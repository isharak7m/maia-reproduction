"""Evaluate full-scale models using random game-positions with proper history."""
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
NUM_POSITIONS = 1000
_CACHE = {}


def get_bin_data(bin_lower: int):
    """Load and cache game-grouped data for a bin."""
    if bin_lower in _CACHE:
        return _CACHE[bin_lower]
    trimmed = Path(f"data/parquet/records_{bin_lower}_trimmed.json")
    default = Path(f"data/parquet/records_{bin_lower}.json")
    path = trimmed if trimmed.exists() else default
    with open(path) as f:
        records = json.load(f)
    games = defaultdict(list)
    for r in records:
        games[r["game_id"]].append(r)
    for gid in games:
        games[gid].sort(key=lambda x: x["ply"])
    valid = {gid: g for gid, g in games.items() if len(g) > HISTORY}
    _CACHE[bin_lower] = valid
    return valid


def load_random_positions(bin_lower: int, n: int = NUM_POSITIONS):
    games = get_bin_data(bin_lower)
    game_ids = list(games.keys())
    random.seed(42 + bin_lower)
    items = []
    while len(items) < n and game_ids:
        gid = random.choice(game_ids)
        g = games[gid]
        idx = random.randint(HISTORY, len(g) - 1)
        start = idx - HISTORY
        items.append({
            "fen": g[idx]["fen"],
            "hist_fens": [g[j]["fen"] for j in range(start, idx)],
            "actual_uci": g[idx]["move_uci"],
        })
    return items


def compute_accuracy(model, items):
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
        items = load_random_positions(b, NUM_POSITIONS)
        n = len(items)
        print(f"Bin {b}: {n} random positions across {len(get_bin_data(b))} games")

        ckpt = f"checkpoints/maia_full_{b}_best.pt"
        model = MaiaNet(in_channels=IN_CH, channels=256, blocks=15)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model = model.to(DEVICE)

        correct = compute_accuracy(model, items)
        acc = correct / n * 100
        print(f"  Maia-{b}: {correct}/{n} = {acc:.1f}%")
        results[str(b)] = {"full_maia": {"correct": correct, "total": n, "accuracy": round(correct/n, 4)}}

        for tb in BINS:
            if tb == b:
                continue
            t_items = load_random_positions(tb, NUM_POSITIONS)
            correct = compute_accuracy(model, t_items)
            acc = correct / len(t_items) * 100
            print(f"  Maia-{b} on bin {tb}: {correct}/{len(t_items)} = {acc:.1f}%")
            results[str(b)][f"full_maia_on_{tb}"] = {
                "correct": correct, "total": len(t_items),
                "accuracy": round(correct/len(t_items), 4)}

    with open("reports/full_model_results2.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS (self-bin bold) ===")
    for b in BINS:
        sb = str(b)
        r = results[sb]
        self_acc = r["full_maia"]["accuracy"] * 100
        accs = [self_acc]
        for tb in BINS:
            if tb != b:
                accs.append(r.get(f"full_maia_on_{tb}", {}).get("accuracy", 0) * 100)
        is_peak = all(self_acc >= a for a in accs)
        prefix = "V" if is_peak else "X"
        print(f"  {prefix} Maia-{b}: self={self_acc:.1f}%  |  cross: {' | '.join(f'{a:.1f}%' for a in accs[1:])}")


if __name__ == "__main__":
    main()
