"""Show sample predictions from trained model."""
import sys, json, torch, chess
sys.path.insert(0, ".")

from src.models.maia_net import MaiaNet
from src.encoding.board import board_to_tensor
from src.encoding.move import move_to_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = MaiaNet(in_channels=17, channels=32, blocks=6)
model.load_state_dict(torch.load("checkpoints/maia_1100_best.pt"))
model = model.to(DEVICE).eval()

with open("data/parquet/records_1100.json") as f:
    records = json.load(f)

val = records[1:5001]
print("=" * 60)
print("SAMPLE PREDICTIONS (10 validation positions)")
print("=" * 60)

with torch.no_grad():
    for i in range(10):
        rec = val[i]
        board = chess.Board(rec["fen"])
        tensor = board_to_tensor(board)
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

        pred_uci = "?"
        for move in board.legal_moves:
            try:
                if move_to_index(move) == best_idx:
                    pred_uci = move.uci()
                    break
            except ValueError:
                continue

        actual_uci = rec["move_uci"]
        ok = pred_uci == actual_uci
        print(f"  {i+1}. Pred: {pred_uci:6s} | Actual: {actual_uci:6s} | {'OK' if ok else 'NO'}")
        print(f"     Fen: {board.fen()[:50]}")

    print()
    # Accuracy on 500
    correct = 0
    for rec in val[:500]:
        board = chess.Board(rec["fen"])
        tensor = board_to_tensor(board)
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
                    if move.uci() == rec["move_uci"]:
                        correct += 1
                    break
            except ValueError:
                continue

    print(f"  Move-matching accuracy (first 500): {correct}/500 = {correct/5:.1f}%")
    print("=" * 60)
