"""Show 1500 bin predictions and accuracy."""
import sys, json, torch, chess
sys.path.insert(0, ".")

from src.models.maia_net import MaiaNet
from src.encoding.board import board_to_tensor
from src.encoding.move import move_to_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

model = MaiaNet(in_channels=17, channels=32, blocks=6)
model.load_state_dict(torch.load("checkpoints/maia_1500_best.pt"))
model = model.to(DEVICE).eval()

trimmed = "data/parquet/records_1500_trimmed.json"
default = "data/parquet/records_1500.json"
path = trimmed if __import__("os").path.exists(trimmed) else default
with open(path) as f:
    records = json.load(f)

print(f"Records: {len(records):,}")
val = records[1:5001]

print("=" * 60)
print("SAMPLE PREDICTIONS (1500 bin, 10 positions)")
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

    print(f"\n  Move-matching accuracy (first 500): {correct}/500 = {correct/5:.1f}%")
    print("=" * 60)
