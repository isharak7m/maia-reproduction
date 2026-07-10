"""Train a single Maia bin model with 4GB-GPU-friendly config."""

import json, logging, sys, time, os
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.models.maia_net import MaiaNet
from src.encoding.board import board_to_tensor
from src.encoding.move import move_to_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train_bin")

BATCH_SIZE = 64
GRAD_ACCUM = 2
EFFECTIVE_BS = BATCH_SIZE * GRAD_ACCUM
TOTAL_STEPS = 30000
CHANNELS = 32
BLOCKS = 6
LR = 0.1
LR_DECAY_STEPS = [6000, 15000, 27000]
LR_DECAY = 0.1
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MoveDataset(Dataset):
    """Simple move-matching dataset from JSON records."""
    def __init__(self, records, limit=None):
        self.records = records if limit is None else records[:limit]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        board = __import__("chess").Board(rec["fen"])
        board_t = board_to_tensor(board)  # (8,8,17)
        move_idx = move_to_index(__import__("chess").Move.from_uci(rec["move_uci"]))
        return torch.from_numpy(board_t).float(), move_idx


def load_split(records: list, val_frac=0.02):
    """Split records into train/val."""
    n_val = max(1, int(len(records) * val_frac))
    train = records[n_val:]
    val = records[:n_val]
    return train, val


def train_bin(bin_lower: int, records: list):
    logger.info(f"=== Training Maia-{bin_lower} ===")
    logger.info(f"  Records: {len(records):,}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Batch: {BATCH_SIZE} (eff {EFFECTIVE_BS}), Steps: {TOTAL_STEPS}")
    logger.info(f"  Channels: {CHANNELS}, Blocks: {BLOCKS}")

    train_rec, val_rec = load_split(records)

    model = MaiaNet(in_channels=17, channels=CHANNELS, blocks=BLOCKS)
    model = model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_ds = MoveDataset(train_rec)
    # Smaller validation
    val_ds = MoveDataset(val_rec[:min(5000, len(val_rec))])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    step = 0
    epoch = 0
    losses = []
    val_losses = []
    start = time.time()

    while step < TOTAL_STEPS:
        epoch += 1
        model.train()
        epoch_losses = []
        for batch_boards, batch_moves in train_loader:
            if step >= TOTAL_STEPS:
                break

            batch_boards = batch_boards.permute(0, 3, 1, 2).to(DEVICE)
            batch_moves = batch_moves.to(DEVICE)

            policy_logits, _ = model(batch_boards)
            loss = criterion(policy_logits, batch_moves)
            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

            # LR decay
            if step in LR_DECAY_STEPS:
                for pg in optimizer.param_groups:
                    pg["lr"] *= LR_DECAY
                logger.info(f"  Step {step}: LR -> {optimizer.param_groups[0]['lr']:.6f}")

            losses.append(loss.item() * GRAD_ACCUM)

            if step % 500 == 0:
                elapsed = time.time() - start
                avg_loss = np.mean(losses[-500:]) if losses else 0
                logger.info(
                    f"  Step {step}/{TOTAL_STEPS} | loss: {avg_loss:.4f} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"{elapsed:.0f}s | epoch {epoch}"
                )

            # Validation
            if step % 2000 == 0 and step > 0:
                model.eval()
                v_losses = []
                with torch.no_grad():
                    for vb, vm in val_loader:
                        vb = vb.permute(0, 3, 1, 2).to(DEVICE)
                        vm = vm.to(DEVICE)
                        vp, _ = model(vb)
                        vloss = criterion(vp, vm)
                        v_losses.append(vloss.item())
                avg_v = np.mean(v_losses)
                val_losses.append((step, avg_v))
                logger.info(f"  --- Val loss: {avg_v:.4f} (best: {best_val_loss:.4f})")
                if avg_v < best_val_loss:
                    best_val_loss = avg_v
                    ckpt = f"{CHECKPOINT_DIR}/maia_{bin_lower}_best.pt"
                    torch.save(model.state_dict(), ckpt)

            step += 1

    total_time = time.time() - start
    logger.info(f"Training complete: {total_time:.0f}s, best val loss: {best_val_loss:.4f}")

    # Save final
    ckpt = f"{CHECKPOINT_DIR}/maia_{bin_lower}_final.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info(f"Saved final: {ckpt}")

    return model, losses, val_losses


def show_sample_predictions(model, val_records, n=10):
    """Show sample predicted vs actual moves."""
    import chess
    model.eval()
    print(f"\n{'='*60}")
    print(f"SAMPLE PREDICTIONS (first {n} validation positions)")
    print(f"{'='*60}")
    matches = 0
    with torch.no_grad():
        for i in range(n):
            rec = val_records[i]
            board = chess.Board(rec["fen"])
            tensor = board_to_tensor(board)
            x = torch.from_numpy(tensor).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            policy, _ = model(x)
            probs = torch.softmax(policy, dim=1).squeeze(0)

            # Mask illegal moves
            legal_mask = torch.zeros(64 * 73, dtype=torch.bool, device=DEVICE)
            for move in board.legal_moves:
                try:
                    idx = move_to_index(move)
                    legal_mask[idx] = True
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
            match = "[OK]" if pred_uci == actual_uci else "[MISMATCH]"
            if pred_uci == actual_uci:
                matches += 1
            print(f"  {i+1}. Fen: {board.fen()[:40]}...")
            print(f"     Predicted: {pred_uci} | Actual: {actual_uci} {match}")

    print(f"\n  Accuracy on sample: {matches}/{n} = {100*matches/n:.1f}%")
    print(f"{'='*60}\n")


def plot_loss_curve(losses, val_losses, bin_lower):
    """Plot and save loss curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses, alpha=0.3, color="blue", label="train")
    # Smoothed
    window = 500
    if len(losses) > window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax1.plot(range(window-1, len(losses)), smoothed, color="blue", linewidth=2, label=f"smoothed ({window})")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss"); ax1.set_title(f"Maia-{bin_lower} Training Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    if val_losses:
        steps, vals = zip(*val_losses)
        ax2.plot(steps, vals, "ro-", markersize=4, label="val")
        ax2.set_xlabel("Step"); ax2.set_ylabel("Val Loss")
        ax2.set_title(f"Maia-{bin_lower} Validation Loss")
        ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"reports/loss_curve_{bin_lower}.png"
    plt.savefig(out, dpi=150)
    logger.info(f"Saved loss curve: {out}")
    return fig


if __name__ == "__main__":
    bin_lower = int(sys.argv[1]) if len(sys.argv) > 1 else 1100

    # Try trimmed file first, then fall back to default
    trimmed = Path(f"data/parquet/records_{bin_lower}_trimmed.json")
    default = Path(f"data/parquet/records_{bin_lower}.json")
    records_file = trimmed if trimmed.exists() else default
    logger.info(f"Loading {records_file}")
    with open(records_file) as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records):,} records")

    model, losses, val_losses = train_bin(bin_lower, records)
    plot_loss_curve(losses, val_losses, bin_lower)

    _, val_rec = load_split(records)
    show_sample_predictions(model, val_rec, n=10)
