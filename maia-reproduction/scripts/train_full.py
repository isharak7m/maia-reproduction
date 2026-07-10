"""Full-scale Maia training: 256ch, 15 blocks, 8 history planes."""

import json, logging, sys, time, os
from pathlib import Path
from collections import defaultdict

import torch
import gc
torch.backends.cudnn.enabled = False  # RTX 2050 cuDNN compat issue

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.models.maia_net import MaiaNet
from src.encoding.board import board_to_tensor_with_history
from src.encoding.move import move_to_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train_full")

CHANNELS = 256
BLOCKS = 15
HISTORY = 8
IN_CHANNELS = 17 + 12 * HISTORY
BATCH_SIZE = 8
GRAD_ACCUM = 8
EFFECTIVE_BS = BATCH_SIZE * GRAD_ACCUM
TOTAL_STEPS = 15000
LR = 0.01
LR_DECAY_STEPS = [5000, 10000, 14000]
LR_DECAY = 0.1
WEIGHT_DECAY = 1e-4
CHECKPOINT_DIR = "checkpoints"
# Retry CUDA detection (OS can block GPU after crash)
def _detect_device():
    for attempt in range(3):
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "cuda"
        except RuntimeError:
            pass
        if attempt < 2:
            import subprocess, time
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            except:
                pass
            time.sleep(2)
    return "cpu"

DEVICE = _detect_device()


def group_by_game(records: list):
    """Group records by game_id, sort by ply."""
    groups = defaultdict(list)
    for rec in records:
        groups[rec["game_id"]].append(rec)
    result = []
    for gid, recs in groups.items():
        recs.sort(key=lambda r: r["ply"])
        result.append(recs)
    return result


class FullMoveDataset(Dataset):
    """Dataset with history planes from grouped games."""
    def __init__(self, games: list):
        self.items = []
        for game in games:
            for i, rec in enumerate(game):
                start = max(0, i - HISTORY)
                # Most recent first (reverse chronological)
                history_fens = [g["fen"] for g in reversed(game[start:i])]
                self.items.append((rec["fen"], history_fens, rec["move_uci"], rec["ply"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import chess
        fen, hist_fens, move_uci, ply = self.items[idx]
        board = chess.Board(fen)
        hist_boards = [chess.Board(f) for f in hist_fens]
        tensor = board_to_tensor_with_history(board, hist_boards, num_history=HISTORY)
        move_idx = move_to_index(chess.Move.from_uci(move_uci))
        return torch.from_numpy(tensor).float(), move_idx


def load_split_games(records: list, val_frac=0.02):
    """Group records into games, split by game."""
    import random
    random.seed(42)
    games = group_by_game(records)
    random.shuffle(games)
    n_val = max(1, int(len(games) * val_frac))
    val_games = games[:n_val]
    train_games = games[n_val:]
    return train_games, val_games


def train_bin(bin_lower: int, records: list):
    logger.info(f"=== Full-scale Maia-{bin_lower} ===")
    logger.info(f"  Records: {len(records):,}")
    logger.info(f"  CH={CHANNELS}, BL={BLOCKS}, HIST={HISTORY}, IN={IN_CHANNELS}")
    logger.info(f"  Batch={BATCH_SIZE}, Accum={GRAD_ACCUM}, Eff={EFFECTIVE_BS}")
    logger.info(f"  Steps={TOTAL_STEPS}, Device={DEVICE}")

    train_games, val_games = load_split_games(records)
    train_ds = FullMoveDataset(train_games)
    val_ds = FullMoveDataset(val_games[:max(1, len(val_games)//8)])
    logger.info(f"  Train items: {len(train_ds):,}, Val items: {len(val_ds):,}")

    model = MaiaNet(in_channels=IN_CHANNELS, channels=CHANNELS, blocks=BLOCKS)
    model = model.to(DEVICE)
    logger.info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    start_step = 0
    step = 0
    epoch = 0
    losses = []
    val_losses = []
    start = time.time()

    # Resume from snapshot if available
    resume_ckpt = Path(f"{CHECKPOINT_DIR}/maia_full_{bin_lower}_snap.pt")
    if resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location=DEVICE)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        best_val_loss = state["best_val_loss"]
        losses = state.get("losses", [])
        val_losses = state.get("val_losses", [])
        epoch = state.get("epoch", 0)
        start = time.time() - state.get("elapsed", 0)
        logger.info(f"Resumed from step {step} (best val: {best_val_loss:.4f})")

    while step < TOTAL_STEPS:
        epoch += 1
        model.train()
        for boards, moves in train_loader:
            if step >= TOTAL_STEPS:
                break
            boards = boards.permute(0, 3, 1, 2).to(DEVICE)
            moves = moves.to(DEVICE)

            try:
                policy, _ = model(boards)
                loss = criterion(policy, moves) / GRAD_ACCUM
                loss.backward()
            except RuntimeError as e:
                err_str = str(e)
                if "cuDNN" in err_str or "CUDNN" in err_str:
                    logger.warning(f"  cuDNN crash at step {step}, disabling cuDNN and retrying")
                    torch.backends.cudnn.enabled = False
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Retry this step without cuDNN
                    optimizer.zero_grad()
                    try:
                        policy, _ = model(boards)
                        loss = criterion(policy, moves) / GRAD_ACCUM
                        loss.backward()
                    except RuntimeError as e2:
                        logger.warning(f"  Second attempt also failed: {e2}, skipping batch")
                        optimizer.zero_grad()
                        step += 1
                        continue
                else:
                    logger.warning(f"  Step {step} failed: {e}, skipping")
                    optimizer.zero_grad()
                    step += 1
                    continue

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

            if step in LR_DECAY_STEPS:
                for pg in optimizer.param_groups:
                    pg["lr"] *= LR_DECAY
                logger.info(f"  Step {step}: LR -> {optimizer.param_groups[0]['lr']:.6f}")

            losses.append(loss.item() * GRAD_ACCUM)

            if step % 500 == 0:
                elapsed = time.time() - start
                avg_loss = np.mean(losses[-500:]) if losses else 0
                steps_per_sec = (step + 1) / max(1, elapsed)
                remaining = (TOTAL_STEPS - step - 1) / max(0.1, steps_per_sec)
                logger.info(
                    f"  Step {step}/{TOTAL_STEPS} | loss: {avg_loss:.4f} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"{steps_per_sec:.2f} step/s | est {remaining/3600:.1f}h remain | epoch {epoch}"
                )

            if step % 1000 == 0 and step > 0:
                torch.cuda.empty_cache()
                gc.collect()

            if step % 2000 == 0 and step > 0:
                # Save snapshot before potentially expensive val run
                _save_snapshot(model, optimizer, step, best_val_loss, losses, val_losses, epoch, start, bin_lower)
                model.eval()
                v_losses = []
                with torch.no_grad():
                    for vb, vm in val_loader:
                        vb = vb.permute(0, 3, 1, 2).to(DEVICE)
                        vm = vm.to(DEVICE)
                        vp, _ = model(vb)
                        v_losses.append(criterion(vp, vm).item())
                avg_v = np.mean(v_losses)
                val_losses.append((step, avg_v))
                logger.info(f"  --- Val loss: {avg_v:.4f} (best: {best_val_loss:.4f})")
                if avg_v < best_val_loss:
                    best_val_loss = avg_v
                    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/maia_full_{bin_lower}_best.pt")
                model.train()

            if step % 2500 == 0 and step > 0:
                _save_snapshot(model, optimizer, step, best_val_loss, losses, val_losses, epoch, start, bin_lower)

            step += 1

    total_time = time.time() - start
    logger.info(f"Done: {total_time/3600:.1f}h, best val: {best_val_loss:.4f}")
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/maia_full_{bin_lower}_final.pt")
    return model, losses, val_losses


def _save_snapshot(model, optimizer, step, best_val_loss, losses, val_losses, epoch, start, bin_lower):
    snap = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "losses": losses[-5000:],
        "val_losses": val_losses,
        "epoch": epoch,
        "elapsed": time.time() - start,
    }
    p = f"{CHECKPOINT_DIR}/maia_full_{bin_lower}_snap.pt"
    torch.save(snap, p)
    logger.info(f"  Snapshot saved at step {step}")


def plot_loss_curve(losses, val_losses, bin_lower):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses, alpha=0.3, color="blue", label="train")
    window = 500
    if len(losses) > window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax1.plot(range(window-1, len(losses)), smoothed, color="blue", linewidth=2)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Full Maia-{bin_lower} Training")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    if val_losses:
        steps, vals = zip(*val_losses)
        ax2.plot(steps, vals, "ro-", markersize=4)
        ax2.set_xlabel("Step"); ax2.set_ylabel("Val Loss")
        ax2.set_title("Validation Loss"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"reports/loss_full_{bin_lower}.png", dpi=150)
    logger.info(f"Loss curve saved")


if __name__ == "__main__":
    bin_lower = int(sys.argv[1]) if len(sys.argv) > 1 else 1100
    trimmed = Path(f"data/parquet/records_{bin_lower}_trimmed.json")
    default = Path(f"data/parquet/records_{bin_lower}.json")
    path = trimmed if trimmed.exists() else default

    logger.info(f"Loading {path}")
    with open(path) as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records):,} records")

    model, losses, val_losses = train_bin(bin_lower, records)
    plot_loss_curve(losses, val_losses, bin_lower)
