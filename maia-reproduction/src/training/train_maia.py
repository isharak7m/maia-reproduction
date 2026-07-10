"""Training loop for Maia move-matching models."""

import itertools
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from src.models.maia_net import MaiaNet, MaiaLoss, create_maia_model
from src.data_pipeline.dataset import create_move_dataloader

logger = logging.getLogger(__name__)


def train_maia(
    model: MaiaNet,
    train_records: list[dict],
    val_records: list[dict],
    rating_bin: int,
    config: dict | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "checkpoints",
    debug: bool = False,
    use_history: bool = True,
    num_history: int = 12,
) -> MaiaNet:
    """Train a Maia move-matching model.

    Args:
        model: MaiaNet instance.
        train_records: Training ply records.
        val_records: Validation ply records.
        rating_bin: Rating bin (e.g., 1100), used for naming.
        config: Training config dict with keys:
            - batch_size (default 1024)
            - total_steps (default 400000)
            - lr (default 0.1)
            - lr_decay_steps (default [80000, 200000, 360000])
            - lr_decay_factor (default 0.1)
            - shuffle_buffer (default 250000)
            - subsample_rate (default 1/32)
        device: Device to train on.
        output_dir: Directory for checkpoints.
        debug: If True, use tiny data for quick validation.

    Returns trained model.
    """
    if config is None:
        config = {}

    batch_size = config.get("batch_size", 1024)
    total_steps = config.get("total_steps", 400000)
    lr = config.get("lr", 0.1)
    lr_decay_steps = config.get("lr_decay_steps", [80000, 200000, 360000])
    lr_decay_factor = config.get("lr_decay_factor", 0.1)
    shuffle_buffer = config.get("shuffle_buffer", 250_000)
    subsample_rate = config.get("subsample_rate", 1.0 / 32.0)

    if debug:
        total_steps = min(total_steps, 1000)
        train_records = train_records[:batch_size * 10]

    model = model.to(device)
    criterion = MaiaLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda" if "cuda" in device else "cpu")

    train_loader = create_move_dataloader(
        train_records,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        subsample_rate=subsample_rate,
        use_history=use_history,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute outcome targets from records
    outcome_map = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
    val_outcomes = torch.tensor([
        [1.0, 0.0, 0.0] if r.get("result") == "1-0" else
        [0.0, 1.0, 0.0] if r.get("result") == "1/2-1/2" else
        [0.0, 0.0, 1.0]
        for r in val_records
    ], device=device).float()

    val_boards = []
    val_moves = []
    for rec in val_records[:5000]:  # Limit validation size
        from src.encoding.board import board_to_tensor, board_to_tensor_with_history
        import chess
        from src.encoding.move import move_to_index

        board = chess.Board(rec["fen"])
        if use_history:
            val_boards.append(board_to_tensor_with_history(board, num_history=num_history))
        else:
            val_boards.append(board_to_tensor(board))
        val_moves.append(move_to_index(chess.Move.from_uci(rec["move_uci"])))

    if val_boards:
        val_boards_t = torch.from_numpy(np.stack(val_boards)).float().to(device)
        val_moves_t = torch.tensor(val_moves, device=device)
    else:
        val_boards_t = None
        val_moves_t = None

    step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    model.train()
    for boards, moves in itertools.cycle(train_loader):
        if step >= total_steps:
            break

        # Adjust learning rate
        if step in lr_decay_steps:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay_factor
            logger.info(f"Step {step}: LR -> {optimizer.param_groups[0]['lr']:.6f}")

        boards = boards.permute(0, 3, 1, 2).to(device)  # (B, C, 8, 8)
        moves = moves.to(device)

        # Build outcome targets (simplified: use result from records)
        # Since we don't track per-batch outcomes, we use an approximation
        outcome_targets = torch.zeros((boards.shape[0], 3), device=device)
        outcome_targets[:, 0] = 0.5  # Default: equal probability
        outcome_targets[:, 1] = 0.3
        outcome_targets[:, 2] = 0.2

        optimizer.zero_grad()

        with autocast("cuda" if "cuda" in device else "cpu", enabled="cuda" in device):
            policy_logits, value_probs = model(boards)
            loss = criterion(policy_logits, value_probs, moves, outcome_targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            logger.info(
                f"Bin {rating_bin} | Step {step}/{total_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Steps/s: {steps_per_sec:.1f}"
            )

            # Validation
            if val_boards_t is not None and step % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    v_policy, v_value = model(val_boards_t)
                    v_loss = criterion(v_policy, v_value, val_moves_t,
                                       val_outcomes[:val_moves_t.shape[0]])
                    if v_loss.item() < best_val_loss:
                        best_val_loss = v_loss.item()
                        ckpt_path = output_dir / f"maia_{rating_bin}_best.pt"
                        torch.save(model.state_dict(), ckpt_path)
                        logger.info(f"Saved best checkpoint: {ckpt_path}")
                model.train()

    if step >= total_steps:
        ckpt_path = output_dir / f"maia_{rating_bin}_final.pt"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved final checkpoint: {ckpt_path}")

    total_time = time.time() - start_time
    logger.info(f"Training complete for bin {rating_bin}: {total_time:.1f}s, {step} steps")

    return model


def train_maia_all_bins(
    train_records_by_bin: dict[int, list[dict]],
    val_records_by_bin: dict[int, list[dict]],
    config: dict | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "checkpoints",
    debug: bool = False,
) -> dict[int, MaiaNet]:
    """Train all 9 Maia models (one per rating bin)."""
    models = {}
    for bin_lower in range(1100, 1900, 100):
        logger.info(f"Training Maia-{bin_lower}...")
        model = create_maia_model(bin_lower)
        model = train_maia(
            model,
            train_records_by_bin.get(bin_lower, []),
            val_records_by_bin.get(bin_lower, []),
            rating_bin=bin_lower,
            config=config,
            device=device,
            output_dir=output_dir,
            debug=debug,
        )
        models[bin_lower] = model

    return models
