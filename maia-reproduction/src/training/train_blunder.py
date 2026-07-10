"""Training loop for blunder prediction models."""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models.blunder_fc import BlunderFC
from src.models.blunder_rescnn import BlunderResCNN, DeepBlunderResCNN
from src.data_pipeline.dataset import create_blunder_dataloader

logger = logging.getLogger(__name__)


def train_blunder_model(
    model: nn.Module,
    train_records: list[dict],
    val_records: list[dict],
    model_name: str,
    config: dict | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "checkpoints",
    include_metadata: bool = False,
    debug: bool = False,
) -> nn.Module:
    """Train a blunder prediction model.

    Args:
        model: Blunder model (FC or ResCNN).
        train_records: Training records with is_blunder labels.
        val_records: Validation records.
        model_name: Name for checkpoint file.
        config: Training config with keys:
            - batch_size (default 2000)
            - total_steps (default 1400000)
            - lr (default 0.002 for FC, 0.0002 for ResCNN)
            - lr_decay_steps (default [20000, 1000000, 1300000])
            - lr_decay_factor (default 0.1)
        include_metadata: Whether model uses board+metadata input.
        debug: If True, use minimal data.

    Returns trained model.
    """
    if config is None:
        config = {}

    # Determine if FC or ResCNN for default LR
    is_fc = isinstance(model, BlunderFC)
    default_lr = config.get("lr", 0.002 if is_fc else 0.0002)

    batch_size = config.get("batch_size", 2000)
    total_steps = config.get("total_steps", 1_400_000)
    lr = default_lr
    lr_decay_steps = config.get("lr_decay_steps", [20000, 1_000_000, 1_300_000])
    lr_decay_factor = config.get("lr_decay_factor", 0.1)

    if debug:
        total_steps = min(total_steps, 2000)
        train_records = train_records[:max(batch_size * 5, len(train_records) // 100)]
        val_records = val_records[:max(batch_size * 2, len(val_records) // 100)]

    in_channels = 22 if include_metadata else 17

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = create_blunder_dataloader(
        train_records,
        batch_size=batch_size,
        balanced=True,
        include_metadata=include_metadata,
        shuffle=True,
    )
    val_loader = create_blunder_dataloader(
        val_records,
        batch_size=batch_size,
        balanced=True,
        include_metadata=include_metadata,
        shuffle=False,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_val_auc = 0.0
    start_time = time.time()

    model.train()
    for batch_idx, (boards, labels) in enumerate(train_loader):
        if batch_idx >= total_steps:
            break

        # Adjust learning rate
        if batch_idx in lr_decay_steps:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay_factor
            logger.info(f"Step {batch_idx}: LR -> {optimizer.param_groups[0]['lr']:.6f}")

        boards = boards.permute(0, 3, 1, 2).to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(boards)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        step = batch_idx + 1

        if step % 200 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"[{model_name}] Step {step}/{total_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {elapsed:.0f}s"
            )

        # Periodic validation
        if step % 2000 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for v_boards, v_labels in val_loader:
                    v_boards = v_boards.permute(0, 3, 1, 2).to(device)
                    v_labels = v_labels.to(device).view(-1, 1)
                    v_out = model(v_boards)
                    all_preds.append(v_out.cpu())
                    all_labels.append(v_labels.cpu())

            if all_preds:
                all_preds = torch.cat(all_preds).numpy()
                all_labels = torch.cat(all_labels).numpy()
                val_auc = roc_auc_score(all_labels, all_preds)
                val_acc = accuracy_score(all_labels, (all_preds > 0.5).astype(float))

                logger.info(
                    f"[{model_name}] Validation | Step {step} | "
                    f"AUC: {val_auc:.4f} | Acc: {val_acc:.4f}"
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    ckpt_path = output_dir / f"{model_name}_best.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info(f"Saved best checkpoint: {ckpt_path}")

            model.train()

    # Save final
    ckpt_path = output_dir / f"{model_name}_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved final checkpoint: {ckpt_path}")

    total_time = time.time() - start_time
    logger.info(
        f"[{model_name}] Training complete: {total_time:.0f}s, {step} steps, "
        f"best val AUC: {best_val_auc:.4f}"
    )

    return model
