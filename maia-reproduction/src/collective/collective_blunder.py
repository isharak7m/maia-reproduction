"""Collective blunder prediction.

Predicts whether >10% of people who reach a given exact position will blunder.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from src.encoding.board import board_to_tensor, flip_board_tensor
from src.models.blunder_fc import BlunderFC
from src.models.blunder_rescnn import BlunderResCNN, DeepBlunderResCNN

logger = logging.getLogger(__name__)


def normalize_position(fen: str) -> str:
    """Normalize a FEN by standardizing to White's perspective.

    Flips the board if it's Black's turn so identical positions
    from either color perspective are merged. Also discards
    move count/ply count fields.

    Returns a canonical FEN string.
    """
    board = chess.Board(fen)

    # If it's Black's turn, flip the board
    if board.turn == chess.BLACK:
        board = board.mirror()

    # Return FEN with only position + turn (strip move counters)
    # Keep castling and en passant, strip ply/move counters
    parts = board.fen().split(" ")
    canonical = " ".join(parts[:4]) + " w - 0 1"
    return canonical


def build_collective_dataset(
    records: list[dict],
    min_occurrences: int = 10,
    blunder_threshold: float = 0.10,
    win_prob_converter=None,
) -> tuple[list[dict], list[dict]]:
    """Build collective blunder prediction dataset from individual move records.

    Groups records by normalized position, counts blunder rate,
    and creates position-level labels.

    Args:
        records: Ply records with fen, move_uci, centipawn_eval, result.
        min_occurrences: Minimum times a position must appear.
        blunder_threshold: Win-prob drop threshold for labeling.
        win_prob_converter: CentipawnWinProbability instance.

    Returns:
        (position_records, position_labels) where position_records are
        dicts per unique position and labels are bools (True if >10% blunder rate).
    """
    # Group by normalized position
    position_groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        norm_fen = normalize_position(rec["fen"])
        position_groups[norm_fen].append(rec)

    # Filter by min occurrences
    position_records = []
    position_labels = []

    for norm_fen, group in position_groups.items():
        if len(group) < min_occurrences:
            continue

        # Count blunders
        blunder_count = 0
        for rec in group:
            cp_before = rec.get("centipawn_eval")
            if cp_before is None:
                continue
            # Compute eval after move (approximate: use Stockfish eval of new position)
            # For simplicity, use the is_blunder flag if computed earlier
            if rec.get("is_blunder", False):
                blunder_count += 1

        if len(group) == 0:
            continue

        blunder_rate = blunder_count / len(group)
        is_high_blunder = blunder_rate > blunder_threshold

        position_records.append({
            "norm_fen": norm_fen,
            "total_occurrences": len(group),
            "blunder_count": blunder_count,
            "blunder_rate": blunder_rate,
        })
        position_labels.append(is_high_blunder)

    logger.info(
        f"Collective dataset: {len(position_records)} positions, "
        f"{sum(position_labels)} high-blunder ({sum(position_labels)/max(len(position_labels),1):.1%})"
    )

    return position_records, position_labels


class CollectiveBlunderDataset(Dataset):
    """Dataset for collective blunder prediction.

    Each sample: (board_tensor, label) for a normalized position.
    """

    def __init__(
        self,
        position_records: list[dict],
        labels: list[bool],
        balance: bool = True,
    ):
        self.records = []
        self.labels = []

        if balance:
            pos_recs = [r for r, l in zip(position_records, labels) if l]
            neg_recs = [r for r, l in zip(position_records, labels) if not l]
            min_len = min(len(pos_recs), len(neg_recs))
            rng = np.random.RandomState(42)
            pos_idx = rng.choice(len(pos_recs), min_len, replace=False)
            neg_idx = rng.choice(len(neg_recs), min_len, replace=False)
            combined = (
                [(pos_recs[i], True) for i in pos_idx]
                + [(neg_recs[i], False) for i in neg_idx]
            )
            rng.shuffle(combined)
            self.records = [r for r, _ in combined]
            self.labels = [l for _, l in combined]
        else:
            self.records = position_records
            self.labels = labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        fen = rec["norm_fen"]
        board = chess.Board(fen)

        # Get board tensor (board-only: 17 channels; no metadata)
        tensor = board_to_tensor(board)

        label = 1.0 if self.labels[idx] else 0.0
        return torch.from_numpy(tensor).float(), torch.tensor(label, dtype=torch.float32)


def train_collective_model(
    model: torch.nn.Module,
    train_positions: list[dict],
    train_labels: list[bool],
    val_positions: list[dict],
    val_labels: list[bool],
    model_name: str,
    config: dict | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "checkpoints",
    debug: bool = False,
) -> torch.nn.Module:
    """Train a collective blunder prediction model.

    Uses early stopping: evaluate every 200 steps, stop if no
    improvement for 64 consecutive evaluations.

    Args:
        model: PyTorch blunder model.
        train_positions: Training position records.
        train_labels: Training labels.
        val_positions: Validation position records.
        val_labels: Validation labels.
        model_name: Name for checkpoint.
        config: Training config.
        device: Device.
        output_dir: Checkpoint directory.
        debug: Use minimal data.

    Returns trained model.
    """
    if config is None:
        config = {}

    batch_size = config.get("batch_size", 2000)
    lr = config.get("lr", 0.0001 if isinstance(model, BlunderFC) else 0.00001)
    lr_decay_steps = config.get("lr_decay_steps", [20000, 100000, 130000])
    lr_decay_factor = config.get("lr_decay_factor", 0.1)
    eval_interval = config.get("eval_interval", 200)
    patience = config.get("patience", 64)
    max_steps = config.get("max_steps", 1_400_000)

    if debug:
        max_steps = min(max_steps, 5000)
        train_positions = train_positions[:max(batch_size * 5, len(train_positions) // 100)]
        train_labels = train_labels[:len(train_positions)]
        val_positions = val_positions[:max(batch_size * 2, len(val_positions) // 100)]
        val_labels = val_labels[:len(val_positions)]

    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = CollectiveBlunderDataset(train_positions, train_labels, balance=True)
    val_dataset = CollectiveBlunderDataset(val_positions, val_labels, balance=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_val_auc = 0.0
    no_improve_count = 0
    all_preds = []
    all_labels = []

    for boards, labels in train_loader:
        if step >= max_steps:
            break

        # LR decay
        if step in lr_decay_steps:
            for pg in optimizer.param_groups:
                pg["lr"] *= lr_decay_factor
            logger.info(f"Step {step}: LR -> {optimizer.param_groups[0]['lr']:.6f}")

        boards = boards.permute(0, 3, 1, 2).to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(boards)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        step += 1

        # Early stopping evaluation
        if step % eval_interval == 0:
            model.eval()
            eval_preds = []
            eval_labels = []

            with torch.no_grad():
                for v_boards, v_labels in val_loader:
                    v_boards = v_boards.permute(0, 3, 1, 2).to(device)
                    v_out = model(v_boards)
                    eval_preds.append(v_out.cpu().numpy())
                    eval_labels.append(v_labels.numpy())

            if eval_preds:
                eval_preds = np.concatenate(eval_preds)
                eval_labels = np.concatenate(eval_labels)
                val_auc = roc_auc_score(eval_labels, eval_preds)
                val_acc = accuracy_score(eval_labels, (eval_preds > 0.5).astype(float))

                logger.info(
                    f"[{model_name}] Step {step} | Loss: {loss.item():.4f} | "
                    f"Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f}"
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    no_improve_count = 0
                    ckpt = output_dir / f"{model_name}_collective_best.pt"
                    torch.save(model.state_dict(), ckpt)
                    logger.info(f"Saved best: {ckpt}")
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    logger.info(f"Early stopping at step {step} (no improvement for {patience} evals)")
                    model.load_state_dict(
                        torch.load(output_dir / f"{model_name}_collective_best.pt")
                    )
                    return model

            model.train()

    # Save final
    ckpt = output_dir / f"{model_name}_collective_final.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info(f"Saved final: {ckpt}")

    return model


def evaluate_blunder_models(
    models: dict[str, tuple[torch.nn.Module, str]],
    test_positions: list[dict],
    test_labels: list[bool],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, dict]:
    """Evaluate multiple blunder models on a test set.

    Args:
        models: Dict of model_name -> (model, model_type)
                where model_type is 'fc', 'rescnn', or 'deep_rescnn'.
        test_positions: Position records for testing.
        test_labels: Ground truth labels.

    Returns dict of model_name -> {accuracy, auc, ...}.
    """
    dataset = CollectiveBlunderDataset(test_positions, test_labels, balance=False)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, pin_memory=True)

    results = {}
    for name, (model, _) in models.items():
        model = model.to(device)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for boards, labels in loader:
                boards = boards.permute(0, 3, 1, 2).to(device)
                out = model(boards)
                all_preds.append(out.cpu().numpy())
                all_labels.append(labels.numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(float))

        results[name] = {
            "accuracy": acc,
            "auc": auc,
            "predictions": all_preds,
            "labels": all_labels,
        }
        logger.info(f"[{name}] Test Acc: {acc:.4f} | AUC: {auc:.4f}")

    return results
