"""Move-matching evaluation: accuracy, model maxima, agreement matrices."""

import logging
from typing import Optional

import chess
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from src.encoding.board import board_to_tensor
from src.encoding.move import move_to_index, mask_illegal_moves, NUM_MOVE_PLANES
from src.models.maia_net import MaiaNet

logger = logging.getLogger(__name__)


def evaluate_maia_accuracy(
    model: MaiaNet,
    positions: list[tuple[chess.Board, str]],  # (board, actual_move_uci)
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 512,
) -> float:
    """Compute move-matching accuracy for a Maia model.

    Args:
        model: Trained MaiaNet.
        positions: List of (board, actual_move_uci) tuples.
        device: Device for inference.
        batch_size: Batch size for batched inference.

    Returns accuracy as fraction [0, 1].
    """
    model = model.to(device)
    model.eval()

    matches = 0
    total = len(positions)

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = positions[i:i + batch_size]

            # Build tensors
            boards_t = []
            legal_masks = []
            target_indices = []

            for board, actual_uci in positions[i:i + batch_size]:
                tensor = board_to_tensor(board)
                boards_t.append(tensor)

                # Target move index
                target_move = chess.Move.from_uci(actual_uci)
                target_idx = move_to_index(target_move)
                target_indices.append(target_idx)

                # Legal moves mask
                mask = torch.zeros(64 * NUM_MOVE_PLANES, dtype=torch.bool)
                for legal_move in board.legal_moves:
                    try:
                        idx = move_to_index(legal_move)
                        mask[idx] = True
                    except ValueError:
                        continue
                legal_masks.append(mask)

            boards_t = torch.from_numpy(np.stack(boards_t)).float().to(device)
            boards_t = boards_t.permute(0, 3, 1, 2)
            legal_masks = torch.stack(legal_masks).to(device)
            target_indices = torch.tensor(target_indices, device=device)

            # Forward
            policy_logits, _ = model(boards_t)
            policy_logits = policy_logits.masked_fill(~legal_masks, -float("inf"))
            pred_indices = torch.argmax(policy_logits, dim=1)

            matches += (pred_indices == target_indices).sum().item()

    accuracy = matches / total if total > 0 else 0.0
    logger.info(f"Maia accuracy: {matches}/{total} = {accuracy:.4f}")
    return accuracy


def compute_accuracy_by_bin(
    models: dict[int, MaiaNet],
    test_sets: dict[int, list[tuple[chess.Board, str]]],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[int, dict[int, float]]:
    """Compute accuracy for each model on each test set.

    Args:
        models: Dict mapping rating bin -> MaiaNet.
        test_sets: Dict mapping rating bin -> list of (board, move) pairs.

    Returns:
        Dict[model_bin][test_bin] -> accuracy.
    """
    results: dict[int, dict[int, float]] = {}

    for model_bin, model in sorted(models.items()):
        results[model_bin] = {}
        for test_bin, positions in sorted(test_sets.items()):
            if not positions:
                continue
            acc = evaluate_maia_accuracy(model, positions, device=device)
            results[model_bin][test_bin] = acc
            logger.info(f"Maia-{model_bin} on bin {test_bin}: {acc:.4f}")

    return results


def compute_agreement_matrix(
    model_predictors: dict[str, object],
    positions: list[chess.Board],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """Compute pairwise model agreement matrix.

    Args:
        model_predictors: Dict mapping model name -> callable
                         that takes (board) and returns move UCI string.
        positions: List of board positions.

    Returns:
        N x N numpy array where entry (i,j) = fraction of positions
        where model i and model j predict the same move.
    """
    names = list(model_predictors.keys())
    n = len(names)
    predictions: dict[str, list[str | None]] = {name: [] for name in names}

    logger.info(f"Computing agreement matrix for {n} models on {len(positions)} positions...")

    for board in positions:
        for name in names:
            pred = model_predictors[name](board)
            predictions[name].append(pred)

    matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matches = sum(
                    1 for a, b in zip(predictions[ni], predictions[nj])
                    if a is not None and b is not None and a == b
                )
                total = sum(
                    1 for a, b in zip(predictions[ni], predictions[nj])
                    if a is not None and b is not None
                )
                matrix[i, j] = matches / total if total > 0 else 0.0

    return matrix


def compute_maxima_table(
    results: dict[str, dict[int, float]],
    test_bins: list[int],
) -> list[dict]:
    """Compute which test bin each model peaks at.

    Args:
        results: Dict mapping model_name -> {test_bin: accuracy}.
        test_bins: List of test bin labels.

    Returns list of dicts with model name, peak bin, peak accuracy.
    """
    table = []
    for model_name, accuracies in results.items():
        best_bin = max(accuracies, key=accuracies.get)
        best_acc = accuracies[best_bin]
        table.append({
            "model": model_name,
            "peak_bin": best_bin,
            "peak_accuracy": best_acc,
            "accuracies": accuracies,
        })
    return table
