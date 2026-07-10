"""Dataset construction: raw parsed records -> model input tensors.

Converts ply-level records from Parquet/CSV into batched PyTorch tensors
for move-matching, blunder prediction, and evaluation.
"""

import logging
import random
from pathlib import Path
from typing import Iterator

import chess
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from src.encoding.board import board_to_tensor, board_to_tensor_with_history
from src.encoding.move import move_to_index, NUM_MOVE_PLANES

logger = logging.getLogger(__name__)


class MoveMatchingDataset(IterableDataset):
    """Iterable dataset for move-matching training.

    Reads ply records, converts them to (board_tensor, move_index) pairs,
    and applies subsampling (1/32) and shuffling.
    """

    def __init__(
        self,
        records: list[dict],
        shuffle_buffer: int = 250_000,
        subsample_rate: float = 1.0 / 32.0,
        use_history: bool = True,
        num_history: int = 12,
        seed: int = 42,
    ):
        self.records = records
        self.shuffle_buffer = shuffle_buffer
        self.subsample_rate = subsample_rate
        self.use_history = use_history
        self.num_history = num_history
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random(self.seed)
        buffer = []

        for rec in self.records:
            if self.subsample_rate < 1.0 and rng.random() > self.subsample_rate:
                continue

            try:
                board_tensor = self._record_to_tensor(rec)
                move_idx = move_to_index(chess.Move.from_uci(rec["move_uci"]))

                buffer.append((board_tensor, move_idx))
                if len(buffer) >= self.shuffle_buffer:
                    rng.shuffle(buffer)
                    yield from buffer
                    buffer = []
            except Exception:
                continue

        # Yield remaining
        if buffer:
            rng.shuffle(buffer)
            yield from buffer

    def _record_to_tensor(self, rec: dict) -> np.ndarray:
        board = chess.Board(rec["fen"])
        if self.use_history:
            return board_to_tensor_with_history(board, num_history=self.num_history)
        return board_to_tensor(board)


def collate_move_matching(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for move-matching data loader."""
    boards = torch.from_numpy(np.stack([b for b, _ in batch]))
    moves = torch.tensor([m for _, m in batch], dtype=torch.long)
    return boards, moves


def create_move_dataloader(
    records: list[dict],
    batch_size: int = 1024,
    shuffle_buffer: int = 250_000,
    subsample_rate: float = 1.0 / 32.0,
    use_history: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for move-matching training.

    Args:
        records: List of ply records (dicts with fen, move_uci).
        batch_size: Batch size.
        shuffle_buffer: Size of shuffle buffer for iterable dataset.
        subsample_rate: Fraction of moves to keep (1/32).
        use_history: Whether to include history planes.
        num_workers: Number of worker processes (0 = main process).

    Returns PyTorch DataLoader.
    """
    dataset = MoveMatchingDataset(
        records=records,
        shuffle_buffer=shuffle_buffer,
        subsample_rate=subsample_rate,
        use_history=use_history,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_move_matching,
        num_workers=num_workers,
        pin_memory=True,
    )


class BlunderDataset(Dataset):
    """Map-style dataset for blunder prediction.

    Each sample: (board_tensor, label) where label is 1 if blunder else 0.
    Supports balanced sampling (50/50).
    """

    def __init__(
        self,
        records: list[dict],
        balanced: bool = True,
        include_metadata: bool = False,
    ):
        self.include_metadata = include_metadata

        blunders = [r for r in records if r.get("is_blunder", False)]
        non_blunders = [r for r in records if not r.get("is_blunder", False)]

        if balanced:
            # Balance to 50/50
            min_len = min(len(blunders), len(non_blunders))
            rng = random.Random(42)
            self.records = rng.sample(blunders, min_len) + rng.sample(non_blunders, min_len)
            rng.shuffle(self.records)
        else:
            self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        board = chess.Board(rec["fen"])
        board_t = board_to_tensor(board)

        if self.include_metadata:
            # Add metadata as extra channels
            meta = self._build_metadata(rec)
            board_t = np.concatenate([board_t, meta], axis=-1)

        label = 1.0 if rec.get("is_blunder", False) else 0.0
        return torch.from_numpy(board_t).float(), torch.tensor(label, dtype=torch.float32)

    def _build_metadata(self, rec: dict) -> np.ndarray:
        """Build normalized metadata channels: ratings, clock, eval."""
        meta = np.zeros((8, 8, 5), dtype=np.float32)

        # White rating (normalized to [0, 1])
        wr = rec.get("white_rating", 1500)
        meta[:, :, 0] = (wr - 1000) / 2000.0

        # Black rating
        br = rec.get("black_rating", 1500)
        meta[:, :, 1] = (br - 1000) / 2000.0

        # White time remaining fraction
        wc = rec.get("clock_remaining")
        if wc is not None and rec.get("tc_initial", 0) > 0:
            meta[:, :, 2] = min(wc / rec["tc_initial"], 1.0)

        # Black time remaining fraction
        bc = rec.get("clock_remaining")
        if bc is not None and rec.get("tc_initial", 0) > 0:
            meta[:, :, 3] = min(bc / rec["tc_initial"], 1.0)

        # Stockfish eval (normalized to [-1, 1])
        eval_cp = rec.get("centipawn_eval")
        if eval_cp is not None:
            meta[:, :, 4] = np.clip(eval_cp / 1000.0, -1.0, 1.0)

        return meta


def create_blunder_dataloader(
    records: list[dict],
    batch_size: int = 2000,
    balanced: bool = True,
    include_metadata: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for blunder prediction training."""
    dataset = BlunderDataset(
        records=records,
        balanced=balanced,
        include_metadata=include_metadata,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
