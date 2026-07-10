"""Tests for dataset construction and data loading."""

import chess
import numpy as np
import torch
import pytest

from src.data_pipeline.dataset import (
    MoveMatchingDataset,
    BlunderDataset,
    create_move_dataloader,
    create_blunder_dataloader,
)
from src.encoding.board import board_to_tensor
from src.encoding.move import move_to_index


class TestMoveMatchingDataset:
    def test_dataset_yields_tensors(self):
        records = [
            {
                "fen": chess.STARTING_FEN,
                "move_uci": "e2e4",
                "white_rating": 1500,
                "black_rating": 1500,
                "result": "*",
            },
            {
                "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "move_uci": "e7e5",
                "white_rating": 1500,
                "black_rating": 1500,
                "result": "*",
            },
        ]

        dataset = MoveMatchingDataset(
            records,
            shuffle_buffer=10,
            subsample_rate=1.0,
            use_history=False,
        )

        items = list(dataset)
        assert len(items) == 2

        for board_t, move_idx in items:
            assert isinstance(board_t, np.ndarray)
            assert isinstance(move_idx, int)
            assert 0 <= move_idx < 64 * 73

    def test_subsampling(self):
        records = [{
            "fen": chess.STARTING_FEN,
            "move_uci": "e2e4",
            "white_rating": 1500,
            "black_rating": 1500,
            "result": "*",
        } for _ in range(100)]

        dataset = MoveMatchingDataset(
            records,
            shuffle_buffer=10,
            subsample_rate=0.5,
            use_history=False,
        )
        items = list(dataset)
        # With rate=0.5, expect ~50 items (with variance)
        assert len(items) < 90, f"Expected more subsampling, got {len(items)}"

    def test_history_tensor_shape(self):
        records = [{
            "fen": chess.STARTING_FEN,
            "move_uci": "e2e4",
            "white_rating": 1500,
            "black_rating": 1500,
            "result": "*",
        }]

        dataset = MoveMatchingDataset(
            records,
            shuffle_buffer=10,
            subsample_rate=1.0,
            use_history=True,
            num_history=12,
        )
        items = list(dataset)
        board_t, _ = items[0]
        # 17 base channels + 12 history * 12 piece channels
        expected_channels = 17 + 12 * 12
        assert board_t.shape == (8, 8, expected_channels), (
            f"Expected (8,8,{expected_channels}), got {board_t.shape}"
        )


class TestBlunderDataset:
    def test_balanced_dataset(self):
        records = [
            {"fen": chess.STARTING_FEN, "move_uci": "e2e4",
             "is_blunder": True, "white_rating": 1500, "black_rating": 1500,
             "result": "*"},
            {"fen": chess.STARTING_FEN, "move_uci": "e2e4",
             "is_blunder": False, "white_rating": 1500, "black_rating": 1500,
             "result": "*"},
            {"fen": chess.STARTING_FEN, "move_uci": "e2e4",
             "is_blunder": False, "white_rating": 1500, "black_rating": 1500,
             "result": "*"},
            {"fen": chess.STARTING_FEN, "move_uci": "e2e4",
             "is_blunder": False, "white_rating": 1500, "black_rating": 1500,
             "result": "*"},
        ]

        dataset = BlunderDataset(records, balanced=True)
        assert len(dataset) == 2  # balanced: 1 blunder + 1 non-blunder

    def test_unbalanced_dataset(self):
        records = [
            {"fen": chess.STARTING_FEN, "move_uci": "e2e4",
             "is_blunder": True, "white_rating": 1500, "black_rating": 1500,
             "result": "*"},
            {"fen": chess.STARTING_FEN, "move_uci": "e2e4",
             "is_blunder": False, "white_rating": 1500, "black_rating": 1500,
             "result": "*"},
        ]

        dataset = BlunderDataset(records, balanced=False)
        assert len(dataset) == 2  # all records kept

    def test_get_item(self):
        records = [{
            "fen": chess.STARTING_FEN,
            "move_uci": "e2e4",
            "is_blunder": True,
            "white_rating": 1500,
            "black_rating": 1500,
            "result": "*",
        }]

        dataset = BlunderDataset(records, balanced=False)
        board_t, label = dataset[0]
        assert board_t.shape == (8, 8, 17)
        assert label.item() == 1.0

    def test_get_item_with_metadata(self):
        records = [{
            "fen": chess.STARTING_FEN,
            "move_uci": "e2e4",
            "is_blunder": False,
            "white_rating": 1500,
            "black_rating": 1500,
            "clock_remaining": 120.0,
            "tc_initial": 600,
            "centipawn_eval": 50.0,
            "result": "*",
        }]

        dataset = BlunderDataset(records, balanced=False, include_metadata=True)
        board_t, label = dataset[0]
        # 17 + 5 metadata channels
        assert board_t.shape == (8, 8, 22), f"Expected (8,8,22), got {board_t.shape}"
        assert label.item() == 0.0


class TestDataLoaderCreation:
    def test_create_move_dataloader(self):
        records = [{
            "fen": chess.STARTING_FEN,
            "move_uci": "e2e4",
            "white_rating": 1500,
            "black_rating": 1500,
            "result": "*",
        } for _ in range(32)]

        loader = create_move_dataloader(
            records,
            batch_size=16,
            shuffle_buffer=100,
            subsample_rate=1.0,
            use_history=False,
            num_workers=0,
        )

        batch = next(iter(loader))
        boards, moves = batch
        assert boards.shape[0] <= 16
        assert moves.shape[0] == boards.shape[0]

    def test_create_blunder_dataloader(self):
        records = [{
            "fen": chess.STARTING_FEN,
            "move_uci": "e2e4",
            "is_blunder": i < 16,
            "white_rating": 1500,
            "black_rating": 1500,
            "result": "*",
        } for i in range(64)]

        loader = create_blunder_dataloader(
            records,
            batch_size=32,
            balanced=True,
        )

        batch = next(iter(loader))
        boards, labels = batch
        # Balanced: 16 blunders + 16 non-blunders (or min of both)
        assert labels.shape[0] == boards.shape[0]
