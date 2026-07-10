"""Tests for board and move encoding, including round-trip verification."""

import chess
import numpy as np
import pytest

from src.encoding.board import (
    board_to_tensor,
    tensor_to_board,
    flip_board_tensor,
    board_to_tensor_with_history,
    NUM_BOARD_CHANNELS,
)
from src.encoding.move import (
    move_to_index,
    index_to_move,
    encode_move_policy,
    mask_illegal_moves,
    NUM_MOVE_PLANES,
)


class TestBoardEncoding:
    def test_tensor_shape(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (8, 8, 17), f"Expected (8,8,17), got {tensor.shape}"

    def test_initial_position(self):
        board = chess.Board()
        tensor = board_to_tensor(board)

        # Python-chess squares: 0=a1 (row0,col0) ... 63=h8 (row7,col7)
        # Row 1 (rank 2) = white pawns (squares 8-15)
        for col in range(8):
            assert tensor[1, col, 0] == 1.0, f"White pawn at row 1 col {col}"

        # Row 6 (rank 7) = black pawns (squares 48-55)
        for col in range(8):
            assert tensor[6, col, 6] == 1.0, f"Black pawn at row 6 col {col}"

        # Row 7 (rank 8) = black back rank (squares 56-63)
        # Black king at e8 (square 60 = row 7, col 4)
        assert tensor[7, 4, 11] == 1.0, "Black king at e8"

        # White king at e1 (square 4 = row 0, col 4)
        assert tensor[0, 4, 5] == 1.0, "White king at e1"

        # Side to move (white)
        assert tensor[:, :, 16].sum() == 64.0, "Side to move should be all 1s for white"

    def _fen_position(self, fen: str) -> str:
        """Return only position+castling+enpassant fields, ignoring move clocks."""
        parts = fen.split(" ")
        return " ".join(parts[:4])

    def test_round_trip(self):
        """Test that board -> tensor -> board preserves the position."""
        def pos_equal(b1, b2, label):
            f1 = self._fen_position(b1.fen())
            f2 = self._fen_position(b2.fen())
            assert f1 == f2, f"Round-trip failed for {label}: {f1} != {f2}"

        # Starting position
        board = chess.Board()
        tensor = board_to_tensor(board)
        recovered = tensor_to_board(tensor)
        pos_equal(board, recovered, "starting position")

        # After e4
        board2 = chess.Board()
        board2.push_san("e4")
        tensor2 = board_to_tensor(board2)
        recovered2 = tensor_to_board(tensor2)
        pos_equal(board2, recovered2, "after e4")

        # Complex position with castling rights lost
        board3 = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        tensor3 = board_to_tensor(board3)
        recovered3 = tensor_to_board(tensor3)
        pos_equal(board3, recovered3, "complex position")

    def test_castling_rights(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # Initial position has all castling rights
        assert tensor[0, 0, 12] == 1.0, "Missing white K-side castling"
        assert tensor[0, 0, 13] == 1.0, "Missing white Q-side castling"
        assert tensor[0, 0, 14] == 1.0, "Missing black K-side castling"
        assert tensor[0, 0, 15] == 1.0, "Missing black Q-side castling"

        # After e4, the rights should be the same
        board.push_san("e4")
        tensor = board_to_tensor(board)
        assert tensor[0, 0, 12] == 1.0
        assert tensor[0, 0, 13] == 1.0
        assert tensor[0, 0, 14] == 1.0
        assert tensor[0, 0, 15] == 1.0

    def test_en_passant(self):
        """En passant is not directly encoded in the tensor but the board
        state (pawn positions) should be preserved."""
        board = chess.Board("rnbqkbnr/ppppppp1/8/7p/4P3/8/PPPP1PPP/RNBQKBNR w KQkq h6 0 2")
        tensor = board_to_tensor(board)
        recovered = tensor_to_board(tensor)
        # Compare piece positions (first 3 fields of FEN, ignoring en passant and move clocks)
        f1 = " ".join(board.fen().split(" ")[:3])
        f2 = " ".join(recovered.fen().split(" ")[:3])
        assert f1 == f2, f"En passant position not preserved: {f1} != {f2}"

    def test_flip_board_tensor(self):
        """Test that flipping then applying from the other perspective yields same semantics."""
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        tensor = board_to_tensor(board)
        flipped = flip_board_tensor(tensor)

        # Flipped should have black to move
        assert flipped[:, :, 16].sum() < 64.0, "Flipped should have black to move"

        # Flip again should recover original (except for side-to-move state)
        double_flipped = flip_board_tensor(flipped)
        assert np.allclose(tensor[:, :, :16], double_flipped[:, :, :16])

    def test_history_stacking_shape(self):
        board = chess.Board()
        moves = ["e4", "e5", "Nf3", "Nc6", "Bb5"]
        history = []
        b = chess.Board()
        for m in moves:
            b.push_san(m)
            history.append(chess.Board(b.fen()))

        # The current board is after Bb5
        current = history[-1]
        # History: positions before each move, most recent first
        tensor = board_to_tensor_with_history(current, history[:-1], num_history=12)
        expected_channels = 17 + 12 * 12  # 17 + 144 = 161
        assert tensor.shape == (8, 8, expected_channels), (
            f"Expected (8,8,{expected_channels}), got {tensor.shape}"
        )

    def test_history_no_history_given(self):
        board = chess.Board()
        tensor = board_to_tensor_with_history(board, num_history=12)
        expected_channels = 17 + 12 * 12
        assert tensor.shape == (8, 8, expected_channels)


class TestMoveEncoding:
    def test_move_to_index_range(self):
        board = chess.Board()
        for move in board.legal_moves:
            idx = move_to_index(move)
            assert 0 <= idx < 64 * 73, f"Index {idx} out of range for move {move}"

    def test_index_to_move_round_trip(self):
        """Every legal move should round-trip through index_to_move(move_to_index(m)) == m."""
        board = chess.Board()
        for move in board.legal_moves:
            idx = move_to_index(move)
            decoded = index_to_move(idx, board)
            assert decoded in board.legal_moves, f"Decoded {decoded} not legal for {move}"
            assert decoded.uci() == move.uci(), f"Round-trip failed: {move} -> {idx} -> {decoded}"

    def test_round_trip_complex_positions(self):
        positions = [
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
            "8/8/8/8/8/8/8/8 w - -",  # empty board
            "k7/8/8/8/8/8/8/K7 w - -",  # just kings
        ]
        for fen in positions:
            board = chess.Board(fen)
            for move in board.legal_moves:
                idx = move_to_index(move)
                decoded = index_to_move(idx, board)
                assert decoded in board.legal_moves, f"FEN {fen}: {decoded} not legal for {move}"
                assert decoded.uci() == move.uci(), (
                    f"FEN {fen}: round-trip {move} -> {idx} -> {decoded}"
                )

    def test_promotion_encoding(self):
        """Test underpromotions specifically."""
        # Position where white can promote
        fen = "k7/4P3/8/8/8/8/8/K7 w - - 0 1"
        board = chess.Board(fen)
        # e7-e8=Q is a queen move
        queen_move = chess.Move.from_uci("e7e8q")
        knight_move = chess.Move.from_uci("e7e8n")
        bishop_move = chess.Move.from_uci("e7e8b")
        rook_move = chess.Move.from_uci("e7e8r")

        for move in [queen_move, knight_move, bishop_move, rook_move]:
            idx = move_to_index(move)
            decoded = index_to_move(idx, board)
            assert decoded.uci() == move.uci(), f"Promotion round-trip: {move} -> {idx} -> {decoded}"

    def test_castling_encoding(self):
        """Castling should encode as king moves in queen-move planes."""
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        board = chess.Board(fen)
        for move in board.legal_moves:
            if board.is_castling(move):
                idx = move_to_index(move)
                decoded = index_to_move(idx, board)
                assert decoded.uci() == move.uci(), f"Castling: {move} -> {idx} -> {decoded}"

    def test_mask_illegal_moves(self):
        board = chess.Board()
        policy = np.random.randn(8, 8, 73).astype(np.float32)
        masked = mask_illegal_moves(policy, board)

        # All legal move positions should be unchanged
        for move in board.legal_moves:
            idx = move_to_index(move)
            from_sq = idx // NUM_MOVE_PLANES
            plane = idx % NUM_MOVE_PLANES
            row, col = divmod(from_sq, 8)
            assert masked[row, col, plane] == policy[row, col, plane], (
                f"Legal move {move} was masked"
            )

        # At least some positions should have -inf
        assert np.any(masked == -float("inf")), "No illegal moves found to mask"

    def test_policy_tensor_shape(self):
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        move = chess.Move.from_uci("e2e4")
        policy = encode_move_policy(move)
        assert policy.shape == (8, 8, 73), f"Expected (8,8,73), got {policy.shape}"
        assert policy.sum() == 1.0, "Policy tensor should sum to 1"
