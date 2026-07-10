"""Board state encoding as 8x8x17 tensor.

Representation:
- 12 channels: one-hot piece type (6) x color (2)
- 4 channels: castling rights (WK, WQ, BK, BQ)
- 1 channel: side to move
"""

import chess
import numpy as np

PIECE_TO_IDX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

NUM_PIECE_PLANES = 12  # 6 piece types x 2 colors
NUM_CASTLING_PLANES = 4
NUM_SIDE_PLANE = 1
NUM_BOARD_CHANNELS = NUM_PIECE_PLANES + NUM_CASTLING_PLANES + NUM_SIDE_PLANE  # 17


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert a chess.Board to an 8x8x17 numpy array.

    Channels 0-11: piece planes (6 types x 2 colors, white then black)
    Channels 12-15: castling rights (WK, WQ, BK, BQ)
    Channel 16: side to move (1.0 if white, 0.0 if black)
    """
    tensor = np.zeros((8, 8, NUM_BOARD_CHANNELS), dtype=np.float32)

    # Piece planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_idx = PIECE_TO_IDX[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            channel = color_offset + piece_idx
            row, col = divmod(square, 8)
            tensor[row, col, channel] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 12] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 13] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 14] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 15] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        tensor[:, :, 16] = 1.0

    return tensor


def board_to_tensor_with_history(
    board: chess.Board, history_boards: list[chess.Board] | None = None, num_history: int = 12
) -> np.ndarray:
    """Convert a board + previous positions to a stacked tensor.

    Each historical board is encoded as 12 piece-only planes (no castling/turn),
    so total channels = 17 (current) + 12 * num_history.

    Args:
        board: Current board position.
        history_boards: Previous board positions, most recent first.
                        If None, padded with empty boards.
        num_history: Number of historical ply to include (default 12).

    Returns:
        Tensor of shape (8, 8, 17 + 12 * num_history).
    """
    if history_boards is None:
        history_boards = []

    # Pad history if needed
    padded_history = list(history_boards[:num_history])
    while len(padded_history) < num_history:
        padded_history.append(chess.Board())  # empty/starting position as padding

    # Current board tensor
    channels = [board_to_tensor(board)]

    # History planes: just piece positions (12 channels each)
    for hb in padded_history:
        history_channels = np.zeros((8, 8, 12), dtype=np.float32)
        for square in chess.SQUARES:
            piece = hb.piece_at(square)
            if piece is not None:
                piece_idx = PIECE_TO_IDX[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                channel = color_offset + piece_idx
                row, col = divmod(square, 8)
                history_channels[row, col, channel] = 1.0
        channels.append(history_channels)

    return np.concatenate(channels, axis=-1)


def tensor_to_board(tensor: np.ndarray) -> chess.Board:
    """Convert an 8x8x17 tensor back to a chess.Board (for testing round-trips)."""
    board = chess.Board()
    board.clear()

    # Piece planes
    for row in range(8):
        for col in range(8):
            square = row * 8 + col
            piece_encoded = False
            for channel in range(12):
                if tensor[row, col, channel] > 0.5:
                    piece_type_idx = channel % 6
                    color = chess.WHITE if channel < 6 else chess.BLACK
                    piece_type = [
                        chess.PAWN, chess.KNIGHT, chess.BISHOP,
                        chess.ROOK, chess.QUEEN, chess.KING,
                    ][piece_type_idx]
                    board.set_piece_at(square, chess.Piece(piece_type, color))
                    piece_encoded = True
                    break
            if not piece_encoded:
                board.remove_piece_at(square)

    # Castling rights
    if tensor[0, 0, 12] > 0.5:
        board.castling_rights |= chess.BB_H1
    if tensor[0, 0, 13] > 0.5:
        board.castling_rights |= chess.BB_A1
    if tensor[0, 0, 14] > 0.5:
        board.castling_rights |= chess.BB_H8
    if tensor[0, 0, 15] > 0.5:
        board.castling_rights |= chess.BB_A8

    # Side to move
    if tensor[0, 0, 16] > 0.5:
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK

    return board


def flip_board_tensor(tensor: np.ndarray) -> np.ndarray:
    """Flip board perspective (for color normalization in collective blunder)."""
    # Mirror rows (vertical flip)
    flipped = tensor[::-1, :, :].copy()
    # Swap white/black piece planes
    white_planes = flipped[:, :, :6].copy()
    flipped[:, :, :6] = flipped[:, :, 6:12]
    flipped[:, :, 6:12] = white_planes
    # Swap castling rights
    wk = flipped[:, :, 12].copy()
    flipped[:, :, 12] = flipped[:, :, 14]
    flipped[:, :, 14] = wk
    wq = flipped[:, :, 13].copy()
    flipped[:, :, 13] = flipped[:, :, 15]
    flipped[:, :, 15] = wq
    # Flip side to move
    flipped[:, :, 16] = 1.0 - flipped[:, :, 16]
    return flipped
