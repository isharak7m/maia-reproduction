"""AlphaZero-style move encoding: 8x8x73 policy head output.

For each of 64 source squares, 73 planes encode:
- 56 queen-move planes (8 directions x 7 distances)
- 8 knight-move planes
- 9 underpromotion planes (3 directions x 3 piece types)
"""

import chess
import numpy as np

# Direction offsets for queen moves (8 directions)
QUEEN_DIRECTIONS = [
    (-1, 0),   # north
    (1, 0),    # south
    (0, 1),    # east
    (0, -1),   # west
    (-1, 1),   # northeast
    (-1, -1),  # northwest
    (1, 1),    # southeast
    (1, -1),   # southwest
]

# Knight move offsets
KNIGHT_OFFSETS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]

# Underpromotion piece types (queen is covered by queen-move planes)
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

NUM_QUEEN_PLANES = 56   # 8 directions x 7 distances
NUM_KNIGHT_PLANES = 8
NUM_UNDERPROMOTION_PLANES = 9  # 3 directions x 3 piece types
NUM_MOVE_PLANES = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + NUM_UNDERPROMOTION_PLANES  # 73


def move_to_index(move: chess.Move) -> int:
    """Convert a chess.Move to a flat index in [0, 73*64).

    Returns index in the 8x8x73 policy space.
    This is a unique encoding; if the move has a promotion piece,
    it goes in the underpromotion or queen-move planes.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    from_row, from_col = divmod(from_sq, 8)
    to_row, to_col = divmod(to_sq, 8)
    dr, dc = to_row - from_row, to_col - from_col

    # Try queen-move directions first
    for dir_idx, (dir_dr, dir_dc) in enumerate(QUEEN_DIRECTIONS):
        if dr == 0 and dc == 0:
            continue
        # Check if move is aligned with this direction
        if dir_dr == 0 and dir_dc != 0:
            if dr != 0 or abs(dc) > 7:
                continue
            dist = abs(dc) - 1
            if dc // abs(dc) != dir_dc // abs(dir_dc):
                continue
        elif dir_dc == 0 and dir_dr != 0:
            if dc != 0 or abs(dr) > 7:
                continue
            dist = abs(dr) - 1
            if dr // abs(dr) != dir_dr // abs(dir_dr):
                continue
        elif dir_dr != 0 and dir_dc != 0:
            if abs(dr) != abs(dc) or abs(dr) > 7:
                continue
            if dr // abs(dr) != dir_dr // abs(dir_dr) or dc // abs(dc) != dir_dc // abs(dir_dc):
                continue
            dist = abs(dr) - 1
        else:
            continue

        plane = dir_idx * 7 + dist
        # Promotion to queen goes here; underpromotions handled separately
        if move.promotion is not None and move.promotion != chess.QUEEN:
            continue
        return from_sq * NUM_MOVE_PLANES + plane

    # Try knight moves
    for knight_idx, (kdr, kdc) in enumerate(KNIGHT_OFFSETS):
        if dr == kdr and dc == kdc and move.promotion is None:
            plane = NUM_QUEEN_PLANES + knight_idx
            return from_sq * NUM_MOVE_PLANES + plane

    # Try underpromotions
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # Determine direction of the underpromotion
        if dr == -1:  # white pawn push north
            dir_idx = 0
        elif dr == 1:  # black pawn push south
            dir_idx = 1
        elif dr == -1 and abs(dc) == 1:  # white capture
            dir_idx = 2 if dc == 1 else 3
        elif dr == 1 and abs(dc) == 1:  # black capture
            dir_idx = 4 if dc == 1 else 5
        else:
            # Fallback: try to infer direction
            dir_idx = 0
            for pi, piece in enumerate(UNDERPROMOTION_PIECES):
                if move.promotion == piece:
                    plane = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + dir_idx * 3 + pi
                    return from_sq * NUM_MOVE_PLANES + plane

        for pi, piece in enumerate(UNDERPROMOTION_PIECES):
            if move.promotion == piece:
                plane = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + dir_idx * 3 + pi
                return from_sq * NUM_MOVE_PLANES + plane

    # Move not found in any category; this can happen for castling which is
    # encoded as a king move in one of the queen-move planes.
    # Castling kingside: e1->g1 or e8->g8
    # Castling queenside: e1->c1 or e8->c8
    # These fall through to the direction check above which handles them.
    # If still not found, try direct direction matching.
    for dir_idx, (dir_dr, dir_dc) in enumerate(QUEEN_DIRECTIONS):
        if dr == dir_dr and dc == dir_dc:
            dist = 0
            plane = dir_idx * 7 + dist
            if move.promotion is not None and move.promotion != chess.QUEEN:
                continue
            return from_sq * NUM_MOVE_PLANES + plane

    raise ValueError(f"Could not encode move {move} (uci={move.uci()})")


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert a flat policy index back to a legal chess.Move.

    The board is needed to determine move legality and
    to resolve ambiguities (e.g., underpromotion directions).
    """
    from_sq = index // NUM_MOVE_PLANES
    plane = index % NUM_MOVE_PLANES

    # Find all legal moves from this source square
    candidates = [m for m in board.legal_moves if m.from_square == from_sq]

    if len(candidates) == 1:
        return candidates[0]

    # Try queen-move planes (0-55)
    if plane < NUM_QUEEN_PLANES:
        dir_idx = plane // 7
        dist = plane % 7
        dr, dc = QUEEN_DIRECTIONS[dir_idx]
        to_row = (from_sq // 8) + dr * (dist + 1)
        to_col = (from_sq % 8) + dc * (dist + 1)
        if 0 <= to_row < 8 and 0 <= to_col < 8:
            to_sq = to_row * 8 + to_col
            for m in candidates:
                if m.to_square == to_sq and (m.promotion is None or m.promotion == chess.QUEEN):
                    return m

    # Try knight planes (56-63)
    elif plane < NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES:
        knight_idx = plane - NUM_QUEEN_PLANES
        kdr, kdc = KNIGHT_OFFSETS[knight_idx]
        to_row = (from_sq // 8) + kdr
        to_col = (from_sq % 8) + kdc
        if 0 <= to_row < 8 and 0 <= to_col < 8:
            to_sq = to_row * 8 + to_col
            for m in candidates:
                if m.to_square == to_sq and m.promotion is None:
                    return m

    # Try underpromotion planes (64-72)
    else:
        up_idx = plane - (NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES)
        dir_idx = up_idx // 3
        piece_idx = up_idx % 3
        promotion_piece = UNDERPROMOTION_PIECES[piece_idx]
        # Direction: 0=north(white), 1=south(black), 2=NE capture, 3=NW capture, 4=SE capture, 5=SW capture
        underpromotion_dirs = [
            (-1, 0), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1)
        ]
        if dir_idx < len(underpromotion_dirs):
            udr, udc = underpromotion_dirs[dir_idx]
            to_row = (from_sq // 8) + udr
            to_col = (from_sq % 8) + udc
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                to_sq = to_row * 8 + to_col
                for m in candidates:
                    if m.to_square == to_sq and m.promotion == promotion_piece:
                        return m

    # Fallback: match by distance
    # Try all candidates
    for m in candidates:
        try:
            if move_to_index(m) == index:
                return m
        except ValueError:
            continue

    raise ValueError(f"Could not decode index {index} (from_sq={from_sq}, plane={plane})")


def encode_move_policy(move: chess.Move) -> np.ndarray:
    """Encode a single move as an 8x8x73 binary policy tensor."""
    policy = np.zeros((8, 8, NUM_MOVE_PLANES), dtype=np.float32)
    idx = move_to_index(move)
    from_sq = idx // NUM_MOVE_PLANES
    plane = idx % NUM_MOVE_PLANES
    row, col = divmod(from_sq, 8)
    policy[row, col, plane] = 1.0
    return policy


def mask_illegal_moves(policy: np.ndarray, board: chess.Board) -> np.ndarray:
    """Zero out policy probabilities for illegal moves and renormalize.

    Args:
        policy: Raw 8x8x73 policy tensor (pre-softmax logits or probs).
        board: Board position to check legality against.

    Returns:
        Masked policy tensor with illegal moves set to -inf (for logits) or 0 (for probs).
    """
    masked = policy.copy()
    legal_flags = np.zeros((8, 8, NUM_MOVE_PLANES), dtype=bool)

    for move in board.legal_moves:
        try:
            idx = move_to_index(move)
            from_sq = idx // NUM_MOVE_PLANES
            plane = idx % NUM_MOVE_PLANES
            row, col = divmod(from_sq, 8)
            legal_flags[row, col, plane] = True
        except ValueError:
            continue

    masked[~legal_flags] = -float("inf")
    return masked


def get_policy_vector(policy: np.ndarray) -> np.ndarray:
    """Flatten 8x8x73 policy to (73*64,) vector."""
    return policy.flatten()
