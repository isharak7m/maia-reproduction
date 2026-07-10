"""PGN parsing: extract ply-level data from Lichess PGN dumps.

Each PGN game yields one row per ply with:
- FEN before the move
- Move played (UCI)
- Player ratings
- Clock info
- Stockfish eval annotations (if present)
"""

import logging
import re
import io
from typing import Iterator

import chess
import chess.pgn
import numpy as np

logger = logging.getLogger(__name__)

# Time control categories (initial time in seconds per player)
BULLET_MAX = 179   # < 3 min
BLITZ_MIN = 180
BLITZ_MAX = 479
RAPID_MIN = 480
RAPID_MAX = 1499
CLASSICAL_MIN = 1500

TC_CATEGORIES = {
    "bullet": (0, BULLET_MAX),
    "blitz": (BLITZ_MIN, BLITZ_MAX),
    "rapid": (RAPID_MIN, RAPID_MAX),
    "classical": (CLASSICAL_MIN, float("inf")),
}


def parse_time_control(headers: dict) -> dict | None:
    """Parse Lichess time control from game headers.

    Returns dict with 'initial', 'increment', 'category' or None if unparseable.
    Lichess TimeControl header format: "300+3" or "600+0" etc.
    """
    tc_str = headers.get("TimeControl", "")
    if not tc_str or tc_str == "-":
        return None
    try:
        parts = tc_str.split("+")
        initial = int(parts[0])
        increment = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None

    total_time = initial + 40 * increment  # rough estimate per player
    if total_time <= BULLET_MAX:
        category = "bullet"
    elif total_time <= BLITZ_MAX:
        category = "blitz"
    elif total_time <= RAPID_MAX:
        category = "rapid"
    else:
        category = "classical"

    return {"initial": initial, "increment": increment, "category": category}


def parse_clock_comment(comment: str) -> float | None:
    """Extract clock time in seconds from a [%clk ...] comment."""
    m = re.search(r"\[%clk\s+(\d+):(\d+):(\d+)\]", comment)
    if m:
        h, m, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return h * 3600 + m * 60 + s
    return None


def parse_eval_comment(comment: str) -> float | None:
    """Extract Stockfish centipawn evaluation from a [%eval ...] comment.

    Handles both centipawn (#) and mate (M) evaluations.
    Returns centipawn score (positive = good for side to move).
    """
    m = re.search(r"\[%eval\s+([\d.\-#+]+)\]", comment)
    if not m:
        return None
    val_str = m.group(1)
    if val_str.startswith("#"):
        # Mate evaluation: #+5 means mate in 5 for side to move
        try:
            mate_in = int(val_str[1:])
            return 10000 * (1 if mate_in > 0 else -1)  # large CP value
        except ValueError:
            return None
    try:
        return float(val_str)
    except ValueError:
        return None


def parse_game(
    game: chess.pgn.Game,
    min_clock: float = 30.0,
    skip_first_n_ply: int = 0,
) -> list[dict]:
    """Parse a single PGN game into ply-level records.

    Args:
        game: Parsed chess.pgn.Game object.
        min_clock: Drop moves where mover had less than this many seconds.
        skip_first_n_ply: Skip first N ply (e.g., 10 to skip book openings).

    Returns:
        List of dicts, one per ply, containing:
            fen, move_uci, move_san, white_rating, black_rating,
            time_control_category, clock_remaining (seconds, or None),
            centipawn_eval (or None), ply_number, game_id, result
    """
    headers = game.headers
    white_elo = headers.get("WhiteElo")
    black_elo = headers.get("BlackElo")
    try:
        white_rating = int(white_elo) if white_elo else None
        black_rating = int(black_elo) if black_elo else None
    except ValueError:
        return []

    tc = parse_time_control(headers)
    if tc is None:
        return []

    result = headers.get("Result", "*")

    records = []
    board = game.board()
    ply = 0

    # Game ID: use a hash of the headers
    game_id = str(hash(str(dict(headers)))) if headers else "unknown"

    node: chess.pgn.Game | None = game
    while node.variations:
        node = node.variations[0]
        move = node.move
        if move is None:
            continue
        ply += 1

        if skip_first_n_ply > 0 and ply <= skip_first_n_ply:
            board.push(move)
            continue

        # Clock check
        clock = None
        if node.comment:
            clock = parse_clock_comment(node.comment)

        if clock is not None and clock < min_clock:
            board.push(move)
            continue

        # Eval check
        eval_cp = None
        if node.comment:
            eval_cp = parse_eval_comment(node.comment)

        records.append({
            "game_id": game_id,
            "fen": board.fen(),
            "move_uci": move.uci(),
            "move_san": board.san(move),
            "ply": ply,
            "white_rating": white_rating,
            "black_rating": black_rating,
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "tc_category": tc["category"],
            "tc_initial": tc["initial"],
            "tc_increment": tc["increment"],
            "clock_remaining": clock,
            "centipawn_eval": eval_cp,
            "result": result,
            "white_clock": clock if board.turn == chess.BLACK else None,
            "black_clock": clock if board.turn == chess.WHITE else None,
        })

        board.push(move)

    return records


def parse_pgn_stream(
    stream: Iterator[bytes],
    min_clock: float = 30.0,
    skip_first_n_ply: int = 0,
) -> Iterator[dict]:
    """Parse a stream of (potentially compressed) PGN data into ply records.

    Args:
        stream: Iterator of bytes chunks (e.g., from decompress_stream).
        min_clock: Drop moves below this clock threshold.
        skip_first_n_ply: Skip first N ply.

    Yields dict records, one per ply.
    """
    buffer = io.BytesIO()
    for chunk in stream:
        buffer.write(chunk)

    buffer.seek(0)
    text_stream = io.TextIOWrapper(buffer, encoding="utf-8")

    game_count = 0
    while True:
        try:
            game = chess.pgn.read_game(text_stream)
        except Exception:
            logger.warning("Error reading game, skipping to next")
            # Skip to next game marker
            for line in text_stream:
                if line.startswith("1."):
                    break
            continue

        if game is None:
            break

        game_count += 1
        try:
            records = parse_game(game, min_clock, skip_first_n_ply)
            yield from records
        except Exception:
            logger.debug(f"Skipping game {game_count} due to parse error", exc_info=True)

        if game_count % 10_000 == 0:
            logger.info(f"Parsed {game_count} games...")

    logger.info(f"Done: parsed {game_count} total games")
