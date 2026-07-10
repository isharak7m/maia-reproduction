"""Filtering and binning operations on parsed chess data.

Provides:
- Time control filtering (exclude bullet/hyperbullet)
- Minimum clock filtering
- Rating bin assignment
- First-N-ply skipping
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Rating bins: 1100-1199, 1200-1299, ..., 1800-1899
RATING_BINS = list(range(1100, 1900, 100))  # [1100, 1200, ..., 1800]
RATING_BIN_LABELS = [f"{lo}-{lo+99}" for lo in RATING_BINS]

# Extended test ranges
EXTENDED_RANGES = [1000, 2500]

BULLET_MAX_TIME = 179  # seconds total initial time


def get_rating_bin(rating: int) -> int | None:
    """Get the bin lower bound for a rating, or None if out of range.

    Args:
        rating: Player Elo rating.

    Returns:
        Bin lower bound (e.g., 1100, 1200, ...) or None if outside 1100-1899.
    """
    if rating < 1100 or rating >= 1900:
        return None
    return (rating // 100) * 100


def is_bullet(tc_category: str) -> bool:
    """Check if a time control category is bullet or hyperbullet."""
    return tc_category == "bullet"


def passes_global_filters(
    record: dict,
    min_clock: float = 30.0,
    skip_bullet: bool = True,
) -> bool:
    """Check if a ply record passes all global filtering rules.

    Rules:
    - No bullet/hyperbullet games
    - No moves with < min_clock seconds remaining
    """
    if skip_bullet and is_bullet(record.get("tc_category", "")):
        return False

    clock = record.get("clock_remaining")
    if clock is not None and clock < min_clock:
        return False

    return True


def filter_to_rating_bin(
    records: list[dict],
    bin_lower: int,
    require_both_players: bool = True,
) -> list[dict]:
    """Keep only records where players fall within a specific rating bin.

    Args:
        records: List of ply records.
        bin_lower: Lower bound of the rating bin (e.g., 1100).
        require_both_players: If True, both players must be in the bin.
                              If False, only the side to move must be.

    Returns filtered records.
    """
    bin_upper = bin_lower + 99
    filtered = []
    for rec in records:
        white = rec.get("white_rating")
        black = rec.get("black_rating")
        if white is None or black is None:
            continue
        if require_both_players:
            if bin_lower <= white <= bin_upper and bin_lower <= black <= bin_upper:
                filtered.append(rec)
        else:
            side = rec.get("side_to_move")
            rating = white if side == "white" else black
            if rating is not None and bin_lower <= rating <= bin_upper:
                filtered.append(rec)
    return filtered


def assign_bin_label(rating: int) -> str | None:
    """Return bin label string for a rating, or None."""
    lower = get_rating_bin(rating)
    if lower is None:
        return None
    return f"{lower}-{lower+99}"


def sample_games_by_year_block(
    records: list[dict],
    year: int,
    block_size: int = 200_000,
    num_blocks: int = 20,
    reserved_last_n_blocks: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Sample games from a single year using block sampling.

    Splits games into blocks of block_size, reserves the last N blocks
    for validation, and randomly selects num_blocks for training.

    Args:
        records: List of ply records from one year (must have game_id).
        year: Calendar year (for logging).
        block_size: Number of games per block.
        num_blocks: Number of blocks to select for training.
        reserved_last_n_blocks: Number of final blocks to reserve for validation.

    Returns:
        (train_records, val_records)
    """
    import random

    # Group by game_id
    game_ids = list(set(r["game_id"] for r in records))
    random.shuffle(game_ids)

    blocks = [game_ids[i:i + block_size] for i in range(0, len(game_ids), block_size)]

    if len(blocks) < reserved_last_n_blocks:
        logger.warning(f"Year {year}: only {len(blocks)} blocks, can't reserve {reserved_last_n_blocks}")
        val_blocks = []
        train_blocks = blocks
    else:
        val_blocks = blocks[-reserved_last_n_blocks:]
        train_blocks = blocks[:-reserved_last_n_blocks]

    # Select num_blocks for training
    if len(train_blocks) > num_blocks:
        train_blocks = random.sample(train_blocks, num_blocks)

    # Map game_id -> records
    game_records: dict[str, list[dict]] = {}
    for r in records:
        game_records.setdefault(r["game_id"], []).append(r)

    def extract(block_list):
        result = []
        for block in block_list:
            for gid in block:
                result.extend(game_records.get(gid, []))
        return result

    train_records = extract(train_blocks)
    val_records = extract(val_blocks)

    logger.info(
        f"Year {year}: {len(train_records)} train / {len(val_records)} val "
        f"from {len(blocks)} blocks ({len(train_blocks)} train / {len(val_blocks)} val blocks)"
    )
    return train_records, val_records
