#!/usr/bin/env python3
"""Train Maia move-matching models and/or blunder prediction models.

Usage:
    python scripts/train.py --config configs/maia_default.yaml --task maia
    python scripts/train.py --config configs/maia_default.yaml --task blunder
    python scripts/train.py --config configs/debug.yaml --task all --debug
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_maia_models(config: dict, debug: bool = False):
    """Train all 9 Maia models."""
    from src.models.maia_net import create_maia_model
    from src.training.train_maia import train_maia

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    rating_bins = config.get("rating_bins", [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800])

    for bin_lower in rating_bins:
        logger.info(f"=== Training Maia-{bin_lower} ===")
        model = create_maia_model(
            bin_lower,
            use_history=config["model"]["use_history"],
            num_history=config["model"]["num_history"],
            channels=config["model"]["channels"],
            blocks=config["model"]["blocks"],
        )

        # In a real run, load train/val records from Parquet
        # For now, train with a note that data must be built first
        train_records = config.get("_train_records", {}).get(bin_lower, [])
        val_records = config.get("_val_records", {}).get(bin_lower, [])

        if not train_records:
            logger.warning(f"No training data for bin {bin_lower}. "
                           f"Run build_dataset.py first, or use synthetic data.")
            # Create minimal synthetic data for testing
            import chess
            from src.encoding.move import move_to_index
            train_records = _synthetic_records(1000, bin_lower)
            val_records = _synthetic_records(100, bin_lower)

        model = train_maia(
            model,
            train_records,
            val_records,
            rating_bin=bin_lower,
            config=config.get("training"),
            device=device,
            output_dir="checkpoints",
            debug=debug,
            use_history=config["model"]["use_history"],
            num_history=config["model"]["num_history"],
        )

        logger.info(f"Maia-{bin_lower} training complete")


def _synthetic_records(n: int, rating_bin: int) -> list[dict]:
    """Generate synthetic training records for testing."""
    import chess
    records = []
    base_fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    ]
    boards = [chess.Board(f) for f in base_fens]
    for i in range(n):
        b = boards[i % len(boards)]
        moves = list(b.legal_moves)
        if moves:
            m = moves[i % len(moves)]
            records.append({
                "fen": b.fen(),
                "move_uci": m.uci(),
                "white_rating": rating_bin + 50,
                "black_rating": rating_bin + 50,
                "result": "1-0" if i % 2 == 0 else "0-1",
                "side_to_move": "white" if b.turn else "black",
            })
    return records


def train_blunder_models(config: dict, debug: bool = False):
    """Train blunder prediction models."""
    from src.models.blunder_fc import BlunderFC
    from src.models.blunder_rescnn import BlunderResCNN
    from src.training.train_blunder import train_blunder_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    win_prob_threshold = config.get("blunder", {}).get("win_prob_threshold", 0.10)

    # Synthetic data for testing
    train_records = _synthetic_blunder_records(5000, win_prob_threshold)
    val_records = _synthetic_blunder_records(500, win_prob_threshold)

    # Train FC (board-only)
    fc_model = BlunderFC(in_channels=17)
    train_blunder_model(
        fc_model, train_records, val_records,
        model_name="blunder_fc_board",
        config=config.get("blunder", {}).get("fc", {}),
        device=device,
        include_metadata=False,
        debug=debug,
    )

    # Train FC (board+metadata)
    fc_meta = BlunderFC(in_channels=22)
    train_blunder_model(
        fc_meta, train_records, val_records,
        model_name="blunder_fc_meta",
        config=config.get("blunder", {}).get("fc", {}),
        device=device,
        include_metadata=True,
        debug=debug,
    )

    # Train ResCNN (board-only)
    rescnn = BlunderResCNN(in_channels=17)
    train_blunder_model(
        rescnn, train_records, val_records,
        model_name="blunder_rescnn_board",
        config=config.get("blunder", {}).get("rescnn", {}),
        device=device,
        include_metadata=False,
        debug=debug,
    )

    # Train ResCNN (board+metadata)
    rescnn_meta = BlunderResCNN(in_channels=22)
    train_blunder_model(
        rescnn_meta, train_records, val_records,
        model_name="blunder_rescnn_meta",
        config=config.get("blunder", {}).get("rescnn", {}),
        device=device,
        include_metadata=True,
        debug=debug,
    )


def _synthetic_blunder_records(n: int, threshold: float = 0.10) -> list[dict]:
    """Generate synthetic blunder records for testing."""
    import chess
    import random
    records = []
    base_fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    ]
    rng = random.Random(42)
    for i in range(n):
        b = chess.Board(base_fens[i % len(base_fens)])
        moves = list(b.legal_moves)
        if moves:
            m = rng.choice(moves)
            records.append({
                "fen": b.fen(),
                "move_uci": m.uci(),
                "is_blunder": rng.random() < 0.3,
                "white_rating": 1500,
                "black_rating": 1500,
                "clock_remaining": 120.0,
                "tc_initial": 600,
                "centipawn_eval": rng.uniform(-300, 300),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--config", default="configs/maia_default.yaml")
    parser.add_argument("--task", choices=["maia", "blunder", "all", "collective"],
                        default="maia")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if args.debug:
        config_path = "configs/debug.yaml"

    config = load_config(config_path)

    if args.task in ("maia", "all"):
        logger.info("=== Training Maia models ===")
        train_maia_models(config, debug=args.debug)

    if args.task in ("blunder", "all"):
        logger.info("=== Training blunder models ===")
        train_blunder_models(config, debug=args.debug)


if __name__ == "__main__":
    main()
