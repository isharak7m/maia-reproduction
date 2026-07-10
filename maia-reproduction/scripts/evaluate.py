#!/usr/bin/env python3
"""
Evaluate models: move-matching accuracy, agreement matrices,
blunder prediction metrics, and generate plots.

Usage:
    python scripts/evaluate.py --config configs/maia_default.yaml --task move_matching
    python scripts/evaluate.py --config configs/maia_default.yaml --task blunder
    python scripts/evaluate.py --config configs/maia_default.yaml --task all
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import numpy as np
import chess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_test_positions(
    config: dict, rating_bins: list[int] | None = None, num_per_bin: int = 1000
) -> dict[int, list[tuple[chess.Board, str]]]:
    """Generate synthetic test positions for evaluation.

    In a real run, load from the pre-built test datasets.
    For now, generate using synthetic data for the pipeline to work.
    """
    import random
    rng = random.Random(42)
    base_fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3",
        "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    ]
    boards = [chess.Board(f) for f in base_fens]

    if rating_bins is None:
        rating_bins = config.get("rating_bins", [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800])

    test_sets = {}
    for bin_lower in rating_bins:
        positions = []
        for i in range(num_per_bin):
            b = chess.Board(base_fens[i % len(base_fens)])
            moves = list(b.legal_moves)
            if moves:
                m = rng.choice(moves)
                positions.append((b, m.uci()))
        test_sets[bin_lower] = positions
        logger.info(f"Bin {bin_lower}: {len(positions)} positions")

    return test_sets


def evaluate_move_matching(config: dict, debug: bool = False):
    """Evaluate move-matching accuracy for Maia models."""
    from src.models.maia_net import MaiaNet, create_maia_model
    from src.evaluation.move_matching import (
        evaluate_maia_accuracy,
        compute_accuracy_by_bin,
        compute_maxima_table,
    )
    from src.evaluation.plots import (
        plot_accuracy_curves,
        plot_combined_accuracy,
        generate_results_table,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rating_bins = config.get("rating_bins", [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800])
    num_test = 100 if debug else 1000

    test_sets = generate_test_positions(config, rating_bins, num_per_bin=num_test)

    # Load or create models
    models = {}
    for bin_lower in rating_bins:
        ckpt_path = Path(f"checkpoints/maia_{bin_lower}_best.pt")
        if ckpt_path.exists():
            model = create_maia_model(
                bin_lower,
                use_history=config["model"]["use_history"],
                num_history=config["model"]["num_history"],
                channels=config["model"]["channels"],
                blocks=config["model"]["blocks"],
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            models[bin_lower] = model
            logger.info(f"Loaded Maia-{bin_lower} from {ckpt_path}")
        else:
            logger.warning(f"No checkpoint for Maia-{bin_lower}, using untrained model")
            models[bin_lower] = create_maia_model(
                bin_lower,
                use_history=config["model"]["use_history"],
                num_history=config["model"]["num_history"],
                channels=config["model"]["channels"],
                blocks=config["model"]["blocks"],
            )

    # Compute accuracy by bin
    results = compute_accuracy_by_bin(models, test_sets, device=device)

    # Plot results
    from src.evaluation.plots import plot_accuracy_curves
    ax = plot_accuracy_curves(
        {str(k): v for k, v in results.items()},
        family="Maia",
        test_bins=list(test_sets.keys()),
    )
    import matplotlib.pyplot as plt
    plt.savefig("reports/maia_accuracy_curves.png", dpi=150, bbox_inches="tight")
    logger.info("Saved accuracy curves to reports/maia_accuracy_curves.png")

    # Generate results table
    maxima = compute_maxima_table(
        {f"Maia-{k}": v for k, v in results.items()},
        list(test_sets.keys()),
    )
    logger.info("Model maxima:")
    for entry in maxima:
        logger.info(f"  {entry['model']}: peaks at {entry['peak_bin']} "
                    f"({entry['peak_accuracy']:.4f})")

    logger.info("Move-matching evaluation complete")


def evaluate_blunder(config: dict, debug: bool = False):
    """Evaluate blunder prediction models."""
    from src.evaluation.move_matching import evaluate_maia_accuracy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Blunder evaluation would load trained models and test data here")

    # In real usage:
    # 1. Load test set from Parquet
    # 2. Load each trained model
    # 3. Compute accuracy, AUC, precision/recall
    # 4. Print results table
    # 5. Generate ROC curves

    # Simulate results for pipeline validation
    results_table = {
        "Model": ["FC Board", "FC Meta", "ResCNN Board", "ResCNN Meta"],
        "Accuracy": [0.56, 0.63, 0.68, 0.72],
        "AUC": [0.58, 0.65, 0.70, 0.74],
    }
    logger.info("Blunder prediction results (simulated):")
    logger.info(f"  {results_table}")

    logger.info("Blunder evaluation complete")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--config", default="configs/maia_default.yaml")
    parser.add_argument("--task", choices=["move_matching", "blunder", "all", "agreement"],
                        default="move_matching")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.task in ("move_matching", "all"):
        evaluate_move_matching(config, debug=args.debug)

    if args.task in ("blunder", "all"):
        evaluate_blunder(config, debug=args.debug)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
