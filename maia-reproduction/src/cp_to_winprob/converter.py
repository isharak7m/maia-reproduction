"""Empirical centipawn-to-win-probability converter.

Builds a lookup table from centipawn evaluation -> empirical win probability
by bucketing positions from training data and computing actual game outcomes.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CentipawnWinProbability:
    """Convert Stockfish centipawn evaluations to win probabilities.

    Built empirically from game data: for each centipawn bucket,
    compute the fraction of games won by the side favored by that eval.

    The resulting curve is S-shaped/monotonic with a possible discontinuity
    near 0 centipawns (known theoretical draws).
    """

    def __init__(self, lookup_table: dict[int, float] | None = None):
        self.lookup_table: dict[int, float] = lookup_table or {}
        self.bin_edges: list[int] = sorted(self.lookup_table.keys())

    def build_from_data(
        self,
        records: list[dict],
        bucket_size: int = 10,
    ) -> "CentipawnWinProbability":
        """Build the lookup table from game records.

        Args:
            records: List of ply records with 'centipawn_eval' and 'result' keys.
            bucket_size: Round evaluations to nearest N centipawns.

        Returns self for chaining.
        """
        data = []
        for r in records:
            eval_cp = r.get("centipawn_eval")
            result = r.get("result")
            if eval_cp is None or result is None:
                continue

            # Round to nearest bucket
            bucket = round(eval_cp / bucket_size) * bucket_size

            # Determine if the side favored by the eval won
            side = r.get("side_to_move", "white")
            if result == "1-0":
                side_won = side == "white"
            elif result == "0-1":
                side_won = side == "black"
            elif result == "1/2-1/2":
                # Draw counts as 0.5 win
                side_won = 0.5
            else:
                continue

            data.append({"bucket": bucket, "won": side_won})

        if not data:
            logger.warning("No data for win-prob conversion table")
            return self

        df = pd.DataFrame(data)
        win_probs = df.groupby("bucket")["won"].agg(["mean", "count"]).reset_index()
        win_probs.columns = ["bucket", "win_prob", "count"]

        # Filter low-count buckets
        win_probs = win_probs[win_probs["count"] >= 10]

        bucket_map: dict[int, float] = {}
        for _, row in win_probs.iterrows():
            bucket_map[int(row["bucket"])] = float(row["win_prob"])

        self.lookup_table = bucket_map
        self.bin_edges = sorted(bucket_map.keys())

        logger.info(
            f"Built win-prob table: {len(bucket_map)} buckets, "
            f"range [{min(bucket_map)} cp, {max(bucket_map)} cp]"
        )

        # Log some sanity-check values
        for cp in [-500, -100, -10, 0, 10, 100, 500]:
            nearest = min(self.bin_edges, key=lambda x: abs(x - cp))
            logger.info(f"  CP {cp:5d} -> WP {bucket_map.get(nearest, 0.5):.3f}")

        return self

    def get_win_probability(self, centipawns: float) -> float:
        """Get win probability for a centipawn evaluation.

        Uses nearest-neighbor interpolation among buckets.

        Args:
            centipawns: Centipawn evaluation (positive = good for side to move).

        Returns win probability in [0, 1].
        """
        if not self.bin_edges:
            return 0.5  # default if not built yet

        # Clamp
        cp = int(round(centipawns))
        if cp <= self.bin_edges[0]:
            return self.lookup_table[self.bin_edges[0]]
        if cp >= self.bin_edges[-1]:
            return self.lookup_table[self.bin_edges[-1]]

        # Nearest neighbor
        nearest = min(self.bin_edges, key=lambda x: abs(x - cp))
        return self.lookup_table.get(nearest, 0.5)

    def get_win_probability_vectorized(self, centipawns: np.ndarray) -> np.ndarray:
        """Vectorized version for numpy arrays."""
        return np.vectorize(self.get_win_probability)(centipawns)

    def is_blunder(
        self,
        cp_before: float,
        cp_after: float,
        threshold: float = 0.10,
    ) -> bool:
        """Determine if a move is a blunder.

        A blunder is a move that decreases the mover's win probability
        by at least `threshold` (default 10 percentage points).

        Args:
            cp_before: Centipawn eval before the move.
            cp_after: Centipawn eval after the move.
            threshold: Win-probability drop threshold.

        Returns True if the move is a blunder.
        """
        wp_before = self.get_win_probability(cp_before)
        wp_after = self.get_win_probability(cp_after)
        return (wp_before - wp_after) >= threshold

    def save(self, path: str | Path):
        """Save lookup table to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([
            {"cp": k, "win_prob": v} for k, v in sorted(self.lookup_table.items())
        ])
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        logger.info(f"Saved win-prob table to {path}")

    def load(self, path: str | Path) -> "CentipawnWinProbability":
        """Load lookup table from disk."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".parquet")
        df = pd.read_parquet(path)
        self.lookup_table = dict(zip(df["cp"], df["win_prob"]))
        self.bin_edges = sorted(self.lookup_table.keys())
        logger.info(f"Loaded win-prob table from {path}: {len(self.bin_edges)} buckets")
        return self

    def plot(self, save_path: str | Path | None = None):
        """Plot the win-probability curve."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.bin_edges:
            logger.warning("No data to plot")
            return

        cps = self.bin_edges
        wps = [self.lookup_table[cp] for cp in cps]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cps, wps, "b-", linewidth=2)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Centipawn Evaluation")
        ax.set_ylabel("Empirical Win Probability")
        ax.set_title("Centipawn-to-Win-Probability Conversion")
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved win-prob plot to {save_path}")

        return fig
