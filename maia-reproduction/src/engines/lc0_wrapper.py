"""Wrapper for running Leela Chess Zero (lc0) as a baseline engine.

Supports loading different network checkpoints and evaluating
move-matching accuracy across strength levels.
"""

import logging
from pathlib import Path
from typing import Optional

import chess
import chess.engine

logger = logging.getLogger(__name__)


class LC0Wrapper:
    """Wrapper around lc0 UCI engine for move-matching evaluation.

    Args:
        engine_path: Path to lc0 executable.
        weights_path: Path to network weights file.
        playouts: Number of playouts per move (default 1 for raw net output).
        name: Human-readable name for this config.
    """

    def __init__(
        self,
        engine_path: str | Path,
        weights_path: str | Path,
        playouts: int = 1,
        name: str | None = None,
    ):
        self.engine_path = Path(engine_path)
        self.weights_path = Path(weights_path)
        self.playouts = playouts
        self._name = name or f"lc0_playouts{playouts}"
        self._engine: Optional[chess.engine.SimpleEngine] = None

    @property
    def name(self) -> str:
        return self._name

    def start(self):
        """Start the engine with the configured weights."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                str(self.engine_path),
                extra_args=[f"--weights={self.weights_path}"],
            )

    def stop(self):
        """Stop the engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def get_best_move(self, board: chess.Board) -> str | None:
        """Get the best move for a position.

        Args:
            board: Chess board position.

        Returns UCI string of the best move, or None.
        """
        self.start()
        try:
            result = self._engine.play(
                board, chess.engine.Limit(nodes=self.playouts)
            )
            if result.move is not None:
                return result.move.uci()
        except Exception as e:
            logger.warning(f"lc0 error: {e}")
        return None

    def compute_move_matching_accuracy(
        self,
        positions: list[tuple[chess.Board, str]],
    ) -> float:
        """Compute move-matching accuracy.

        Args:
            positions: List of (board, actual_move_uci) tuples.

        Returns fraction matching.
        """
        matches = 0
        total = len(positions)

        for board, actual_uci in positions:
            predicted_uci = self.get_best_move(board)
            if predicted_uci == actual_uci:
                matches += 1

        accuracy = matches / total if total > 0 else 0.0
        logger.info(f"{self._name}: {matches}/{total} = {accuracy:.4f}")
        return accuracy

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
