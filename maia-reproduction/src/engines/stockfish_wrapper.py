"""Wrapper for running Stockfish as a baseline engine.

Supports fixed-depth search (Depth 1 through Depth 15) and
evaluates move-matching accuracy against human moves.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import chess
import chess.engine

logger = logging.getLogger(__name__)


class StockfishWrapper:
    """Wrapper around Stockfish UCI engine for move-matching evaluation.

    Args:
        engine_path: Path to Stockfish executable.
        depth: Search depth (1-15).
    """

    def __init__(self, engine_path: str | Path, depth: int = 15):
        self.engine_path = Path(engine_path)
        self.depth = depth
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._cache: dict[tuple[str, int], str] = {}  # (fen, depth) -> bestmove UCI

    def start(self):
        """Start the engine."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))

    def stop(self):
        """Stop the engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def get_best_move(self, board: chess.Board) -> str | None:
        """Get the best move for a position at the configured depth.

        Args:
            board: Chess board position.

        Returns UCI string of the best move, or None if no move found.
        """
        cache_key = (board.fen(), self.depth)
        if cache_key in self._cache:
            return self._cache[cache_key]

        self.start()
        try:
            result = self._engine.play(board, chess.engine.Limit(depth=self.depth))
            best_move = result.move
            if best_move is not None:
                best_uci = best_move.uci()
                self._cache[cache_key] = best_uci
                return best_uci
        except Exception as e:
            logger.warning(f"Stockfish error at depth {self.depth}: {e}")
        return None

    def get_best_move_batch(
        self, boards: list[chess.Board], clear_cache: bool = False
    ) -> list[str | None]:
        """Get best moves for multiple positions.

        Args:
            boards: List of board positions.
            clear_cache: Whether to clear the internal cache first.

        Returns list of UCI strings (or None for failures).
        """
        if clear_cache:
            self._cache.clear()

        return [self.get_best_move(b) for b in boards]

    def evaluate_position(self, board: chess.Board) -> float | None:
        """Get Stockfish centipawn evaluation of a position.

        Returns centipawn score from the perspective of the side to move.
        """
        self.start()
        try:
            info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info.get("score")
            if score is not None:
                return score.relative.score(mate_score=10000)
        except Exception as e:
            logger.warning(f"Stockfish eval error: {e}")
        return None

    def get_top_moves(
        self, board: chess.Board, num_moves: int = 3
    ) -> list[tuple[str, float]]:
        """Get the top N moves with their scores (in centipawns).

        Returns list of (uci, cp_score) tuples, sorted by score descending.
        """
        self.start()
        try:
            result = self._engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth),
                multipv=num_moves,
            )
            moves_and_scores = []
            for info in result:
                move = info.get("pv", [None])[0]
                score = info.get("score")
                if move is not None and score is not None:
                    cp = score.relative.score(mate_score=10000)
                    moves_and_scores.append((move.uci(), cp))
            return moves_and_scores
        except Exception as e:
            logger.warning(f"Stockfish multi-PV error: {e}")
        return []

    def compute_move_matching_accuracy(
        self,
        positions: list[tuple[chess.Board, str]],  # (board, actual_move_uci)
    ) -> float:
        """Compute move-matching accuracy on a set of positions.

        Args:
            positions: List of (board, actual_move_uci) tuples.

        Returns fraction of positions where Stockfish's best move
        matches the human's actual move.
        """
        matches = 0
        total = len(positions)

        for board, actual_uci in positions:
            predicted_uci = self.get_best_move(board)
            if predicted_uci == actual_uci:
                matches += 1

        accuracy = matches / total if total > 0 else 0.0
        logger.info(
            f"Stockfish d={self.depth}: {matches}/{total} = {accuracy:.4f}"
        )
        return accuracy

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
