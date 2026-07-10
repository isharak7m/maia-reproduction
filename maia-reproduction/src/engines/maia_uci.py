"""Minimal UCI protocol wrapper for Maia models.

Allows Maia models to be used as pseudo-UCI engines that can be
interfaced with through python-chess's UCI engine interface.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

import chess
import torch
import numpy as np

from src.encoding.board import board_to_tensor
from src.encoding.move import mask_illegal_moves, NUM_MOVE_PLANES
from src.models.maia_net import MaiaNet

logger = logging.getLogger(__name__)


class MaiaUCIEngine:
    """UCI-compatible wrapper around a Maia model.

    While not a full UCI implementation, provides the key methods
    needed by python-chess's UCI interface for evaluation purposes.

    Args:
        model: Trained MaiaNet.
        name: Engine name (e.g., "Maia-1100").
        use_history: Whether model uses history planes.
    """

    def __init__(
        self,
        model: MaiaNet,
        name: str = "Maia",
        use_history: bool = False,
    ):
        self.model = model
        self._name = name
        self.use_history = use_history
        self.device = next(model.parameters()).device

    @property
    def name(self) -> str:
        return self._name

    def play(self, board: chess.Board) -> chess.Move | None:
        """Get the highest-probability legal move for the position.

        Args:
            board: Current board position.

        Returns the most likely human move according to Maia.
        """
        self.model.eval()
        with torch.no_grad():
            tensor = board_to_tensor(board)
            tensor_t = torch.from_numpy(tensor).float().to(self.device)
            tensor_t = tensor_t.permute(2, 0, 1).unsqueeze(0)  # (1, C, 8, 8)

            policy_logits, _ = self.model(tensor_t)
            policy_logits = policy_logits.squeeze(0)  # (73*64,)

            # Mask illegal moves
            legal_mask = torch.zeros(64 * NUM_MOVE_PLANES, dtype=torch.bool, device=self.device)
            for move in board.legal_moves:
                from src.encoding.move import move_to_index
                try:
                    idx = move_to_index(move)
                    legal_mask[idx] = True
                except ValueError:
                    continue

            policy_logits = policy_logits.masked_fill(~legal_mask, -float("inf"))
            probs = torch.softmax(policy_logits, dim=0)

            best_idx = torch.argmax(probs).item()

            from src.encoding.move import index_to_move
            try:
                return index_to_move(best_idx, board)
            except ValueError:
                # Fallback: pick any legal move
                moves = list(board.legal_moves)
                return moves[0] if moves else None

    def get_move_probabilities(
        self, board: chess.Board
    ) -> dict[str, float]:
        """Get probability distribution over legal moves.

        Returns dict mapping UCI string to probability.
        """
        self.model.eval()
        result = {}
        with torch.no_grad():
            tensor = board_to_tensor(board)
            tensor_t = torch.from_numpy(tensor).float().to(self.device)
            tensor_t = tensor_t.permute(2, 0, 1).unsqueeze(0)

            policy_logits, _ = self.model(tensor_t)
            policy_logits = policy_logits.squeeze(0)

            from src.encoding.move import move_to_index, index_to_move

            legal_moves = list(board.legal_moves)
            legal_indices = []
            for m in legal_moves:
                try:
                    legal_indices.append(move_to_index(m))
                except ValueError:
                    continue

            legal_mask = torch.zeros(64 * NUM_MOVE_PLANES, dtype=torch.bool, device=self.device)
            for idx in legal_indices:
                legal_mask[idx] = True

            policy_logits = policy_logits.masked_fill(~legal_mask, -float("inf"))
            probs = torch.softmax(policy_logits, dim=0)

            for move, idx in zip(legal_moves, legal_indices):
                result[move.uci()] = probs[idx].item()

        return result
