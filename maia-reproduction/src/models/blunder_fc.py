"""Fully-connected neural network for blunder prediction.

Architecture: 1028 -> 512 -> 256 -> 1 output (sigmoid)
"""

import torch
import torch.nn as nn


class BlunderFC(nn.Module):
    """Fully-connected network for blunder prediction.

    Takes flattened board tensor (8*8*C) and outputs a single scalar
    representing probability of a blunder.
    """

    def __init__(self, in_channels: int = 17):
        super().__init__()
        input_dim = 8 * 8 * in_channels
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1028),
            nn.Tanh(),
            nn.Linear(1028, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, C, 8, 8).

        Returns:
            Blunder probability (batch, 1) in [0, 1].
        """
        b = x.shape[0]
        x = x.reshape(b, -1)
        return self.net(x)
