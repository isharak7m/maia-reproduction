"""Maia residual CNN with policy and value heads.

Architecture:
- Trunk: 6 residual blocks with SE blocks, 64 filters, 3x3 conv
- Policy head: to 8x8x73 policy distribution
- Value head: to 3-class (win/draw/loss) outcome prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.encoding.board import NUM_BOARD_CHANNELS
from src.encoding.move import NUM_MOVE_PLANES


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1))
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convs, batch norm, and SE block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out


class MaiaNet(nn.Module):
    """Maia neural network: residual CNN with policy and value heads.

    Args:
        in_channels: Number of input channels (17 for board-only,
                     17 + 12*num_history for history variant).
        channels: Number of conv filters (default 64).
        blocks: Number of residual blocks (default 6).
    """

    def __init__(
        self,
        in_channels: int = NUM_BOARD_CHANNELS,
        channels: int = 64,
        blocks: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = blocks

        # Initial convolution
        self.conv_init = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(channels)

        # Residual trunk
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 80, 3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(80)
        self.policy_final = nn.Conv2d(80, NUM_MOVE_PLANES, 3, padding=1, bias=False)

        # Value head
        self.value_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, C, 8, 8).

        Returns:
            (policy_logits, value_probs) where policy_logits has shape
            (batch, 73*64) and value_probs has shape (batch, 3).
        """
        b = x.shape[0]

        # Initial conv
        x = F.relu(self.bn_init(self.conv_init(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.policy_final(policy)  # (b, 73, 8, 8)
        policy = policy.permute(0, 2, 3, 1)  # (b, 8, 8, 73)
        policy = policy.reshape(b, -1)  # (b, 73*64)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(b, -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)
        value = F.softmax(value, dim=1)

        return policy, value

    def get_move_probs(
        self, x: torch.Tensor, legal_moves_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get move probability distribution over the board.

        Args:
            x: Input tensor (batch, C, 8, 8).
            legal_moves_mask: Optional boolean mask of legal moves (batch, 73*64).

        Returns:
            Probability distribution over legal moves (batch, 73*64).
        """
        self.eval()
        with torch.no_grad():
            policy_logits, _ = self.forward(x)

            if legal_moves_mask is not None:
                policy_logits = policy_logits.masked_fill(~legal_moves_mask, -float("inf"))

            probs = F.softmax(policy_logits, dim=1)
            return probs

    def win_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Compute win probability from the value head.

        Returns P(win) + 0.5 * P(draw).
        """
        self.eval()
        with torch.no_grad():
            _, value = self.forward(x)
            # value is (win, draw, loss) from perspective of side to move
            win_prob = value[:, 0] + 0.5 * value[:, 1]
            return win_prob


def create_maia_model(
    rating_bin: int,
    use_history: bool = True,
    num_history: int = 12,
    channels: int = 64,
    blocks: int = 6,
) -> MaiaNet:
    """Create a Maia model configured for a specific rating bin.

    Args:
        rating_bin: Lower bound of the rating bin (e.g., 1100).
        use_history: Whether to include history planes.
        num_history: Number of historical ply (if use_history).

    Returns a MaiaNet instance.
    """
    if use_history:
        in_channels = NUM_BOARD_CHANNELS + 12 * num_history
    else:
        in_channels = NUM_BOARD_CHANNELS

    model = MaiaNet(in_channels=in_channels, channels=channels, blocks=blocks)
    return model


class MaiaLoss(nn.Module):
    """Combined policy + value loss for Maia training.

    Policy: cross-entropy against the human's move.
    Value: MSE against actual game outcome.
    Equally weighted.
    """

    def __init__(self):
        super().__init__()
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()

    def forward(
        self,
        policy_logits: torch.Tensor,
        value_probs: torch.Tensor,
        target_move: torch.Tensor,
        target_outcome: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            policy_logits: Raw logits (batch, 73*64).
            value_probs: Value head output (batch, 3) after softmax.
            target_move: Ground-truth move indices (batch,).
            target_outcome: Ground-truth outcomes (batch, 3) as
                          one-hot [win, draw, loss].

        Returns scalar loss.
        """
        p_loss = self.policy_loss(policy_logits, target_move)
        v_loss = self.value_loss(value_probs, target_outcome)
        return p_loss + v_loss
