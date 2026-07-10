"""Residual CNN for blunder prediction.

Same trunk architecture as MaiaNet (6 residual blocks, 64 channels, 3x3 kernels)
but with a single scalar output head (blunder probability) instead of policy+value heads.

Includes a deeper variant (8 blocks, 256 filters) for collective blunder prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.maia_net import ResidualBlock


class BlunderResCNN(nn.Module):
    """Residual CNN for blunder prediction.

    Args:
        in_channels: Number of input channels (17 for board-only,
                     22 for board+metadata).
        channels: Number of conv filters (default 64).
        blocks: Number of residual blocks (default 6).
        kernel_size: Convolution kernel size (default 3).
    """

    def __init__(
        self,
        in_channels: int = 17,
        channels: int = 64,
        blocks: int = 6,
    ):
        super().__init__()

        # Initial convolution
        self.conv_init = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(channels)

        # Residual trunk
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(blocks)
        ])

        # Output head
        self.head_conv = nn.Conv2d(channels, 16, 3, padding=1, bias=False)
        self.head_bn = nn.BatchNorm2d(16)
        self.head_fc = nn.Linear(16 * 8 * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, C, 8, 8).

        Returns:
            Blunder probability (batch, 1) in [0, 1].
        """
        x = F.relu(self.bn_init(self.conv_init(x)))

        for block in self.res_blocks:
            x = block(x)

        x = F.relu(self.head_bn(self.head_conv(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.head_fc(x)
        x = torch.sigmoid(x)
        return x


class DeepBlunderResCNN(nn.Module):
    """Deeper residual CNN for collective blunder prediction.

    Args:
        in_channels: Number of input channels.
        channels: Number of conv filters (default 256).
        blocks: Number of residual blocks (default 8).
    """

    def __init__(
        self,
        in_channels: int = 17,
        channels: int = 256,
        blocks: int = 8,
    ):
        super().__init__()

        self.conv_init = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(blocks)
        ])

        self.head_conv = nn.Conv2d(channels, 32, 3, padding=1, bias=False)
        self.head_bn = nn.BatchNorm2d(32)
        self.head_fc = nn.Linear(32 * 8 * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_init(self.conv_init(x)))

        for block in self.res_blocks:
            x = block(x)

        x = F.relu(self.head_bn(self.head_conv(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.head_fc(x)
        x = torch.sigmoid(x)
        return x
