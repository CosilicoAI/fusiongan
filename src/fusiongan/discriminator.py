"""Discriminator: classifies records as real (from source) or synthetic."""

from __future__ import annotations

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator network for distinguishing real from synthetic records.

    Uses a simple MLP with LeakyReLU activations and sigmoid output
    to produce probability scores in [0, 1].

    Attributes:
        input_dim: Dimension of input records.
        hidden_dim: Width of hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ):
        """Initialize the Discriminator.

        Args:
            input_dim: Dimension of input records.
            hidden_dim: Width of hidden layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify records as real or fake.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Probability scores of shape (batch, 1), where higher
            values indicate "more real".
        """
        return self.net(x)
