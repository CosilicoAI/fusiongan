"""Generator: produces synthetic records from latent noise."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator network for producing synthetic records.

    Maps latent noise vectors to synthetic records. Optionally
    supports conditional generation.

    Attributes:
        latent_dim: Dimension of latent noise vectors.
        output_dim: Dimension of output records.
        hidden_dim: Width of hidden layers.
        condition_dim: Dimension of conditioning vectors (if any).
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        condition_dim: Optional[int] = None,
    ):
        """Initialize the Generator.

        Args:
            latent_dim: Dimension of latent noise vectors.
            output_dim: Dimension of output records.
            hidden_dim: Width of hidden layers.
            condition_dim: Dimension of conditioning vectors (optional).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # Input dimension includes condition if provided
        input_dim = latent_dim + (condition_dim or 0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate records from latent vectors.

        Args:
            z: Latent noise tensor of shape (batch, latent_dim).
            condition: Optional conditioning tensor of shape (batch, condition_dim).

        Returns:
            Generated records of shape (batch, output_dim).
        """
        if condition is not None:
            z = torch.cat([z, condition], dim=1)
        return self.net(z)

    def sample(
        self,
        n: int,
        condition: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Sample n synthetic records.

        Args:
            n: Number of records to generate.
            condition: Optional conditioning tensor of shape (n, condition_dim).
            device: Device to generate on.

        Returns:
            Generated records of shape (n, output_dim).
        """
        z = torch.randn(n, self.latent_dim, device=device)
        return self.forward(z, condition)
