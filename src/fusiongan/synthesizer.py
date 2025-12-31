"""MultiSourceSynthesizer: GAN-based synthesis from multiple data sources."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fusiongan.data_source import DataSource
from fusiongan.discriminator import Discriminator
from fusiongan.generator import Generator
from fusiongan.metrics import compute_coverage, compute_discriminator_accuracy


class MultiSourceSynthesizer:
    """Multi-source adversarial synthesizer.

    Uses a single generator that produces complete records, with multiple
    discriminators (one per source) that evaluate projections to each
    source's observed columns.

    Attributes:
        sources: List of DataSource objects.
        full_schema: Combined column names across all sources.
        generator: Generator network.
        discriminators: Dict mapping source name to Discriminator.
        weighting: Sample weighting strategy.
    """

    def __init__(
        self,
        sources: List[DataSource],
        weighting: str = "uniform",
        latent_dim: int = 32,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        """Initialize the synthesizer.

        Args:
            sources: List of DataSource objects.
            weighting: Sample weighting strategy ("uniform", "cluster", "density").
            latent_dim: Dimension of generator latent space.
            hidden_dim: Hidden layer width for networks.
            device: Device to train on.
        """
        self.sources = sources
        self.weighting = weighting
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Build full schema from all sources
        self._build_schema()

        # Create networks
        self.generator = Generator(
            latent_dim=latent_dim,
            output_dim=len(self.full_schema),
            hidden_dim=hidden_dim,
        ).to(device)

        self.discriminators: Dict[str, Discriminator] = {}
        for source in sources:
            self.discriminators[source.name] = Discriminator(
                input_dim=len(source.columns),
                hidden_dim=hidden_dim,
            ).to(device)

        # Training state
        self.is_fitted = False
        self.holdouts: Dict[str, pd.DataFrame] = {}
        self.train_data: Dict[str, pd.DataFrame] = {}
        self.weights: Dict[str, np.ndarray] = {}
        self.history: Dict = {
            "generator_loss": [],
            "discriminator_loss": defaultdict(list),
        }

    def _build_schema(self) -> None:
        """Build full schema and set column indices for each source."""
        # Collect all unique columns preserving order
        seen = set()
        self.full_schema: List[str] = []
        for source in self.sources:
            for col in source.columns:
                if col not in seen:
                    self.full_schema.append(col)
                    seen.add(col)

        # Map column names to indices
        col_to_idx = {col: i for i, col in enumerate(self.full_schema)}

        # Set column indices for each source
        for source in self.sources:
            source.column_indices = [col_to_idx[col] for col in source.columns]

    def fit(
        self,
        epochs: int = 100,
        holdout_frac: float = 0.2,
        batch_size: int = 64,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        n_critic: int = 1,
        verbose: bool = True,
    ) -> "MultiSourceSynthesizer":
        """Train the synthesizer.

        Args:
            epochs: Number of training epochs.
            holdout_frac: Fraction of data to hold out for evaluation.
            batch_size: Training batch size.
            lr_g: Generator learning rate.
            lr_d: Discriminator learning rate.
            n_critic: Discriminator steps per generator step.
            verbose: Whether to print progress.

        Returns:
            self for chaining.
        """
        # Split data and compute weights
        for source in self.sources:
            train, holdout = source.split(holdout_frac=holdout_frac, seed=42)
            self.train_data[source.name] = train
            self.holdouts[source.name] = holdout

            # Compute sample weights on training data
            train_source = DataSource(source.name, train, source.columns)
            self.weights[source.name] = train_source.compute_weights(self.weighting)

        # Setup optimizers
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        d_optimizers = {
            name: torch.optim.Adam(d.parameters(), lr=lr_d, betas=(0.5, 0.999))
            for name, d in self.discriminators.items()
        }

        for epoch in range(epochs):
            g_losses = []
            d_losses = defaultdict(list)

            # Sample batches from each source
            for source in self.sources:
                train_df = self.train_data[source.name]
                weights = self.weights[source.name]
                n_train = len(train_df)

                # Number of batches for this source
                n_batches = max(1, n_train // batch_size)

                for _ in range(n_batches):
                    # Sample batch with weights
                    probs = weights / weights.sum()
                    batch_idx = np.random.choice(n_train, size=min(batch_size, n_train), p=probs)
                    real_batch = torch.tensor(
                        train_df.iloc[batch_idx].values,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    batch_weights = torch.tensor(
                        weights[batch_idx],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    # Train discriminator for this source
                    for _ in range(n_critic):
                        d_optimizer = d_optimizers[source.name]
                        d_optimizer.zero_grad()

                        # Real samples
                        real_scores = self.discriminators[source.name](real_batch)

                        # Fake samples - generate full records and project
                        with torch.no_grad():
                            fake_full = self.generator.sample(len(real_batch), device=self.device)
                        fake_projected = fake_full[:, source.column_indices]
                        fake_scores = self.discriminators[source.name](fake_projected)

                        # Weighted BCE loss
                        real_loss = -(batch_weights * torch.log(real_scores.squeeze() + 1e-8)).mean()
                        fake_loss = -torch.log(1 - fake_scores + 1e-8).mean()
                        d_loss = real_loss + fake_loss

                        d_loss.backward()
                        d_optimizer.step()

                        d_losses[source.name].append(d_loss.item())

                    # Train generator
                    g_optimizer.zero_grad()

                    # Generate and project
                    fake_full = self.generator.sample(batch_size, device=self.device)
                    fake_projected = fake_full[:, source.column_indices]
                    fake_scores = self.discriminators[source.name](fake_projected)

                    g_loss = -torch.log(fake_scores + 1e-8).mean()
                    g_loss.backward()
                    g_optimizer.step()

                    g_losses.append(g_loss.item())

            # Record epoch losses
            self.history["generator_loss"].append(np.mean(g_losses))
            for name in d_losses:
                self.history["discriminator_loss"][name].append(np.mean(d_losses[name]))

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - G loss: {np.mean(g_losses):.4f}")

        self.is_fitted = True
        return self

    def generate(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic records.

        Args:
            n: Number of records to generate.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with full schema columns.
        """
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            samples = self.generator.sample(n, device=self.device)

        return pd.DataFrame(
            samples.cpu().numpy(),
            columns=self.full_schema,
        )

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Evaluate synthetic data quality per source.

        Returns:
            Dict mapping source name to metrics dict with:
                - coverage: Mean nearest-neighbor distance to holdout
                - discriminator_accuracy: How well discriminator distinguishes
        """
        metrics = {}

        # Generate samples for evaluation
        n_eval = max(len(h) for h in self.holdouts.values())
        synthetic = self.generate(n=n_eval)

        for source in self.sources:
            holdout = self.holdouts[source.name]

            # Project synthetic to this source's columns
            synthetic_proj = synthetic[source.columns].values
            holdout_vals = holdout.values

            # Coverage
            coverage = compute_coverage(holdout_vals, synthetic_proj)

            # Discriminator accuracy
            with torch.no_grad():
                real_tensor = torch.tensor(holdout_vals, dtype=torch.float32, device=self.device)
                fake_tensor = torch.tensor(synthetic_proj, dtype=torch.float32, device=self.device)

                real_scores = self.discriminators[source.name](real_tensor).cpu().numpy()
                fake_scores = self.discriminators[source.name](fake_tensor).cpu().numpy()

            disc_acc = compute_discriminator_accuracy(real_scores, fake_scores)

            metrics[source.name] = {
                "coverage": coverage,
                "discriminator_accuracy": disc_acc,
            }

        return metrics
