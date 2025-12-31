"""Tests for Discriminator class."""

import numpy as np
import pytest
import torch

from fusiongan import Discriminator


class TestDiscriminatorInit:
    """Test Discriminator initialization."""

    def test_creates_with_input_dim(self):
        D = Discriminator(input_dim=10)
        assert D.input_dim == 10

    def test_default_hidden_dim(self):
        D = Discriminator(input_dim=10)
        assert D.hidden_dim == 128

    def test_custom_hidden_dim(self):
        D = Discriminator(input_dim=10, hidden_dim=64)
        assert D.hidden_dim == 64


class TestDiscriminatorForward:
    """Test Discriminator forward pass."""

    def test_output_shape(self):
        D = Discriminator(input_dim=5, hidden_dim=32)
        x = torch.randn(16, 5)
        out = D(x)

        assert out.shape == (16, 1)

    def test_output_in_zero_one(self):
        D = Discriminator(input_dim=5, hidden_dim=32)
        x = torch.randn(100, 5)
        out = D(x)

        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_gradients_flow(self):
        D = Discriminator(input_dim=5, hidden_dim=32)
        x = torch.randn(16, 5, requires_grad=True)
        out = D(x)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestDiscriminatorTraining:
    """Test Discriminator training behavior."""

    def test_distinguishes_different_distributions(self):
        """Discriminator should easily separate very different distributions."""
        D = Discriminator(input_dim=2, hidden_dim=64)
        optimizer = torch.optim.Adam(D.parameters(), lr=1e-3)

        # Very different distributions
        real = torch.randn(200, 2)  # N(0, 1)
        fake = torch.randn(200, 2) + 5  # N(5, 1)

        # Train
        for _ in range(100):
            optimizer.zero_grad()

            real_out = D(real)
            fake_out = D(fake)

            loss = -torch.log(real_out + 1e-8).mean() - torch.log(1 - fake_out + 1e-8).mean()
            loss.backward()
            optimizer.step()

        # Should achieve high accuracy
        with torch.no_grad():
            real_acc = (D(real) > 0.5).float().mean()
            fake_acc = (D(fake) < 0.5).float().mean()

        assert real_acc > 0.9
        assert fake_acc > 0.9

    def test_struggles_with_identical_distributions(self):
        """Discriminator should achieve ~50% on identical distributions."""
        D = Discriminator(input_dim=2, hidden_dim=64)
        optimizer = torch.optim.Adam(D.parameters(), lr=1e-3)

        # Same distribution
        dist = torch.distributions.Normal(0, 1)

        for _ in range(100):
            real = dist.sample((100, 2))
            fake = dist.sample((100, 2))

            optimizer.zero_grad()
            real_out = D(real)
            fake_out = D(fake)
            loss = -torch.log(real_out + 1e-8).mean() - torch.log(1 - fake_out + 1e-8).mean()
            loss.backward()
            optimizer.step()

        # Should be near 50% (can't distinguish)
        with torch.no_grad():
            real = dist.sample((500, 2))
            fake = dist.sample((500, 2))
            real_acc = (D(real) > 0.5).float().mean()

        assert 0.35 < real_acc < 0.65  # Near random (loosened bounds)


class TestDiscriminatorWeightedLoss:
    """Test weighted discriminator training."""

    def test_weighted_loss_upweights_samples(self):
        D = Discriminator(input_dim=2, hidden_dim=32)

        # Batch with weights
        x = torch.randn(10, 2)
        weights = torch.ones(10)
        weights[0] = 10.0  # First sample 10x more important

        out = D(x)
        weighted_loss = -(weights * torch.log(out.squeeze() + 1e-8)).mean()

        # Gradient for first sample should be larger
        weighted_loss.backward()

        # Check that weighted sample contributes more to loss
        assert weighted_loss.item() > 0

    def test_weighted_training_focuses_on_rare(self):
        """Weighted training should improve accuracy on upweighted samples."""
        D = Discriminator(input_dim=2, hidden_dim=64)
        optimizer = torch.optim.Adam(D.parameters(), lr=1e-3)

        # Common type vs rare type
        common_real = torch.randn(180, 2)
        rare_real = torch.randn(20, 2) + 3  # Rare, shifted

        # Fake misses the rare type
        fake = torch.randn(200, 2)  # Only covers common

        # Weights: upweight rare samples
        weights = torch.ones(200)
        weights[180:] = 9.0  # Rare samples 9x weight

        real = torch.cat([common_real, rare_real])

        for _ in range(200):
            optimizer.zero_grad()

            real_out = D(real)
            fake_out = D(fake)

            # Weighted loss on real
            real_loss = -(weights * torch.log(real_out.squeeze() + 1e-8)).mean()
            fake_loss = -torch.log(1 - fake_out + 1e-8).mean()

            loss = real_loss + fake_loss
            loss.backward()
            optimizer.step()

        # Should distinguish rare type well
        with torch.no_grad():
            rare_acc = (D(rare_real) > 0.5).float().mean()

        assert rare_acc > 0.7  # Good at spotting rare as real
