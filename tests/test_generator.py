"""Tests for Generator class."""

import numpy as np
import pytest
import torch

from fusiongan import Generator


class TestGeneratorInit:
    """Test Generator initialization."""

    def test_creates_with_dims(self):
        G = Generator(latent_dim=32, output_dim=10)
        assert G.latent_dim == 32
        assert G.output_dim == 10

    def test_default_hidden_dim(self):
        G = Generator(latent_dim=32, output_dim=10)
        assert G.hidden_dim == 128

    def test_custom_hidden_dim(self):
        G = Generator(latent_dim=32, output_dim=10, hidden_dim=256)
        assert G.hidden_dim == 256


class TestGeneratorForward:
    """Test Generator forward pass."""

    def test_output_shape(self):
        G = Generator(latent_dim=32, output_dim=10)
        z = torch.randn(16, 32)
        out = G(z)

        assert out.shape == (16, 10)

    def test_output_is_finite(self):
        G = Generator(latent_dim=32, output_dim=10)
        z = torch.randn(100, 32)
        out = G(z)

        assert torch.isfinite(out).all()

    def test_gradients_flow(self):
        G = Generator(latent_dim=32, output_dim=10)
        z = torch.randn(16, 32, requires_grad=True)
        out = G(z)
        loss = out.sum()
        loss.backward()

        assert z.grad is not None


class TestGeneratorSample:
    """Test Generator sampling."""

    def test_sample_shape(self):
        G = Generator(latent_dim=32, output_dim=10)
        samples = G.sample(n=100)

        assert samples.shape == (100, 10)

    def test_sample_is_deterministic_with_seed(self):
        G = Generator(latent_dim=32, output_dim=10)

        torch.manual_seed(42)
        samples1 = G.sample(n=50)

        torch.manual_seed(42)
        samples2 = G.sample(n=50)

        torch.testing.assert_close(samples1, samples2)

    def test_sample_varies_without_seed(self):
        G = Generator(latent_dim=32, output_dim=10)

        samples1 = G.sample(n=50)
        samples2 = G.sample(n=50)

        assert not torch.allclose(samples1, samples2)


class TestGeneratorConditional:
    """Test conditional generation."""

    def test_conditional_forward(self):
        G = Generator(latent_dim=32, output_dim=10, condition_dim=5)
        z = torch.randn(16, 32)
        cond = torch.randn(16, 5)
        out = G(z, condition=cond)

        assert out.shape == (16, 10)

    def test_conditional_sample(self):
        G = Generator(latent_dim=32, output_dim=10, condition_dim=5)
        cond = torch.randn(100, 5)
        samples = G.sample(n=100, condition=cond)

        assert samples.shape == (100, 10)

    def test_different_conditions_produce_different_output(self):
        G = Generator(latent_dim=32, output_dim=10, condition_dim=5)

        torch.manual_seed(42)
        z = torch.randn(10, 32)

        cond1 = torch.zeros(10, 5)
        cond2 = torch.ones(10, 5)

        out1 = G(z, condition=cond1)
        out2 = G(z, condition=cond2)

        # Same noise, different condition → different output
        assert not torch.allclose(out1, out2)


class TestGeneratorTraining:
    """Test Generator training with adversarial loss."""

    def test_generator_improves_with_training(self):
        """Generator should produce samples closer to target distribution."""
        from fusiongan import Discriminator

        torch.manual_seed(42)  # For reproducibility

        G = Generator(latent_dim=32, output_dim=2)
        D = Discriminator(input_dim=2, hidden_dim=64)

        g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Target: N(3, 0.5)
        target_dist = torch.distributions.Normal(3, 0.5)

        for step in range(300):
            # Train discriminator
            d_opt.zero_grad()
            real = target_dist.sample((64, 2))
            fake = G.sample(64)

            d_loss = -torch.log(D(real) + 1e-8).mean() - torch.log(1 - D(fake.detach()) + 1e-8).mean()
            d_loss.backward()
            d_opt.step()

            # Train generator
            g_opt.zero_grad()
            fake = G.sample(64)
            g_loss = -torch.log(D(fake) + 1e-8).mean()
            g_loss.backward()
            g_opt.step()

        # Generated samples should be near target mean (more robust than loss)
        with torch.no_grad():
            samples = G.sample(1000)
            sample_mean = samples.mean().item()

        assert abs(sample_mean - 3) < 1.5  # Within 1.5 of target mean
