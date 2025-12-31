"""Tests for evaluation metrics."""

import numpy as np
import pytest
import torch

from fusiongan.metrics import (
    compute_coverage,
    compute_discriminator_accuracy,
    compute_mmd,
    pairwise_distances,
)


class TestPairwiseDistances:
    """Test distance computation."""

    def test_pairwise_shape(self):
        a = np.random.randn(10, 5)
        b = np.random.randn(20, 5)
        dists = pairwise_distances(a, b)

        assert dists.shape == (10, 20)

    def test_self_distance_zero(self):
        a = np.random.randn(10, 5)
        dists = pairwise_distances(a, a)

        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(dists), 0)

    def test_symmetric(self):
        a = np.random.randn(10, 5)
        b = np.random.randn(10, 5)

        dists_ab = pairwise_distances(a, b)
        dists_ba = pairwise_distances(b, a)

        np.testing.assert_array_almost_equal(dists_ab, dists_ba.T)

    def test_known_distance(self):
        a = np.array([[0, 0]])
        b = np.array([[3, 4]])
        dists = pairwise_distances(a, b)

        assert dists[0, 0] == pytest.approx(5.0)  # 3-4-5 triangle


class TestCoverage:
    """Test coverage metric."""

    def test_perfect_coverage(self):
        """Identical distributions should have minimal coverage distance."""
        np.random.seed(42)
        real = np.random.randn(100, 5)
        synthetic = real.copy()  # Exact match

        coverage = compute_coverage(real, synthetic)
        assert coverage < 0.01

    def test_poor_coverage(self):
        """Disjoint distributions should have high coverage distance."""
        real = np.random.randn(100, 5)
        synthetic = np.random.randn(100, 5) + 10  # Shifted far away

        coverage = compute_coverage(real, synthetic)
        assert coverage > 5.0

    def test_coverage_is_mean_min_distance(self):
        """Coverage should be mean distance to nearest synthetic."""
        real = np.array([[0, 0], [1, 1]])
        synthetic = np.array([[0.1, 0.1], [10, 10]])

        coverage = compute_coverage(real, synthetic)

        # First real point: nearest is [0.1, 0.1], dist ≈ 0.14
        # Second real point: nearest is [0.1, 0.1], dist ≈ 1.27
        expected = (np.sqrt(0.02) + np.sqrt(1.62)) / 2
        assert coverage == pytest.approx(expected, rel=0.01)

    def test_coverage_with_k(self):
        """Coverage with k>1 should average k nearest distances."""
        real = np.array([[0, 0]])
        synthetic = np.array([[1, 0], [2, 0], [3, 0]])

        coverage_k1 = compute_coverage(real, synthetic, k=1)
        coverage_k2 = compute_coverage(real, synthetic, k=2)
        coverage_k3 = compute_coverage(real, synthetic, k=3)

        assert coverage_k1 == pytest.approx(1.0)  # Nearest is at dist 1
        assert coverage_k2 == pytest.approx(1.5)  # Mean of 1, 2
        assert coverage_k3 == pytest.approx(2.0)  # Mean of 1, 2, 3


class TestDiscriminatorAccuracy:
    """Test discriminator accuracy computation."""

    def test_perfect_discrimination(self):
        """Should return 1.0 if discriminator perfectly separates."""
        real_scores = np.array([0.9, 0.95, 0.85, 0.99])
        fake_scores = np.array([0.1, 0.05, 0.15, 0.01])

        acc = compute_discriminator_accuracy(real_scores, fake_scores)
        assert acc == 1.0

    def test_random_discrimination(self):
        """Should return ~0.5 if discriminator is random."""
        np.random.seed(42)
        real_scores = np.random.rand(1000)
        fake_scores = np.random.rand(1000)

        acc = compute_discriminator_accuracy(real_scores, fake_scores)
        assert 0.45 < acc < 0.55

    def test_inverted_discrimination(self):
        """Should return 0.0 if discriminator is completely wrong."""
        real_scores = np.array([0.1, 0.05])
        fake_scores = np.array([0.9, 0.95])

        acc = compute_discriminator_accuracy(real_scores, fake_scores)
        assert acc == 0.0


class TestMMD:
    """Test Maximum Mean Discrepancy."""

    def test_same_distribution_low_mmd(self):
        """MMD should be near zero for same distribution."""
        np.random.seed(42)
        a = np.random.randn(500, 5)
        b = np.random.randn(500, 5)

        mmd = compute_mmd(a, b)
        assert mmd < 0.1

    def test_different_distribution_high_mmd(self):
        """MMD should be high for different distributions."""
        a = np.random.randn(500, 5)
        b = np.random.randn(500, 5) + 3

        mmd = compute_mmd(a, b)
        # With median heuristic, MMD for shifted distributions should be substantial
        assert mmd > 0.5

    def test_mmd_is_symmetric(self):
        """MMD(a, b) should equal MMD(b, a)."""
        a = np.random.randn(100, 5)
        b = np.random.randn(100, 5) + 1

        mmd_ab = compute_mmd(a, b)
        mmd_ba = compute_mmd(b, a)

        assert mmd_ab == pytest.approx(mmd_ba, rel=0.01)

    def test_mmd_zero_for_identical(self):
        """MMD should be exactly zero for identical samples."""
        a = np.random.randn(100, 5)

        mmd = compute_mmd(a, a)
        assert mmd == pytest.approx(0, abs=1e-6)
