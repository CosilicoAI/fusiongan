"""Evaluation metrics for synthetic data quality."""

from __future__ import annotations

import numpy as np


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between two sets of points.

    Args:
        a: Array of shape (n, d).
        b: Array of shape (m, d).

    Returns:
        Distance matrix of shape (n, m) where entry [i, j] is ||a[i] - b[j]||.
    """
    # Efficient computation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    a_sq = np.sum(a**2, axis=1, keepdims=True)  # (n, 1)
    b_sq = np.sum(b**2, axis=1, keepdims=True)  # (m, 1)
    ab = a @ b.T  # (n, m)

    dists_sq = a_sq + b_sq.T - 2 * ab
    # Clip small negative values from numerical error
    dists_sq = np.maximum(dists_sq, 0)
    return np.sqrt(dists_sq)


def compute_coverage(
    real: np.ndarray,
    synthetic: np.ndarray,
    k: int = 1,
) -> float:
    """Compute coverage: mean k-NN distance from real to synthetic.

    Lower is better - indicates synthetic data covers the real distribution.

    Args:
        real: Real data of shape (n, d).
        synthetic: Synthetic data of shape (m, d).
        k: Number of nearest neighbors to average.

    Returns:
        Mean distance from each real point to its k nearest synthetic points.
    """
    dists = pairwise_distances(real, synthetic)

    # For each real point, find k nearest synthetic
    if k == 1:
        min_dists = np.min(dists, axis=1)
    else:
        # Get k smallest distances for each real point
        k_nearest = np.partition(dists, k - 1, axis=1)[:, :k]
        min_dists = np.mean(k_nearest, axis=1)

    return float(np.mean(min_dists))


def compute_discriminator_accuracy(
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute discriminator classification accuracy.

    Args:
        real_scores: Discriminator outputs for real samples.
        fake_scores: Discriminator outputs for fake samples.
        threshold: Classification threshold (default 0.5).

    Returns:
        Accuracy in [0, 1]. Near 0.5 indicates generator fools discriminator.
    """
    real_correct = np.sum(real_scores >= threshold)
    fake_correct = np.sum(fake_scores < threshold)
    total = len(real_scores) + len(fake_scores)

    return float((real_correct + fake_correct) / total)


def compute_mmd(
    a: np.ndarray,
    b: np.ndarray,
    gamma: float | None = None,
) -> float:
    """Compute Maximum Mean Discrepancy with RBF kernel.

    MMD measures distribution difference: 0 for identical distributions,
    larger values for more different distributions.

    Args:
        a: First sample of shape (n, d).
        b: Second sample of shape (m, d).
        gamma: RBF kernel bandwidth parameter. If None, uses median heuristic.

    Returns:
        MMD^2 value (non-negative).
    """
    # Use median heuristic for gamma if not specified
    if gamma is None:
        all_data = np.vstack([a, b])
        # Compute pairwise distances and use median as bandwidth
        dists = pairwise_distances(all_data, all_data)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist**2 + 1e-8)

    # RBF kernel: k(x, y) = exp(-gamma * ||x - y||^2)
    def rbf_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dists_sq = pairwise_distances(x, y) ** 2
        return np.exp(-gamma * dists_sq)

    k_aa = rbf_kernel(a, a)
    k_bb = rbf_kernel(b, b)
    k_ab = rbf_kernel(a, b)

    n, m = len(a), len(b)

    # MMD^2 = E[k(a,a')] + E[k(b,b')] - 2*E[k(a,b)]
    # For unbiased estimate, exclude diagonal for self-comparisons
    mmd_sq = (
        (k_aa.sum() - np.trace(k_aa)) / (n * (n - 1))
        + (k_bb.sum() - np.trace(k_bb)) / (m * (m - 1))
        - 2 * k_ab.mean()
    )

    # Return sqrt for interpretability, clamp negatives from numerical issues
    return float(np.sqrt(max(mmd_sq, 0)))
