"""DataSource: represents a single survey data source."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


class DataSource:
    """A single data source (e.g., CPS, PUF) with observed columns.

    Attributes:
        name: Identifier for this source.
        data: The underlying DataFrame.
        columns: List of column names observed in this source.
        column_indices: Indices of these columns in the full schema.
    """

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ):
        """Initialize a DataSource.

        Args:
            name: Identifier for this source (e.g., "cps", "puf").
            data: DataFrame containing the source data.
            columns: Columns observed in this source. If None, uses all columns.

        Raises:
            ValueError: If specified columns are not found in data.
        """
        self.name = name
        self._data = data

        if columns is None:
            self.columns = list(data.columns)
        else:
            # Validate columns exist
            missing = set(columns) - set(data.columns)
            if missing:
                raise ValueError(f"Columns not found in data: {missing}")
            self.columns = columns

        # Will be set when integrated into full schema
        self.column_indices: List[int] = []

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> pd.DataFrame:
        """Return data with only observed columns."""
        return self._data[self.columns]

    def split(
        self,
        holdout_frac: float = 0.2,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and holdout sets.

        Args:
            holdout_frac: Fraction of data to hold out.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train, holdout) DataFrames.
        """
        rng = np.random.RandomState(seed)
        n = len(self)
        n_holdout = int(n * holdout_frac)

        indices = rng.permutation(n)
        holdout_idx = indices[:n_holdout]
        train_idx = indices[n_holdout:]

        return (
            self.data.iloc[train_idx].reset_index(drop=True),
            self.data.iloc[holdout_idx].reset_index(drop=True),
        )

    def project(self, full_data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Project full records to this source's columns.

        Args:
            full_data: Array of shape (n, full_dim) with all variables.

        Returns:
            Array of shape (n, len(self.columns)) with only this source's columns.
        """
        if isinstance(full_data, torch.Tensor):
            return full_data[:, self.column_indices]
        else:
            return full_data[:, self.column_indices]

    def compute_weights(
        self,
        method: str = "uniform",
        n_clusters: int = 20,
    ) -> np.ndarray:
        """Compute sample weights for training.

        Args:
            method: Weighting method - "uniform", "cluster", or "density".
            n_clusters: Number of clusters for cluster-based weighting.

        Returns:
            Array of weights, one per sample.
        """
        n = len(self)

        if method == "uniform":
            return np.ones(n)

        elif method == "cluster":
            from sklearn.cluster import KMeans

            X = self.data.values
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # Weight inversely by cluster size
            cluster_sizes = np.bincount(labels, minlength=n_clusters)
            weights = 1.0 / np.maximum(cluster_sizes[labels], 1)

            # Normalize to sum to n
            weights = weights / weights.sum() * n
            return weights

        elif method == "density":
            from sklearn.neighbors import KernelDensity

            X = self.data.values
            # Standardize for KDE
            X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

            kde = KernelDensity(bandwidth=0.5)
            kde.fit(X_std)
            log_density = kde.score_samples(X_std)

            # Weight inversely by density
            weights = 1.0 / (np.exp(log_density) + 1e-8)

            # Clip extreme weights
            weights = np.clip(weights, 0, np.percentile(weights, 99))

            # Normalize
            weights = weights / weights.sum() * n
            return weights

        else:
            raise ValueError(f"Unknown weighting method: {method}")

    def to_tensor(
        self,
        indices: Optional[List[int]] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Convert data to PyTorch tensor.

        Args:
            indices: Optional subset of row indices.
            device: Device to place tensor on.

        Returns:
            Float32 tensor of shape (n, n_columns).
        """
        if indices is not None:
            data = self.data.iloc[indices].values
        else:
            data = self.data.values

        return torch.tensor(data, dtype=torch.float32, device=device)
