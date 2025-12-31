"""Tests for DataSource class."""

import numpy as np
import pandas as pd
import pytest

from fusiongan import DataSource


class TestDataSourceInit:
    """Test DataSource initialization."""

    def test_creates_from_dataframe(self):
        df = pd.DataFrame({"age": [25, 30, 35], "income": [50000, 60000, 70000]})
        source = DataSource("test", df)
        assert source.name == "test"
        assert len(source) == 3

    def test_infers_columns_from_dataframe(self):
        df = pd.DataFrame({"age": [25, 30], "income": [50000, 60000]})
        source = DataSource("test", df)
        assert source.columns == ["age", "income"]

    def test_accepts_explicit_columns(self):
        df = pd.DataFrame({"age": [25], "income": [50000], "extra": [1]})
        source = DataSource("test", df, columns=["age", "income"])
        assert source.columns == ["age", "income"]
        assert "extra" not in source.columns

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"age": [25]})
        with pytest.raises(ValueError, match="not found"):
            DataSource("test", df, columns=["age", "income"])

    def test_stores_column_indices(self):
        """Column indices start empty and are set by synthesizer."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        source = DataSource("test", df, columns=["a", "c"])
        # Indices are empty until integrated into a schema
        assert source.column_indices == []
        # After integration, they map to positions in full schema
        source.column_indices = [0, 2]
        assert source.column_indices == [0, 2]


class TestDataSourceSplit:
    """Test train/holdout splitting."""

    def test_splits_into_train_holdout(self):
        df = pd.DataFrame({"x": range(100)})
        source = DataSource("test", df)
        train, holdout = source.split(holdout_frac=0.2, seed=42)

        assert len(train) == 80
        assert len(holdout) == 20

    def test_split_is_deterministic_with_seed(self):
        df = pd.DataFrame({"x": range(100)})
        source = DataSource("test", df)

        train1, holdout1 = source.split(holdout_frac=0.2, seed=42)
        train2, holdout2 = source.split(holdout_frac=0.2, seed=42)

        pd.testing.assert_frame_equal(holdout1, holdout2)

    def test_split_is_different_with_different_seeds(self):
        df = pd.DataFrame({"x": range(100)})
        source = DataSource("test", df)

        _, holdout1 = source.split(holdout_frac=0.2, seed=42)
        _, holdout2 = source.split(holdout_frac=0.2, seed=43)

        assert not holdout1.equals(holdout2)

    def test_split_preserves_columns(self):
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        source = DataSource("test", df, columns=["a"])
        train, holdout = source.split(holdout_frac=0.2)

        assert list(train.columns) == ["a"]
        assert list(holdout.columns) == ["a"]


class TestDataSourceProject:
    """Test projection of full records to source columns."""

    def test_projects_tensor_to_source_columns(self):
        import torch

        # Source only observes columns 0 and 2
        df = pd.DataFrame({"a": [1], "c": [3]})
        source = DataSource("test", df, columns=["a", "c"])
        source.column_indices = [0, 2]  # In full schema

        full = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # 4 columns
        projected = source.project(full)

        assert projected.shape == (1, 2)
        assert projected[0, 0] == 1.0  # column 0
        assert projected[0, 1] == 3.0  # column 2

    def test_projects_numpy_array(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        source = DataSource("test", df)
        source.column_indices = [1, 3]

        full = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        projected = source.project(full)

        assert projected.shape == (2, 2)
        np.testing.assert_array_equal(projected, [[1, 3], [6, 8]])


class TestDataSourceWeights:
    """Test weight computation for rare types."""

    def test_computes_cluster_weights(self):
        # Create data with one rare cluster
        common = np.random.randn(90, 2)
        rare = np.random.randn(10, 2) + 10  # Shifted cluster
        data = np.vstack([common, rare])
        df = pd.DataFrame(data, columns=["x", "y"])

        source = DataSource("test", df)
        weights = source.compute_weights(method="cluster", n_clusters=2)

        assert len(weights) == 100
        assert weights.sum() == pytest.approx(100, rel=0.01)  # Normalized
        # Rare cluster should have higher per-record weight
        assert weights[90:].mean() > weights[:90].mean()

    def test_computes_uniform_weights(self):
        df = pd.DataFrame({"x": range(100)})
        source = DataSource("test", df)
        weights = source.compute_weights(method="uniform")

        assert len(weights) == 100
        assert (weights == 1.0).all()

    def test_computes_density_weights(self):
        # Dense region vs sparse region
        dense = np.zeros((80, 2))
        sparse = np.random.randn(20, 2) * 5
        data = np.vstack([dense, sparse])
        df = pd.DataFrame(data, columns=["x", "y"])

        source = DataSource("test", df)
        weights = source.compute_weights(method="density")

        assert len(weights) == 100
        # Sparse region should have higher weight
        assert weights[80:].mean() > weights[:80].mean()


class TestDataSourceTensor:
    """Test conversion to tensors."""

    def test_to_tensor(self):
        import torch

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        source = DataSource("test", df)
        tensor = source.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
        assert tensor.dtype == torch.float32

    def test_to_tensor_subset(self):
        import torch

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        source = DataSource("test", df)
        tensor = source.to_tensor(indices=[0])

        assert tensor.shape == (1, 2)
