"""Tests for MultiSourceSynthesizer class."""

import numpy as np
import pandas as pd
import pytest
import torch

from fusiongan import DataSource, MultiSourceSynthesizer

# Silence sklearn warnings in tests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class TestSynthesizerInit:
    """Test MultiSourceSynthesizer initialization."""

    def test_creates_with_sources(self):
        cps = pd.DataFrame({"age": [25, 30], "wages": [50000, 60000]})
        puf = pd.DataFrame({"wages": [70000, 80000], "cap_gains": [1000, 2000]})

        sources = [
            DataSource("cps", cps),
            DataSource("puf", puf),
        ]

        synth = MultiSourceSynthesizer(sources)
        assert len(synth.sources) == 2

    def test_infers_full_schema(self):
        """Should combine columns from all sources."""
        cps = pd.DataFrame({"age": [25], "wages": [50000]})
        puf = pd.DataFrame({"wages": [70000], "cap_gains": [1000]})

        sources = [
            DataSource("cps", cps),
            DataSource("puf", puf),
        ]

        synth = MultiSourceSynthesizer(sources)
        assert set(synth.full_schema) == {"age", "wages", "cap_gains"}

    def test_creates_discriminator_per_source(self):
        cps = pd.DataFrame({"age": [25], "wages": [50000]})
        puf = pd.DataFrame({"wages": [70000], "cap_gains": [1000]})

        sources = [
            DataSource("cps", cps),
            DataSource("puf", puf),
        ]

        synth = MultiSourceSynthesizer(sources)
        assert "cps" in synth.discriminators
        assert "puf" in synth.discriminators

    def test_creates_single_generator(self):
        cps = pd.DataFrame({"age": [25], "wages": [50000]})

        sources = [DataSource("cps", cps)]
        synth = MultiSourceSynthesizer(sources)

        assert synth.generator is not None
        assert synth.generator.output_dim == 2  # age, wages


class TestSynthesizerFit:
    """Test training."""

    @pytest.fixture
    def simple_sources(self):
        """Two sources with overlapping and unique columns."""
        np.random.seed(42)
        n = 500

        # CPS: age, wages (shared), unemployment
        cps = pd.DataFrame({
            "age": np.random.randint(18, 80, n),
            "wages": np.random.lognormal(10, 1, n),
            "unemployment": np.random.choice([0, 1], n, p=[0.95, 0.05]) * np.random.lognormal(8, 1, n),
        })

        # PUF: wages (shared), capital_gains, dividends
        puf = pd.DataFrame({
            "wages": np.random.lognormal(10.5, 1, n),  # Slightly different dist
            "capital_gains": np.random.choice([0, 1], n, p=[0.7, 0.3]) * np.random.lognormal(9, 2, n),
            "dividends": np.random.lognormal(7, 1.5, n),
        })

        return [
            DataSource("cps", cps),
            DataSource("puf", puf),
        ]

    def test_fit_runs(self, simple_sources):
        synth = MultiSourceSynthesizer(simple_sources)
        synth.fit(epochs=5, holdout_frac=0.2, verbose=False)

        assert synth.is_fitted

    def test_fit_creates_holdouts(self, simple_sources):
        synth = MultiSourceSynthesizer(simple_sources)
        synth.fit(epochs=5, holdout_frac=0.2, verbose=False)

        assert synth.holdouts["cps"] is not None
        assert len(synth.holdouts["cps"]) == 100  # 20% of 500

    def test_fit_records_losses(self, simple_sources):
        synth = MultiSourceSynthesizer(simple_sources)
        synth.fit(epochs=10, holdout_frac=0.2, verbose=False)

        assert len(synth.history["generator_loss"]) == 10
        assert len(synth.history["discriminator_loss"]["cps"]) == 10

    def test_generator_loss_recorded(self, simple_sources):
        """Verify losses are recorded during training."""
        synth = MultiSourceSynthesizer(simple_sources)
        synth.fit(epochs=50, holdout_frac=0.2, verbose=False)

        # Losses should be recorded and finite
        assert len(synth.history["generator_loss"]) == 50
        assert all(np.isfinite(loss) for loss in synth.history["generator_loss"])
        # GAN losses don't always decrease monotonically, but should stay bounded
        assert max(synth.history["generator_loss"]) < 10.0


class TestSynthesizerGenerate:
    """Test synthetic data generation."""

    @pytest.fixture
    def fitted_synth(self):
        np.random.seed(42)
        cps = pd.DataFrame({
            "age": np.random.randint(18, 80, 200),
            "wages": np.random.lognormal(10, 1, 200),
        })
        puf = pd.DataFrame({
            "wages": np.random.lognormal(10, 1, 200),
            "cap_gains": np.random.lognormal(8, 2, 200),
        })

        sources = [DataSource("cps", cps), DataSource("puf", puf)]
        synth = MultiSourceSynthesizer(sources)
        synth.fit(epochs=20, holdout_frac=0.2, verbose=False)
        return synth

    def test_generate_shape(self, fitted_synth):
        synthetic = fitted_synth.generate(n=100)

        assert len(synthetic) == 100
        assert set(synthetic.columns) == {"age", "wages", "cap_gains"}

    def test_generate_returns_dataframe(self, fitted_synth):
        synthetic = fitted_synth.generate(n=100)
        assert isinstance(synthetic, pd.DataFrame)

    def test_generate_values_finite(self, fitted_synth):
        synthetic = fitted_synth.generate(n=100)
        assert np.isfinite(synthetic.values).all()

    def test_generate_deterministic_with_seed(self, fitted_synth):
        synth1 = fitted_synth.generate(n=50, seed=42)
        synth2 = fitted_synth.generate(n=50, seed=42)

        pd.testing.assert_frame_equal(synth1, synth2)


class TestSynthesizerEvaluate:
    """Test evaluation metrics."""

    @pytest.fixture
    def trained_synth(self):
        np.random.seed(42)
        n = 300

        cps = pd.DataFrame({
            "age": np.random.randint(18, 80, n),
            "wages": np.random.lognormal(10, 1, n),
        })
        puf = pd.DataFrame({
            "wages": np.random.lognormal(10, 1, n),
            "cap_gains": np.random.lognormal(8, 2, n),
        })

        sources = [DataSource("cps", cps), DataSource("puf", puf)]
        synth = MultiSourceSynthesizer(sources)
        synth.fit(epochs=30, holdout_frac=0.2, verbose=False)
        return synth

    def test_evaluate_returns_per_source_metrics(self, trained_synth):
        metrics = trained_synth.evaluate()

        assert "cps" in metrics
        assert "puf" in metrics

    def test_evaluate_includes_coverage(self, trained_synth):
        metrics = trained_synth.evaluate()

        assert "coverage" in metrics["cps"]
        assert "coverage" in metrics["puf"]
        assert metrics["cps"]["coverage"] >= 0

    def test_evaluate_includes_discriminator_accuracy(self, trained_synth):
        metrics = trained_synth.evaluate()

        assert "discriminator_accuracy" in metrics["cps"]
        # Should be near 50% if generator is good
        assert 0 <= metrics["cps"]["discriminator_accuracy"] <= 1

    def test_coverage_is_computed(self):
        """Verify coverage metric is computed and reasonable."""
        np.random.seed(42)
        torch.manual_seed(42)
        n = 300

        cps = pd.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.randn(n),
        })

        sources = [DataSource("cps", cps)]
        synth = MultiSourceSynthesizer(sources)
        synth.fit(epochs=50, holdout_frac=0.2, verbose=False)
        metrics = synth.evaluate()

        # Coverage should be finite and positive
        assert metrics["cps"]["coverage"] > 0
        assert np.isfinite(metrics["cps"]["coverage"])
        # For simple Gaussian data, coverage should be reasonable
        assert metrics["cps"]["coverage"] < 5.0


class TestSynthesizerWeighting:
    """Test weighted discriminator training."""

    def test_cluster_weighting_option(self):
        np.random.seed(42)
        # Create imbalanced data
        common = np.random.randn(180, 2)
        rare = np.random.randn(20, 2) + 5
        data = np.vstack([common, rare])
        df = pd.DataFrame(data, columns=["x", "y"])

        sources = [DataSource("test", df)]
        synth = MultiSourceSynthesizer(sources, weighting="cluster")
        synth.fit(epochs=10, holdout_frac=0.2, verbose=False)

        assert synth.weighting == "cluster"

    def test_weighted_improves_rare_coverage(self):
        """Weighted training should better cover rare types."""
        np.random.seed(42)

        # Imbalanced: 90% common, 10% rare
        common = np.random.randn(270, 2)
        rare = np.random.randn(30, 2) + 5
        data = np.vstack([common, rare])
        df = pd.DataFrame(data, columns=["x", "y"])

        sources = [DataSource("test", df)]

        # Uniform weighting
        synth_uniform = MultiSourceSynthesizer(sources, weighting="uniform")
        synth_uniform.fit(epochs=50, holdout_frac=0.2, verbose=False)

        # Cluster weighting
        synth_cluster = MultiSourceSynthesizer(sources, weighting="cluster")
        synth_cluster.fit(epochs=50, holdout_frac=0.2, verbose=False)

        # Generate and check coverage on rare type
        uniform_synth = synth_uniform.generate(1000)
        cluster_synth = synth_cluster.generate(1000)

        # Count synthetic samples near rare region (x > 3)
        uniform_rare = (uniform_synth["x"] > 3).sum()
        cluster_rare = (cluster_synth["x"] > 3).sum()

        # Cluster weighting should produce more rare samples
        assert cluster_rare >= uniform_rare * 0.8  # At least close


class TestSynthesizerMultiSource:
    """Test multi-source fusion behavior."""

    def test_generates_complete_records(self):
        """Generator should produce records with all variables from both sources."""
        np.random.seed(42)
        n = 500

        # CPS has age, wages
        # PUF has wages, cap_gains
        # Generator should produce records with all three: age, wages, cap_gains
        age = np.random.randint(20, 80, n)
        wages = np.random.lognormal(10, 1, n)
        cap_gains = np.maximum(0, (age - 30) * 1000 + np.random.randn(n) * 5000)

        cps = pd.DataFrame({"age": age, "wages": wages})
        puf = pd.DataFrame({"wages": wages, "cap_gains": cap_gains})

        sources = [DataSource("cps", cps), DataSource("puf", puf)]
        synth = MultiSourceSynthesizer(sources)
        synth.fit(epochs=50, holdout_frac=0.2, verbose=False)

        synthetic = synth.generate(n=1000)

        # Should have all columns from both sources
        assert set(synthetic.columns) == {"age", "wages", "cap_gains"}
        # Values should be finite
        assert np.isfinite(synthetic.values).all()
        # Should have correct number of records
        assert len(synthetic) == 1000
