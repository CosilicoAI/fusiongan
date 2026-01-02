"""SCF Benchmark: Ground truth validation using Survey of Consumer Finances.

This experiment uses SCF as the complete "ground truth" population, then creates
artificial surveys with different variable subsets and row samples to evaluate
how well FusionGAN recovers the true joint distribution.

Design:
    1. Load full SCF as ground truth (age, income, wages, assets, debts, etc.)
    2. Create "Survey A" - sample 50% of rows, observe only demographics + income
    3. Create "Survey B" - sample 50% of rows (different), observe only income + wealth
    4. Hold out 20% of full population as test set
    5. Train FusionGAN on Survey A + Survey B
    6. Evaluate: Do cross-survey correlations match held-out truth?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from fusiongan import DataSource, MultiSourceSynthesizer

# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent / ".cache"


@dataclass
class SCFBenchmarkConfig:
    """Configuration for SCF benchmark experiment."""

    # SCF loading
    year: int = 2022

    # Survey splits - which columns each artificial survey observes
    survey_a_cols: Tuple[str, ...] = ("age", "educ", "income", "wageinc")
    survey_b_cols: Tuple[str, ...] = ("wageinc", "asset", "networth", "debt")

    # Shared column(s) that appear in both surveys
    shared_cols: Tuple[str, ...] = ("wageinc",)

    # Sampling
    sample_frac_a: float = 0.5  # Fraction of population in survey A
    sample_frac_b: float = 0.5  # Fraction of population in survey B
    overlap_frac: float = 0.0   # Fraction overlap between A and B (0 = no overlap)
    holdout_frac: float = 0.2   # Fraction held out for evaluation

    # Training
    epochs: int = 300
    latent_dim: int = 32
    hidden_dim: int = 128
    weighting: str = "cluster"
    batch_size: int = 64

    # Reproducibility
    seed: int = 42


def download_scf(year: int = 2022) -> pd.DataFrame:
    """Download SCF summary extract from the Federal Reserve.

    The summary extract contains derived variables that are easier to work with
    than the full public dataset.

    Args:
        year: SCF survey year (2019, 2022, etc.)

    Returns:
        Raw DataFrame from Stata file.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"scf{year}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Summary extract URL pattern
    url = f"https://www.federalreserve.gov/econres/files/scfp{year}s.zip"
    print(f"Downloading SCF {year} from {url}...")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    # Extract Stata file from zip
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Find the .dta file
        dta_files = [f for f in z.namelist() if f.endswith(".dta")]
        if not dta_files:
            raise ValueError(f"No .dta file found in {url}")

        with z.open(dta_files[0]) as f:
            df = pd.read_stata(io.BytesIO(f.read()))

    # Cache for future use
    df.to_parquet(cache_path)
    print(f"Cached to {cache_path}")

    return df


def load_scf(year: int = 2022, seed: int = 42) -> pd.DataFrame:
    """Load and preprocess SCF data.

    Args:
        year: SCF survey year.
        seed: Random seed for implicate selection.

    Returns:
        DataFrame with key variables, one row per household.
    """
    raw = download_scf(year)

    # SCF has 5 implicates per household (multiple imputation)
    # y1 encodes case_id * 10 + implicate, so implicate = y1 % 10
    if "y1" in raw.columns:
        raw["_implicate"] = raw["y1"] % 10
        df = raw[raw["_implicate"] == 1].copy()
        df = df.drop(columns=["_implicate"])
    elif "Y1" in raw.columns:
        raw["_implicate"] = raw["Y1"] % 10
        df = raw[raw["_implicate"] == 1].copy()
        df = df.drop(columns=["_implicate"])
    else:
        # No implicate indicator, use all rows
        df = raw.copy()

    # Key variables from SCF summary extract
    # See: https://www.federalreserve.gov/econres/files/codebk2022.txt
    var_map = {
        "AGE": "age",           # Age of reference person
        "EDCL": "educ",         # Education level (1-4)
        "INCOME": "income",     # Total income
        "WAGEINC": "wageinc",   # Wage and salary income
        "ASSET": "asset",       # Total assets
        "NETWORTH": "networth", # Net worth
        "DEBT": "debt",         # Total debt
        "HOUSES": "houses",     # Value of primary residence
        "STOCKS": "stocks",     # Direct stock holdings
        "NFIN": "nfin",         # Non-financial assets
        "FIN": "fin",           # Financial assets
        "WGT": "weight",        # Survey weight
    }

    # Find columns (case-insensitive)
    raw_cols = {c.upper(): c for c in df.columns}
    selected = {}
    for scf_name, our_name in var_map.items():
        if scf_name in raw_cols:
            selected[raw_cols[scf_name]] = our_name

    df = df[list(selected.keys())].rename(columns=selected)

    # Clean: replace negative values with 0 for wealth variables
    for col in ["asset", "networth", "houses", "stocks", "nfin", "fin"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Log-transform skewed variables for better GAN training
    for col in ["income", "wageinc", "asset", "networth", "debt", "houses", "stocks"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # Standardize all numeric columns for GAN training
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std

    return df.reset_index(drop=True)


def create_artificial_surveys(
    full_data: pd.DataFrame,
    config: SCFBenchmarkConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split full population into artificial surveys.

    Args:
        full_data: Complete SCF data.
        config: Benchmark configuration.

    Returns:
        Tuple of (survey_a, survey_b, holdout) DataFrames.
    """
    rng = np.random.RandomState(config.seed)
    n = len(full_data)

    # First, hold out test set
    indices = rng.permutation(n)
    n_holdout = int(n * config.holdout_frac)
    holdout_idx = indices[:n_holdout]
    available_idx = indices[n_holdout:]

    # Split remaining into surveys
    n_available = len(available_idx)
    n_a = int(n_available * config.sample_frac_a)
    n_b = int(n_available * config.sample_frac_b)

    if config.overlap_frac > 0:
        # Allow some overlap between surveys
        n_overlap = int(min(n_a, n_b) * config.overlap_frac)
        overlap_idx = available_idx[:n_overlap]
        remaining_idx = available_idx[n_overlap:]

        a_only = remaining_idx[:n_a - n_overlap]
        b_only = remaining_idx[n_a - n_overlap:n_a - n_overlap + n_b - n_overlap]

        survey_a_idx = np.concatenate([overlap_idx, a_only])
        survey_b_idx = np.concatenate([overlap_idx, b_only])
    else:
        # No overlap - disjoint samples
        survey_a_idx = available_idx[:n_a]
        survey_b_idx = available_idx[n_a:n_a + n_b]

    # Extract data for each survey (only their observed columns)
    survey_a = full_data.iloc[survey_a_idx][list(config.survey_a_cols)].reset_index(drop=True)
    survey_b = full_data.iloc[survey_b_idx][list(config.survey_b_cols)].reset_index(drop=True)
    holdout = full_data.iloc[holdout_idx].reset_index(drop=True)

    return survey_a, survey_b, holdout


def compute_prdc(
    real: np.ndarray,
    fake: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    """Compute Precision, Recall, Density, Coverage (PRDC) metrics.

    Standard metrics from "Reliable Fidelity and Diversity Metrics for
    Generative Models" (Naeem et al., 2020). Used by synthcity.

    Args:
        real: Real data array (n_real, n_features)
        fake: Synthetic data array (n_fake, n_features)
        k: Number of neighbors for radius estimation

    Returns:
        Dict with precision, recall, density, coverage (all 0-1, higher = better)
    """
    from sklearn.neighbors import NearestNeighbors

    # Compute k-NN distances within each set to establish "radius"
    nn_real = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(real)
    real_distances, _ = nn_real.kneighbors(real)
    real_radii = real_distances[:, -1]  # k-th neighbor distance

    nn_fake = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(fake)
    fake_distances, _ = nn_fake.kneighbors(fake)
    fake_radii = fake_distances[:, -1]

    # Distance from each fake point to nearest real
    dist_fake_to_real, _ = nn_real.kneighbors(fake)
    dist_fake_to_real = dist_fake_to_real[:, 0]  # Nearest only

    # Distance from each real point to nearest fake
    nn_fake_1 = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(fake)
    dist_real_to_fake, _ = nn_fake_1.kneighbors(real)
    dist_real_to_fake = dist_real_to_fake[:, 0]

    # For each fake sample, find the nearest real and check if within that real's radius
    nearest_real_idx = nn_real.kneighbors(fake, n_neighbors=1, return_distance=False)[:, 0]
    precision = (dist_fake_to_real <= real_radii[nearest_real_idx]).mean()

    # For each real sample, find the nearest fake and check if within that fake's radius
    nearest_fake_idx = nn_fake.kneighbors(real, n_neighbors=1, return_distance=False)[:, 0]
    recall = (dist_real_to_fake <= fake_radii[nearest_fake_idx]).mean()

    # Coverage: fraction of real samples with a nearby fake sample (within real's own radius)
    coverage = (dist_real_to_fake <= real_radii).mean()

    # Density: for each fake sample, count how many real samples have it within their radius
    # Then average and normalize by k
    dist_fake_to_all_real = nn_real.kneighbors(fake, n_neighbors=len(real), return_distance=True)[0]
    density = (dist_fake_to_all_real <= real_radii).sum(axis=1).mean() / k

    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }


def compute_record_realism_metrics(
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame,
    k: int = 5,
) -> Dict[str, float]:
    """Compute record-level realism metrics.

    Uses PRDC (Precision, Recall, Density, Coverage) plus mean distances.

    Args:
        synthetic: Generated synthetic data.
        holdout: Held-out ground truth.
        k: Number of neighbors for k-NN metrics.

    Returns:
        Dict with PRDC metrics plus mean distances.
    """
    from sklearn.neighbors import NearestNeighbors

    # Align columns
    common_cols = list(set(synthetic.columns) & set(holdout.columns))
    synth_vals = synthetic[common_cols].values
    hold_vals = holdout[common_cols].values

    # PRDC metrics (standard, 0-1 scale, higher = better)
    prdc = compute_prdc(hold_vals, synth_vals, k=k)

    # Also compute mean distances (for interpretability)
    nn_synth = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(synth_vals)
    dist_to_synth, _ = nn_synth.kneighbors(hold_vals)
    mean_dist_to_synth = dist_to_synth[:, 0].mean()

    nn_hold = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(hold_vals)
    dist_to_hold, _ = nn_hold.kneighbors(synth_vals)
    mean_dist_to_hold = dist_to_hold[:, 0].mean()

    return {
        **prdc,
        "mean_dist_real_to_synth": mean_dist_to_synth,
        "mean_dist_synth_to_real": mean_dist_to_hold,
    }


@dataclass
class BenchmarkResults:
    """Results from SCF benchmark experiment.

    Uses PRDC metrics (Precision, Recall, Density, Coverage) which are
    standard in synthetic data evaluation (Naeem et al., 2020).
    """

    # PRDC metrics (0-1 scale, higher = better)
    precision: float    # Fraction of synthetic within real manifold
    recall: float       # Fraction of real within synthetic manifold
    density: float      # How densely synthetic clusters around real
    coverage: float     # Fraction of real with nearby synthetic (KEY METRIC)

    # Mean distances (for interpretability)
    mean_dist_real_to_synth: float
    mean_dist_synth_to_real: float

    # Per-source discriminator accuracy
    per_source_disc_acc: Dict[str, float]

    # Data sizes
    n_train: int
    n_holdout: int
    n_synthetic: int

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "SCF Benchmark Results",
            "=" * 55,
            f"Training: {self.n_train:,} | Holdout: {self.n_holdout:,} | Synthetic: {self.n_synthetic:,}",
            "",
            "PRDC Metrics (0-1, higher = better):",
            f"  Coverage:  {self.coverage:.1%}  ← KEY: fraction of real covered by synthetic",
            f"  Precision: {self.precision:.1%}  (synthetic within real manifold)",
            f"  Recall:    {self.recall:.1%}  (real within synthetic manifold)",
            f"  Density:   {self.density:.2f}",
            "",
            "Mean Distances (lower = better):",
            f"  Real → Synth: {self.mean_dist_real_to_synth:.4f}",
            f"  Synth → Real: {self.mean_dist_synth_to_real:.4f}",
        ]

        if self.per_source_disc_acc:
            lines.append("")
            lines.append("Discriminator Accuracy (ideal = 50%):")
            for source, acc in self.per_source_disc_acc.items():
                quality = "✓" if 0.45 <= acc <= 0.55 else "○" if 0.4 <= acc <= 0.6 else "✗"
                lines.append(f"  {source}: {acc:.1%} {quality}")

        return "\n".join(lines)


def run_benchmark(config: Optional[SCFBenchmarkConfig] = None) -> BenchmarkResults:
    """Run the SCF benchmark experiment.

    Args:
        config: Benchmark configuration. Uses defaults if None.

    Returns:
        BenchmarkResults with all metrics.
    """
    if config is None:
        config = SCFBenchmarkConfig()

    print("Loading SCF data...")
    full_data = load_scf(year=config.year, seed=config.seed)
    print(f"  Loaded {len(full_data):,} households")

    print("Creating artificial surveys...")
    survey_a, survey_b, holdout = create_artificial_surveys(full_data, config)
    print(f"  Survey A: {len(survey_a):,} records, columns: {list(survey_a.columns)}")
    print(f"  Survey B: {len(survey_b):,} records, columns: {list(survey_b.columns)}")
    print(f"  Holdout: {len(holdout):,} records")

    # Create DataSources
    sources = [
        DataSource("survey_a", survey_a),
        DataSource("survey_b", survey_b),
    ]

    print(f"Training FusionGAN ({config.epochs} epochs)...")
    synth = MultiSourceSynthesizer(
        sources=sources,
        weighting=config.weighting,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    )
    synth.fit(
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=True,
    )

    # Generate synthetic data
    n_synthetic = len(holdout)
    print(f"Generating {n_synthetic:,} synthetic records...")
    synthetic = synth.generate(n=n_synthetic, seed=config.seed)

    # Evaluate per-source discriminator accuracy
    print("Evaluating record-level realism (PRDC metrics)...")
    eval_metrics = synth.evaluate()
    per_source_disc_acc = {k: v["discriminator_accuracy"] for k, v in eval_metrics.items()}

    # Evaluate full-record realism against holdout using PRDC
    all_cols = list(set(config.survey_a_cols) | set(config.survey_b_cols))
    holdout_subset = holdout[all_cols]

    metrics = compute_record_realism_metrics(
        synthetic=synthetic,
        holdout=holdout_subset,
        k=5,
    )

    # Get training size for context
    n_train = len(survey_a) + len(survey_b)

    results = BenchmarkResults(
        precision=metrics["precision"],
        recall=metrics["recall"],
        density=metrics["density"],
        coverage=metrics["coverage"],
        mean_dist_real_to_synth=metrics["mean_dist_real_to_synth"],
        mean_dist_synth_to_real=metrics["mean_dist_synth_to_real"],
        per_source_disc_acc=per_source_disc_acc,
        n_train=n_train,
        n_holdout=len(holdout),
        n_synthetic=n_synthetic,
    )

    print(results.summary())
    return results


def compute_nn_distances(
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nearest-neighbor distances for coverage analysis.

    Args:
        synthetic: Generated synthetic data.
        holdout: Held-out ground truth.

    Returns:
        Tuple of (holdout_to_synth_distances, synth_to_holdout_distances)
    """
    from sklearn.neighbors import NearestNeighbors

    common_cols = list(set(synthetic.columns) & set(holdout.columns))
    synth_vals = synthetic[common_cols].values
    hold_vals = holdout[common_cols].values

    # Distance from each holdout record to nearest synthetic
    nn_synth = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_synth.fit(synth_vals)
    hold_to_synth, _ = nn_synth.kneighbors(hold_vals)

    # Distance from each synthetic record to nearest holdout
    nn_hold = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_hold.fit(hold_vals)
    synth_to_hold, _ = nn_hold.kneighbors(synth_vals)

    return hold_to_synth.flatten(), synth_to_hold.flatten()


def plot_coverage_distribution(
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot distribution of nearest-neighbor distances.

    Shows how well synthetic data covers the real population.
    Lower distances = better coverage.

    Args:
        synthetic: Generated synthetic data.
        holdout: Held-out ground truth.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    hold_to_synth, synth_to_hold = compute_nn_distances(synthetic, holdout)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coverage: holdout → synthetic (primary metric)
    ax1 = axes[0]
    ax1.hist(hold_to_synth, bins=50, color="#00d4ff", alpha=0.7, edgecolor="white")
    ax1.axvline(np.mean(hold_to_synth), color="red", linestyle="--",
                label=f"Mean: {np.mean(hold_to_synth):.3f}")
    ax1.axvline(np.median(hold_to_synth), color="orange", linestyle="--",
                label=f"Median: {np.median(hold_to_synth):.3f}")
    ax1.set_xlabel("Distance to Nearest Synthetic", fontsize=11)
    ax1.set_ylabel("Count (Holdout Records)", fontsize=11)
    ax1.set_title("Coverage: Holdout → Synthetic\n(Every real person has a similar synthetic)", fontsize=12)
    ax1.legend()

    # Authenticity: synthetic → holdout
    ax2 = axes[1]
    ax2.hist(synth_to_hold, bins=50, color="#7b2cbf", alpha=0.7, edgecolor="white")
    ax2.axvline(np.mean(synth_to_hold), color="red", linestyle="--",
                label=f"Mean: {np.mean(synth_to_hold):.3f}")
    ax2.axvline(np.median(synth_to_hold), color="orange", linestyle="--",
                label=f"Median: {np.median(synth_to_hold):.3f}")
    ax2.set_xlabel("Distance to Nearest Holdout", fontsize=11)
    ax2.set_ylabel("Count (Synthetic Records)", fontsize=11)
    ax2.set_title("Authenticity: Synthetic → Holdout\n(Synthetic records look real)", fontsize=12)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Run with default config
    results = run_benchmark()

    # Save figure
    fig = plot_correlation_recovery(results, save_path="correlation_recovery.png")
    plt.show()
