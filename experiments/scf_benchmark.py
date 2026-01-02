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


def compute_record_realism_metrics(
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame,
    k: int = 5,
) -> Dict[str, float]:
    """Compute record-level realism metrics.

    These metrics evaluate whether synthetic records look like they could
    have come from the real population. Aggregate statistics (correlations,
    totals) are handled separately by calibration.

    Args:
        synthetic: Generated synthetic data.
        holdout: Held-out ground truth.
        k: Number of neighbors for k-NN metrics.

    Returns:
        Dict with:
            - coverage: Mean distance from holdout to nearest synthetic
                       (lower = synthetic covers real distribution)
            - authenticity: Mean distance from synthetic to nearest holdout
                           (lower = synthetic records look real, not mode-collapsed)
            - density_ratio: Ratio of authenticity to coverage
                            (close to 1.0 = balanced quality)
    """
    from sklearn.neighbors import NearestNeighbors

    # Align columns
    common_cols = list(set(synthetic.columns) & set(holdout.columns))
    synth_vals = synthetic[common_cols].values
    hold_vals = holdout[common_cols].values

    # Coverage: for each holdout record, find distance to nearest synthetic
    nn_synth = NearestNeighbors(n_neighbors=min(k, len(synth_vals)), metric="euclidean")
    nn_synth.fit(synth_vals)
    distances_to_synth, _ = nn_synth.kneighbors(hold_vals)
    coverage = distances_to_synth[:, 0].mean()  # Distance to nearest

    # Authenticity: for each synthetic record, find distance to nearest holdout
    nn_hold = NearestNeighbors(n_neighbors=min(k, len(hold_vals)), metric="euclidean")
    nn_hold.fit(hold_vals)
    distances_to_hold, _ = nn_hold.kneighbors(synth_vals)
    authenticity = distances_to_hold[:, 0].mean()  # Distance to nearest

    # Density ratio: balanced if close to 1.0
    density_ratio = authenticity / (coverage + 1e-8)

    return {
        "coverage": coverage,
        "authenticity": authenticity,
        "density_ratio": density_ratio,
    }


@dataclass
class BenchmarkResults:
    """Results from SCF benchmark experiment.

    Focus is on record-level realism, not aggregate statistics.
    Correlations and totals are handled by calibration downstream.
    """

    # Per-source metrics (projected to each survey's columns)
    per_source_coverage: Dict[str, float]
    per_source_disc_acc: Dict[str, float]

    # Full-record realism (comparing complete synthetic to holdout)
    coverage: float          # Mean NN dist from holdout → synthetic
    authenticity: float      # Mean NN dist from synthetic → holdout
    density_ratio: float     # authenticity / coverage (ideal ≈ 1.0)

    # Data sizes
    n_survey_a: int
    n_survey_b: int
    n_holdout: int
    n_synthetic: int

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "SCF Benchmark Results",
            "=" * 50,
            f"Survey A: {self.n_survey_a:,} records",
            f"Survey B: {self.n_survey_b:,} records",
            f"Holdout: {self.n_holdout:,} records (ground truth)",
            f"Synthetic: {self.n_synthetic:,} records",
            "",
            "Record-Level Realism (lower distance = better):",
            f"  Coverage:     {self.coverage:.4f}  (holdout → synth)",
            f"  Authenticity: {self.authenticity:.4f}  (synth → holdout)",
            f"  Density Ratio: {self.density_ratio:.2f}  (ideal ≈ 1.0)",
            "",
            "Per-Source Discriminator Accuracy (ideal = 50%):",
        ]

        for source in self.per_source_disc_acc:
            acc = self.per_source_disc_acc[source]
            # 50% = generator fools discriminator completely
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

    # Evaluate per-source metrics (projected to each survey's columns)
    print("Evaluating record-level realism...")
    eval_metrics = synth.evaluate()
    per_source_coverage = {k: v["coverage"] for k, v in eval_metrics.items()}
    per_source_disc_acc = {k: v["discriminator_accuracy"] for k, v in eval_metrics.items()}

    # Evaluate full-record realism against holdout
    # This is the key metric: do complete synthetic records look real?
    all_cols = list(set(config.survey_a_cols) | set(config.survey_b_cols))
    holdout_subset = holdout[all_cols]

    realism_metrics = compute_record_realism_metrics(
        synthetic=synthetic,
        holdout=holdout_subset,
        k=5,
    )

    results = BenchmarkResults(
        per_source_coverage=per_source_coverage,
        per_source_disc_acc=per_source_disc_acc,
        coverage=realism_metrics["coverage"],
        authenticity=realism_metrics["authenticity"],
        density_ratio=realism_metrics["density_ratio"],
        n_survey_a=len(survey_a),
        n_survey_b=len(survey_b),
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
