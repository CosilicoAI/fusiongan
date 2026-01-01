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


def compute_cross_survey_correlation(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> float:
    """Compute correlation between two columns.

    Args:
        df: DataFrame containing both columns.
        col_a: First column (originally from survey A only).
        col_b: Second column (originally from survey B only).

    Returns:
        Pearson correlation coefficient.
    """
    valid_mask = df[col_a].notna() & df[col_b].notna()
    if valid_mask.sum() < 2:
        return np.nan
    return np.corrcoef(df.loc[valid_mask, col_a], df.loc[valid_mask, col_b])[0, 1]


def compute_all_cross_correlations(
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame,
    survey_a_cols: List[str],
    survey_b_cols: List[str],
    shared_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute all cross-survey correlations.

    Args:
        synthetic: Generated synthetic data.
        holdout: Held-out ground truth.
        survey_a_cols: Columns unique to survey A.
        survey_b_cols: Columns unique to survey B.
        shared_cols: Columns in both surveys.

    Returns:
        Dict with correlation comparisons.
    """
    # Find unique columns (not in both surveys)
    a_unique = [c for c in survey_a_cols if c not in shared_cols]
    b_unique = [c for c in survey_b_cols if c not in shared_cols]

    results = {}
    for col_a in a_unique:
        for col_b in b_unique:
            if col_a in synthetic.columns and col_b in synthetic.columns:
                key = f"{col_a}_vs_{col_b}"
                results[key] = {
                    "true": compute_cross_survey_correlation(holdout, col_a, col_b),
                    "synthetic": compute_cross_survey_correlation(synthetic, col_a, col_b),
                }

    return results


@dataclass
class BenchmarkResults:
    """Results from SCF benchmark experiment."""

    # Per-source metrics
    coverage: Dict[str, float]
    discriminator_accuracy: Dict[str, float]

    # Cross-survey correlation recovery
    correlations: Dict[str, Dict[str, float]]
    correlation_rmse: float

    # Data sizes
    n_survey_a: int
    n_survey_b: int
    n_holdout: int
    n_synthetic: int

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "SCF Benchmark Results",
            "=" * 40,
            f"Survey A: {self.n_survey_a:,} records",
            f"Survey B: {self.n_survey_b:,} records",
            f"Holdout: {self.n_holdout:,} records",
            f"Synthetic: {self.n_synthetic:,} records",
            "",
            "Per-Source Metrics:",
        ]

        for source in self.coverage:
            lines.append(
                f"  {source}: coverage={self.coverage[source]:.4f}, "
                f"disc_acc={self.discriminator_accuracy[source]:.1%}"
            )

        lines.extend([
            "",
            "Cross-Survey Correlation Recovery:",
            f"  Overall RMSE: {self.correlation_rmse:.4f}",
        ])

        for key, vals in self.correlations.items():
            lines.append(
                f"  {key}: true={vals['true']:.3f}, synth={vals['synthetic']:.3f}"
            )

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

    # Evaluate per-source metrics
    print("Evaluating...")
    eval_metrics = synth.evaluate()
    coverage = {k: v["coverage"] for k, v in eval_metrics.items()}
    disc_acc = {k: v["discriminator_accuracy"] for k, v in eval_metrics.items()}

    # Evaluate cross-survey correlations
    all_cols = list(set(config.survey_a_cols) | set(config.survey_b_cols))
    holdout_subset = holdout[all_cols]

    correlations = compute_all_cross_correlations(
        synthetic=synthetic,
        holdout=holdout_subset,
        survey_a_cols=list(config.survey_a_cols),
        survey_b_cols=list(config.survey_b_cols),
        shared_cols=list(config.shared_cols),
    )

    # Compute correlation RMSE
    corr_errors = []
    for key, vals in correlations.items():
        if not np.isnan(vals["true"]) and not np.isnan(vals["synthetic"]):
            corr_errors.append((vals["true"] - vals["synthetic"]) ** 2)

    correlation_rmse = np.sqrt(np.mean(corr_errors)) if corr_errors else np.nan

    results = BenchmarkResults(
        coverage=coverage,
        discriminator_accuracy=disc_acc,
        correlations=correlations,
        correlation_rmse=correlation_rmse,
        n_survey_a=len(survey_a),
        n_survey_b=len(survey_b),
        n_holdout=len(holdout),
        n_synthetic=n_synthetic,
    )

    print(results.summary())
    return results


def plot_correlation_recovery(
    results: BenchmarkResults,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot true vs recovered correlations.

    Args:
        results: Benchmark results.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    true_corrs = []
    synth_corrs = []
    labels = []

    for key, vals in results.correlations.items():
        if not np.isnan(vals["true"]) and not np.isnan(vals["synthetic"]):
            true_corrs.append(vals["true"])
            synth_corrs.append(vals["synthetic"])
            labels.append(key.replace("_vs_", "\nvs\n"))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(true_corrs, synth_corrs, s=100, alpha=0.7, c="#00d4ff", edgecolors="white")

    # Perfect recovery line
    lims = [min(min(true_corrs), min(synth_corrs)) - 0.1,
            max(max(true_corrs), max(synth_corrs)) + 0.1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect recovery")

    # Labels
    for i, label in enumerate(labels):
        ax.annotate(label, (true_corrs[i], synth_corrs[i]),
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=8, alpha=0.8)

    ax.set_xlabel("True Correlation (Holdout)", fontsize=12)
    ax.set_ylabel("Recovered Correlation (Synthetic)", fontsize=12)
    ax.set_title(f"Cross-Survey Correlation Recovery\n(RMSE: {results.correlation_rmse:.4f})",
                fontsize=14)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.grid(True, alpha=0.3)

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
