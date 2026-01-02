"""Compare synthesis methods on SCF benchmark.

Methods:
- Identity: training set as-is (upper bound)
- FusionGAN: adversarial multi-source fusion
- QRF: Quantile Random Forest imputation (via microimpute)
- ZeroInflatedQRF: Two-stage QRF for zero-inflated variables
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from experiments.scf_benchmark import (
    SCFBenchmarkConfig,
    load_scf,
    create_artificial_surveys,
    compute_prdc,
    MultiSourceSynthesizer,
    DataSource,
)


@dataclass
class MethodResult:
    """Results for a single method."""
    name: str
    precision: float
    recall: float
    density: float
    coverage: float
    n_synthetic: int


def run_identity(
    train_full: pd.DataFrame,
    holdout_full: pd.DataFrame,
    all_cols: List[str],
) -> MethodResult:
    """Identity baseline: use training set as synthetic."""
    prdc = compute_prdc(
        holdout_full[all_cols].values,
        train_full[all_cols].values,
        k=5,
    )
    return MethodResult(
        name="Identity",
        precision=prdc["precision"],
        recall=prdc["recall"],
        density=prdc["density"],
        coverage=prdc["coverage"],
        n_synthetic=len(train_full),
    )


def run_fusiongan(
    survey_a: pd.DataFrame,
    survey_b: pd.DataFrame,
    holdout_full: pd.DataFrame,
    all_cols: List[str],
    config: SCFBenchmarkConfig,
) -> MethodResult:
    """FusionGAN: adversarial multi-source synthesis."""
    sources = [
        DataSource("survey_a", survey_a),
        DataSource("survey_b", survey_b),
    ]
    synth = MultiSourceSynthesizer(
        sources=sources,
        weighting=config.weighting,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    )
    synth.fit(epochs=config.epochs, batch_size=config.batch_size, verbose=False)

    n_synthetic = len(survey_a) + len(survey_b)
    synthetic = synth.generate(n=n_synthetic, seed=config.seed)

    prdc = compute_prdc(
        holdout_full[all_cols].values,
        synthetic[all_cols].values,
        k=5,
    )
    return MethodResult(
        name="FusionGAN",
        precision=prdc["precision"],
        recall=prdc["recall"],
        density=prdc["density"],
        coverage=prdc["coverage"],
        n_synthetic=n_synthetic,
    )


def run_qrf_imputation(
    survey_a: pd.DataFrame,
    survey_b: pd.DataFrame,
    holdout_full: pd.DataFrame,
    all_cols: List[str],
    config: SCFBenchmarkConfig,
) -> MethodResult:
    """QRF: Impute missing columns using Quantile Random Forests.

    OLD strategy (data fusion, not synthesis):
    - Keep original rows, impute missing columns
    - Problem: 100% coverage because we keep real rows!

    NEW strategy (true synthesis):
    1. Sample new shared variable values (bootstrap from combined surveys)
    2. Use QRF to predict ALL columns from shared variables
    3. Result: completely new synthetic rows
    """
    try:
        from quantile_forest import RandomForestQuantileRegressor
    except ImportError:
        print("quantile-forest not installed, skipping QRF")
        return None

    shared_cols = list(set(config.survey_a_cols) & set(config.survey_b_cols))
    a_only_cols = [c for c in config.survey_a_cols if c not in shared_cols]
    b_only_cols = [c for c in config.survey_b_cols if c not in shared_cols]

    # Combine shared columns from both surveys for sampling
    shared_a = survey_a[shared_cols].copy()
    shared_b = survey_b[shared_cols].copy()
    shared_combined = pd.concat([shared_a, shared_b], ignore_index=True)

    # Bootstrap sample new shared variable combinations
    rng = np.random.RandomState(config.seed)
    n_synthetic = len(survey_a) + len(survey_b)
    sample_idx = rng.choice(len(shared_combined), size=n_synthetic, replace=True)

    # Add small noise to break exact matches with original records
    sampled_shared = shared_combined.iloc[sample_idx].values.copy()
    noise_scale = 0.1  # Small noise relative to standardized data
    sampled_shared += rng.normal(0, noise_scale, sampled_shared.shape)

    synthetic = pd.DataFrame(sampled_shared, columns=shared_cols)

    # Train QRF models to predict each column from shared variables
    # Use stochastic quantile sampling for diversity
    quantiles_to_sample = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Predict A-only columns (train on survey A)
    for col in a_only_cols:
        qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        qrf.fit(survey_a[shared_cols].values, survey_a[col].values)
        # Sample random quantile per record for diversity
        all_preds = qrf.predict(sampled_shared, quantiles=quantiles_to_sample)
        quantile_choices = rng.choice(len(quantiles_to_sample), size=n_synthetic)
        synthetic[col] = all_preds[np.arange(n_synthetic), quantile_choices]

    # Predict B-only columns (train on survey B)
    for col in b_only_cols:
        qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        qrf.fit(survey_b[shared_cols].values, survey_b[col].values)
        all_preds = qrf.predict(sampled_shared, quantiles=quantiles_to_sample)
        quantile_choices = rng.choice(len(quantiles_to_sample), size=n_synthetic)
        synthetic[col] = all_preds[np.arange(n_synthetic), quantile_choices]

    prdc = compute_prdc(
        holdout_full[all_cols].values,
        synthetic[all_cols].values,
        k=5,
    )
    return MethodResult(
        name="QRF",
        precision=prdc["precision"],
        recall=prdc["recall"],
        density=prdc["density"],
        coverage=prdc["coverage"],
        n_synthetic=len(synthetic),
    )


def run_zero_inflated_qrf(
    survey_a: pd.DataFrame,
    survey_b: pd.DataFrame,
    holdout_full: pd.DataFrame,
    all_cols: List[str],
    config: SCFBenchmarkConfig,
) -> MethodResult:
    """Zero-inflated QRF: Two-stage model for zero-heavy variables.

    TRUE SYNTHESIS approach:
    1. Sample new shared variable values (bootstrap + noise)
    2. For each column, use two-stage model:
       - Stage 1: Classify zero vs non-zero (probabilistic)
       - Stage 2: QRF on non-zero values with quantile sampling
    3. Result: completely new synthetic rows

    Note: Data is standardized, so we detect zeros by finding the minimum
    (which corresponds to original zero after (0 - mean) / std transformation).
    """
    from sklearn.ensemble import RandomForestClassifier

    try:
        from quantile_forest import RandomForestQuantileRegressor
    except ImportError:
        print("quantile-forest not installed, skipping ZI-QRF")
        return None

    shared_cols = list(set(config.survey_a_cols) & set(config.survey_b_cols))
    a_only_cols = [c for c in config.survey_a_cols if c not in shared_cols]
    b_only_cols = [c for c in config.survey_b_cols if c not in shared_cols]

    # Combine shared columns from both surveys for sampling
    shared_a = survey_a[shared_cols].copy()
    shared_b = survey_b[shared_cols].copy()
    shared_combined = pd.concat([shared_a, shared_b], ignore_index=True)

    # Bootstrap sample new shared variable combinations
    rng = np.random.RandomState(config.seed)
    n_synthetic = len(survey_a) + len(survey_b)
    sample_idx = rng.choice(len(shared_combined), size=n_synthetic, replace=True)

    # Add small noise to break exact matches with original records
    sampled_shared = shared_combined.iloc[sample_idx].values.copy()
    noise_scale = 0.1
    sampled_shared += rng.normal(0, noise_scale, sampled_shared.shape)

    synthetic = pd.DataFrame(sampled_shared, columns=shared_cols)
    quantiles_to_sample = [0.1, 0.25, 0.5, 0.75, 0.9]

    def synthesize_zero_inflated_col(X_train, y_train, X_synth, rng, zero_inflation_threshold=0.1):
        """Two-stage zero-inflated synthesis for a single column."""
        n = len(X_synth)
        results = np.zeros(n)

        # Detect zeros in standardized data
        min_val = y_train.min()
        at_min = np.isclose(y_train, min_val, atol=1e-6)
        zero_frac = at_min.sum() / len(y_train)

        if zero_frac < zero_inflation_threshold:
            # Not zero-inflated, use standard QRF with quantile sampling
            qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            qrf.fit(X_train, y_train)
            all_preds = qrf.predict(X_synth, quantiles=quantiles_to_sample)
            quantile_choices = rng.choice(len(quantiles_to_sample), size=n)
            return all_preds[np.arange(n), quantile_choices], False

        # Zero-inflated: use two-stage model
        is_nonzero = (~at_min).astype(int)

        if is_nonzero.sum() < 10:
            return np.full(n, min_val), True

        # Stage 1: Classify zero vs non-zero (use probabilities for stochastic sampling)
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_train, is_nonzero)
        probs = clf.predict_proba(X_synth)[:, 1]  # P(non-zero)
        pred_nonzero = rng.random(n) < probs  # Stochastic classification

        # Stage 2: QRF on non-zero values
        nonzero_mask = ~at_min
        results.fill(min_val)  # Default to standardized zero

        if nonzero_mask.sum() > 10 and pred_nonzero.sum() > 0:
            qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            qrf.fit(X_train[nonzero_mask], y_train[nonzero_mask])
            # Predict with quantile sampling for diversity
            all_preds = qrf.predict(X_synth[pred_nonzero], quantiles=quantiles_to_sample)
            quantile_choices = rng.choice(len(quantiles_to_sample), size=pred_nonzero.sum())
            results[pred_nonzero] = all_preds[np.arange(pred_nonzero.sum()), quantile_choices]

        return results, True

    zero_inflated_count = 0

    # Synthesize A-only columns (train on survey A)
    for col in a_only_cols:
        preds, is_zi = synthesize_zero_inflated_col(
            survey_a[shared_cols].values,
            survey_a[col].values,
            sampled_shared,
            rng,
        )
        if is_zi:
            zero_inflated_count += 1
        synthetic[col] = preds

    # Synthesize B-only columns (train on survey B)
    for col in b_only_cols:
        preds, is_zi = synthesize_zero_inflated_col(
            survey_b[shared_cols].values,
            survey_b[col].values,
            sampled_shared,
            rng,
        )
        if is_zi:
            zero_inflated_count += 1
        synthetic[col] = preds

    print(f"  (ZI-QRF used two-stage model for {zero_inflated_count} zero-inflated columns)")

    prdc = compute_prdc(
        holdout_full[all_cols].values,
        synthetic[all_cols].values,
        k=5,
    )
    return MethodResult(
        name="ZI-QRF",
        precision=prdc["precision"],
        recall=prdc["recall"],
        density=prdc["density"],
        coverage=prdc["coverage"],
        n_synthetic=len(synthetic),
    )


def run_comparison(config: Optional[SCFBenchmarkConfig] = None) -> List[MethodResult]:
    """Run all methods and compare."""
    if config is None:
        config = SCFBenchmarkConfig(epochs=300, hidden_dim=256, weighting='cluster', latent_dim=64)

    print("Loading SCF data...")
    full_data = load_scf(year=config.year, seed=config.seed)

    # Split into train/holdout
    rng = np.random.RandomState(config.seed)
    n = len(full_data)
    indices = rng.permutation(n)
    n_holdout = int(n * config.holdout_frac)
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]

    all_cols = list(set(config.survey_a_cols) | set(config.survey_b_cols))
    train_full = full_data.iloc[train_idx][all_cols].reset_index(drop=True)
    holdout_full = full_data.iloc[holdout_idx][all_cols].reset_index(drop=True)

    # Create artificial surveys
    survey_a, survey_b, _ = create_artificial_surveys(full_data, config)

    print(f"Train: {len(train_full):,} | Holdout: {len(holdout_full):,}")
    print(f"Survey A: {len(survey_a):,} cols={list(survey_a.columns)}")
    print(f"Survey B: {len(survey_b):,} cols={list(survey_b.columns)}")
    print()

    results = []

    # Identity
    print("Running Identity baseline...")
    results.append(run_identity(train_full, holdout_full, all_cols))

    # QRF
    print("Running QRF imputation...")
    qrf_result = run_qrf_imputation(survey_a, survey_b, holdout_full, all_cols, config)
    if qrf_result:
        results.append(qrf_result)

    # Zero-inflated QRF
    print("Running Zero-Inflated QRF...")
    ziqrf_result = run_zero_inflated_qrf(survey_a, survey_b, holdout_full, all_cols, config)
    if ziqrf_result:
        results.append(ziqrf_result)

    # FusionGAN (slowest, run last)
    print(f"Running FusionGAN ({config.epochs} epochs)...")
    results.append(run_fusiongan(survey_a, survey_b, holdout_full, all_cols, config))

    # Print comparison table
    print()
    print("=" * 60)
    print(f"{'Method':<15} {'Coverage':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<15} {r.coverage:>10.1%} {r.precision:>10.1%} {r.recall:>10.1%}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_comparison()
