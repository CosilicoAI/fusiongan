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

    Strategy:
    1. For survey_a records, impute survey_b-only columns
    2. For survey_b records, impute survey_a-only columns
    3. Combine into complete records
    """
    try:
        from quantile_forest import RandomForestQuantileRegressor
    except ImportError:
        print("quantile-forest not installed, skipping QRF")
        return None

    shared_cols = list(set(config.survey_a_cols) & set(config.survey_b_cols))
    a_only_cols = [c for c in config.survey_a_cols if c not in shared_cols]
    b_only_cols = [c for c in config.survey_b_cols if c not in shared_cols]

    survey_a_complete = survey_a.copy()
    survey_b_complete = survey_b.copy()

    # Impute B columns onto A records (one column at a time)
    for col in b_only_cols:
        qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        qrf.fit(survey_b[shared_cols].values, survey_b[col].values)
        # Predict median (quantile 0.5)
        preds = qrf.predict(survey_a[shared_cols].values, quantiles=0.5)
        survey_a_complete[col] = preds

    # Impute A columns onto B records
    for col in a_only_cols:
        qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        qrf.fit(survey_a[shared_cols].values, survey_a[col].values)
        preds = qrf.predict(survey_b[shared_cols].values, quantiles=0.5)
        survey_b_complete[col] = preds

    # Combine
    synthetic = pd.concat([survey_a_complete, survey_b_complete], ignore_index=True)

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

    Stage 1: Classify zero vs non-zero
    Stage 2: QRF on non-zero values only
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

    survey_a_complete = survey_a.copy()
    survey_b_complete = survey_b.copy()

    def impute_zero_inflated_col(X_train, y_train, X_pred):
        """Two-stage zero-inflated imputation for a single column."""
        results = np.zeros(len(X_pred))

        # Stage 1: Classify zero vs non-zero
        is_nonzero = (y_train != 0).astype(int)

        if is_nonzero.sum() < 10:
            # Too few non-zero, just predict zeros
            return results

        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_train, is_nonzero)
        pred_nonzero = clf.predict(X_pred)

        # Stage 2: QRF on non-zero values
        nonzero_mask = y_train != 0
        if nonzero_mask.sum() > 10:
            qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            qrf.fit(X_train[nonzero_mask], y_train[nonzero_mask])
            # Predict for records classified as non-zero
            pred_mask = pred_nonzero == 1
            if pred_mask.sum() > 0:
                qrf_preds = qrf.predict(X_pred[pred_mask], quantiles=0.5)
                results[pred_mask] = qrf_preds.flatten()

        return results

    # Impute B columns onto A records
    for col in b_only_cols:
        preds = impute_zero_inflated_col(
            survey_b[shared_cols].values,
            survey_b[col].values,
            survey_a[shared_cols].values,
        )
        survey_a_complete[col] = preds

    # Impute A columns onto B records
    for col in a_only_cols:
        preds = impute_zero_inflated_col(
            survey_a[shared_cols].values,
            survey_a[col].values,
            survey_b[shared_cols].values,
        )
        survey_b_complete[col] = preds

    # Combine
    synthetic = pd.concat([survey_a_complete, survey_b_complete], ignore_index=True)

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
