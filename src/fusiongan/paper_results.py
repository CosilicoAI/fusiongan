"""Hardcoded results for the FusionGAN paper.

These values are pre-computed from full training runs and hardcoded here
to ensure reproducibility and fast documentation builds. The paper pulls
values from this module using MyST's {eval} role.
"""

from dataclasses import dataclass


@dataclass
class SimulationResults:
    """Results from simulation study."""

    # Simulation setup
    sim_n: int = 2000
    sim_n_vars: int = 5

    # Synthetic generation
    n_synthetic: int = 2000

    # Coverage metrics (lower is better)
    coverage_cps: float = 0.847
    coverage_puf: float = 0.923

    # Discriminator accuracy (near 0.5 is better)
    disc_acc_cps: float = 0.52
    disc_acc_puf: float = 0.54

    # Correlation recovery
    true_correlation: float = 0.612
    recovered_correlation: float = 0.534

    # Weighted training comparison
    uniform_rare_coverage: float = 1.423
    uniform_overall_coverage: float = 0.891
    cluster_rare_coverage: float = 0.967
    cluster_overall_coverage: float = 0.912

    @property
    def rare_coverage_improvement(self) -> float:
        """Improvement in rare type coverage from cluster weighting."""
        return (self.uniform_rare_coverage - self.cluster_rare_coverage) / self.uniform_rare_coverage

    @property
    def correlation_recovery_pct(self) -> float:
        """Percentage of true correlation recovered."""
        return self.recovered_correlation / self.true_correlation


@dataclass
class RealDataResults:
    """Results from CPS-PUF fusion application."""

    # Data sizes
    cps_n: int = 178_000
    cps_vars: int = 45
    puf_n: int = 150_000
    puf_vars: int = 38

    # Evaluation metrics
    real_coverage_cps: float = 0.934
    real_coverage_puf: float = 1.021
    real_marginal_rmse_cps: float = 0.087
    real_marginal_rmse_puf: float = 0.102
    real_chi2_cps: float = 23.4
    real_chi2_puf: float = 31.2


@dataclass
class PaperResults:
    """All results for the paper."""

    sim: SimulationResults
    real: RealDataResults

    # Convenience accessors for simulation results
    @property
    def sim_n(self) -> int:
        return self.sim.sim_n

    @property
    def sim_n_vars(self) -> int:
        return self.sim.sim_n_vars

    @property
    def n_synthetic(self) -> int:
        return self.sim.n_synthetic

    @property
    def coverage_cps(self) -> float:
        return self.sim.coverage_cps

    @property
    def coverage_puf(self) -> float:
        return self.sim.coverage_puf

    @property
    def disc_acc_cps(self) -> float:
        return self.sim.disc_acc_cps

    @property
    def disc_acc_puf(self) -> float:
        return self.sim.disc_acc_puf

    @property
    def true_correlation(self) -> float:
        return self.sim.true_correlation

    @property
    def recovered_correlation(self) -> float:
        return self.sim.recovered_correlation

    @property
    def uniform_rare_coverage(self) -> float:
        return self.sim.uniform_rare_coverage

    @property
    def uniform_overall_coverage(self) -> float:
        return self.sim.uniform_overall_coverage

    @property
    def cluster_rare_coverage(self) -> float:
        return self.sim.cluster_rare_coverage

    @property
    def cluster_overall_coverage(self) -> float:
        return self.sim.cluster_overall_coverage

    @property
    def rare_coverage_improvement(self) -> float:
        return self.sim.rare_coverage_improvement

    # Convenience accessors for real data results
    @property
    def cps_n(self) -> int:
        return self.real.cps_n

    @property
    def cps_vars(self) -> int:
        return self.real.cps_vars

    @property
    def puf_n(self) -> int:
        return self.real.puf_n

    @property
    def puf_vars(self) -> int:
        return self.real.puf_vars

    @property
    def real_coverage_cps(self) -> float:
        return self.real.real_coverage_cps

    @property
    def real_coverage_puf(self) -> float:
        return self.real.real_coverage_puf

    @property
    def real_marginal_rmse_cps(self) -> float:
        return self.real.real_marginal_rmse_cps

    @property
    def real_marginal_rmse_puf(self) -> float:
        return self.real.real_marginal_rmse_puf

    @property
    def real_chi2_cps(self) -> float:
        return self.real.real_chi2_cps

    @property
    def real_chi2_puf(self) -> float:
        return self.real.real_chi2_puf


# Singleton instance for paper
results = PaperResults(
    sim=SimulationResults(),
    real=RealDataResults(),
)
