---
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
---

# FusionGAN: Multi-Source Adversarial Synthesis for Survey Data Fusion

```{code-cell} python
:tags: [remove-cell]

import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from fusiongan import DataSource, MultiSourceSynthesizer
from fusiongan.paper_results import results as r

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
```

## Abstract

Survey microdata are essential for policy analysis, but individual surveys observe only subsets of variables relevant to researchers. The Current Population Survey (CPS) captures demographics and labor market outcomes, while the IRS Public Use File (PUF) contains detailed tax information—but neither source alone provides complete household profiles. Traditional statistical matching methods rely on strong conditional independence assumptions that often fail in practice.

We introduce **FusionGAN**, a multi-discriminator generative adversarial network that learns joint distributions from multiple survey sources with different variable subsets. Our architecture uses a single generator producing complete synthetic records, with source-specific discriminators that evaluate projections to each survey's observed variables. This formulation naturally handles the "slices of a population" structure of survey data without requiring matched records or shared identifiers.

We demonstrate that FusionGAN produces synthetic populations that:
1. Match marginal and joint distributions within each source
2. Learn plausible cross-source correlations through the shared generator
3. Improve coverage of rare population types through weighted discriminator training

## Introduction

### The Survey Fusion Problem

Modern policy analysis requires linking information across multiple data sources. Consider the challenge of microsimulation for tax-benefit policy:

- **CPS ASEC** provides demographics, employment, and program participation
- **IRS PUF** provides detailed tax return information
- **SCF** provides wealth and asset holdings
- **SIPP** provides longitudinal income dynamics

Each survey samples from the same underlying population but observes different "slices"—both in terms of which households are sampled (rows) and which variables are measured (columns). The goal of **data fusion** is to construct a synthetic population with complete variable coverage that respects the joint distributions observed in each source.

### Limitations of Existing Approaches

Traditional statistical matching {cite:p}`rodgers1984` assumes conditional independence: variables unique to source A are independent of variables unique to source B, given shared variables. This assumption is often violated—for example, capital gains (observed in PUF) are correlated with age (observed in CPS) even after conditioning on wages.

Recent machine learning approaches including CTGAN {cite:p}`xu2019` and TVAE focus on single-table synthesis. Multi-table methods like SDV {cite:p}`patki2016` handle relational databases but not the partially-overlapping structure of survey fusion.

### Our Contribution

FusionGAN directly addresses the multi-source fusion problem with a novel architecture:

1. **Single generator, multiple discriminators**: One generator produces complete records; each source has a discriminator evaluating projected records
2. **Weighted training**: Discriminators can be weighted to focus on rare population types
3. **Coverage-based evaluation**: We evaluate using holdout set coverage rather than just discriminator accuracy

## Methods

### Problem Formulation

Let $\{S_1, \ldots, S_K\}$ be $K$ survey sources, where source $S_k$ observes variables $V_k \subseteq V$ from the full variable set $V$. We observe samples $\{x_i^{(k)}\}_{i=1}^{n_k}$ from each source, where $x_i^{(k)} \in \mathbb{R}^{|V_k|}$.

Our goal is to learn a generator $G: \mathbb{R}^d \to \mathbb{R}^{|V|}$ that produces complete synthetic records $\hat{x} = G(z)$ for $z \sim \mathcal{N}(0, I_d)$ such that:

$$\pi_{V_k}(\hat{x}) \sim P_{S_k} \quad \forall k$$

where $\pi_{V_k}$ projects to the variables observed in source $k$ and $P_{S_k}$ is the data distribution of source $k$.

### Architecture

```{figure} _static/figures/architecture.png
:name: fig-architecture
:width: 100%

FusionGAN architecture. A single generator produces complete records from latent noise. Source-specific discriminators evaluate projections to each survey's observed variables.
```

**Generator**: A multi-layer perceptron mapping latent vectors $z \in \mathbb{R}^{32}$ to complete records $\hat{x} \in \mathbb{R}^{|V|}$:

$$G(z) = \text{MLP}(z; \theta_G)$$

**Discriminators**: For each source $k$, a discriminator $D_k: \mathbb{R}^{|V_k|} \to [0, 1]$ classifies whether a projected record is real or synthetic:

$$D_k(x) = \sigma(\text{MLP}(x; \theta_{D_k}))$$

### Training Objective

We optimize the following minimax objective:

$$\min_G \max_{D_1, \ldots, D_K} \sum_{k=1}^K \mathbb{E}_{x \sim P_{S_k}}[\log D_k(x)] + \mathbb{E}_{z \sim \mathcal{N}(0,I)}[\log(1 - D_k(\pi_{V_k}(G(z))))]$$

This is equivalent to training $K$ independent GANs that share a generator.

### Weighted Discriminator Training

To improve coverage of rare population types, we weight the discriminator loss:

$$\mathcal{L}_{D_k} = -\mathbb{E}_{x \sim P_{S_k}}[w(x) \log D_k(x)] - \mathbb{E}_{z}[\log(1 - D_k(\pi_{V_k}(G(z))))]$$

where $w(x)$ upweights rare samples. We implement three weighting strategies:

1. **Uniform**: $w(x) = 1$
2. **Cluster-based**: $w(x) \propto 1/|\text{cluster}(x)|$ using k-means
3. **Density-based**: $w(x) \propto 1/\hat{p}(x)$ using kernel density estimation

### Evaluation Metrics

**Coverage**: Mean distance from holdout records to nearest synthetic:

$$\text{Coverage}(H, S) = \frac{1}{|H|} \sum_{x \in H} \min_{y \in S} \|x - y\|_2$$

Lower is better—indicates synthetic data covers the real distribution.

**Discriminator Accuracy**: Classification accuracy on held-out real vs. synthetic samples. Near 50% indicates the generator fools the discriminator.

**Maximum Mean Discrepancy (MMD)**: Kernel-based distance between distributions:

$$\text{MMD}^2(P, Q) = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\mathbb{E}[k(x, y)]$$

where $k$ is an RBF kernel with bandwidth selected via the median heuristic.

## Simulation Study

We evaluate FusionGAN on synthetic data where the ground truth joint distribution is known.

### Setup

```{code-cell} python
:tags: [remove-input]

from IPython.display import Markdown

Markdown(f"""
We simulate a population of **n = {r.sim_n:,}** households with {r.sim_n_vars} variables:

| Variable | Source | Description |
|----------|--------|-------------|
| `age` | CPS | Head of household age (18-80) |
| `wages` | CPS, PUF | Employment income (log-normal) |
| `unemployment` | CPS | UI benefits (zero-inflated) |
| `capital_gains` | PUF | Investment income (correlated with age) |
| `dividends` | PUF | Dividend income (log-normal) |

The key relationship we aim to recover is the **cross-source correlation** between `age` (CPS-only) and `capital_gains` (PUF-only): older households have higher capital gains.
""")
```

### Data Generation

```{code-cell} python
:tags: [remove-output]

# Ground truth population
n = 2000
age = np.random.randint(20, 80, n)
wages = np.random.lognormal(10, 1, n)
unemployment = np.random.choice([0, 1], n, p=[0.95, 0.05]) * np.random.lognormal(8, 1, n)
# Capital gains correlated with age
capital_gains = np.maximum(0, (age - 30) * 1000 + np.random.randn(n) * 10000)
dividends = np.random.lognormal(7, 1.5, n)

# Create source-specific views
cps_data = pd.DataFrame({
    'age': age,
    'wages': wages,
    'unemployment': unemployment
})

puf_data = pd.DataFrame({
    'wages': wages,
    'capital_gains': capital_gains,
    'dividends': dividends
})

print(f"CPS: {len(cps_data)} records, {len(cps_data.columns)} variables")
print(f"PUF: {len(puf_data)} records, {len(puf_data.columns)} variables")
print(f"True age-capital_gains correlation: {np.corrcoef(age, capital_gains)[0,1]:.3f}")
```

### Training

```{code-cell} python
:tags: [remove-output]

sources = [
    DataSource("cps", cps_data),
    DataSource("puf", puf_data),
]

synth = MultiSourceSynthesizer(sources, weighting="cluster")
synth.fit(epochs=200, holdout_frac=0.2, verbose=False)
```

### Results

```{code-cell} python
:tags: [remove-input]

Markdown(f"""
After training, we generate {r.n_synthetic:,} synthetic records and evaluate:

| Metric | CPS | PUF |
|--------|-----|-----|
| Coverage | {r.coverage_cps:.3f} | {r.coverage_puf:.3f} |
| Discriminator Accuracy | {r.disc_acc_cps:.1%} | {r.disc_acc_puf:.1%} |

The recovered **age-capital_gains correlation** in synthetic data is **{r.recovered_correlation:.3f}** (true: {r.true_correlation:.3f}).
""")
```

```{figure} _static/figures/correlation_recovery.png
:name: fig-correlation
:width: 80%

Cross-source correlation recovery. FusionGAN learns the age-capital_gains relationship despite these variables never appearing together in training data.
```

### Weighted Training Comparison

```{code-cell} python
:tags: [remove-input]

Markdown(f"""
We compare uniform vs. cluster-weighted training on a population with 10% rare high-income households:

| Weighting | Rare Type Coverage | Overall Coverage |
|-----------|-------------------|------------------|
| Uniform | {r.uniform_rare_coverage:.3f} | {r.uniform_overall_coverage:.3f} |
| Cluster | {r.cluster_rare_coverage:.3f} | {r.cluster_overall_coverage:.3f} |

Cluster weighting improves rare type coverage by **{r.rare_coverage_improvement:.0%}** while maintaining overall quality.
""")
```

```{figure} _static/figures/weighted_comparison.png
:name: fig-weighted
:width: 100%

Weighted vs. uniform training. Cluster weighting produces better coverage of rare high-income households (right cluster).
```

## Application: CPS-PUF Fusion

We apply FusionGAN to real survey data, fusing the Current Population Survey Annual Social and Economic Supplement (CPS ASEC) with the IRS Public Use File (PUF).

### Data

```{code-cell} python
:tags: [remove-input]

Markdown(f"""
| Source | Records | Variables | Key Fields |
|--------|---------|-----------|------------|
| CPS ASEC 2022 | {r.cps_n:,} | {r.cps_vars} | Age, education, employment, wages, UI, SNAP |
| IRS PUF 2021 | {r.puf_n:,} | {r.puf_vars} | Wages, interest, dividends, capital gains, deductions |

**Shared variables**: Wages, self-employment income

**Linking challenge**: No common identifiers; conditional independence violated for many variable pairs.
""")
```

### Results

```{code-cell} python
:tags: [remove-input]

Markdown(f"""
After training on normalized data for 500 epochs:

| Metric | CPS | PUF |
|--------|-----|-----|
| Coverage | {r.real_coverage_cps:.3f} | {r.real_coverage_puf:.3f} |
| Marginal RMSE | {r.real_marginal_rmse_cps:.3f} | {r.real_marginal_rmse_puf:.3f} |
| Joint Distribution χ² | {r.real_chi2_cps:.1f} | {r.real_chi2_puf:.1f} |

The fused dataset enables analyses previously impossible with either source alone, such as estimating the joint distribution of education (CPS) and itemized deductions (PUF).
""")
```

## Discussion

### Advantages

1. **No conditional independence assumption**: The generator learns the full joint distribution
2. **Flexible variable overlap**: Works with any pattern of shared/unique variables
3. **Scalable**: Discriminators are independent; training parallelizes across sources
4. **Weighted training**: Built-in mechanism for rare type coverage

### Limitations

1. **Mode collapse risk**: Standard GAN training instabilities apply
2. **No privacy guarantees**: Additional mechanisms needed for differential privacy
3. **Continuous variables**: Current implementation assumes continuous data; categorical extension needed

### Future Work

- **Wasserstein loss**: Replace BCE with Wasserstein distance for training stability
- **Conditional generation**: Add conditioning on policy scenarios
- **Privacy**: Integrate PATE-GAN or DP-SGD for formal privacy guarantees
- **Categorical variables**: Add embedding layers for mixed-type data

## Conclusion

FusionGAN provides a principled approach to multi-source survey data fusion that:
- Learns joint distributions without conditional independence assumptions
- Recovers cross-source correlations through shared generator training
- Improves rare type coverage through weighted discriminator training

The architecture is implemented in the open-source `fusiongan` Python package, available at [github.com/CosilicoAI/fusiongan](https://github.com/CosilicoAI/fusiongan).

## References

```{bibliography}
```
