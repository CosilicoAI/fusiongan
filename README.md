# fusiongan

Multi-source adversarial synthesis for survey data fusion.

## The Problem

Survey data sources like CPS, PUF, SCF, ACS each observe different slices of the population:
- **Row slices**: Different sampled individuals
- **Column slices**: Different variables observed

No single source has complete information. Traditional statistical matching assumes conditional independence. We can do better.

## The Approach

Train a generator that produces complete synthetic records which simultaneously fool discriminators for each data source:

```
Generator → Full synthetic record (all variables)
    ↓
    ├── Project to CPS columns → CPS Discriminator: "Real or fake?"
    ├── Project to PUF columns → PUF Discriminator: "Real or fake?"
    └── Project to SCF columns → SCF Discriminator: "Real or fake?"
```

The generator must produce records that look realistic when projected to ANY source's observed variables.

## Key Features

- **Multi-source**: Train on CPS + PUF + SCF + ... simultaneously
- **Weighted discriminators**: Rare population types get proper attention
- **Coverage-aware**: Explicitly optimize for holdout coverage, not just discriminator loss
- **Holdout evaluation**: Test on held-out data from each source

## Installation

```bash
pip install fusiongan
```

## Quick Start

```python
from fusiongan import MultiSourceSynthesizer, DataSource

# Define sources with their observed columns
sources = [
    DataSource("cps", cps_df, columns=["age", "sex", "wages", "uc", "ss"]),
    DataSource("puf", puf_df, columns=["wages", "cap_gains", "dividends"]),
]

# Train
synth = MultiSourceSynthesizer(sources)
synth.fit(holdout_frac=0.2, epochs=100)

# Evaluate coverage on holdouts
metrics = synth.evaluate()
# {'cps': {'coverage': 0.12, 'disc_acc': 0.52}, 'puf': {...}}

# Generate complete synthetic records
synthetic = synth.generate(n=100_000)
```

## How It Works

1. **Split each source** into train/holdout
2. **Train generator** to produce full records
3. **Train discriminators** (one per source) to distinguish real holdout from projected synthetic
4. **Weight training** so rare population types get attention (cluster-based or density-based)
5. **Evaluate** coverage: can we generate records near every holdout record?

## Citation

```bibtex
@software{fusiongan,
  author = {Cosilico},
  title = {fusiongan: Multi-source adversarial synthesis for survey data fusion},
  year = {2025},
  url = {https://github.com/CosilicoAI/fusiongan}
}
```

## License

MIT
