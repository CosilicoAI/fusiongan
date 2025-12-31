"""fusiongan: Multi-source adversarial synthesis for survey data fusion."""

from fusiongan.data_source import DataSource
from fusiongan.discriminator import Discriminator
from fusiongan.generator import Generator
from fusiongan.metrics import (
    compute_coverage,
    compute_discriminator_accuracy,
    compute_mmd,
    pairwise_distances,
)
from fusiongan.synthesizer import MultiSourceSynthesizer

__version__ = "0.1.0"

__all__ = [
    "DataSource",
    "Discriminator",
    "Generator",
    "MultiSourceSynthesizer",
    "compute_coverage",
    "compute_discriminator_accuracy",
    "compute_mmd",
    "pairwise_distances",
]
