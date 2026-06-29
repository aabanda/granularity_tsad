"""granularity_tsad: granularity-aware evaluation for time-series anomaly detection.

Reusable, paper-clean implementation of the metrics, aggregations, clustering
and figures from the *Granularity* study:

* :mod:`granularity_tsad.metrics` — point-wise, sequence-wise and PAK metrics.
* :mod:`granularity_tsad.data` — UCR loading and score/target alignment.
* :mod:`granularity_tsad.aggregation` — score aggregation and detectability profiles.
* :mod:`granularity_tsad.clustering` — hierarchical clustering of series.
* :mod:`granularity_tsad.plots` — publication-quality figures.
* :mod:`granularity_tsad.easytsad_runner` — run EasyTSAD detectors from scratch.
"""

from . import aggregation, clustering, data, metrics, plots
from .config import PaperConfig, load_config

__version__ = "0.1.0"

__all__ = [
    "metrics",
    "data",
    "aggregation",
    "clustering",
    "plots",
    "load_config",
    "PaperConfig",
]
