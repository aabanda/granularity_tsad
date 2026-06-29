"""Configuration and repository paths for the granularity_tsad package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "paper.yaml"

# Point-wise detectors aggregated to build point-detectability profiles
# (paper Section 3.2). Order matters only for reproducibility of aggregations.
DEFAULT_POINTWISE_METHODS: list[str] = [
    "MyAlgo_ESN",
    "LSTMADbeta",
    "LSTMADalpha",
    "MyAlgo_TAE",
    "FITS",
    "MyAlgo_DM",
    "EncDecAD",
    "FCVAE",
    "TimesNet",
    "AE",
    "Donut",
    "SRCNN",
]


@dataclass
class PaperConfig:
    repo_root: Path
    data_dir: Path
    results_dir: Path
    figures_dir: Path
    methods_point_wise: list[str] = field(default_factory=list)
    methods_sequence_wise: list[str] = field(default_factory=list)
    training_schema: str = "naive"
    preprocess: str = "min-max"
    k_values: list[int] = field(default_factory=list)
    exclude_point_anomalies: bool = True
    min_anomaly_length: int = 2
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def all_methods(self) -> list[str]:
        return self.methods_point_wise + self.methods_sequence_wise

    def filter_sequence_anomalies(self, anomaly_df):
        """Drop point-anomaly series (paper Section 3.1) when configured.

        Keeps rows whose ``Anomaly Length`` (end - start) is at least
        :attr:`min_anomaly_length`. With the default threshold this removes the
        3 single-point and 17 two-point series, leaving the 230 sequence
        anomalies analysed in the paper.
        """
        if not self.exclude_point_anomalies:
            return anomaly_df
        return anomaly_df[anomaly_df["Anomaly Length"] >= self.min_anomaly_length]


def load_config(path: Path | None = None) -> PaperConfig:
    cfg_path = path or DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    benchmark = raw.get("benchmark", {})
    dataset = raw.get("dataset", {})
    methods = raw.get("methods", {})
    figures = raw.get("figures", {})
    metrics = raw.get("metrics", {})

    return PaperConfig(
        repo_root=REPO_ROOT,
        data_dir=REPO_ROOT / dataset.get("data_dir", "data"),
        results_dir=REPO_ROOT / benchmark.get("results_dir", "Results"),
        figures_dir=REPO_ROOT / figures.get("output_dir", "figures"),
        methods_point_wise=list(methods.get("point_wise", [])),
        methods_sequence_wise=list(methods.get("sequence_wise", [])),
        training_schema=benchmark.get("training_schema", "naive"),
        preprocess=benchmark.get("preprocess", "min-max"),
        k_values=list(metrics.get("pak_k_values", [])),
        exclude_point_anomalies=bool(dataset.get("exclude_point_anomalies", True)),
        min_anomaly_length=int(dataset.get("min_anomaly_length", 2)),
        raw=raw,
    )
