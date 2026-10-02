"""Configuration and repository paths for the granularity_tsad package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "paper.yaml"

# Point-wise detectors used to build point-detectability profiles
# (paper Section 3.2 / main_paper ``alg_list_pred``). Overridden by config.
DEFAULT_POINTWISE_METHODS: list[str] = [
    "AE",
    "FCVAE",
    "EncDecAD",
    "Donut",
    "SRCNN",
    "MyAlgo_DM",
    "MyAlgo_TAE",
    "LSTMADalpha",
    "LSTMADbeta",
    "TimesNet",
    "FITS",
    "MyAlgo_ESN",
]


@dataclass
class PaperConfig:
    repo_root: Path
    data_dir: Path
    results_dir: Path
    figures_dir: Path
    methods_point_wise: list[str] = field(default_factory=list)
    methods_point_wise_for_clustering: list[str] = field(default_factory=list)
    methods_point_wise_for_analysis: list[str] = field(default_factory=list)
    methods_sequence_wise: list[str] = field(default_factory=list)
    methods_sequence_wise_extra: list[str] = field(default_factory=list)
    sequence_wise_extended: bool = False
    split_point_wise: bool = True
    training_schema: str = "naive"
    preprocess: str = "min-max"
    k_values: list[int] = field(default_factory=list)
    exclude_point_anomalies: bool = True
    min_anomaly_length: int = 2
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def all_methods(self) -> list[str]:
        seen: list[str] = []
        for name in (
            self.methods_point_wise
            + self.methods_point_wise_for_clustering
            + self.methods_point_wise_for_analysis
            + self.methods_sequence_wise
        ):
            if name not in seen:
                seen.append(name)
        return seen

    @property
    def analysis_methods(self) -> list[str]:
        """Point-wise (analysis split) + active sequence-wise catalogue."""
        seen: list[str] = []
        for name in self.point_methods_for_analysis + self.methods_sequence_wise:
            if name not in seen:
                seen.append(name)
        return seen

    @property
    def point_methods_for_clustering(self) -> list[str]:
        """Detectors used to build profiles / clusters (Fig 6–7)."""
        if self.split_point_wise:
            return self.methods_point_wise_for_clustering or self.methods_point_wise
        return self.methods_point_wise

    @property
    def point_methods_for_analysis(self) -> list[str]:
        """Detectors used in boxplots / Bayesian P-vs-S (Fig 9–12)."""
        if self.split_point_wise:
            return self.methods_point_wise_for_analysis or self.methods_point_wise
        return self.methods_point_wise

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

    point_wise = list(methods.get("point_wise", []))
    for_clustering = list(
        methods.get("point_wise_for_clustering", point_wise or DEFAULT_POINTWISE_METHODS)
    )
    for_analysis = list(methods.get("point_wise_for_analysis", point_wise))
    if not point_wise:
        # Keep legacy ``point_wise`` as the ordered union of the two splits.
        seen: list[str] = []
        for name in for_clustering + for_analysis:
            if name not in seen:
                seen.append(name)
        point_wise = seen
    split_point_wise = bool(methods.get("split_point_wise", True))
    sequence_wise = list(methods.get("sequence_wise", []))
    sequence_wise_extra = list(
        methods.get(
            "sequence_wise_extra",
            methods.get("sequence_wise_extended_methods", ["DAMP", "CARLA"]),
        )
    )
    sequence_wise_extended = bool(methods.get("sequence_wise_extended", False))
    if sequence_wise_extended:
        for name in sequence_wise_extra:
            if name not in sequence_wise:
                sequence_wise.append(name)

    def _resolve(path_value: str | Path, default: str) -> Path:
        path = Path(path_value if path_value is not None else default)
        return path if path.is_absolute() else (REPO_ROOT / path)

    return PaperConfig(
        repo_root=REPO_ROOT,
        data_dir=_resolve(dataset.get("data_dir", "data"), "data"),
        results_dir=_resolve(benchmark.get("results_dir", "Results"), "Results"),
        figures_dir=_resolve(figures.get("output_dir", "figures"), "figures"),
        methods_point_wise=point_wise,
        methods_point_wise_for_clustering=for_clustering,
        methods_point_wise_for_analysis=for_analysis,
        methods_sequence_wise=sequence_wise,
        methods_sequence_wise_extra=sequence_wise_extra,
        sequence_wise_extended=sequence_wise_extended,
        split_point_wise=split_point_wise,
        training_schema=benchmark.get("training_schema", "naive"),
        preprocess=benchmark.get("preprocess", "min-max"),
        k_values=list(metrics.get("pak_k_values", [])),
        exclude_point_anomalies=bool(dataset.get("exclude_point_anomalies", True)),
        min_anomaly_length=int(dataset.get("min_anomaly_length", 2)),
        raw=raw,
    )
