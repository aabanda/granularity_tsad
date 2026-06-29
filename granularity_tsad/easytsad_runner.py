"""Thin wrapper around EasyTSAD to run point-wise detectors and read scores.

EasyTSAD is treated as an external dependency. Depending on how it was
installed the controller lives at ``EasyTSAD.Controller`` or
``lib.EasyTSAD.EasyTSAD.Controller``; both import paths are attempted.

EasyTSAD persists anomaly scores to ``<workspace>/Results/Scores/<method>/
<schema>/<dataset>/<curve>.npy``. After running, :func:`load_score` reads
those arrays back so they can be aggregated into a heatmap.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def _import_controller():
    for module in ("EasyTSAD.Controller", "lib.EasyTSAD.EasyTSAD.Controller"):
        try:
            return importlib.import_module(module).TSADController
        except ImportError:
            continue
    raise ImportError(
        "Could not import EasyTSAD. Install it or make `EasyTSAD` importable "
        "(e.g. add the EasyTSAD checkout to PYTHONPATH)."
    )


def _import_methods(method_names: list[str]) -> None:
    """Ensure every requested detector class is registered with EasyTSAD.

    EasyTSAD auto-registers any ``BaseMethod`` subclass through a metaclass at
    import time (``BaseMethodMeta.registry``). Importing ``EasyTSAD.Methods``
    registers the built-ins; the custom ``MyAlgo_*`` detectors register
    themselves when their module is imported from the local ``MyAlgo`` package.
    """
    builtins_mod = None
    for module in ("EasyTSAD.Methods", "lib.EasyTSAD.EasyTSAD.Methods"):
        try:
            builtins_mod = importlib.import_module(module)
            break
        except ImportError:
            continue
    if builtins_mod is None:
        raise ImportError("Could not import EasyTSAD.Methods.")

    for name in method_names:
        if name in getattr(builtins_mod, "__dict__", {}):
            continue
        # Custom detectors: importing the module triggers metaclass registration.
        for module in (f"lib.MyAlgo.{name}", f"MyAlgo.{name}", name):
            try:
                importlib.import_module(module)
                break
            except ImportError:
                continue


class EasyTSADRunner:
    """Run EasyTSAD point-wise detectors on UCR curves and fetch their scores."""

    def __init__(
        self,
        data_dir: Path,
        workspace: Path | None = None,
        dataset: str = "UCR",
        training_schema: str = "naive",
        preprocess: str = "min-max",
    ):
        self.data_dir = Path(data_dir)
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.dataset = dataset
        self.training_schema = training_schema
        self.preprocess = preprocess
        self._controller_cls = _import_controller()

    def run(self, methods: list[str], curves: list[str], replace: bool = True) -> None:
        """Run the given detectors on the specified UCR curve(s)."""
        gctrl = self._controller_cls()
        gctrl.set_dataset(
            dataset_type="UTS",
            dirname=str(self.data_dir),
            datasets=[self.dataset],
            specify_curves=True,
            curve_names=list(curves),
        )
        # Ensure the method classes are importable even if not auto-registered.
        _import_methods(methods)
        for method in methods:
            gctrl.run_exps(
                method=method,
                training_schema=self.training_schema,
                replace=replace,
                preprocess=self.preprocess,
            )

    def score_path(self, method: str, curve: str) -> Path:
        return (
            self.workspace
            / "Results"
            / "Scores"
            / method
            / self.training_schema
            / self.dataset
            / f"{curve}.npy"
        )

    def load_score(self, method: str, curve: str) -> np.ndarray:
        path = self.score_path(method, curve)
        if not path.exists():
            raise FileNotFoundError(
                f"Score file not found: {path}. Did the EasyTSAD run complete?"
            )
        return np.load(path)

    def load_scores(self, methods: list[str], curve: str) -> list[np.ndarray]:
        return [self.load_score(m, curve) for m in methods]
