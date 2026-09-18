"""Import compatibility shim for the custom ``MyAlgo_*`` detectors.

The detectors depend on two external libraries that may be available under
different import roots:

* **EasyTSAD** — either installed as the ``EasyTSAD`` package or vendored under
  the development namespace ``lib.EasyTSAD.EasyTSAD``.
* **tsdalia** — TECNALIA's anomaly-detection library, installable as ``tsdalia``
  (``pip install -e <path>/tsadalia``) or vendored as ``lib.tsadalia.tsdalia``.

This module resolves both layouts so the detector files have a single, stable
import surface regardless of how the environment is set up.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_EASYTSAD_ROOTS = ("EasyTSAD", "lib.EasyTSAD.EasyTSAD")
_TSDALIA_ROOTS = ("tsdalia", "lib.tsadalia.tsdalia")


def _first_module(candidates: tuple[str, ...]) -> ModuleType:
    last_error: Exception | None = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last_error = exc
    raise ImportError(
        "Could not import any of: " + ", ".join(candidates)
        + (f". Last error: {last_error}" if last_error else "")
    )


def easytsad(submodule: str) -> ModuleType:
    """Import ``<EasyTSAD root>.<submodule>`` from whichever layout is present."""
    return _first_module(tuple(f"{root}.{submodule}" for root in _EASYTSAD_ROOTS))


def tsdalia(submodule: str) -> ModuleType:
    """Import ``<tsdalia root>.<submodule>`` from whichever layout is present."""
    return _first_module(tuple(f"{root}.{submodule}" for root in _TSDALIA_ROOTS))


# EasyTSAD's ``Methods`` package has a circular import with ``Controller`` /
# ``TrainingSchema`` that only resolves when the Controller (the documented
# entry point) is imported first. Import it up front so ``Methods`` is safe to
# import regardless of order.
try:
    easytsad("Controller")
except ImportError:
    pass

_methods = easytsad("Methods")
BaseMethod = _methods.BaseMethod
TSData = _methods.TSData
BaseMethodMeta = _methods.BaseMethodMeta


def pick_torch_device():
    """Return a usable torch device; never raise on broken CUDA drivers."""
    import torch

    # Some hosts report a CUDA driver error inside ``is_available()`` itself
    # (e.g. Error 304). Patch once so the rest of the stack stays on CPU.
    if not getattr(torch.cuda, "_granularity_safe_is_available", False):
        _orig = torch.cuda.is_available

        def _safe_is_available() -> bool:
            try:
                return bool(_orig())
            except Exception:
                return False

        torch.cuda.is_available = _safe_is_available  # type: ignore[method-assign]
        torch.cuda._granularity_safe_is_available = True

    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def pick_torch_device_str() -> str:
    return "cuda" if pick_torch_device().type == "cuda" else "cpu"


# Apply the CUDA safety patch as soon as this shim is imported.
try:
    pick_torch_device()
except Exception:
    pass
