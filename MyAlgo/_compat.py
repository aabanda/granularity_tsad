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
