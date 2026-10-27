"""Deprecated alias for :mod:`pe_nbeats`.

The package was renamed from ``lightningnbeats`` to ``pe_nbeats`` (PyPI:
``pe-nbeats``). This module keeps old imports working, including submodule
imports such as ``from lightningnbeats.loaders import TimeSeriesDataModule``.
"""
import importlib
import sys
import warnings

warnings.warn(
    "`lightningnbeats` has been renamed to `pe_nbeats`; "
    "update imports to `import pe_nbeats`.",
    DeprecationWarning,
    stacklevel=2,
)

import pe_nbeats as _pe_nbeats
from pe_nbeats import *  # noqa: F401,F403
from pe_nbeats import __version__, get_best_accelerator  # noqa: F401

for _name in (
    "blocks",
    "blocks.blocks",
    "constants",
    "data",
    "loaders",
    "losses",
    "models",
    "preflight",
):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"pe_nbeats.{_name}")


def __getattr__(name):
    return getattr(_pe_nbeats, name)
