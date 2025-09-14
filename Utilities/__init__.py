# Utilities/__init__.py
import importlib
import os
import sys

# Default backend (overridden in run.py)
_backend = os.environ.get("UTILITIES_BACKEND", "ch")

def set_backend(backend: str):
    global _backend
    if backend not in ("ch", "ch_cmdp"):
        raise ValueError(f"Invalid backend {backend}")
    _backend = backend
    _reload_utilities()

def _import(name: str):
    """Helper to dynamically import from the chosen backend"""
    return importlib.import_module(f"Utilities.{_backend}.{name}")

def _reload_utilities():
    """
    Re-import and register Utilities.<module> pointing
    to Utilities.<backend>.<module>
    """

    submodules = [
        "BaseUtility",
        "BikeUtility",
        "CarUtility",
        "Parameters",
        "PtUtility",
        "TourUtility",
        "WalkUtility",
        "ZeroUtility"
    ]

    for name in submodules:
        mod = _import(name)
        sys.modules[f"Utilities.{name}"] = mod
        globals()[name] = mod   # so `import Utilities; Utilities.carUtility` works too

# Initialize on first import
_reload_utilities()
