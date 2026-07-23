"""Fixtures and environment pinning for the characterization harness.

Goldens are generated under single-threaded BLAS/OpenMP with a fixed hash seed.
The comparison lanes are tolerance-based and therefore robust to threading, so
running without the pin does not hard-fail here -- it only emits a warning. The
goldens *generator* asserts the pin (see ``generate_goldens.py``).
"""

from __future__ import annotations

import os
import warnings

import pytest

# Matplotlib must never try to open a display in the harness.
os.environ.setdefault("MPLBACKEND", "Agg")

from . import _environment, _manifest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _pinned_numeric_environment():
    """Warn (do not fail) if BLAS/OpenMP threading is not pinned."""
    if not _environment.threads_pinned():
        warnings.warn(
            "characterization comparisons run at tolerance and tolerate this, "
            "but goldens were generated with OPENBLAS_NUM_THREADS=1 and "
            "OMP_NUM_THREADS=1; export those for the tightest agreement.",
            RuntimeWarning,
            stacklevel=1,
        )
    yield


@pytest.fixture(scope="session")
def manifest() -> _manifest.Manifest:
    """Return the loaded goldens manifest, or skip if it is absent."""
    if not _manifest.manifest_exists():
        pytest.skip("goldens manifest not generated yet")
    return _manifest.Manifest.load()


@pytest.fixture(scope="session")
def goldens_dir():
    """Return the directory holding canonical golden artifacts."""
    return _manifest.GOLDENS_DIR
