"""Pin and fingerprint the execution environment.

Reproducible goldens require a pinned numeric environment: single-threaded
BLAS/OpenMP (so reduction order is fixed), a stable hash seed, and a known set
of package versions. :func:`env_lock` returns a sha256 over the resolved lock
file and the versions of the packages that actually touch the numbers, so a
dependency bump is visible in the manifest. :func:`require_pinned_threads`
raises unless the threading environment is set the way goldens were generated.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path

# Packages whose version can move the last bits of a fit result.
NUMERIC_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "xarray",
    "pandas",
    "openpyxl",
    "pyquaternion",
)

REQUIRED_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")


def repo_root() -> Path:
    """Return the repository root (three levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def package_versions() -> dict[str, str]:
    """Return the installed versions of the numeric packages."""
    versions: dict[str, str] = {}
    for name in NUMERIC_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    return dict(sorted(versions.items()))


def env_lock() -> str:
    """Return a sha256 fingerprint of the resolved environment."""
    digest = hashlib.sha256()
    lock = repo_root() / "uv.lock"
    if lock.exists():
        digest.update(lock.read_bytes())
    for name, version in package_versions().items():
        digest.update(f"{name}=={version}\n".encode())
    return digest.hexdigest()


def threads_pinned() -> bool:
    """Return True when single-threaded BLAS/OpenMP is configured."""
    return all(os.environ.get(var) == "1" for var in REQUIRED_THREAD_VARS)


def require_pinned_threads() -> None:
    """Raise unless the threading environment matches the golden baseline."""
    if not threads_pinned():
        missing = {var: os.environ.get(var) for var in REQUIRED_THREAD_VARS}
        raise RuntimeError(
            "characterization goldens require single-threaded numerics; set "
            "OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 "
            f"(currently {missing})"
        )
