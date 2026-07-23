"""Reduce arbitrary fit results to a canonical, hashable numeric form.

Every characterized entry point returns something -- a numpy array, an xarray
Dataset/DataArray, a mapping, a dataclass holding a ``data`` attribute, or a
scalar. :func:`canonicalize` flattens any of these into a sorted mapping of
``str -> float64 ndarray`` with a deterministic layout, so two runs of the same
code produce byte-identical serialized output under the pinned environment.

The serialized artifact is a compressed ``.npz`` written with sorted keys; its
sha256 is the fingerprint recorded in the manifest.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence

import numpy as np


def _walk(prefix: str, obj, out: dict[str, np.ndarray]) -> None:
    """Recursively flatten ``obj`` into ``out`` keyed by dotted path."""
    # Lazy imports: xarray is a hard dep of the assembly code but keep the
    # canonicaliser importable without it for pure-array callers.
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - xarray is always present here
        xr = None

    if xr is not None and isinstance(obj, xr.Dataset):
        for name in sorted(map(str, obj.data_vars)):
            _walk(f"{prefix}.{name}" if prefix else name, obj[name], out)
        # Coordinates are part of the observable output; capture numeric ones.
        for name in sorted(map(str, obj.coords)):
            coord = obj[name]
            if np.issubdtype(np.asarray(coord.values).dtype, np.number):
                _walk(
                    f"{prefix}.coord.{name}" if prefix else f"coord.{name}",
                    coord.values,
                    out,
                )
        return
    if xr is not None and isinstance(obj, xr.DataArray):
        _walk(prefix, np.asarray(obj.values), out)
        return

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is present in this env
        pd = None
    if pd is not None and isinstance(obj, pd.DataFrame):
        _store(prefix, obj.to_numpy(dtype=float, na_value=np.nan), out)
        return
    if pd is not None and isinstance(obj, pd.Series):
        _store(prefix, obj.to_numpy(dtype=float, na_value=np.nan), out)
        return

    if isinstance(obj, Mapping):
        for key in sorted(obj, key=str):
            _walk(f"{prefix}.{key}" if prefix else str(key), obj[key], out)
        return
    if isinstance(obj, (str, bytes)):
        # Non-numeric leaves are dropped from the canonical numeric form; the
        # entry-point registry decides what is golden-worthy.
        return
    if isinstance(obj, Sequence):
        arr = np.asarray(obj)
        if arr.dtype == object:
            for i, item in enumerate(obj):
                _walk(f"{prefix}[{i}]", item, out)
            return
        _store(prefix, arr, out)
        return
    _store(prefix, np.asarray(obj), out)


def _store(key: str, arr: np.ndarray, out: dict[str, np.ndarray]) -> None:
    if arr.dtype == object:
        # Object arrays (shapely polygons, vtk handles) are not numeric goldens.
        return
    if not np.issubdtype(arr.dtype, np.number):
        return
    out[key or "value"] = np.ascontiguousarray(arr, dtype=np.float64)


def canonicalize(obj) -> dict[str, np.ndarray]:
    """Return a sorted mapping of ``str -> float64 ndarray`` for ``obj``."""
    out: dict[str, np.ndarray] = {}
    _walk("", obj, out)
    return dict(sorted(out.items()))


def to_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize ``arrays`` to compressed ``.npz`` bytes with sorted keys."""
    buffer = io.BytesIO()
    ordered = {k: arrays[k] for k in sorted(arrays)}
    np.savez_compressed(buffer, **ordered)
    return buffer.getvalue()


def sha256_bytes(payload: bytes) -> str:
    """Return the hex sha256 of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    """Return the hex sha256 of the file at ``path`` (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(payload: bytes) -> dict[str, np.ndarray]:
    """Load a canonical ``.npz`` payload back into a mapping of arrays."""
    with np.load(io.BytesIO(payload)) as data:
        return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}
