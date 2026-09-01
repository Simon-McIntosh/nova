"""Enable JAX dtype capability once and select precision on each runtime object.

The process permits both native float32 and float64 arithmetic.  Domain code
constructs arrays in an explicitly resolved dtype, so one process may compile
separate fp32 and fp64 variants without changing configuration around a solve.
``auto`` is resolved by the solver or operator that owns the numerical contract.
"""

from dataclasses import dataclass
from enum import StrEnum
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any


_PERSISTENT_CACHE_KEY_FIELDS = (
    "serialized computation",
    "jaxlib version",
    "backend version",
    "XLA flags",
    "compile options",
    "accelerator topology",
)
_PERSISTENT_CACHE_MAXIMUM_BYTES = 8 << 30
_PERSISTENT_CACHE_MINIMUM_COMPILE_SECONDS = 1.0


class Precision(StrEnum):
    """Runtime precision selected before arrays are constructed or traced."""

    AUTOMATIC = "auto"
    SINGLE = "float32"
    DOUBLE = "float64"


__all__ = [
    "CompilationRelease",
    "PersistentCompilationCache",
    "Precision",
    "bound_compilation_retention",
    "compilation_release_history",
    "configure_dtypes",
    "configure_persistent_compilation_cache",
    "default_persistent_compilation_cache_root",
    "resolve_precision",
]

_dtypes_configured = False


@dataclass(frozen=True)
class CompilationRelease:
    """Executable counts around one threshold-triggered cache release."""

    before: int
    after: int
    collected_objects: int


@dataclass(frozen=True)
class PersistentCompilationCache:
    """One explicitly selected persistent-cache namespace and its key contract."""

    root: Path
    directory: Path
    version_key: str
    runtime_identity: dict[str, Any]
    minimum_compile_seconds: float
    maximum_bytes: int

    def receipt(self) -> dict[str, Any]:
        """Return the directory layout and JAX entry-key invalidation contract."""
        return {
            "selection": "explicit_per_recipe",
            "root": str(self.root),
            "directory": str(self.directory),
            "layout": "<root>/nova/jax-compilation/<runtime-version-key>",
            "version_key": self.version_key,
            "runtime_identity": self.runtime_identity,
            "entry_key_fields": list(_PERSISTENT_CACHE_KEY_FIELDS),
            "invalidation_scope": {
                "directory_version_key": [
                    "backend and compiler options",
                    "device topology",
                    "jax and jaxlib/XLA build",
                ],
                "jax_entry_key": list(_PERSISTENT_CACHE_KEY_FIELDS),
            },
            "minimum_compile_seconds": self.minimum_compile_seconds,
            "maximum_bytes": self.maximum_bytes,
        }


_compilation_releases: list[CompilationRelease] = []


def default_persistent_compilation_cache_root() -> Path:
    """Return the tracked per-user parent selected by explicit launch recipes."""
    return Path.home() / ".cache"


def _persistent_cache_runtime_identity() -> dict[str, Any]:
    """Return compiler and topology values that version the shared directory."""
    import jax
    import jaxlib

    devices = jax.local_devices()
    if not devices:
        raise RuntimeError("persistent compilation cache requires a local JAX device")
    client = devices[0].client
    return {
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "backend": devices[0].platform,
        "backend_version": getattr(client, "platform_version", None),
        "compiler_options": {
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "jax_compilation_cache_include_metadata_in_key": bool(
                jax.config.jax_compilation_cache_include_metadata_in_key
            ),
            "jax_persistent_cache_enable_xla_caches": (
                jax.config.jax_persistent_cache_enable_xla_caches
            ),
            "xla_flags": os.environ.get("XLA_FLAGS"),
        },
        "device_topology": [
            {
                "platform": device.platform,
                "kind": device.device_kind,
                "process_index": int(device.process_index),
                "id": int(device.id),
            }
            for device in devices
        ],
    }


def configure_persistent_compilation_cache(
    root: Path | str,
    *,
    minimum_compile_seconds: float = _PERSISTENT_CACHE_MINIMUM_COMPILE_SECONDS,
    maximum_bytes: int = _PERSISTENT_CACHE_MAXIMUM_BYTES,
) -> PersistentCompilationCache:
    """Explicitly select the shared versioned cache for one launch recipe.

    The directory version isolates runtime/compiler and topology changes. JAX's
    entry key then hashes the serialized computation and the same compiler and
    accelerator inputs, so data-only changes reuse an entry while changed traced
    programs cannot. Nothing calls this function at import time: each launch
    recipe opts in at its executable boundary.
    """
    if minimum_compile_seconds < 0:
        raise ValueError("minimum compile seconds cannot be negative")
    if maximum_bytes < 1:
        raise ValueError("maximum cache bytes must be positive")

    import jax

    resolved_root = Path(root).expanduser().resolve()
    runtime_identity = _persistent_cache_runtime_identity()
    encoded = json.dumps(
        runtime_identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    version_key = f"runtime-{hashlib.sha256(encoded).hexdigest()[:20]}"
    directory = resolved_root / "nova" / "jax-compilation" / version_key
    directory.mkdir(parents=True, exist_ok=True)
    configured = jax.config.jax_compilation_cache_dir
    if configured is None or Path(configured).expanduser().resolve() != directory:
        from jax._src import compilation_cache

        compilation_cache.reset_cache()
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_compilation_cache_max_size", maximum_bytes)
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs", minimum_compile_seconds
    )
    return PersistentCompilationCache(
        root=resolved_root,
        directory=directory,
        version_key=version_key,
        runtime_identity=runtime_identity,
        minimum_compile_seconds=minimum_compile_seconds,
        maximum_bytes=maximum_bytes,
    )


def compilation_release_history() -> tuple[CompilationRelease, ...]:
    """Return cache-release measurements accumulated by this process."""
    return tuple(_compilation_releases)


def _live_executable_count() -> int:
    """Return loaded executable count without initialising a JAX backend."""
    import sys

    if "jax" not in sys.modules:
        return 0

    from jax._src import xla_bridge

    if not xla_bridge.backends_are_initialized():
        return 0
    return len(xla_bridge.get_backend().live_executables())


def bound_compilation_retention(
    live_executable_ceiling: int,
) -> CompilationRelease | None:
    """Release process-wide JAX caches after they cross ``ceiling``.

    The check is safe before JAX import and backend initialisation.  Collection
    is deliberately threshold-driven: reusable compiled functions remain hot
    below the bound, while a long-lived process cannot retain an unbounded set
    of loaded executables through JAX's staging and dispatch caches.
    """
    if live_executable_ceiling < 1:
        raise ValueError("live executable ceiling must be positive")

    before = _live_executable_count()
    if before <= live_executable_ceiling:
        return None

    import jax

    jax.clear_caches()
    collected = gc.collect()
    release = CompilationRelease(
        before=before,
        after=_live_executable_count(),
        collected_objects=collected,
    )
    _compilation_releases.append(release)
    return release


def configure_dtypes() -> None:
    """Establish the no-toggle runtime dtype policy once per process.

    Enabling x64 makes double arithmetic numerically identical to JAX's ordinary
    fp64 mode; it does not force arrays to use it.  Precision choices live on
    solver objects, and their explicit array dtypes allow single- and
    double-precision JIT variants to coexist.
    """
    global _dtypes_configured
    if _dtypes_configured:
        return

    import jax

    jax.config.update("jax_enable_x64", True)
    _dtypes_configured = True


def resolve_precision(
    requested: Precision | str,
    automatic: Precision,
) -> Precision:
    """Resolve one solver's requested precision against its automatic policy."""
    choice = Precision(requested)
    try:
        configure_dtypes()
    except ModuleNotFoundError as error:
        if error.name != "jax":
            raise
    if choice is Precision.AUTOMATIC:
        return automatic
    return choice
