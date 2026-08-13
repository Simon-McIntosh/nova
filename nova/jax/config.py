"""Enable JAX dtype capability once and select precision on each runtime object.

The process permits both native float32 and float64 arithmetic.  Domain code
constructs arrays in an explicitly resolved dtype, so one process may compile
separate fp32 and fp64 variants without changing configuration around a solve.
``auto`` is resolved by the solver or operator that owns the numerical contract.
"""

from dataclasses import dataclass
from enum import StrEnum
import gc


class Precision(StrEnum):
    """Runtime precision selected before arrays are constructed or traced."""

    AUTOMATIC = "auto"
    SINGLE = "float32"
    DOUBLE = "float64"


__all__ = [
    "CompilationRelease",
    "Precision",
    "bound_compilation_retention",
    "compilation_release_history",
    "configure_dtypes",
    "resolve_precision",
]

_dtypes_configured = False


@dataclass(frozen=True)
class CompilationRelease:
    """Executable counts around one threshold-triggered cache release."""

    before: int
    after: int
    collected_objects: int


_compilation_releases: list[CompilationRelease] = []


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
