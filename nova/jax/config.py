"""Configure JAX precision explicitly before constructing runtime arrays.

The runtime default remains single precision.  Explicit double-precision
arrays are nevertheless honoured, so one process may compile separate fp32 and
fp64 variants without changing configuration around a solve.  Domain code
resolves ``auto`` against its measured numerical contract; configuration only
records the operator choice and establishes the process-wide dtype policy.
"""

from enum import StrEnum


class Precision(StrEnum):
    """Runtime precision selected before arrays are constructed or traced."""

    AUTOMATIC = "auto"
    SINGLE = "float32"
    DOUBLE = "float64"


__all__ = ["Precision", "configure_dtypes"]

_dtypes_configured = False
_enabled = False


def configure_dtypes() -> None:
    """Establish the no-toggle runtime dtype policy once per process.

    Explicit fp64 remains available while ordinary JAX array construction
    continues to default to fp32.  Precision choices live on solver objects,
    allowing single- and double-precision JIT variants to coexist.
    """
    global _dtypes_configured
    if _dtypes_configured:
        return

    import jax

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    _dtypes_configured = True


def _resolve_precision(
    requested: Precision | str,
    automatic: Precision,
) -> Precision:
    """Resolve one solver's requested precision against its automatic policy."""
    if not _dtypes_configured:
        raise RuntimeError(
            "call nova.jax.config.configure_dtypes() before constructing "
            "precision-sensitive arrays"
        )
    choice = Precision(requested)
    if choice is Precision.AUTOMATIC:
        return automatic
    return choice


def enable_x64() -> bool:
    """Enable JAX double precision once per process.

    Idempotent: repeat calls after the first are free, so a module may call this
    at import and a caller may call it again before tracing without cost.

    Returns
    -------
    bool
        True once fp64 is enabled, False when JAX is not installed. Returning
        rather than raising lets a module whose JAX support is optional call
        this unconditionally from inside its import guard.
    """
    global _enabled
    if _enabled:
        return True
    try:
        import jax
    except ModuleNotFoundError:
        return False
    jax.config.update("jax_enable_x64", True)
    _enabled = True
    return True
