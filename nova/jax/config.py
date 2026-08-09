"""Enable JAX dtype capability once and select precision on each runtime object.

The process permits both native float32 and float64 arithmetic.  Domain code
constructs arrays in an explicitly resolved dtype, so one process may compile
separate fp32 and fp64 variants without changing configuration around a solve.
``auto`` is resolved by the solver or operator that owns the numerical contract.
"""

from enum import StrEnum


class Precision(StrEnum):
    """Runtime precision selected before arrays are constructed or traced."""

    AUTOMATIC = "auto"
    SINGLE = "float32"
    DOUBLE = "float64"


__all__ = ["Precision", "configure_dtypes", "resolve_precision"]

_dtypes_configured = False


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
