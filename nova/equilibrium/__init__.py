"""Axisymmetric equilibrium solvers.

Classes are named from a two-axis constructor, ``ProblemRepresentation``: the
problem token says what is given and what is sought (forward: sources to flux;
inverse: shape targets to currents; reconstruct: measurements to state) and the
representation token says how the plasma current is carried (profile: a
:math:`j_\\phi(\\psi_N)` ladder; moment: current moments; harmonic: vacuum ring
functions).

All quantities are raw SI. Poloidal flux is the total flux
:math:`\\Phi = 2 \\pi R A_\\phi` in Wb and :math:`\\mu_0` is always written
explicitly.
"""

from typing import TYPE_CHECKING

from nova.equilibrium.diagnostics import (
    DECAY_INDEX_WINDOW,
    decay_index,
    shafranov_vertical_field,
    shafranov_vertical_field_elongated,
)

if TYPE_CHECKING:
    from nova.equilibrium.profile import (
        ProfileDegrees,
        ProfilePrior,
        ProfileResult,
        ReconstructProfile,
    )

_PROFILE_EXPORTS = frozenset(
    {
        "ProfileDegrees",
        "ProfilePrior",
        "ProfileResult",
        "ReconstructProfile",
    }
)

__all__ = [
    "DECAY_INDEX_WINDOW",
    "ProfileDegrees",
    "ProfilePrior",
    "ProfileResult",
    "ReconstructProfile",
    "decay_index",
    "shafranov_vertical_field",
    "shafranov_vertical_field_elongated",
]


def __getattr__(name: str):
    """Load the JAX reconstruction API only when one of its names is requested."""
    if name not in _PROFILE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from nova.equilibrium import profile

    value = getattr(profile, name)
    globals()[name] = value
    return value
