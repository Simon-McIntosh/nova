"""Carrier selection and session construction for the playable app.

The Solov'ev machine is the default document.  A session argument selects the
MAST frozen-six response carrier instead — the shared, digest-verified
response the millisecond route is measured on — so the same app serves one
machine today and the carrier-backed layout when the mast profile is
assembled.
"""

from __future__ import annotations

from pathlib import Path

from apps.playable.production import ForwardMachine, ProductionSolver
from apps.playable.session import PlayableSession
from apps.playable.solovev import build_machine as build_solovev_machine
from apps.playable.shape import PlasmaShape

#: The shared MAST frozen-six response carrier (digest-verified at load).
MAST_FROZEN_SIX_CARRIER = Path(
    "/work/projects/imas_gpu/sophelio/mast_frozen_six_response_carriers"
)

#: Machine names the session argument may select.
AVAILABLE_MACHINES = ("solovev", "mast")


class MachineUnavailable(RuntimeError):
    """The named carrier exists but its forward profile is not assembled here."""


def machine_argument(arguments, default: str = "solovev") -> str:
    """Resolve the ``machine`` session argument from Bokeh request arguments.

    Bokeh passes application arguments as lists of decoded bytestrings;
    ``machine=mast`` on the session URL selects the MAST carrier and the
    default (no argument) stays on the Solov'ev machine.
    """
    values = arguments.get("machine")
    if not values:
        return default
    value = values[0].decode() if isinstance(values[0], bytes) else str(values[0])
    if value not in AVAILABLE_MACHINES:
        raise ValueError(f"unknown machine {value!r}; choose from {AVAILABLE_MACHINES}")
    return value


def mast_carrier_path() -> Path:
    """Return the resolved frozen-six carrier directory, verified present."""
    directory = MAST_FROZEN_SIX_CARRIER
    if not directory.is_dir():
        raise FileNotFoundError(
            f"the MAST frozen-six response carrier is not reachable at {directory}"
        )
    return directory


def build_machine(machine: str) -> ForwardMachine:
    """Return the forward machine the named carrier supports."""
    if machine == "solovev":
        return build_solovev_machine()
    if machine == "mast":
        carrier = mast_carrier_path()
        raise MachineUnavailable(
            f"the MAST frozen-six carrier is reachable at {carrier}, but "
            "assembling its forward profile from the carrier is the "
            "carrier-backed receipt node's work, not the playable session's"
        )
    raise ValueError(f"unknown machine {machine!r}; choose from {AVAILABLE_MACHINES}")


def build_session(
    machine: str = "solovev", *, shape: PlasmaShape | None = None
) -> PlayableSession:
    """Build a playable session over the machine the argument selects."""
    carrier = build_machine(machine)
    radius, height, _shape = carrier.profile.operator.raster_geometry()
    return PlayableSession(
        solver=ProductionSolver(carrier),
        shape=shape if shape is not None else PlasmaShape(),
        machine=machine,
        wall=carrier.wall,
        raster_bounds=(
            (float(radius[0]), float(radius[-1])),
            (float(height[0]), float(height[-1])),
        ),
    )
