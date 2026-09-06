"""Server-side session holder over the forward solve.

The session owns the current equilibrium, the commanded control-point set,
the key map, and a solve callable typed as a protocol, so the production
constrained solve can be swapped for the constrained reduced route without
the app changing.  Each key press steps one control parameter by its stated
signed size, re-solves as a warm start from the previous equilibrium, and
records a receipt row of keyframe wall and trips.  The session also carries
the compiled-program handle the reduced route returns, handing it back on
every later solve so a keyframe chain re-enters one program.  ``frame_push``
reduces the session to the ``ColumnDataSource`` channels the shared poloidal
renderers bound, so a keyframe is pushed by writing those columns verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import numpy as np

from apps.playable.shape import PlasmaShape, keymap

if TYPE_CHECKING:
    from nova.equilibrium.forward import ForwardEquilibrium


class KeyframeReceipt(NamedTuple):
    """One keyframe action and the solve receipt row it produced."""

    key: str
    parameter: str | None
    delta: float
    wall: float  # seconds inside the solve callable
    trips: int  # active-set trips spent by the solve
    reused: bool  # whether the solve re-entered a carried compiled program


class SolveResult(NamedTuple):
    """One warm-started solve outcome against a commanded shape."""

    equilibrium: ForwardEquilibrium | object
    wall: float
    trips: int
    program: object | None = None  # the compiled program a chain re-enters
    reused: bool = False  # whether a carried program was re-entered this solve


@runtime_checkable
class KeyframeSolver(Protocol):
    """Solve one commanded shape warm-started from the previous equilibrium.

    The production implementation runs the constrained reduced route and
    hands its compiled-program handle back so the session carries it and the
    next solve re-enters one program; the constrained Newton-Krylov route
    stays reachable as the reference.  ``action`` is the ``(parameter,
    delta)`` pair the key press named, ``program`` is the handle the session
    carried from the previous keyframe, and the result carries the receipt
    row of wall and trips plus the program to carry on.
    """

    def __call__(
        self,
        previous: ForwardEquilibrium | None,
        commanded: PlasmaShape,
        *,
        action: tuple[str, float] | None = None,
        program: object | None = None,
    ) -> SolveResult: ...


@dataclass
class PlayableSession:
    """Hold the current equilibrium, commanded shape and key map.

    ``machine`` names the carrier the session was built on — the Solov'ev
    default or the MAST frozen-six response carrier selected by a session
    argument — and is recorded on every receipt so a replay can state what
    produced each frame.  ``program`` is the compiled-program handle the
    production solver returns; it is handed back on every later solve so a
    keyframe chain re-enters one program after the first build.
    """

    solver: KeyframeSolver
    shape: PlasmaShape = field(default_factory=PlasmaShape)
    machine: str = "solovev"
    keys: dict[str, tuple[str, float]] = field(default_factory=keymap)
    equilibrium: ForwardEquilibrium | object | None = None
    receipts: list[KeyframeReceipt] = field(default_factory=list)
    wall: np.ndarray | None = None
    #: (radius bounds, height bounds) of the carrier's raster flux image.
    raster_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    program: object | None = None

    def prime(self) -> KeyframeReceipt:
        """Solve the commanded shape from a cold start as the first frame."""
        return self.step(None)

    def step(self, key: str | None) -> KeyframeReceipt:
        """Step one named control, warm re-solve, and record the receipt row.

        A ``None`` key primes the session: the commanded shape is solved as-is
        from the previous equilibrium (or a cold seed) and the frame it names
        is the initial view.  The compiled program carried from the previous
        solve is handed back in and the one the solve returns is stored, so
        the second press onwards re-enters one program.
        """
        if key is None:
            action = None
            commanded = self.shape
        else:
            try:
                action = self.keys[key]
            except KeyError as error:
                raise KeyError(
                    f"unknown key {key!r}; bound keys: {sorted(self.keys)}"
                ) from error
            parameter, delta = action
            commanded = self.shape.apply(parameter, delta)
        result = self.solver(
            self.equilibrium, commanded, action=action, program=self.program
        )
        self.equilibrium = result.equilibrium
        self.program = result.program
        self.shape = commanded
        parameter, delta = (None, 0.0) if action is None else action
        receipt = KeyframeReceipt(
            key=key if key is not None else "prime",
            parameter=parameter,
            delta=delta,
            wall=result.wall,
            trips=result.trips,
            reused=result.reused,
        )
        self.receipts.append(receipt)
        return receipt


def frame_push(session: PlayableSession) -> dict[str, dict[str, np.ndarray]]:
    """Return the keyframe channels reduced to the renderers' bound columns.

    The channel sources mirror :mod:`apps.pulsedesign.poloidal_view`: the
    raster flux image, the separatrix polyline, the commanded control points,
    the topology X-points, the compensating currents per circuit and the
    latest keyframe receipt row.  Every column is shaped exactly as its
    renderer binds it.
    """
    equilibrium = session.equilibrium
    raster = equilibrium.raster_flux
    labelled = equilibrium.labelled_flux

    radius = np.asarray(raster.radius, dtype=float)
    height = np.asarray(raster.height, dtype=float)
    n_radius, n_height = radius.size, height.size
    psi = np.asarray(raster.psi, dtype=float).reshape(n_radius, n_height).T
    # Bokeh draws an image with row zero at the top; the raster rows increase
    # with height, so a flipped array displays the machine upright.
    psi = psi[::-1, :]

    points = session.shape.control_points()

    x_points = np.stack(
        (
            np.asarray(labelled.primary_x_point, dtype=float),
            np.asarray(labelled.secondary_x_point, dtype=float),
        )
    )
    x_points = x_points[np.isfinite(x_points).all(axis=1)]

    receipt = session.receipts[-1]
    return {
        "flux": {"psi": psi},
        "separatrix": {
            "x": np.asarray(raster.separatrix[:, 0], dtype=float),
            "z": np.asarray(raster.separatrix[:, 1], dtype=float),
        },
        "points": {"x": points[0], "z": points[1]},
        "x_points": {"x": x_points[:, 0], "z": x_points[:, 1]},
        "compensation": compensating_currents(equilibrium),
        "receipt": {
            "action": [f"{receipt.key}: {receipt.parameter} {receipt.delta:+.4g}"],
            "wall": [float(receipt.wall)],
            "trips": [int(receipt.trips)],
        },
    }


def compensating_currents(equilibrium: object) -> dict[str, np.ndarray]:
    """Return the per-circuit compensating current the constraint rows drove.

    Every registered row contributes its recorded physical unknown spread
    over its derived circuit direction, so the returned channel is one scalar
    per prescribed circuit (zero where a frame carried no constraint rows).
    """
    records = list(getattr(equilibrium, "constraints", ()) or ())
    directions = [
        np.asarray(getattr(record, "compensator_direction", None)) for record in records
    ]
    physical = [
        np.asarray(getattr(record, "physical_unknown", None)) for record in records
    ]
    circuit_count = 0
    for direction in directions:
        if direction is not None and direction.size:
            circuit_count = max(circuit_count, direction.shape[0])
    current = np.zeros(circuit_count)
    for direction, row_value in zip(directions, physical, strict=False):
        if (
            direction is None
            or row_value is None
            or not direction.size
            or not row_value.size
        ):
            continue
        if direction.ndim == 2 and direction.shape[0] == circuit_count:
            current += direction @ row_value
    return {"circuit": np.arange(circuit_count, dtype=float), "current": current}
