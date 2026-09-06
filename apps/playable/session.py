"""Server-side session holder over the forward solve.

The session owns the current equilibrium, the commanded control-point set,
the key map, and a solve callable typed as a protocol, so the production
constrained solve can be swapped for the constrained reduced route without
the app changing.  Each key press steps one control parameter by its stated
signed size, re-solves as a warm start from the previous equilibrium, and
records a receipt row of keyframe wall and trips.  The session also carries
the compiled-program handle the reduced route returns, handing it back on
every later solve so a keyframe chain re-enters one program.  The session also
carries the camera :class:`~apps.playable.camera.FrameDecoder` loaded once per
session; ``decode_frame`` runs after the poloidal push so a slow decode delays
only the picture, and records the decode wall and decoder identity beside each
frame in ``decoded_frames``.  ``frame_push`` reduces the session to the
``ColumnDataSource`` channels the shared poloidal renderers bound, so a
keyframe is pushed by writing those columns verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import numpy as np

from apps.playable.camera import DecodedFrame, FrameDecoder
from apps.playable.shape import PlasmaShape, keymap

if TYPE_CHECKING:
    from nova.equilibrium.forward import ForwardEquilibrium
    from nova.equilibrium.steering_frames import SteeringFrame


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
    #: The camera decoder loaded once per session (the placeholder by
    #: default); ``decode_frame`` calls it after each poloidal push.
    decoder: FrameDecoder | None = None
    #: One decode record per decoded keyframe, in step order, carrying the
    #: decode wall and the decoder identity beside the receipts.
    decoded_frames: list[DecodedFrame] = field(default_factory=list)
    #: Whether the record and playback strip is currently recording.
    recording: bool = False

    @property
    def frame_index(self) -> int:
        """Return the current keyframe index (the prime is frame one)."""
        return len(self.receipts)

    def current_frame(self) -> SteeringFrame:
        """Return the current equilibrium as the typed steering frame."""
        return steering_frame(self)

    def decode_frame(self) -> DecodedFrame | None:
        """Decode the current frame through the session decoder and record it.

        The app calls this after the poloidal push so a slow decode delays
        only the picture; the returned record carries the decode wall and the
        decoder identity and is appended beside the keyframe receipts, one
        record per decoded frame.
        """
        if self.decoder is None:
            return None
        decoded = self.decoder.decode(self.current_frame())
        self.decoded_frames.append(decoded)
        return decoded

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


def frame_push(
    session: PlayableSession, *, equilibrium: object | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Return the keyframe channels reduced to the renderers' bound columns.

    The channel sources mirror :mod:`apps.pulsedesign.poloidal_view`: the
    raster flux image, the separatrix polyline, the commanded control points,
    the topology X-points, the compensating currents per circuit and the
    latest keyframe receipt row.  Every column is shaped exactly as its
    renderer binds it.

    The view binds the lattice columns only - the raster image, the two
    polyline projections of the separatrix and the scalar marks - so the push
    itself is a device-to-host projection that costs milliseconds regardless
    of how the raster was produced.  ``equilibrium`` defaults to the
    session's current one; a caller feeding the solve result's on-device
    raster (the light-receipt path) passes it here and the same columns come
    out.
    """
    equilibrium = session.equilibrium if equilibrium is None else equilibrium
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


def steering_frame(session: PlayableSession) -> SteeringFrame:
    """Return the current equilibrium as a typed steering frame for the decode.

    The frame is assembled, never computed: the raster channels come from the
    current raster flux image, the labelled points from the consumer map, the
    per-row compensation from the solve's constraint records, and the action,
    keyframe wall and trip count from the latest receipt row.  Absent topology
    slots stay absent (NaN coordinates, False finite-mask flags) per the frame
    schema's no-imputation rule, and a carrier that publishes more fills the
    corresponding fields.  The finished session record of the frame-schema
    followup replaces this seam's policy and version fields when it lands.
    """
    from nova.equilibrium.solve_request import ForwardSolvePolicy
    from nova.equilibrium.steering_frames import (
        SteeringAction,
        SteeringFrame,
        policy_digest,
    )

    equilibrium = session.equilibrium
    raster = getattr(equilibrium, "raster_flux", None)
    labelled = getattr(equilibrium, "labelled_flux", None)

    def channel(name: str, default):
        return np.asarray(getattr(raster, name, default), dtype=np.float64)

    radius = channel("radius", np.linspace(0.6, 1.42, 12))
    height = channel("height", np.linspace(-0.42, 0.42, 10))
    n_radius, n_height = radius.size, height.size
    psi = channel("psi", np.zeros(n_radius * n_height)).reshape(n_radius, n_height)
    psi_norm = channel("psi_norm", np.linspace(0.0, 1.0, n_radius * n_height)).reshape(
        n_radius, n_height
    )
    domain = channel("domain_label", np.zeros(n_radius * n_height))
    shape = np.asarray([n_radius, n_height], dtype=np.int32)

    def slot(value, size: int) -> np.ndarray:
        if value is None:
            return np.full(size, np.nan)
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return np.full(size, np.nan)
        if array.size < size:
            array = np.append(array, np.full(size - array.size, np.nan))
        return array[:size]

    axis = slot(getattr(labelled, "o_point", None), 2)
    primary = slot(getattr(labelled, "primary_x_point", None), 2)
    secondary = slot(getattr(labelled, "secondary_x_point", None), 2)
    raw_strike = getattr(labelled, "strike_points", None)
    strike = np.full((2, 2), np.nan)
    if raw_strike is not None and np.asarray(raw_strike).size >= 4:
        strike = np.asarray(raw_strike, dtype=np.float64).reshape(2, 2)[:2]
    lcfs = np.asarray(
        getattr(labelled, "lcfs", np.full((1, 2), np.nan)), dtype=np.float64
    ).reshape(-1, 2)
    boundary_count = int(
        getattr(
            labelled,
            "lcfs_vertex_count",
            np.count_nonzero(np.isfinite(lcfs).all(axis=1)),
        )
    )

    def present(point) -> bool:
        """Return whether one coordinate pair is fully present."""
        return bool(np.isfinite(np.asarray(point, dtype=float)).all())

    finite_mask = np.asarray(
        [
            present(axis),
            present(primary),
            present(secondary),
            present(strike[0]),
            present(strike[1]),
            int(boundary_count) > 0,
        ],
        dtype=bool,
    )

    rows = [
        np.ravel(np.asarray(record.physical_unknown, dtype=np.float64))
        for record in getattr(equilibrium, "constraints", ())
        if getattr(record, "physical_unknown", None) is not None
    ]
    compensation = np.concatenate(rows) if rows else np.empty((0,), dtype=np.float64)

    receipt = session.receipts[-1] if session.receipts else None
    action = SteeringAction(
        name=(None if receipt is None else receipt.parameter) or "prime",
        delta=0.0 if receipt is None else float(receipt.delta),
        commanded_control_points=np.asarray(
            session.shape.control_points(), dtype=np.float64
        ).T,
    )

    import nova

    return SteeringFrame(
        radius=radius,
        height=height,
        shape=shape,
        psi=psi,
        psi_norm=psi_norm,
        domain_label=domain.astype(np.int8),
        separatrix=np.asarray(
            getattr(raster, "separatrix", np.full((30, 2), np.nan)),
            dtype=np.float64,
        ).reshape(-1, 2),
        separatrix_vertex_count=np.int32(getattr(raster, "separatrix_vertex_count", 0)),
        magnetic_axis_r=axis[0],
        magnetic_axis_z=axis[1],
        x_point_r=np.stack((primary[0], secondary[0])),
        x_point_z=np.stack((primary[1], secondary[1])),
        strike_points_r=strike[:, 0],
        strike_points_z=strike[:, 1],
        lcfs_r=lcfs[:, 0],
        lcfs_z=lcfs[:, 1],
        n_boundary_coords=np.int32(boundary_count),
        finite_mask=finite_mask,
        coil_current=np.empty((0,), dtype=np.float64),
        compensating_current=compensation,
        action=action,
        wall_seconds=0.0 if receipt is None else float(receipt.wall),
        trip_count=0 if receipt is None else int(receipt.trips),
        carrier_identity=session.machine,
        nova_version=nova.__version__,
        policy_digest=policy_digest(ForwardSolvePolicy()),
    )


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
