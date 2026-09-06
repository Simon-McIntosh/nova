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
frame in ``decoded_frames``.  One ``SteeringFrame`` is assembled after each
solve and shared by rendering, decoding and optional recording. ``frame_push``
only reduces that frame to the ``ColumnDataSource`` channels the shared
poloidal renderers bind.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import numpy as np

from apps.playable.camera import DecodedFrame, FrameDecoder
from apps.playable.shape import PlasmaShape, keymap

if TYPE_CHECKING:
    from nova.equilibrium.forward import ForwardEquilibrium
    from nova.equilibrium.solve_request import ForwardSolveReceipt
    from nova.equilibrium.steering_frames import SteeringFrame


class KeyframeReceipt(NamedTuple):
    """One keyframe action and the solve receipt row it produced."""

    key: str
    parameter: str | None
    delta: float
    wall: float  # seconds inside the solve callable
    trips: int  # active-set trips spent by the solve
    reused: bool  # whether the solve re-entered a carried compiled program
    frame_assembly_wall: float = 0.0  # seconds spent assembling SteeringFrame


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


@runtime_checkable
class FrameBuilder(Protocol):
    """Wrap an equilibrium in the typed receipt ``assemble_frame`` consumes."""

    def __call__(
        self, equilibrium: ForwardEquilibrium | object, *, wall_seconds: float
    ) -> ForwardSolveReceipt: ...


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
    #: Adapter for solve routes that return an equilibrium rather than a
    #: ``ForwardSolveReceipt``.  The session still calls ``assemble_frame``;
    #: this adapter only supplies its typed receipt input.
    frame_builder: FrameBuilder | None = None
    #: The most recently assembled frame.  Rendering and decoding share this
    #: exact object so neither can re-derive a second view of the solve.
    frame: SteeringFrame | None = None
    #: Frames admitted while recording is enabled, ready for ``write_session``.
    recorded_frames: list[SteeringFrame] = field(default_factory=list)
    #: Per-keyframe assembly timings and the input branch used, retained as
    #: measurable session evidence.
    frame_assembly_walls: list[float] = field(default_factory=list)
    frame_assembly_routes: list[str] = field(default_factory=list)

    @property
    def frame_index(self) -> int:
        """Return the current keyframe index (the prime is frame one)."""
        return len(self.receipts)

    def current_frame(self) -> SteeringFrame:
        """Return the frame assembled for the current solved keyframe."""
        if self.frame is None:
            raise RuntimeError("the session has no solved steering frame")
        return self.frame

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
        from nova.equilibrium.solve_request import ForwardSolveReceipt
        from nova.equilibrium.steering_frames import SteeringAction, assemble_frame

        if isinstance(result.equilibrium, ForwardSolveReceipt):
            solve_receipt = result.equilibrium
            equilibrium = solve_receipt.terminal_state
            assembly_route = "receipt"
        else:
            equilibrium = result.equilibrium
            if self.frame_builder is None:
                solve_receipt = equilibrium_frame_receipt(
                    equilibrium,
                    wall_seconds=float(result.wall),
                    route=getattr(self.solver, "route", None),
                )
            else:
                solve_receipt = self.frame_builder(
                    equilibrium, wall_seconds=float(result.wall)
                )
            if not isinstance(solve_receipt, ForwardSolveReceipt):
                raise TypeError("a frame builder must return a ForwardSolveReceipt")
            assembly_route = "frame-builder"

        parameter, delta = (None, 0.0) if action is None else action
        steering_action = SteeringAction(
            name=parameter or "prime",
            delta=float(delta),
            commanded_control_points=np.asarray(
                commanded.control_points(), dtype=np.float64
            ).T,
        )
        assembly_started = perf_counter()
        frame = assemble_frame(
            solve_receipt,
            action=steering_action,
            **frame_assembly_inputs(self, solve_receipt),
        )
        assembly_wall = perf_counter() - assembly_started
        receipt = KeyframeReceipt(
            key=key if key is not None else "prime",
            parameter=parameter,
            delta=delta,
            wall=float(frame.wall_seconds),
            trips=int(frame.trip_count),
            reused=result.reused,
            frame_assembly_wall=assembly_wall,
        )
        self.equilibrium = equilibrium
        self.program = result.program
        self.shape = commanded
        self.frame = frame
        self.receipts.append(receipt)
        self.frame_assembly_walls.append(assembly_wall)
        self.frame_assembly_routes.append(assembly_route)
        if self.recording:
            self.recorded_frames.append(frame)
        return receipt

    def write_recording(
        self,
        *,
        filename: str,
        dirname: str,
        include_raster: bool = True,
    ):
        """Persist the frames admitted while recording was enabled."""
        from nova.equilibrium.steering_frames import write_session

        if not self.recorded_frames:
            raise ValueError("the session has no recorded steering frames")
        return write_session(
            self.recorded_frames,
            filename=filename,
            dirname=dirname,
            include_raster=include_raster,
        )


def frame_push(
    session: PlayableSession, *, frame: SteeringFrame | None = None
) -> dict[str, dict[str, np.ndarray | list]]:
    """Return the keyframe channels reduced to the renderers' bound columns.

    Every dynamic field except the clipped plasma-cell polygons is reduced
    from the one :class:`SteeringFrame` shared with the decoder and recorder.
    The caller adds the plasma polygons from the operator's solved clipped
    geometry, because those polygons are deliberately not part of the
    machine-independent frame contract.
    """
    frame = session.current_frame() if frame is None else frame
    points = np.asarray(frame.action.commanded_control_points, dtype=float).T
    x_points = np.column_stack((frame.x_point_r, frame.x_point_z))
    primary = x_points[:1]
    primary = primary[np.isfinite(primary).all(axis=1)]
    secondary = x_points[1:]
    secondary = secondary[np.isfinite(secondary).all(axis=1)]
    strikes = np.column_stack((frame.strike_points_r, frame.strike_points_z))
    strikes = strikes[np.isfinite(strikes).all(axis=1)]
    secondary = np.concatenate((secondary, strikes), axis=0)
    axis = np.asarray([[frame.magnetic_axis_r, frame.magnetic_axis_z]], dtype=float)
    axis = axis[np.isfinite(axis).all(axis=1)]

    separatrix_count = int(np.asarray(frame.separatrix_vertex_count))
    separatrix = np.asarray(frame.separatrix, dtype=float)[:separatrix_count]
    surfaces = np.stack((frame.flux_surface_r, frame.flux_surface_z), axis=-1)
    contour_lines = []
    for loop in np.asarray(surfaces[1:], dtype=float):
        if not np.isfinite(loop).all():
            continue
        if loop.size and not np.allclose(loop[0], loop[-1]):
            loop = np.vstack((loop, loop[:1]))
        contour_lines.append(loop)
    compensation = np.asarray(frame.compensating_current, dtype=float).reshape(-1)
    return {
        "separatrix": {
            "x": separatrix[:, 0],
            "z": separatrix[:, 1],
        },
        "points": {"x": points[0], "z": points[1]},
        "levelset": {
            "x": [loop[:, 0].tolist() for loop in contour_lines],
            "z": [loop[:, 1].tolist() for loop in contour_lines],
        },
        "o_points": {"x": axis[:, 0], "z": axis[:, 1]},
        "x_points": {"x": primary[:, 0], "z": primary[:, 1]},
        "x_points_secondary": {
            "x": secondary[:, 0],
            "z": secondary[:, 1],
        },
        "compensation": {
            "circuit": np.arange(compensation.size, dtype=float),
            "current": compensation,
        },
        "receipt": {
            "action": [f"{frame.action.name} {frame.action.delta:+.4g}"],
            "wall": [float(frame.wall_seconds)],
            "trips": [int(frame.trip_count)],
        },
    }


def equilibrium_frame_receipt(
    equilibrium: ForwardEquilibrium | object,
    *,
    wall_seconds: float,
    route: str | None = None,
) -> ForwardSolveReceipt:
    """Wrap an equilibrium returned by an untyped route in a solve receipt."""
    from nova.equilibrium.solve_request import (
        ForwardSolvePolicy,
        ForwardSolveReceipt,
        ResolvedForwardSolveDefaults,
    )

    history = equilibrium.fixed_point
    residual_history = getattr(history, "active_set_residuals", None)
    if residual_history is None:
        residual_history = history.trace
    mask_history = getattr(history, "active_set_mask_differences", None)
    if mask_history is None:
        mask_history = history.shadow_mask_changes
    normalisation = getattr(equilibrium, "normalisation", None)
    amplitude = getattr(normalisation, "amplitude", None)
    policy = ForwardSolvePolicy()
    if route is not None:
        policy = replace(policy, route=route)
    return ForwardSolveReceipt(
        terminal_state=equilibrium,
        qualified=bool(getattr(getattr(equilibrium, "finite", None), "passed", True)),
        termination_reason=history.termination_reason,
        residual_history=residual_history,
        mask_history=mask_history,
        globalisation_decisions=(
            history.inner_iteration_decisions,
            history.inner_iteration_applied_factors,
        ),
        amplitude_history=(
            np.empty((0,), dtype=float)
            if amplitude is None
            else np.atleast_1d(np.asarray(amplitude))
        ),
        topology_read=getattr(equilibrium, "topology", None),
        polish_receipt=None,
        compilation_cache_hit=False,
        wall_seconds=float(wall_seconds),
        resolved_defaults=ResolvedForwardSolveDefaults.from_policy(policy),
    )


def frame_assembly_inputs(
    session: PlayableSession, receipt: ForwardSolveReceipt
) -> dict[str, object]:
    """Return contextual inputs needed to assemble the session's solved frame."""
    equilibrium = receipt.terminal_state
    solver = session.solver
    machine = getattr(solver, "machine", None)
    profile = getattr(machine, "profile", None)
    applied_current = getattr(solver, "prescribed_current", None)
    if applied_current is None:
        applied_current = getattr(equilibrium, "coil_current", np.empty((0,)))

    psi_norm = np.linspace(0.0, 1.0, 65)
    internal_geometry = getattr(equilibrium, "internal_geometry", None)
    if profile is None:
        p_prime = np.zeros_like(psi_norm)
        ff_prime = np.zeros_like(psi_norm)
    else:
        p_prime = np.asarray(profile.source.core.p_prime(psi_norm), dtype=float)
        ff_prime = np.asarray(profile.source.core.ff_prime(psi_norm), dtype=float)
        if internal_geometry is None:
            from nova.equilibrium.flux_surface_geometry import (
                FluxSurfaceGeometry,
                source_field_function,
            )

            topology = equilibrium.topology
            axis = np.asarray(topology.axis, dtype=float)
            internal_geometry = FluxSurfaceGeometry.internal_geometry(
                profile.lattice,
                np.asarray(equilibrium.flux),
                source_field_function(profile.source, float(topology.flux_span)),
                axis=(float(axis[0]), float(axis[1])),
                boundary_flux=float(topology.boundary_flux),
                n_surface=11,
                n_theta=64,
                n_rho=25,
                diverted=bool(getattr(topology, "diverted", False)),
            )

    return {
        "carrier_identity": session.machine,
        "applied_current": np.asarray(applied_current, dtype=float),
        "p_prime_psi_norm": psi_norm,
        "p_prime": p_prime,
        "ff_prime_psi_norm": psi_norm,
        "ff_prime": ff_prime,
        "p_prime_source": str(getattr(equilibrium, "p_prime_source", "efm")),
        "reference_centroid_z": getattr(equilibrium, "reference_centroid_z", None),
        "internal_geometry": internal_geometry,
        "wall": session.wall,
    }
