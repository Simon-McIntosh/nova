"""Evidence one centroid-constrained MAST fixed-point solve and its flux panels.

The converged MAST 22086/43 frame is solved through the compiled-slice reduced
fixed point with the vertical current-centroid row registered: once with no
program in hand, which builds the fixed-shape program, and then re-entering
that same program on two-percent current edits so the receipt carries the
program-building wall beside the warm re-entry wall measured on real trips.
A warm wall taken by re-solving the seed against itself would time a
trivially-satisfied solve, so every warm point moves the current.

The run then persists exactly what a poloidal panel pair needs -- the external
conductor-only flux on the solve lattice, the solved total flux, the control
cells the moment integration actually uses, their domain labels, the wall units
and the read's landmarks -- and ``panels`` draws the pair from that file, so the
figure can be redrawn without a second allocation.

Two facts about the cell representation are recorded rather than assumed,
because a panel that implied otherwise would overstate the mesh: the forward
lattice carries one uniform rectangular control cell per node, and no cell is
cut by the wall or by the separatrix.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import coil_edit_latency as edit
from benchmarks import mast_response_carrier_warm as response_carrier
from nova.catalog.mast_geometry import shaped_section_vertices
from nova.equilibrium.forward import _lattice_cells
from nova.equilibrium.separatrix_branches import assemble_separatrix_branches
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "docs/figures/mast-centroid-capability"
DEFAULT_STATE = FIGURE_ROOT / "panel-fields.npz"
DEFAULT_RECEIPT = FIGURE_ROOT / "centroid-solve.json"
DEFAULT_FIGURE = FIGURE_ROOT / "vacuum-and-plasma-panels.png"
#: Warm points move the current so the timed solve does real trips.
WARM_EDIT_FRACTIONS = (0.02, 0.04, 0.06)
#: Shared level count for the two panels; one physical array serves both.
LEVEL_COUNT = 24
#: Translucent enough that the contours read through the filled cells.
CELL_ALPHA = 0.55
#: The boundary is drawn in the contour ink, only heavier.
BOUNDARY_LINEWIDTH = 2.6 * 0.35
BOUNDARY_SAMPLES = 12
#: Coils are context, not subject: a faint outline the contours read through.
COIL_EDGE_COLOR = "#b8b8b8"
COIL_LINEWIDTH = 0.45
#: p-prime scales by 1+offset and ff-prime by 1-offset, so the pinned net
#: plasma current is unchanged and only its radial distribution moves.
RATIO_OFFSETS = (-0.30, -0.15, 0.0, 0.15, 0.30)
#: ff-prime alone, over a wide amplitude range; p-prime is held and the solve
#: pins the net plasma current, so the current normalisation is the lambda.
FIELD_FUNCTION_SCALES = (0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60)
#: Internal shape witnesses. 3/2 is requested first; on frames whose q minimum
#: sits above it there is no such surface, and 2/1 is the next one that exists.
RATIONAL_ORDERS = (1.5, 2.0)
#: Linestyle per rational order, so two internal surfaces stay distinguishable.
RATIONAL_STYLES = {"1.5": (0, (1, 1.6)), "2": (0, (5, 2, 1, 2))}
#: Drawn plasma tessellation; the wall fit decides the delivered count.
DEFAULT_HEX_CELLS = 4000
CPU_PROVENANCE_MARKER = "NOVA_CENTROID_PANELS_CPU_PROVENANCE"


def _source_revision() -> str:
    """Return the commit the measurement ran from."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _provenance() -> dict[str, Any]:
    """Return the host, scheduler and backend whatever ran is measured on."""
    device = jax.devices()[0]
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "platform": device.platform,
        "device_kind": device.device_kind,
        "devices": [str(item) for item in jax.devices()],
        "tmpdir": os.environ.get("TMPDIR"),
        "cpu_provenance_marker": os.environ.get(CPU_PROVENANCE_MARKER) or None,
        "scheduler": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "memory": os.environ.get("SLURM_MEM_PER_NODE"),
        },
    }


def _require_host() -> None:
    """Require TMPDIR in the job body, and a named reason off the GPU."""
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR=/tmp must be set in the job body")
    device = jax.devices()[0]
    if (
        device.platform != "gpu"
        and not os.environ.get(CPU_PROVENANCE_MARKER, "").strip()
    ):
        raise RuntimeError(
            "a run off the GPU must name its reason in "
            f"{CPU_PROVENANCE_MARKER}; the receipt carries it"
        )


def _coil_outlines(shot: int) -> np.ndarray:
    """Return every stored winding-pack section of one shot as a quadrilateral.

    The outlines are the machine the conductor-only panel is the field of, so
    they come from the same stored geometry the response was built from rather
    than from a drawing table.
    """
    import zarr

    from nova.imas.mast_solve_inputs import SHOT_STORE

    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    fields = {
        name: np.asarray(group[f"fcoil_{name}"], dtype=np.float64).reshape(-1)
        for name in ("r", "z", "width", "height", "ang1", "ang2")
    }
    return np.asarray(
        [
            shaped_section_vertices(
                fields["r"][index],
                fields["z"][index],
                fields["width"][index],
                fields["height"][index],
                fields["ang1"][index],
                fields["ang2"][index],
            )
            for index in range(fields["r"].size)
        ],
        dtype=float,
    )


def _branches(profile: Any, state: Any, topology: Any) -> dict[str, np.ndarray]:
    """Assemble the solved field's own level set at the boundary flux.

    The assembly runs on the SOLUTION'S lattice spline, not on a receiver
    raster: a raster is a resampled image whose contour wanders between nodes,
    and the raw level set of either grid is unsplit and unbounded. Splitting it
    at the polished saddle is what leaves one axis-enclosing cycle to draw and
    to cut the plasma mesh against.
    """
    lattice = profile.lattice
    shape = tuple(int(value) for value in np.asarray(lattice.shape))
    values = np.asarray(state[: lattice.node_count], dtype=float).reshape(shape)
    assembled = jax.device_get(
        assemble_separatrix_branches(
            jnp.asarray(values.T),
            jnp.asarray(np.asarray(lattice.radius, dtype=float)),
            jnp.asarray(np.asarray(lattice.height, dtype=float)),
            jnp.asarray(topology.boundary_flux),
            jnp.asarray(np.asarray(topology.axis, dtype=float)),
        )
    )
    return {
        "closed_controls_rz": np.asarray(assembled["closed_controls_rz"], float),
        "closed_valid": np.asarray(assembled["closed_valid"], bool),
        "open_controls_rz": np.asarray(assembled["open_controls_rz"], float),
        "open_valid": np.asarray(assembled["open_valid"], bool),
        "open_branch_valid": np.asarray(assembled["open_branch_valid"], bool),
        "well_formed": np.asarray(bool(np.asarray(assembled["well_formed"]))),
    }


def _rational_surfaces(
    lattice: Any,
    source: Any,
    flux: np.ndarray,
    topology: Any,
    orders: tuple[float, ...],
) -> dict[str, Any]:
    """Return each requested rational surface, or why it is absent.

    The safety factor is read from this member's own flux-surface-averaged
    geometry and its own diamagnetic gradient, so a surface moves with the
    profile rather than being carried over from a neighbour. The OUTERMOST
    crossing is taken, because q rises towards the boundary on these frames.

    Absence is reported with the q range that produced it rather than as a
    bare None. A frame whose q minimum sits above the requested order simply
    has no such surface, and that is a fact about the equilibrium the reader
    needs; swallowing it leaves a blank curve with no explanation.
    """
    from nova.equilibrium.flux_surface_geometry import (
        FluxSurfaceGeometry,
        source_field_function,
    )

    axis = np.asarray(topology.axis, dtype=float).reshape(-1)[:2]
    axis_flux = float(np.asarray(topology.axis_flux))
    boundary_flux = float(np.asarray(topology.boundary_flux))
    try:
        geometry = FluxSurfaceGeometry.from_flux_map(
            lattice,
            flux,
            source_field_function(source, float(np.asarray(topology.flux_span))),
            axis=(float(axis[0]), float(axis[1])),
            boundary_flux=boundary_flux,
        )
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        return {"status": f"{type(error).__name__}: {error}", "surfaces": {}}
    label = np.asarray(geometry.psi_norm, dtype=float)
    factor = np.abs(np.asarray(geometry.safety_factor, dtype=float))
    finite = np.isfinite(label) & np.isfinite(factor)
    label, factor = label[finite], factor[finite]
    if label.size < 2:
        return {"status": "no finite safety factor", "surfaces": {}}
    found: dict[str, Any] = {}
    for order in orders:
        crossings = np.flatnonzero((factor[:-1] - order) * (factor[1:] - order) < 0.0)
        if crossings.size == 0:
            found[f"{order:g}"] = None
            continue
        index = int(crossings[-1])
        span = factor[index + 1] - factor[index]
        weight = 0.0 if span == 0.0 else (order - factor[index]) / span
        psi_norm = float(label[index] + weight * (label[index + 1] - label[index]))
        found[f"{order:g}"] = {
            "psi_norm": psi_norm,
            "flux": axis_flux + psi_norm * (boundary_flux - axis_flux),
        }
    return {
        "status": "read",
        "safety_factor_minimum": float(factor.min()),
        "safety_factor_maximum": float(factor.max()),
        "surfaces": found,
    }


def _surface_labels(rational: dict[str, Any]) -> str:
    """Return one compact log line of which rational surfaces were found."""
    found = rational.get("surfaces", {})
    return ",".join(
        f"{name}:{'absent' if entry is None else round(entry['psi_norm'], 4)}"
        for name, entry in sorted(found.items())
    )


def _closed_loop(
    lattice: Any, flux: np.ndarray, level: float, axis: np.ndarray
) -> np.ndarray:
    """Return the axis-enclosing closed loop of one level of the solved spline."""
    shape = tuple(int(value) for value in np.asarray(lattice.shape))
    values = np.asarray(flux, dtype=float).reshape(shape)
    assembled = jax.device_get(
        assemble_separatrix_branches(
            jnp.asarray(values.T),
            jnp.asarray(np.asarray(lattice.radius, dtype=float)),
            jnp.asarray(np.asarray(lattice.height, dtype=float)),
            jnp.asarray(float(level)),
            jnp.asarray(np.asarray(axis, dtype=float).reshape(-1)[:2]),
        )
    )
    return poloidal.sample_cubic_controls(
        np.asarray(assembled["closed_controls_rz"], float),
        np.asarray(assembled["closed_valid"], bool),
        BOUNDARY_SAMPLES,
    )


def _class_label(achieved: dict[str, Any]) -> str:
    """Return the one-word topology the read derived, or its refusal."""
    if achieved.get("read_status") != "qualified":
        return str(achieved.get("read_status", "unread"))
    return str(achieved.get("class", "unread"))


def _stored_profiles(shot: int, row: int) -> dict[str, np.ndarray]:
    """Return the stored flux-function tables and their boundary primitives.

    The case builds its two flux functions as closures over these tables, so
    the tables are not reachable from the built source; they are re-read from
    the same store, converted by the same expressions the case uses, and the
    varied source is then assembled through the ordinary constructors rather
    than by reaching inside a closure.
    """
    import zarr

    from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
    from nova.imas.mast_solve_inputs import SHOT_STORE

    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    psi_norm = np.asarray(group["psi_norm"], dtype=np.float64)
    return {
        "psi_norm": psi_norm,
        "p_prime": -np.asarray(group["pprime"][row], dtype=np.float64)
        / TOTAL_FLUX_FACTOR,
        "ff_prime": -np.asarray(group["ffprime"][row], dtype=np.float64)
        / TOTAL_FLUX_FACTOR,
        "boundary_pressure": float(group["ppsi_c"][row, -1]),
        "boundary_field_function": float(group["fpsi_c"][row, -1]),
    }


def _sampled_factory(nodes: np.ndarray, values: np.ndarray) -> Any:
    """Return the array-owned equivalent of the case's interpolating closure.

    The default representation closes the evaluator over its tables, so a
    changed table is a changed program by construction. This one carries the
    coordinate and value tables as leaves, so a series that moves the tables
    shares one compiled program. The end caps differ in form -- a quintic cap
    here against the case's slope-matched cubic -- so a run on this
    representation is not a bit reproduction of the reference case, and the
    receipt says so rather than implying one.
    """
    from nova.equilibrium.solve_request import SampledFluxFunction

    return SampledFluxFunction(np.asarray(nodes, float), np.asarray(values, float))


def _varied_source(
    stored: dict[str, np.ndarray], pressure_scale: float, field_scale: float
) -> Any:
    """Return a source with the two flux-function tables rescaled.

    The solve pins the net plasma current, so scaling the pair in OPPOSITE
    directions leaves the current where it was and changes only how it is
    distributed: p-prime enters the toroidal density weighted by R and
    ff-prime weighted by 1/R, so their ratio is the radial shape lever.

    The tables are carried as array leaves, so every member of the series
    shares one compiled program and the variation costs one solve rather than
    one build.
    """
    from nova.equilibrium.source import DomainProfile, ForwardSource

    return ForwardSource(
        core=DomainProfile(
            p_prime=_sampled_factory(
                stored["psi_norm"], pressure_scale * stored["p_prime"]
            ),
            ff_prime=_sampled_factory(
                stored["psi_norm"], field_scale * stored["ff_prime"]
            ),
        ),
        boundary_pressure=stored["boundary_pressure"],
        boundary_field_function=stored["boundary_field_function"],
    )


def _timed_solve(
    profile: Any,
    state: jax.Array,
    current: jax.Array,
    requested: jax.Array,
    target_current: float,
    program: Any,
    pairs: tuple[Any, ...],
) -> tuple[Any, float]:
    """Return one compiled-slice constrained solve and its drained wall."""
    started = time.perf_counter()
    result = edit._compiled_edit(
        profile,
        state,
        current,
        requested,
        target_current,
        program,
        constraint_pairs=pairs,
    )
    jax.block_until_ready(result.state)
    return result, time.perf_counter() - started


def _solve_record(result: Any, wall_s: float) -> dict[str, Any]:
    """Return the terminal facts one solve is judged on."""
    return {
        "wall_seconds": float(wall_s),
        "wall_milliseconds": float(wall_s * 1.0e3),
        "terminal_residual": float(np.asarray(result.terminal_residual)),
        "converged": bool(np.asarray(result.converged)),
        "termination": result.termination_name,
        "trip_count": int(np.asarray(result.active_set_iterations)),
    }


def _wall_arrays(units: tuple[Any, ...]) -> dict[str, np.ndarray]:
    """Flatten the wall units with their offsets, closure and kind."""
    radius: list[np.ndarray] = []
    height: list[np.ndarray] = []
    offsets = [0]
    for unit in units:
        radius.append(np.asarray(unit.r, dtype=float))
        height.append(np.asarray(unit.z, dtype=float))
        offsets.append(offsets[-1] + radius[-1].size)
    return {
        "wall_radius": np.concatenate(radius),
        "wall_height": np.concatenate(height),
        "wall_offsets": np.asarray(offsets, dtype=int),
        "wall_closed": np.asarray([bool(unit.closed) for unit in units]),
        "wall_kinds": np.asarray([str(unit.kind) for unit in units]),
    }


def _restore_units(data: Any) -> tuple[Any, ...]:
    """Rebuild the typed wall units the panel draws."""
    from nova.equilibrium.wall_mask import WallUnit

    offsets = np.asarray(data["wall_offsets"], dtype=int)
    radius = np.asarray(data["wall_radius"], dtype=float)
    height = np.asarray(data["wall_height"], dtype=float)
    closed = np.asarray(data["wall_closed"], dtype=bool)
    kinds = [str(value) for value in np.asarray(data["wall_kinds"])]
    return tuple(
        WallUnit(
            radius[start:stop],
            height[start:stop],
            kind=kinds[index],
            closed=bool(closed[index]),
        )
        for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:]))
    )


def measure(
    state_path: Path,
    receipt_path: Path,
    carrier_path: Path,
    grid_points: int | None = None,
) -> None:
    """Solve the centroid-constrained frame and persist the panel fields."""
    configure_dtypes()
    if jax.config.jax_enable_x64 is not True:
        raise RuntimeError("extended precision did not take before any array was built")
    _require_host()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    total_started = time.perf_counter()

    profile, prepared, carrier = edit._prepare_case(carrier_path, grid_points)
    operator = profile.operator
    lattice = profile.lattice
    node_count = int(operator.grid.node_number)
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    target_current = float(prepared["target_current"])
    base_current = np.asarray(prepared["prescribed_current"], dtype=np.float64)
    circuit_index = int(prepared["circuit_index"])
    pairs = (prepared["vertical_centroid_pair"],)
    seed = prepared["initial"]

    # The program-building solve: no program in hand, so its wall carries the
    # trace and the compile alongside one execution.
    cold, cold_wall = _timed_solve(
        profile,
        seed,
        jnp.asarray(base_current),
        requested,
        target_current,
        None,
        pairs,
    )
    print(
        f"COLD wall_s={cold_wall:.6f} residual={float(cold.terminal_residual):.3e}",
        flush=True,
    )

    warm: list[dict[str, Any]] = []
    program = cold.program
    state = cold.state
    for fraction in WARM_EDIT_FRACTIONS:
        values = base_current.copy()
        values[circuit_index] *= 1.0 + fraction
        result, wall = _timed_solve(
            profile,
            state,
            jnp.asarray(values),
            requested,
            target_current,
            program,
            pairs,
        )
        record = _solve_record(result, wall)
        record["edit_fraction"] = float(fraction)
        warm.append(record)
        print(
            f"WARM fraction={fraction:.2f} wall_s={wall:.6f} "
            f"residual={record['terminal_residual']:.3e} "
            f"converged={record['converged']}",
            flush=True,
        )
        program = result.program
        state = result.state

    # The panels draw the base converged frame, not an edited one.
    masks, achieved = operator.read(cold.state)
    label = np.asarray(masks.label)[:node_count]
    psi_norm = np.asarray(masks.psi_norm)[:node_count]
    core = np.asarray(masks.core)[:node_count]
    vacuum = np.asarray(
        operator.external(None, jnp.asarray(base_current))[:node_count], dtype=float
    )
    solved = np.asarray(cold.state, dtype=float)[:node_count]
    cells = np.asarray(_lattice_cells(lattice), dtype=float)
    units = edit._wall_units(operator)
    nulls = edit._null_points(profile, cold.state)
    branches = _branches(profile, cold.state, achieved)
    coils = _coil_outlines(edit.SHOT)

    payload: dict[str, Any] = {
        "radius": np.asarray(lattice.radius, dtype=float),
        "height": np.asarray(lattice.height, dtype=float),
        "shape": np.asarray(lattice.shape, dtype=int),
        "vacuum_psi": vacuum,
        "solved_psi": solved,
        "cell_polygons": cells,
        "cell_label": label,
        "cell_psi_norm": psi_norm,
        "cell_core": core,
        "axis": np.asarray(nulls["axis"], dtype=float),
        "x_points": np.asarray(nulls["x_points"], dtype=float),
        "x_point_flux": np.asarray(nulls["x_point_flux"], dtype=float),
        "saddle_index": np.asarray(int(nulls["saddle_index"])),
        "axis_flux": np.asarray(float(np.asarray(achieved.axis_flux))),
        "boundary_flux": np.asarray(float(np.asarray(achieved.boundary_flux))),
        "coil_outlines": coils,
        **{f"branch_{name}": value for name, value in branches.items()},
        **_wall_arrays(units),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_path, **payload)

    walls = [item["wall_milliseconds"] for item in warm]
    receipt = {
        "artifact": "one centroid-constrained MAST fixed-point solve and its panels",
        "identity": f"{edit.SHOT}/{edit.SLICE_INDEX} mixed",
        "source_revision": _source_revision(),
        "runtime": _provenance(),
        "evidence_inputs": {
            "response_carrier": carrier,
            "persistent_compilation_cache": cache.receipt(),
            "shot": edit.SHOT,
            "slice_index": edit.SLICE_INDEX,
            "grid_points": grid_points,
            "lattice_shape": [int(value) for value in np.asarray(lattice.shape)],
            "edited_circuit": circuit_index,
            "coil_mapping": prepared["coil_mapping"],
        },
        "measurement_contract": {
            "route": (
                "reduced_newton.solve_constrained_reduced_newton_compiled through "
                "benchmarks.coil_edit_latency._compiled_edit"
            ),
            "constraint": (
                "one vertical current-centroid row targeted on the frame's own "
                "reference centre height"
            ),
            "centroid_target_m": float(prepared["reference_centroid_z"]),
            "centroid_tolerance_m": edit.VERTICAL_CENTROID_TOLERANCE,
            "requested_class": "diverted",
            "tolerance": edit.COMPILED_SLICE_TOLERANCE,
            "newton_steps": edit.COMPILED_SLICE_NEWTON_STEPS,
            "active_set_steps": edit.COMPILED_SLICE_ACTIVE_SET_STEPS,
            "cold_definition": (
                "no program in hand: the wall carries tracing and compilation "
                "alongside one execution"
            ),
            "warm_definition": (
                "the same fixed-shape program re-entered on a moved current "
                "vector, so the timed solve does real trips"
            ),
        },
        "cell_representation": {
            "source": "nova.equilibrium.forward._lattice_cells over the FluxLattice",
            "shape": "uniform rectangle, one control cell per lattice node",
            "vertices_per_cell": int(cells.shape[1]),
            "cell_count": int(cells.shape[0]),
            "radial_step_m": float(lattice.radial_step),
            "vertical_step_m": float(lattice.vertical_step),
            "distinct_cell_areas": int(np.unique(np.asarray(operator.area)).size),
            "clipped_cell_count": 0,
            "note": (
                "every cell carries the full radial-by-vertical control area; no "
                "cell is cut by the wall or by the separatrix, so a panel must "
                "not be read as showing clipped cells"
            ),
        },
        "cold_solve": _solve_record(cold, cold_wall),
        "warm_solves": warm,
        "summary": {
            "cold_wall_milliseconds": float(cold_wall * 1.0e3),
            "warm_wall_milliseconds_median": float(np.median(walls)),
            "warm_wall_milliseconds_minimum": float(np.min(walls)),
            "warm_wall_milliseconds_maximum": float(np.max(walls)),
            "warm_point_count": len(warm),
            "warm_converged_count": int(sum(item["converged"] for item in warm)),
            "core_cell_count": int(np.count_nonzero(core)),
            "elapsed_seconds": float(time.perf_counter() - total_started),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("RECEIPT=" + str(receipt_path), flush=True)


def _solved_boundary(
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
    level: float,
    axis: np.ndarray,
) -> np.ndarray:
    """Return the closed contour at ``level`` that encloses the magnetic axis.

    Selecting by axis containment rather than by size is what keeps the choice
    polarity-free: the core is the high side of the boundary flux under one
    polarity and the low side under the other, and both cases leave the axis
    inside the same loop.
    """
    import contourpy
    import shapely

    generator = contourpy.contour_generator(
        x=radius, y=height, z=flux, line_type=contourpy.LineType.Separate
    )
    point = shapely.Point(float(axis[0]), float(axis[1]))
    best: tuple[float, np.ndarray] | None = None
    for line in generator.lines(float(level)):
        loop = np.asarray(line, dtype=float)
        if loop.shape[0] < 4 or not np.allclose(loop[0], loop[-1]):
            continue
        polygon = shapely.Polygon(loop)
        if not polygon.is_valid:
            polygon = polygon.buffer(0.0)
        if polygon.is_empty or not polygon.contains(point):
            continue
        if best is None or polygon.area > best[0]:
            best = (float(polygon.area), loop)
    if best is None:
        raise ValueError("no closed boundary contour encloses the magnetic axis")
    return best[1]


def panels(
    state_path: Path, figure_path: Path, receipt_path: Path, hex_cells: int
) -> None:
    """Draw the conductor-only and the solved panel from the persisted fields.

    The pair is a superposition statement: the left panel is the field of the
    conductors alone, which is known exactly from their stored geometry and
    currents, and the right panel is that same field plus the plasma image the
    two low-degree flux functions support. Both are contoured on ONE physical
    level array so the difference between them is the plasma and not a change
    of scale.

    The boundary is the assembled level set of the solved lattice spline, split
    at its saddle -- not a contour of the receiver raster, whose between-node
    wander would show as a wobble and would also be the wrong curve to cut the
    plasma mesh against.
    """
    from nova.media.sources.plasma_mesh import clip_to_boundary, hex_mesh

    data = np.load(state_path, allow_pickle=False)
    shape = tuple(int(value) for value in np.asarray(data["shape"]))
    radius = np.asarray(data["radius"], dtype=float)
    height = np.asarray(data["height"], dtype=float)
    vacuum = np.asarray(data["vacuum_psi"], dtype=float).reshape(shape).T
    solved = np.asarray(data["solved_psi"], dtype=float).reshape(shape).T
    units = _restore_units(data)
    coils = np.asarray(data["coil_outlines"], dtype=float)
    boundary_flux = float(np.asarray(data["boundary_flux"]))
    magnetic_axis = np.asarray(data["axis"], dtype=float)
    x_points = np.asarray(data["x_points"], dtype=float).reshape(-1, 2)
    saddle = int(np.asarray(data["saddle_index"]))
    branches = {
        name: np.asarray(data[f"branch_{name}"])
        for name in (
            "closed_controls_rz",
            "closed_valid",
            "open_controls_rz",
            "open_valid",
            "open_branch_valid",
        )
    }

    boundary = poloidal.sample_cubic_controls(
        branches["closed_controls_rz"], branches["closed_valid"], BOUNDARY_SAMPLES
    )
    mesh, mesh_provenance = hex_mesh(units, cells=hex_cells)
    clipped = clip_to_boundary(mesh, boundary) if boundary.shape[0] >= 3 else ()
    vertex_counts = (
        np.asarray([len(item) for item in clipped])
        if clipped
        else np.zeros(0, dtype=int)
    )

    both = np.concatenate((vacuum.ravel(), solved.ravel()))
    levels = poloidal.contour_levels(both, LEVEL_COUNT, boundary=boundary_flux)

    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 6.4), facecolor=DEFAULT_INK.figure_facecolor
    )
    for panel in axes:
        poloidal_axes(panel)
        poloidal.draw_coils(
            panel, coils, edgecolor=COIL_EDGE_COLOR, linewidth=COIL_LINEWIDTH
        )
        poloidal.draw_wall(panel, units=units)

    poloidal.draw_flux_contours(axes[0], radius, height, vacuum, levels)
    axes[0].set_title("conductors only, no plasma", fontsize=9)

    poloidal.draw_plasma_cells(axes[1], clipped, alpha=CELL_ALPHA)
    poloidal.draw_flux_contours(axes[1], radius, height, solved, levels)
    # The boundary is one of the drawn contours, not a separate red curve: same
    # ink, slightly heavier, so it reads as the surface the level array already
    # contains rather than as an overlay from somewhere else.
    tally = poloidal.draw_separatrix_branches(
        axes[1],
        branches,
        closed_color=DEFAULT_INK.contour_color,
        open_color=DEFAULT_INK.contour_color,
        closed_linewidth=BOUNDARY_LINEWIDTH,
        open_linewidth=BOUNDARY_LINEWIDTH,
    )
    admitted = x_points[saddle : saddle + 1] if x_points.size else None
    other = np.delete(x_points, saddle, axis=0) if x_points.shape[0] > 1 else None
    poloidal.draw_nulls(
        axes[1],
        magnetic_axis=magnetic_axis,
        x_points=admitted,
        other_x_points=other,
        contain=units,
    )
    axes[1].set_title("solved, with plasma", fontsize=9)

    for panel in axes:
        panel.set_xlim(float(radius.min()), float(radius.max()))
        panel.set_ylim(float(height.min()), float(height.max()))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    drawn = {
        **mesh_provenance,
        "boundary_source": (
            "assemble_separatrix_branches on the solved lattice spline, split "
            "at the admitted saddle"
        ),
        "boundary_vertex_count": int(boundary.shape[0]),
        "boundary_branches_drawn": tally,
        "boundary_clipped_cells": int(len(clipped)),
        "cut_at_boundary_cells": int(np.count_nonzero(vertex_counts != 7)),
        "coil_section_count": int(coils.shape[0]),
    }
    print(
        "FIGURE=%s hex_delivered=%d boundary_cells=%d cut=%d coils=%d "
        "boundary_vertices=%d levels=%d"
        % (
            figure_path,
            int(mesh_provenance["delivered_cells"]),
            len(clipped),
            drawn["cut_at_boundary_cells"],
            coils.shape[0],
            boundary.shape[0],
            levels.size,
        ),
        flush=True,
    )
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["figure"] = str(figure_path)
        receipt["drawn_plasma_mesh"] = drawn
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def series(
    receipt_path: Path,
    figure_path: Path,
    carrier_path: Path,
    grid_points: int | None,
    hex_cells: int,
) -> None:
    """Solve one frame repeatedly with the flux-function pair rescaled.

    Every member carries the same conductor currents and the same pinned net
    plasma current; only the ratio of the two low-degree flux-function
    amplitudes moves. p-prime enters the toroidal current density weighted by
    R and ff-prime weighted by 1/R, so the ratio redistributes a fixed current
    radially, which is the lever on the boundary shape and on whether the
    plasma reaches a saddle or leans on the wall.

    The pair cannot break up-down symmetry by itself -- both are functions of
    normalised flux alone -- so an upper against lower saddle can only move
    here by changing WHICH of the machine's own saddles the boundary reaches
    first. That is a real mechanism on a near-double-null frame and no
    mechanism at all on a frame far from one; the receipt records both saddles
    so the reader can see which case this is.
    """
    configure_dtypes()
    if jax.config.jax_enable_x64 is not True:
        raise RuntimeError("extended precision did not take before any array was built")
    _require_host()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    started = time.perf_counter()
    profile, prepared, carrier = edit._prepare_case(
        carrier_path, grid_points, _sampled_factory
    )
    operator = profile.operator
    lattice = profile.lattice
    node_count = int(operator.grid.node_number)
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    target_current = float(prepared["target_current"])
    base_current = jnp.asarray(prepared["prescribed_current"])
    pairs = (prepared["vertical_centroid_pair"],)
    seed = prepared["initial"]
    units = edit._wall_units(operator)
    coils = _coil_outlines(edit.SHOT)

    stored = _stored_profiles(edit.SHOT, edit.SLICE_INDEX)
    members: list[dict[str, Any]] = []
    panels_data: list[dict[str, Any]] = []
    program = None
    for offset in RATIO_OFFSETS:
        varied = _varied_source(stored, 1.0 + offset, 1.0 - offset)
        member_profile = profile._with_source(varied)
        result, wall = _timed_solve(
            member_profile,
            seed,
            base_current,
            requested,
            target_current,
            program,
            pairs,
        )
        program = result.program
        record = _solve_record(result, wall)
        record["ratio_offset"] = float(offset)
        record["pressure_scale"] = 1.0 + float(offset)
        record["field_function_scale"] = 1.0 - float(offset)
        try:
            _masks, achieved = member_profile.operator.read(result.state)
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            record["read_error"] = f"{type(error).__name__}: {error}"
            members.append(record)
            continue
        nulls = edit._null_points(member_profile, result.state)
        record["achieved_class"] = edit._achieved_class(member_profile, result.state)
        record["axis_rz"] = [float(value) for value in nulls["axis"]]
        record["x_points_rz"] = np.asarray(nulls["x_points"], float).tolist()
        record["x_point_flux"] = np.asarray(nulls["x_point_flux"], float).tolist()
        record["saddle_index"] = int(nulls["saddle_index"])
        record["boundary_flux"] = float(np.asarray(achieved.boundary_flux))
        record["axis_flux"] = float(np.asarray(achieved.axis_flux))
        members.append(record)
        panels_data.append(
            {
                "offset": float(offset),
                "flux": np.asarray(result.state, float)[:node_count],
                "branches": _branches(member_profile, result.state, achieved),
                "nulls": nulls,
                "boundary_flux": float(np.asarray(achieved.boundary_flux)),
                "converged": bool(np.asarray(result.converged)),
                "class": _class_label(record["achieved_class"]),
            }
        )
        print(
            f"MEMBER offset={offset:+.2f} wall_s={wall:.3f} "
            f"residual={record['terminal_residual']:.3e} "
            f"converged={record['converged']} "
            f"class={_class_label(record['achieved_class'])}",
            flush=True,
        )

    _render_series(
        panels_data, lattice, units, coils, figure_path, hex_cells, node_count
    )
    receipt = {
        "artifact": "flux-function ratio series at one pinned net plasma current",
        "identity": f"{edit.SHOT}/{edit.SLICE_INDEX} mixed",
        "source_revision": _source_revision(),
        "runtime": _provenance(),
        "evidence_inputs": {
            "response_carrier": carrier,
            "persistent_compilation_cache": cache.receipt(),
            "grid_points": grid_points,
            "lattice_shape": [int(value) for value in np.asarray(lattice.shape)],
        },
        "measurement_contract": {
            "fixed": (
                "conductor currents, the vertical current-centroid row, and the "
                "pinned net plasma current"
            ),
            "varied": (
                "the two flux-function normalisations, scaled by 1+offset and "
                "1-offset, coefficient vectors and evaluator untouched"
            ),
            "target_current_a": target_current,
            "ratio_offsets": [float(value) for value in RATIO_OFFSETS],
            "program_reuse": (
                "one compiled program serves every member: the tables cross the "
                "program boundary as array leaves"
            ),
            "profile_representation": (
                "65-node efm/pprime and efm/ffprime tables carried by "
                "SampledFluxFunction; the reference case closes the same tables "
                "into its evaluator behind a slope-matched cubic cap, so this "
                "series is not a bit reproduction of that case"
            ),
        },
        "members": members,
        "summary": {
            "member_count": len(members),
            "converged_count": int(sum(bool(m.get("converged")) for m in members)),
            "classes": sorted(
                {
                    _class_label(m["achieved_class"])
                    for m in members
                    if "achieved_class" in m
                }
            ),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print("RECEIPT=" + str(receipt_path), flush=True)


def _render_series(
    panels_data: list[dict[str, Any]],
    lattice: Any,
    units: tuple[Any, ...],
    coils: np.ndarray,
    figure_path: Path,
    hex_cells: int,
    node_count: int,
) -> None:
    """Draw the series as one strip on a single shared level array."""
    from nova.media.sources.plasma_mesh import clip_to_boundary, hex_mesh

    if not panels_data:
        raise RuntimeError("the series produced no drawable member")
    shape = tuple(int(value) for value in np.asarray(lattice.shape))
    radius = np.asarray(lattice.radius, float)
    height = np.asarray(lattice.height, float)
    maps = [item["flux"].reshape(shape).T for item in panels_data]
    # One physical level array across the whole strip: a member drawn on its
    # own levels would show a shape change that is only a change of scale.
    levels = poloidal.contour_levels(
        np.concatenate([item.ravel() for item in maps]),
        LEVEL_COUNT,
        boundary=panels_data[len(panels_data) // 2]["boundary_flux"],
    )
    mesh, _provenance = hex_mesh(units, cells=hex_cells)
    figure, axes = plt.subplots(
        1,
        len(panels_data),
        figsize=(3.1 * len(panels_data), 6.4),
        facecolor=DEFAULT_INK.figure_facecolor,
    )
    axes = np.atleast_1d(axes)
    for panel, item, flux in zip(axes, panels_data, maps):
        poloidal_axes(panel)
        poloidal.draw_coils(
            panel, coils, edgecolor=COIL_EDGE_COLOR, linewidth=COIL_LINEWIDTH
        )
        boundary = poloidal.sample_cubic_controls(
            item["branches"]["closed_controls_rz"],
            item["branches"]["closed_valid"],
            BOUNDARY_SAMPLES,
        )
        if boundary.shape[0] >= 3:
            poloidal.draw_plasma_cells(
                panel, clip_to_boundary(mesh, boundary), alpha=CELL_ALPHA
            )
        poloidal.draw_flux_contours(panel, radius, height, flux, levels)
        poloidal.draw_separatrix_branches(
            panel,
            item["branches"],
            closed_color=DEFAULT_INK.contour_color,
            open_color=DEFAULT_INK.contour_color,
            closed_linewidth=BOUNDARY_LINEWIDTH,
            open_linewidth=BOUNDARY_LINEWIDTH,
        )
        x_points = np.asarray(item["nulls"]["x_points"], float).reshape(-1, 2)
        saddle = int(item["nulls"]["saddle_index"])
        poloidal.draw_nulls(
            panel,
            magnetic_axis=np.asarray(item["nulls"]["axis"], float),
            x_points=x_points[saddle : saddle + 1] if x_points.size else None,
            other_x_points=(
                np.delete(x_points, saddle, axis=0) if x_points.shape[0] > 1 else None
            ),
            contain=units,
        )
        mark = "" if item["converged"] else "  (not converged)"
        panel.set_title(
            f"p'x{1.0 + item['offset']:.2f}  FF'x{1.0 - item['offset']:.2f}\n"
            f"{item['class']}{mark}",
            fontsize=8,
        )
        panel.set_xlim(float(radius.min()), float(radius.max()))
        panel.set_ylim(float(height.min()), float(height.max()))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print(f"SERIES_FIGURE={figure_path} members={len(panels_data)}", flush=True)


def family(
    receipt_path: Path,
    figure_path: Path,
    carrier_path: Path,
    grid_points: int | None,
) -> None:
    """Vary the diamagnetic gradient alone and record how the shape answers.

    One parameter moves: the amplitude of ff-prime. p-prime is untouched and
    the solve pins the net plasma current, so the current normalisation is the
    lambda that absorbs the amplitude change and every member carries the SAME
    total current. What is left is shape.

    Two curves are recorded per member because shape is not only the outline:
    the separatrix is the external shape and the q = 3/2 rational surface is an
    internal one, read from each member's own flux-surface-averaged safety
    factor rather than carried over from a neighbour.
    """
    configure_dtypes()
    if jax.config.jax_enable_x64 is not True:
        raise RuntimeError("extended precision did not take before any array was built")
    _require_host()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    started = time.perf_counter()
    profile, prepared, carrier = edit._prepare_case(
        carrier_path, grid_points, _sampled_factory
    )
    operator = profile.operator
    lattice = profile.lattice
    node_count = int(operator.grid.node_number)
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    target_current = float(prepared["target_current"])
    base_current = jnp.asarray(prepared["prescribed_current"])
    pairs = (prepared["vertical_centroid_pair"],)
    seed = prepared["initial"]
    units = edit._wall_units(operator)
    coils = _coil_outlines(edit.SHOT)
    stored = _stored_profiles(edit.SHOT, edit.SLICE_INDEX)

    members: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    program = None
    state = seed
    for scale in FIELD_FUNCTION_SCALES:
        varied = _varied_source(stored, 1.0, scale)
        member_profile = profile._with_source(varied)
        result, wall = _timed_solve(
            member_profile,
            state,
            base_current,
            requested,
            target_current,
            program,
            pairs,
        )
        program = result.program
        record = _solve_record(result, wall)
        record["field_function_scale"] = float(scale)
        flux = np.asarray(result.state, float)[:node_count]
        try:
            _masks, topology = member_profile.operator.read(result.state)
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            record["read_error"] = f"{type(error).__name__}: {error}"
            members.append(record)
            continue
        record["achieved_class"] = _class_label(
            edit._achieved_class(member_profile, result.state)
        )
        record["axis_rz"] = np.asarray(topology.axis, float).reshape(-1)[:2].tolist()
        record["boundary_flux"] = float(np.asarray(topology.boundary_flux))
        record["axis_flux"] = float(np.asarray(topology.axis_flux))
        boundary = _closed_loop(
            lattice, flux, record["boundary_flux"], np.asarray(topology.axis, float)
        )
        rational = _rational_surfaces(lattice, varied, flux, topology, RATIONAL_ORDERS)
        record["rational"] = rational
        surfaces: dict[str, np.ndarray] = {}
        for name, entry in rational["surfaces"].items():
            if entry is None:
                continue
            loop = _closed_loop(
                lattice, flux, entry["flux"], np.asarray(topology.axis, float)
            )
            if loop.shape[0] >= 2:
                surfaces[name] = loop
        record["boundary_vertex_count"] = int(boundary.shape[0])
        record["rational_vertex_counts"] = {
            name: int(loop.shape[0]) for name, loop in surfaces.items()
        }
        members.append(record)
        curves.append(
            {
                "scale": float(scale),
                "boundary": boundary,
                "rational": surfaces,
                "ff_prime": scale * stored["ff_prime"],
                "converged": bool(np.asarray(result.converged)),
                "class": record["achieved_class"],
            }
        )
        print(
            f"MEMBER ff_scale={scale:.2f} wall_s={wall:.3f} "
            f"residual={record['terminal_residual']:.3e} "
            f"converged={record['converged']} class={record['achieved_class']} "
            f"q_range={rational.get('safety_factor_minimum')}"
            f"..{rational.get('safety_factor_maximum')} "
            f"surfaces={_surface_labels(rational)}",
            flush=True,
        )
        # Advance the warm start only through a member that converged: seeding
        # the next member from a failed terminal state carries that failure
        # forward and reads as a property of the next profile.
        if bool(np.asarray(result.converged)):
            state = result.state

    _render_family(curves, stored["psi_norm"], units, coils, lattice, figure_path)
    receipt = {
        "artifact": (
            "diamagnetic-gradient family at one pinned net plasma current, with "
            "external and internal shape recorded"
        ),
        "identity": f"{edit.SHOT}/{edit.SLICE_INDEX} mixed",
        "source_revision": _source_revision(),
        "runtime": _provenance(),
        "evidence_inputs": {
            "response_carrier": carrier,
            "persistent_compilation_cache": cache.receipt(),
            "grid_points": grid_points,
            "lattice_shape": [int(value) for value in np.asarray(lattice.shape)],
        },
        "measurement_contract": {
            "varied": "the ff-prime table amplitude alone; p-prime is untouched",
            "held": (
                "conductor currents, the vertical current-centroid row, and the "
                "net plasma current, which the solve pins so the current "
                "normalisation absorbs the amplitude change"
            ),
            "target_current_a": target_current,
            "field_function_scales": [float(value) for value in FIELD_FUNCTION_SCALES],
            "external_shape": "separatrix, the axis-enclosing lobe at boundary flux",
            "internal_shape": (
                "rational surfaces at q = "
                + ", ".join(f"{order:g}" for order in RATIONAL_ORDERS)
                + ", from each member's own flux-surface-averaged safety "
                "factor at the outermost crossing; a member whose q minimum "
                "sits above an order has no such surface and records its own "
                "q range instead"
            ),
            "program_reuse": (
                "one compiled program serves every member: the tables cross the "
                "program boundary as array leaves"
            ),
        },
        "members": members,
        "summary": {
            "member_count": len(members),
            "converged_count": int(sum(bool(m.get("converged")) for m in members)),
            "classes": sorted({str(m.get("achieved_class")) for m in members}),
            "rational_surface_found": {
                f"{order:g}": int(
                    sum(
                        bool(
                            (m.get("rational") or {})
                            .get("surfaces", {})
                            .get(f"{order:g}")
                        )
                        for m in members
                    )
                )
                for order in RATIONAL_ORDERS
            },
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print("RECEIPT=" + str(receipt_path), flush=True)


def _render_family(
    curves: list[dict[str, Any]],
    psi_norm: np.ndarray,
    units: tuple[Any, ...],
    coils: np.ndarray,
    lattice: Any,
    figure_path: Path,
) -> None:
    """Draw the shape family beside the gradient variation that produced it."""
    from matplotlib import colormaps
    from nova.media.ink import trace_axes

    if not curves:
        raise RuntimeError("the family produced no drawable member")
    radius = np.asarray(lattice.radius, float)
    height = np.asarray(lattice.height, float)
    colours = colormaps["viridis"](np.linspace(0.05, 0.9, len(curves)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 6.6),
        width_ratios=(1.35, 1.0),
        facecolor=DEFAULT_INK.figure_facecolor,
    )
    poloidal_axes(axes[0])
    poloidal.draw_coils(
        axes[0], coils, edgecolor=COIL_EDGE_COLOR, linewidth=COIL_LINEWIDTH
    )
    poloidal.draw_wall(axes[0], units=units)
    for colour, item in zip(colours, curves):
        style = "solid" if item["converged"] else (0, (4, 2))
        if item["boundary"].shape[0] >= 2:
            loop = np.vstack((item["boundary"], item["boundary"][:1]))
            axes[0].plot(
                loop[:, 0],
                loop[:, 1],
                color=colour,
                linewidth=1.6,
                linestyle=style,
                zorder=DEFAULT_INK.zorder_separatrix,
            )
        for name, surface in item["rational"].items():
            loop = np.vstack((surface, surface[:1]))
            axes[0].plot(
                loop[:, 0],
                loop[:, 1],
                color=colour,
                linewidth=1.0,
                linestyle=RATIONAL_STYLES.get(name, (0, (2, 2))),
                zorder=DEFAULT_INK.zorder_separatrix,
            )
    axes[0].set_xlim(float(radius.min()), float(radius.max()))
    axes[0].set_ylim(float(height.min()), float(height.max()))
    drawn = sorted({name for item in curves for name in item["rational"]})
    internal = (
        ", ".join(f"q = {name}" for name in drawn) if drawn else "no rational surface"
    )
    axes[0].set_title(f"separatrix (solid) and {internal}", fontsize=9)

    trace_axes(axes[1])
    for colour, item in zip(colours, curves):
        axes[1].plot(
            np.asarray(psi_norm, float),
            np.asarray(item["ff_prime"], float),
            color=colour,
            linewidth=1.4,
            label=f"x{item['scale']:.2f}",
        )
    axes[1].set_xlabel(r"$\psi_N$", fontsize=9)
    axes[1].set_ylabel(r"$FF^\prime$  [T m / Wb]", fontsize=9)
    axes[1].set_title("the diamagnetic gradient that produced them", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False, title="ff' scale", title_fontsize=7)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print(f"FAMILY_FIGURE={figure_path} members={len(curves)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "panels", "series", "family"):
        item = subparsers.add_parser(name)
        item.add_argument("--state", type=Path, default=DEFAULT_STATE)
        item.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
        item.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
        item.add_argument(
            "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
        )
        item.add_argument("--hex-cells", type=int, default=DEFAULT_HEX_CELLS)
        item.add_argument("--grid-points", type=int, default=None)
    arguments = parser.parse_args()
    if arguments.command == "family":
        family(
            arguments.receipt,
            arguments.figure,
            arguments.carrier,
            arguments.grid_points,
        )
    elif arguments.command == "series":
        series(
            arguments.receipt,
            arguments.figure,
            arguments.carrier,
            arguments.grid_points,
            arguments.hex_cells,
        )
    elif arguments.command == "run":
        measure(
            arguments.state,
            arguments.receipt,
            arguments.carrier,
            arguments.grid_points,
        )
    else:
        panels(
            arguments.state,
            arguments.figure,
            arguments.receipt,
            arguments.hex_cells,
        )


if __name__ == "__main__":
    main()
