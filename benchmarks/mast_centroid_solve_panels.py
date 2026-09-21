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


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "panels"):
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
    if arguments.command == "run":
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
