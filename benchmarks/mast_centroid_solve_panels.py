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
from nova.equilibrium.forward import _lattice_cells
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


def measure(state_path: Path, receipt_path: Path, carrier_path: Path) -> None:
    """Solve the centroid-constrained frame and persist the panel fields."""
    configure_dtypes()
    if jax.config.jax_enable_x64 is not True:
        raise RuntimeError("extended precision did not take before any array was built")
    _require_host()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    total_started = time.perf_counter()

    profile, prepared, carrier = edit._prepare_case(carrier_path)
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


def panels(state_path: Path, figure_path: Path, receipt_path: Path) -> None:
    """Draw the conductor-only and the solved panel from the persisted fields."""
    data = np.load(state_path, allow_pickle=False)
    shape = tuple(int(value) for value in np.asarray(data["shape"]))
    radius = np.asarray(data["radius"], dtype=float)
    height = np.asarray(data["height"], dtype=float)
    vacuum = np.asarray(data["vacuum_psi"], dtype=float).reshape(shape).T
    solved = np.asarray(data["solved_psi"], dtype=float).reshape(shape).T
    cells = np.asarray(data["cell_polygons"], dtype=float)
    core = np.asarray(data["cell_core"], dtype=bool)
    units = _restore_units(data)
    boundary_flux = float(np.asarray(data["boundary_flux"]))
    x_points = np.asarray(data["x_points"], dtype=float).reshape(-1, 2)
    saddle = int(np.asarray(data["saddle_index"]))

    # One physical level array serves both panels: two maps contoured on
    # independently chosen levels can be made to look like anything.
    both = np.concatenate((vacuum.ravel(), solved.ravel()))
    levels = poloidal.contour_levels(both, LEVEL_COUNT, boundary=boundary_flux)

    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 6.4), facecolor=DEFAULT_INK.figure_facecolor
    )
    for panel in axes:
        poloidal_axes(panel)

    poloidal.draw_flux_contours(axes[0], radius, height, vacuum, levels)
    poloidal.draw_wall(axes[0], units=units)
    axes[0].set_title("conductors only, no plasma", fontsize=9)

    poloidal.draw_plasma_cells(
        axes[1], [cell for cell, keep in zip(cells, core) if keep], alpha=CELL_ALPHA
    )
    poloidal.draw_flux_contours(axes[1], radius, height, solved, levels)
    poloidal.draw_wall(axes[1], units=units)
    admitted = x_points[saddle : saddle + 1] if x_points.size else None
    other = np.delete(x_points, saddle, axis=0) if x_points.shape[0] > 1 else None
    poloidal.draw_nulls(
        axes[1],
        magnetic_axis=np.asarray(data["axis"], dtype=float),
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
    print(
        "FIGURE=%s core_cells=%d levels=%d boundary_flux=%.6f"
        % (figure_path, int(np.count_nonzero(core)), levels.size, boundary_flux),
        flush=True,
    )
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["figure"] = str(figure_path)
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
    arguments = parser.parse_args()
    if arguments.command == "run":
        measure(arguments.state, arguments.receipt, arguments.carrier)
    else:
        panels(arguments.state, arguments.figure, arguments.receipt)


if __name__ == "__main__":
    main()
