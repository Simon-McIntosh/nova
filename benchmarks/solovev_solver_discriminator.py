"""Measure solver-level flux error across hexagonal and rectangular carriers.

Each invocation measures one carrier and requested resolution in a fresh
process.  The production solve is followed by two normalization controls: a
post-hoc constant gauge shift that makes the terminal axis flux equal the
closed-form value, and a second production-seam solve whose topology read is
locally instrumented to use the closed-form axis position and flux.  The
instrumentation lives only in this benchmark process and does not change Nova's
solver semantics.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import LineString

from benchmarks import solovev_certificate as certificate
from nova.biot.plasmagrid import PlasmaGrid
from nova.equilibrium import ForwardProfile
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh, ring_condition
from nova.frame.coilset import CoilSet
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/uniform-cell-clip-and-coupling/solver-discriminator"
RECEIPT = OUTPUT_ROOT / "receipt.json"
ERROR_FIGURE = OUTPUT_ROOT / "error-maps.svg"
CONTROL_FIGURE = OUTPUT_ROOT / "exact-axis-control-error-maps.svg"
REQUESTED_CELLS = (110, 300, 500)
TILINGS = ("hexagonal", "rectangular")
CASE_NAME = "weak-rotation-reactor-static"
OUTBOARD_WINDOW = (7.25, 7.75, 0.0, 0.5)
HELD_CLIPPING_TIP = "1c4daf51"


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _array_digest(values: np.ndarray) -> str:
    packed = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _lane_receipt() -> dict[str, Any]:
    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _clipping_stack_receipt(revision: str) -> dict[str, Any]:
    present = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", HELD_CLIPPING_TIP, revision],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    return {
        "held_tip": HELD_CLIPPING_TIP,
        "present_in_measured_revision": present,
        "measurement_interpretation": (
            "the clipping stack is absent; this receipt measures the assigned "
            "production base exactly as dispatched"
            if not present
            else "the held clipping stack is reachable from the measured revision"
        ),
    }


def _rectangular_machine(case: Any, requested_cells: int) -> Any:
    """Build the certificate machine with authored rectangular cells."""
    wall = oracle_fixture.limiter_contour(case, points=oracle_fixture.WALL_POINT_COUNT)
    coilset = CoilSet(dplasma=-requested_cells, tplasma="rectangle")
    coilset.firstwall.insert(wall, turn="rectangle")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    centres = np.c_[
        np.asarray(coilset.subframe.loc[plasma, "x"], dtype=np.float64),
        np.asarray(coilset.subframe.loc[plasma, "z"], dtype=np.float64),
    ]
    polygons = tuple(
        oracle_fixture._clean_vertices(
            np.asarray(item.poly.exterior.coords, dtype=np.float64)[:-1, :2]
        )
        for item in material
    )
    area = np.asarray([item.poly.area for item in material], dtype=np.float64)
    triangulation = Delaunay(centres)
    boundary = LineString(wall)
    boundary_cells = np.asarray(
        [
            position
            for position, item in enumerate(material)
            if item.poly.intersects(boundary)
        ],
        dtype=np.intp,
    )
    stencil, _ = PlasmaGrid.loop_neighbour_vertices(
        centres, triangulation.vertex_neighbor_vertices, boundary_cells
    )
    mesh = StencilMesh(centres, stencil, area)
    sections = np.asarray(coilset.aloc["plasma", "section"], dtype=object).astype(str)
    complete = np.flatnonzero(sections == "rectangle")
    if len(complete) == 0:
        raise RuntimeError("the rectangular carrier has no complete generator")
    dimensions = np.c_[
        np.asarray(coilset.aloc["plasma", "dl"], dtype=np.float64)[complete],
        np.asarray(coilset.aloc["plasma", "dt"], dtype=np.float64)[complete],
    ]
    width, height = dimensions[0]
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    offsets = np.column_stack(
        (0.5 * width * np.cos(angles), 0.5 * height * np.sin(angles))
    )
    sampling = centres[:, None, :] + offsets[None, :, :]
    geometry = MomentGeometry.from_cells(mesh, polygons, sampling_vertices=sampling)
    condition = ring_condition(centres, stencil)
    regular = np.asarray([len(polygon) == 4 for polygon in polygons])
    interior_stencil = stencil[(condition < 1.0e3) & regular[stencil].all(axis=1)]
    sample = geometry.sample_node_coordinates
    with oracle_fixture.polygon_analytic_flux_moment_executor() as executor:
        grid_blocks = oracle_fixture._flux_blocks(
            centres, polygons, geometry.atomic_mesh.centroids, executor=executor
        )
        wall_blocks = oracle_fixture._flux_blocks(
            wall, polygons, geometry.atomic_mesh.centroids, executor=executor
        )
        sample_blocks = oracle_fixture._flux_blocks(
            sample, polygons, geometry.atomic_mesh.centroids, executor=executor
        )
    return oracle_fixture.OracleMachine(
        node=centres,
        area=area,
        cell_polygons=polygons,
        stencil=stencil,
        interior_stencil=interior_stencil,
        wall_node=wall,
        sampling_vertices=sampling,
        sample_coordinates=sample,
        plasma_to_grid=grid_blocks[0],
        plasma_to_grid_r=grid_blocks[1],
        plasma_to_grid_z=grid_blocks[2],
        plasma_to_wall=wall_blocks[0],
        plasma_to_wall_r=wall_blocks[1],
        plasma_to_wall_z=wall_blocks[2],
        plasma_to_sample=sample_blocks[0],
        plasma_to_sample_r=sample_blocks[1],
        plasma_to_sample_z=sample_blocks[2],
        cache={
            "hit": False,
            "store": None,
            "semantic_key": None,
            "builder": "benchmark-local rectangular certificate carrier",
        },
    )


def _machine(case: Any, requested_cells: int, tiling: str) -> Any:
    if tiling == "hexagonal":
        return oracle_fixture.cached_machine(
            case,
            -requested_cells,
            wall_nodes=oracle_fixture.WALL_POINT_COUNT,
        )
    return _rectangular_machine(case, requested_cells)


@contextmanager
def _exact_axis_iteration(case: Any) -> Iterator[None]:
    """Instrument topology reads with the closed-form axis position and flux."""
    original = ForwardFluxOperator._fixed_design_read
    exact_axis = np.asarray(case.magnetic_axis, dtype=np.float64)
    exact_axis_flux = oracle_fixture.TOTAL_FLUX_FACTOR * float(case.axis_flux)

    def fixed_design_read(
        self: ForwardFluxOperator,
        physical: Any,
        requested_class: Any = None,
        private_wall_node_mask: Any = None,
    ) -> tuple[Any, Any, Any, Any]:
        masks, topology, connected, admitted = original(
            self, physical, requested_class, private_wall_node_mask
        )
        controlled = topology._replace(
            axis=jnp.asarray(exact_axis, dtype=topology.axis.dtype),
            axis_flux=jnp.asarray(exact_axis_flux, dtype=topology.axis_flux.dtype),
        )
        return masks, controlled, connected, admitted

    ForwardFluxOperator._fixed_design_read = fixed_design_read
    jax.clear_caches()
    try:
        yield
    finally:
        ForwardFluxOperator._fixed_design_read = original
        jax.clear_caches()


def _outboard_enrichment(coordinates: np.ndarray, error: np.ndarray) -> dict[str, Any]:
    r_min, r_max, z_min, z_max = OUTBOARD_WINDOW
    selected = (
        (coordinates[:, 0] >= r_min)
        & (coordinates[:, 0] <= r_max)
        & (coordinates[:, 1] >= z_min)
        & (coordinates[:, 1] <= z_max)
    )
    squared = np.asarray(error, dtype=np.float64) ** 2
    total = float(np.sum(squared))
    domain_fraction = float(np.mean(selected))
    error_fraction = float(np.sum(squared[selected]) / max(total, np.finfo(float).tiny))
    return {
        "bounds_rz_m": list(OUTBOARD_WINDOW),
        "node_count": int(np.count_nonzero(selected)),
        "domain_fraction": domain_fraction,
        "squared_flux_error_fraction": error_fraction,
        "enrichment": error_fraction / max(domain_fraction, np.finfo(float).tiny),
    }


def _error_metrics(
    coordinates: np.ndarray,
    state: np.ndarray,
    exact: np.ndarray,
    exact_span: float,
) -> dict[str, Any]:
    error = np.asarray(state, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    squared = float(np.sum(error**2))
    return {
        "squared_flux_error_wb2": squared,
        "rms_flux_error_wb": float(np.sqrt(np.mean(error**2))),
        "sup_flux_error_wb": float(np.max(np.abs(error))),
        "rms_fraction_of_exact_span": float(np.sqrt(np.mean(error**2)) / exact_span),
        "sup_fraction_of_exact_span": float(np.max(np.abs(error)) / exact_span),
        "outboard_window": _outboard_enrichment(coordinates, error),
    }


def _solve(
    profile: ForwardProfile,
    seed: np.ndarray,
    target_current: float,
    identity: str,
) -> tuple[Any, float]:
    request = certificate._certificate_solve_request(
        profile, seed, target_current, carrier_identity=identity
    )
    started = perf_counter()
    result = profile.solve(request)
    jax.block_until_ready(result.equilibrium.flux)
    return result, perf_counter() - started


def _topology_metrics(
    operator: ForwardFluxOperator, state: np.ndarray, exact_axis: np.ndarray
) -> dict[str, Any]:
    topology = certificate._topology(operator, state)
    axis = topology["axis_rz_m"]
    return {
        "read": topology,
        "axis_error_mm": (
            float(1.0e3 * np.linalg.norm(np.asarray(axis) - exact_axis))
            if axis is not None
            else None
        ),
    }


def measure_rung(tiling: str, requested_cells: int) -> dict[str, Any]:
    """Measure one carrier-resolution rung and retain its plotting arrays."""
    started = perf_counter()
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the solver discriminator requires JAX extended precision")
    if jax.default_backend() != "cpu":
        raise RuntimeError("the solver discriminator requires the CPU backend")
    compilation_cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    carrier_case, source_case, exact = certificate._case(CASE_NAME)
    print(
        f"DISCRIMINATOR_BUILD tiling={tiling} requested={requested_cells}",
        flush=True,
    )
    machine = _machine(carrier_case, requested_cells, tiling)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(CASE_NAME, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(
        operator, mesh, newton_steps=certificate.recovery.NEWTON_STEPS
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        CASE_NAME, source_case, operator, exact_physical
    )
    seed, requested_class, seed_receipt = certificate._production_seed(
        profile, CASE_NAME, target_current, centroid, current_receipt
    )

    print(
        f"DISCRIMINATOR_SOLVE arm=production tiling={tiling} "
        f"requested={requested_cells}",
        flush=True,
    )
    baseline_receipt, baseline_seconds = _solve(
        profile,
        seed,
        target_current,
        f"solver-discriminator:{tiling}:{requested_cells}:production",
    )
    baseline_equilibrium = baseline_receipt.equilibrium
    baseline_state = np.asarray(baseline_equilibrium.flux, dtype=np.float64)

    print(
        f"DISCRIMINATOR_SOLVE arm=exact_axis tiling={tiling} "
        f"requested={requested_cells}",
        flush=True,
    )
    with _exact_axis_iteration(exact):
        control_receipt, control_seconds = _solve(
            profile,
            seed,
            target_current,
            f"solver-discriminator:{tiling}:{requested_cells}:exact-axis",
        )
    control_equilibrium = control_receipt.equilibrium
    control_state = np.asarray(control_equilibrium.flux, dtype=np.float64)

    cell_count = len(machine.node)
    node = np.asarray(machine.node, dtype=np.float64)
    exact_grid = np.asarray(oracle_state[:cell_count], dtype=np.float64)
    baseline_grid = baseline_state[:cell_count]
    control_grid = control_state[:cell_count]
    exact_axis = np.asarray(exact.magnetic_axis, dtype=np.float64)
    exact_axis_flux = oracle_fixture.TOTAL_FLUX_FACTOR * float(exact.axis_flux)
    exact_span = abs(exact_axis_flux)
    baseline_topology = _topology_metrics(operator, baseline_state, exact_axis)
    control_topology = _topology_metrics(operator, control_state, exact_axis)
    terminal_axis_flux = baseline_topology["read"]["axis_flux_wb"]
    gauge_shift = (
        exact_axis_flux - float(terminal_axis_flux)
        if terminal_axis_flux is not None
        else float("nan")
    )
    renormalised_grid = baseline_grid + gauge_shift

    baseline_error = _error_metrics(node, baseline_grid, exact_grid, exact_span)
    renormalised_error = _error_metrics(node, renormalised_grid, exact_grid, exact_span)
    control_error = _error_metrics(node, control_grid, exact_grid, exact_span)
    baseline_squared = baseline_error["squared_flux_error_wb2"]

    def removed(candidate: dict[str, Any]) -> float:
        return float(
            1.0
            - candidate["squared_flux_error_wb2"]
            / max(baseline_squared, np.finfo(float).tiny)
        )

    revision = _source_revision()
    row = {
        "case": CASE_NAME,
        "tiling": tiling,
        "requested_cells": requested_cells,
        "solver_requested_cells": -requested_cells,
        "realised_cells": cell_count,
        "characteristic_pitch_m": float(np.sqrt(np.median(machine.area))),
        "source_revision": revision,
        "clipping_stack": _clipping_stack_receipt(revision),
        "lane": _lane_receipt(),
        "persistent_compilation_cache": compilation_cache.receipt(),
        "production_path": {
            "operator_builder": (
                "scripts/analytic_oracle_fixtures/measure.py::forward_operator"
            ),
            "solver_call": "ForwardProfile.solve(ForwardSolveRequest)",
            "requested_class": int(requested_class),
            "seed": seed_receipt,
            "target_current_a": target_current,
        },
        "production": {
            "terminal_residual": float(baseline_equilibrium.fixed_point.residual),
            "solve_wall_seconds": baseline_seconds,
            "axis": baseline_topology,
            "flux_error": baseline_error,
            "state_sha256_binary64": _array_digest(baseline_state),
        },
        "exact_axis_flux_renormalisation": {
            "operation": (
                "add exact analytic axis flux minus the production topology axis "
                "flux to every terminal grid value"
            ),
            "exact_axis_flux_wb": exact_axis_flux,
            "production_axis_flux_wb": terminal_axis_flux,
            "constant_gauge_shift_wb": gauge_shift,
            "flux_error": renormalised_error,
            "fraction_squared_flux_error_removed": removed(renormalised_error),
        },
        "exact_axis_iteration_control": {
            "operation": (
                "production solve seam with every topology read instrumented to "
                "substitute the closed-form axis position and flux"
            ),
            "solver_semantics_modified_in_repository": False,
            "exact_axis_rz_m": exact_axis.tolist(),
            "exact_axis_flux_wb": exact_axis_flux,
            "terminal_residual": float(control_equilibrium.fixed_point.residual),
            "solve_wall_seconds": control_seconds,
            "emergent_axis_after_restoring_production_read": control_topology,
            "flux_error": control_error,
            "fraction_squared_flux_error_removed": removed(control_error),
            "state_sha256_binary64": _array_digest(control_state),
        },
        "plot_data": {
            "node_rz_m": node,
            "analytic_flux_wb": exact_grid,
            "production_error_wb": baseline_grid - exact_grid,
            "exact_axis_control_error_wb": control_grid - exact_grid,
        },
        "elapsed_seconds": perf_counter() - started,
    }
    gauge_removed = row["exact_axis_flux_renormalisation"][
        "fraction_squared_flux_error_removed"
    ]
    control_removed = row["exact_axis_iteration_control"][
        "fraction_squared_flux_error_removed"
    ]
    print(
        f"DISCRIMINATOR_RESULT tiling={tiling} requested={requested_cells} "
        f"axis_mm={baseline_topology['axis_error_mm']} "
        f"residual={row['production']['terminal_residual']:.17g} "
        f"gauge_removed={gauge_removed:.9g} "
        f"control_removed={control_removed:.9g}",
        flush=True,
    )
    return row


def _part_path(tiling: str, requested_cells: int) -> Path:
    return OUTPUT_ROOT / "parts" / f"{tiling}-{requested_cells}.json"


def _shared_error_levels(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    absolute = np.concatenate(
        [np.abs(np.asarray(row["plot_data"][key], dtype=np.float64)) for row in rows]
    )
    nonzero = absolute[absolute > 0.0]
    if nonzero.size == 0:
        return np.asarray([np.finfo(float).tiny])
    lower = max(float(np.percentile(nonzero, 5.0)), np.finfo(float).tiny)
    upper = float(np.max(nonzero))
    return np.geomspace(lower, upper, 10) if upper > lower else np.asarray([upper])


def _plot(rows: list[dict[str, Any]], key: str, output: Path, title: str) -> None:
    levels = _shared_error_levels(rows, key)
    figure, axes = plt.subplots(3, 2, figsize=(10.5, 12.0), constrained_layout=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            REQUESTED_CELLS.index(row["requested_cells"]),
            TILINGS.index(row["tiling"]),
        ),
    )
    for axis, row in zip(axes.ravel(), ordered, strict=True):
        node = np.asarray(row["plot_data"]["node_rz_m"], dtype=np.float64)
        analytic = np.asarray(row["plot_data"]["analytic_flux_wb"], dtype=np.float64)
        error = np.abs(np.asarray(row["plot_data"][key], dtype=np.float64))
        analytic_axis = float(
            row["exact_axis_flux_renormalisation"]["exact_axis_flux_wb"]
        )
        analytic_norm = (analytic - analytic_axis) / -analytic_axis
        axis.tricontour(
            node[:, 0],
            node[:, 1],
            analytic_norm,
            levels=np.linspace(0.1, 0.9, 9),
            colors="0.65",
            linewidths=0.55,
        )
        axis.tricontour(
            node[:, 0],
            node[:, 1],
            np.maximum(error, levels[0]),
            levels=levels,
            colors="C3",
            linewidths=0.9,
        )
        exact_axis = np.asarray(
            row["exact_axis_iteration_control"]["exact_axis_rz_m"], dtype=np.float64
        )
        axis.plot(exact_axis[0], exact_axis[1], marker="+", color="black", ms=7)
        axis.set_aspect("equal")
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_title(
            f"{row['tiling']} · {row['requested_cells']} requested · "
            f"{row['realised_cells']} realised"
        )
    figure.suptitle(title + "\nred: |flux error| shared levels; grey: analytic ψN")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def aggregate(output: Path = RECEIPT) -> dict[str, Any]:
    rows = [
        json.loads(_part_path(tiling, cells).read_text(encoding="utf-8"))
        for cells in REQUESTED_CELLS
        for tiling in TILINGS
    ]
    revisions = {row["source_revision"] for row in rows}
    if len(revisions) != 1:
        raise RuntimeError(f"rungs do not share one source revision: {revisions}")
    for row in rows:
        lane = row["lane"]
        if (
            lane["execution"] != "slurm"
            or lane["partition"] != "all_debug"
            or lane["jax_platform"] != "cpu"
            or not lane["jax_enable_x64"]
        ):
            raise RuntimeError("every rung must be an all_debug CPU-x64 measurement")
        if row["clipping_stack"]["present_in_measured_revision"]:
            raise RuntimeError(
                "the assigned-base receipt unexpectedly contains the held stack"
            )
    _plot(rows, "production_error_wb", ERROR_FIGURE, "Production solver discriminator")
    _plot(
        rows,
        "exact_axis_control_error_wb",
        CONTROL_FIGURE,
        "Exact-axis iteration control",
    )
    compact_rows = []
    for row in rows:
        compact = dict(row)
        compact.pop("plot_data")
        compact_rows.append(compact)
    gauge_removed = [
        row["exact_axis_flux_renormalisation"]["fraction_squared_flux_error_removed"]
        for row in compact_rows
    ]
    iteration_removed = [
        row["exact_axis_iteration_control"]["fraction_squared_flux_error_removed"]
        for row in compact_rows
    ]
    receipt = {
        "schema": "nova.solovev-solver-discriminator.v1",
        "case": CASE_NAME,
        "source_revision": revisions.pop(),
        "completed": True,
        "clipping_stack": {
            "held_tip": HELD_CLIPPING_TIP,
            "present_in_measured_revision": False,
            "reason": (
                "the discriminator intentionally measures the assigned base while "
                "the clipping stack remains held outside main"
            ),
        },
        "contract": {
            "requested_cells": list(REQUESTED_CELLS),
            "tilings": list(TILINGS),
            "one_rung_per_fresh_process": True,
            "lane": "all_debug CPU float64",
            "production_solver": "ForwardProfile.solve(ForwardSolveRequest)",
            "outboard_window_rz_m": list(OUTBOARD_WINDOW),
        },
        "rows": compact_rows,
        "figures": [
            str(ERROR_FIGURE.relative_to(ROOT)),
            str(CONTROL_FIGURE.relative_to(ROOT)),
        ],
        "headline": {
            "axis_error_mm_by_tiling_and_requested_cells": {
                row["tiling"]: {
                    str(item["requested_cells"]): item["production"]["axis"][
                        "axis_error_mm"
                    ]
                    for item in compact_rows
                    if item["tiling"] == row["tiling"]
                }
                for row in compact_rows
            },
            "terminal_residual_by_tiling_and_requested_cells": {
                row["tiling"]: {
                    str(item["requested_cells"]): item["production"][
                        "terminal_residual"
                    ]
                    for item in compact_rows
                    if item["tiling"] == row["tiling"]
                }
                for row in compact_rows
            },
            "exact_axis_flux_renormalisation_fraction_removed_range": [
                float(min(gauge_removed)),
                float(max(gauge_removed)),
            ],
            "exact_axis_iteration_fraction_removed_range": [
                float(min(iteration_removed)),
                float(max(iteration_removed)),
            ],
        },
    }
    _write_json(output, receipt)
    return receipt


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiling", choices=TILINGS)
    parser.add_argument("--requested-cells", type=int, choices=REQUESTED_CELLS)
    parser.add_argument("--part", type=Path)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--output", type=Path, default=RECEIPT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    if arguments.aggregate:
        receipt = aggregate(arguments.output)
        print(json.dumps(receipt["headline"], sort_keys=True), flush=True)
        return
    if arguments.tiling is None or arguments.requested_cells is None:
        raise SystemExit("one --tiling and --requested-cells pair is required")
    part = arguments.part or _part_path(arguments.tiling, arguments.requested_cells)
    row = measure_rung(arguments.tiling, arguments.requested_cells)
    _write_json(part, row)
    print(f"DISCRIMINATOR_ROW_EXIT=0 part={part}", flush=True)


if __name__ == "__main__":
    main()
