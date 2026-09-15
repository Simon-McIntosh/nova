"""Census provisional private carriers on limited Solovev rows.

The production topology classifier first labels every closed carrier outside
the magnetic-axis connectivity component as ``PRIVATE_FLUX``.  That raw flood
is retained here as diagnostic evidence.  The forward operator then applies
the physical rule: a private-flux region exists only behind a finite admitted
saddle.  Limited configurations therefore have no private-flux domain and no
flood residual shadow, even when wall-cut shared links split the diagnostic
connectivity graph.

For the persisted weak 1000- and 2500-cell terminal states and for the analytic
flux sampled on the same carriers, this census records every provisional
private carrier with its index, position, domain label, normalised flux,
connectivity component and distance to the wall polygon.  It also verifies the
post-rule domain and residual shadow are empty and draws the spatial evidence
as line contours with the wall and nulls.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
from typing import Any
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.connectivity_boundary import (
    _canonicalize_reciprocal_hex_edges,
)
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.flux_surface_connectivity import (
    hex_edge_admissibility,
    label_saddle_aware_hex_connected_components,
    private_flux_mask,
)
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import poloidal_axes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "docs/figures/cut-cell-current-attribution/limited-shadow"
CASE_NAME = "weak-rotation-reactor-static"
REQUESTED_CELLS = (-1000, -2500)
LIMITED_CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
)
BASE_REVISION = "3082be61cb6dd4a4d9cae1889993717c4e63e58c"
CPU_TEST_MODULES = (
    "tests/test_analytic_oracle_fixture_exterior.py",
    "tests/test_batched_labeller.py",
    "tests/test_batched_operator_boundary.py",
    "tests/test_census_polish_compaction.py",
    "tests/test_cold_seed_portfolio_raster.py",
    "tests/test_connectivity_boundary.py",
    "tests/test_constraint_response_matrix.py",
    "tests/test_domain_participation_continuity.py",
    "tests/test_edge_constraint_demonstration.py",
    "tests/test_equilibrium_constraint_protocol.py",
    "tests/test_equilibrium_flux_surface_geometry.py",
    "tests/test_equilibrium_forward.py",
    "tests/test_equilibrium_forward_constrained.py",
    "tests/test_equilibrium_forward_reference.py",
    "tests/test_equilibrium_forward_solve.py",
    "tests/test_equilibrium_rotation.py",
    "tests/test_equilibrium_sol.py",
    "tests/test_equilibrium_source.py",
    "tests/test_equilibrium_stencil_mesh.py",
    "tests/test_forward_census.py",
    "tests/test_forward_constraints.py",
    "tests/test_forward_moment_class.py",
    "tests/test_forward_operator_axes.py",
    "tests/test_forward_operator_domain.py",
    "tests/test_forward_operator_tangent.py",
    "tests/test_forward_secondary_null.py",
    "tests/test_forward_support_clip_mode.py",
    "tests/test_hex_flood_coil_nulls.py",
    "tests/test_hex_flood_geometries.py",
    "tests/test_hex_flood_sn_secondary.py",
    "tests/test_hex_flood_snowflake.py",
    "tests/test_jax_topology.py",
    "tests/test_limited_row_residual_shadow.py",
    "tests/test_observable_reduction_parity.py",
    "tests/test_plasma_cell_flood_families.py",
    "tests/test_plasma_cell_shadow_telemetry.py",
    "tests/test_plasma_cell_topology_read.py",
    "tests/test_prescribed_current_solve.py",
    "tests/test_prescribed_current_traced.py",
    "tests/test_recovery_frozen_partition.py",
    "tests/test_reduced_newton_constraint_bounds.py",
    "tests/test_reduced_newton_constraints.py",
    "tests/test_shape_constraints.py",
    "tests/test_sol_closure.py",
    "tests/test_sol_support_selection.py",
    "tests/test_stationary_point_admission.py",
    "tests/test_structured_read_census_rows.py",
    "tests/test_topology_domain_partition.py",
    "tests/test_topology_o_qualification.py",
    "tests/test_wall_anchor_shadow.py",
    "tests/test_wall_height_exclusion.py",
    "tests/test_wall_units_solver.py",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(certificate._strict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _part_path(requested_cells: int) -> Path:
    return (
        ROOT
        / "docs/figures/cut-cell-current-attribution/wholecell-rows/parts/rows"
        / f"{CASE_NAME}-production-route-cells-{abs(requested_cells)}.json"
    )


def _rebuild(requested_cells: int, case_name: str = CASE_NAME):
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)
    empty = certificate.oracle_fixture.forward_operator(source_case, machine)
    _physical, exterior, exterior_cache = (
        certificate.oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty, analytic
        )
    )
    operator = certificate.oracle_fixture.forward_operator(
        source_case, machine, exterior
    )
    return machine, operator, coordinates, np.asarray(analytic), exterior_cache


def _map_reapplication(operator, row: dict[str, Any]) -> dict[str, Any]:
    """Compare an unmasked one-map residual with the solver terminal scalar."""
    terminal = np.asarray(row["render_data"]["terminal_flux_wb"], dtype=np.float64)
    terminal_j = jnp.asarray(terminal)
    target = jnp.asarray(row["solver"]["target_current_a"], dtype=jnp.float64)
    physical = operator.external(None, None) + operator.internal(
        terminal_j, None, target
    )
    scalar = float(
        jnp.max(jnp.abs(physical - terminal_j))
        / jnp.maximum(jnp.max(jnp.abs(physical)), jnp.finfo(physical.dtype).tiny)
    )
    solver = float(row["solver"]["terminal_fixed_point_residual"])
    relative_difference = abs(scalar - solver) / max(abs(solver), np.finfo(float).eps)
    return {
        "solver_terminal_residual": solver,
        "unmasked_one_application_residual": scalar,
        "relative_difference": relative_difference,
        "agrees_to_relative_1e_12": relative_difference <= 1.0e-12,
    }


def _row_summary(mode: str, row: dict[str, Any], operator) -> dict[str, Any]:
    topology = row["geometry"]["root_topology"]
    amplitude = row["solver"]["lambda_amplitude_history"]["samples"][-1]["amplitude"]
    summary = {
        "mode": mode,
        "case": row["case"],
        "requested_cells": row["requested_cells"],
        "realised_cells": row["realised_cells"],
        "axis_error_m": row["geometry"]["magnetic_axis_position_error_m"],
        "axis_error_in_pitch": (
            row["geometry"]["magnetic_axis_position_error_m"]
            / row["characteristic_pitch_m"]
        ),
        "terminal_residual": row["solver"]["terminal_fixed_point_residual"],
        "boundary_flux_error_from_analytic_zero_wb": row["geometry"][
            "boundary_flux_error_wb"
        ],
        "plasma_current_amplitude": amplitude,
        "converged": row["solver"]["production_telemetry"]["converged"],
        "o_candidate_count": topology["o_candidate_count"],
        "x_candidate_count": topology["x_candidate_count"],
        "figure": row["figure"],
        "map_reapplication": _map_reapplication(operator, row),
    }
    baseline = _baseline_part(row["case"], abs(row["requested_cells"]))
    summary["before"] = _metric_summary(baseline)
    summary["terminal_flux_bit_identical_to_before"] = bool(
        np.array_equal(
            np.asarray(row["render_data"]["terminal_flux_wb"], dtype=np.float64),
            np.asarray(baseline["render_data"]["terminal_flux_wb"], dtype=np.float64),
        )
    )
    return summary


def _baseline_part(case_name: str, cells: int) -> dict[str, Any]:
    if case_name in LIMITED_CASES and cells == 1000:
        path = (
            ROOT
            / "docs/figures/cut-cell-current-attribution/wholecell-rows/parts/rows"
            / f"{case_name}-production-route-cells-{cells}.json"
        )
    elif case_name == certificate.DIVERTED_CASE_NAME and cells == 500:
        path = (
            ROOT
            / "docs/figures/gs-absolute-accuracy/solovev/production-route-parts"
            / f"{case_name}-production-route-cells-{cells}.json"
        )
    else:
        raise ValueError(f"no committed comparison row for {case_name} at {cells}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_summary(row: dict[str, Any]) -> dict[str, Any]:
    amplitude = row["solver"]["lambda_amplitude_history"]["samples"][-1]["amplitude"]
    telemetry = row["solver"]["production_telemetry"]
    return {
        "axis_error_in_pitch": (
            row["geometry"]["magnetic_axis_position_error_m"]
            / row["characteristic_pitch_m"]
        ),
        "terminal_residual": row["solver"]["terminal_fixed_point_residual"],
        "boundary_flux_error_from_analytic_zero_wb": row["geometry"][
            "boundary_flux_error_wb"
        ],
        "plasma_current_amplitude": amplitude,
        "converged": telemetry["converged"],
    }


def _solve_gate(output_root: Path) -> dict[str, Any]:
    """Run the limited 1000-cell solves and the diverted identity control."""
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the solve gate requires binary64")
    original_mode = support_clip_mode()
    original_figure_root = certificate.FIGURE_ROOT
    original_part_root = certificate.PART_ROOT
    rows: list[dict[str, Any]] = []
    solve_plan = [
        *(("chord", case_name, -1000) for case_name in LIMITED_CASES[:2]),
        ("chord", LIMITED_CASES[2], -1000),
        ("chord", certificate.DIVERTED_CASE_NAME, -500),
        ("exact", CASE_NAME, -1000),
    ]
    receipt_path = output_root / "solve-receipt.json"
    try:
        for mode, case_name, requested_cells in solve_plan:
            set_support_clip_mode(mode)
            certificate.FIGURE_ROOT = output_root / "solve-panels" / mode
            certificate.PART_ROOT = output_root / "solve-parts" / mode
            row = certificate._measure(case_name, requested_cells)
            _machine, operator, _coordinates, _analytic, _cache = _rebuild(
                requested_cells, case_name
            )
            rows.append(_row_summary(mode, row, operator))
            _write_json(
                receipt_path,
                {
                    "schema": "nova.limited-row-shadow-solve-gate",
                    "source_revision": _source_revision(),
                    "completed": False,
                    "rows": rows,
                },
            )
    finally:
        set_support_clip_mode(original_mode)
        certificate.FIGURE_ROOT = original_figure_root
        certificate.PART_ROOT = original_part_root
    limited = [row for row in rows if row["case"] in LIMITED_CASES]
    control = next(row for row in rows if row["case"] == certificate.DIVERTED_CASE_NAME)
    acceptance = {
        "limited_null_census": all(
            row["o_candidate_count"] == 1 and row["x_candidate_count"] == 0
            for row in limited
        ),
        "unmasked_residual_matches_solver": all(
            row["map_reapplication"]["agrees_to_relative_1e_12"] for row in rows
        ),
        "diverted_private_shadow_retained": control["x_candidate_count"] >= 1,
        "diverted_terminal_state_bit_identical": control[
            "terminal_flux_bit_identical_to_before"
        ],
    }
    receipt = {
        "schema": "nova.limited-row-shadow-solve-gate",
        "source_revision": _source_revision(),
        "completed": True,
        "rows": rows,
        "acceptance": acceptance,
    }
    _write_json(receipt_path, receipt)
    if not all(acceptance.values()):
        raise RuntimeError(f"solve gate failed: {acceptance}")
    print(
        "LIMITED_SHADOW_SOLVE_EXIT=0 " + json.dumps(acceptance, sort_keys=True),
        flush=True,
    )
    return receipt


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _run_cpu_module(
    arm: str,
    source_root: Path,
    module: str,
    output_root: Path,
) -> dict[str, Any]:
    """Run one importing test module in an isolated CPU process."""
    module_path = source_root / module
    stem = module_path.stem
    log_path = output_root / arm / f"{stem}.log"
    junit_path = output_root / arm / f"{stem}.junit.xml"
    if not module_path.exists():
        result = {"module": module, "status": "absent"}
        _write_json(output_root / arm / f"{stem}.json", result)
        return result
    with tempfile.TemporaryDirectory(prefix=f"nova-limited-{arm}-{stem}-") as cache:
        environment = os.environ.copy()
        environment.update(
            {
                "TMPDIR": "/tmp",
                "JAX_PLATFORMS": "cpu",
                "JAX_ENABLE_X64": "1",
                "JAX_ENABLE_COMPILATION_CACHE": "1",
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                "JAX_COMPILATION_CACHE_DIR": cache,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONPATH": str(source_root),
                "XLA_FLAGS": (
                    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
                ),
            }
        )
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    [
                        "/home/ITER/mcintos/Code/nova/.venv/bin/python",
                        "-m",
                        "pytest",
                        "-p",
                        "no:cacheprovider",
                        "-m",
                        "not slow",
                        "-q",
                        f"--junitxml={junit_path}",
                        str(module_path),
                    ],
                    cwd=source_root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
                exit_status = completed.returncode
                status = "passed" if exit_status == 0 else "failed"
            except subprocess.TimeoutExpired:
                exit_status = 124
                status = "timed_out"
        result = {
            "module": module,
            "status": status,
            "exit_status": exit_status,
            "wall_seconds": time.monotonic() - started,
            "log": str(log_path),
            "junit": str(junit_path),
        }
        if junit_path.exists():
            result["counts"] = _junit_counts(junit_path)
        _write_json(output_root / arm / f"{stem}.json", result)
        return result


def _cpu_delta_arm(arm: str, output_root: Path, base_revision: str) -> dict[str, Any]:
    """Persist one base or after test-module arm as each process completes."""
    output_root = output_root.resolve()
    (output_root / arm).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nova-limited-shadow-source-") as scratch:
        if arm == "base":
            archive = Path(scratch) / "source.tar"
            subprocess.run(
                ["git", "archive", "--format=tar", "-o", str(archive), base_revision],
                cwd=ROOT,
                check=True,
            )
            source_root = Path(scratch) / "base"
            source_root.mkdir()
            with tarfile.open(archive) as stream:
                stream.extractall(source_root, filter="data")
            revision = base_revision
        else:
            source_root = ROOT
            revision = _source_revision()
        results: list[dict[str, Any]] = []
        receipt_path = output_root / arm / "arm-receipt.json"
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(
                    _run_cpu_module, arm, source_root, module, output_root
                ): module
                for module in CPU_TEST_MODULES
            }
            for future in as_completed(futures):
                results.append(future.result())
                results.sort(key=lambda item: item["module"])
                _write_json(
                    receipt_path,
                    {
                        "schema": "nova.test-module-delta-arm",
                        "arm": arm,
                        "revision": revision,
                        "completed": False,
                        "results": results,
                    },
                )
    failure_ids = [
        result["module"]
        for result in results
        if result["status"] not in {"passed", "absent"}
    ]
    receipt = {
        "schema": "nova.test-module-delta-arm",
        "arm": arm,
        "revision": revision,
        "completed": True,
        "module_count": len(results),
        "failure_count": len(failure_ids),
        "failure_ids": failure_ids,
        "results": results,
    }
    _write_json(receipt_path, receipt)
    print(
        "LIMITED_SHADOW_CPU_ARM_EXIT="
        f"{int(bool(failure_ids))} arm={arm} failures={len(failure_ids)}",
        flush=True,
    )
    if failure_ids:
        raise RuntimeError(f"{arm} CPU module failures: {failure_ids}")
    return receipt


def _raw_read(operator, physical: jax.Array):
    """Reproduce the connectivity labels before saddle qualification."""
    topology = operator._fixed_design_topology
    initial = topology.read_qualification(
        physical, operator.polarity, operator.inside_material
    )
    grid_flux, _wall_flux = topology.split_flux_map(physical)
    vmap_o, _vmap_x = topology.grid(grid_flux)
    rescue_axis = operator._independent_rescue_axis(vmap_o)
    _seed, material = operator.connectivity_axis_seed(rescue_axis)
    result = topology.read_qualification(physical, operator.polarity, material)
    same_axis = jnp.all(jnp.equal(initial.state.axis, result.state.axis))
    admitted = result.axis_admitted & (~initial.axis_admitted | same_axis)
    if not bool(admitted):
        raise RuntimeError("limited-row axis was not admitted")
    return result


def _component_labels(operator, physical: np.ndarray, raw) -> np.ndarray:
    """Rebuild the exact limited-class connectivity component ids."""
    topology = operator._fixed_design_topology
    grid_flux = np.asarray(physical[: operator.grid.node_number], dtype=np.float64)
    confined = np.asarray(raw.masks.core | raw.masks.private_flux, dtype=bool)
    rings = np.asarray(topology.connectivity_rings, dtype=np.int32)
    coordinate = np.asarray(topology.connectivity_coordinate, dtype=np.float64)
    shared_edges = np.asarray(topology.connectivity_shared_edges, dtype=np.float64)
    edge_values = np.sum(
        np.asarray(topology.connectivity_edge_weight)
        * grid_flux[np.asarray(topology.connectivity_edge_gather)],
        axis=-1,
    )
    uncertainty = float(
        np.asarray(raw.polish_receipt["boundary_interpolation_uncertainty"])
    )
    direction = 1.0 if operator.polarity >= 0 else -1.0
    comparison_boundary = float(raw.state.boundary_flux) + direction * uncertainty
    link = np.asarray(
        hex_edge_admissibility(
            jnp.asarray(grid_flux),
            jnp.asarray(coordinate[:, 0]),
            jnp.asarray(coordinate[:, 1]),
            jnp.asarray(comparison_boundary),
            jnp.asarray(raw.state.axis_flux),
            jnp.asarray(shared_edges),
            edge_values=jnp.asarray(edge_values),
        ),
        dtype=bool,
    )
    missing = np.zeros(rings.shape, dtype=bool)
    missing[:, 1:] = rings[:, 1:] == rings[:, :1]
    link = np.asarray(
        _canonicalize_reciprocal_hex_edges(
            jnp.asarray(rings), jnp.asarray(link & ~missing)
        ),
        dtype=bool,
    )
    distance2 = np.sum((coordinate - np.asarray(raw.state.axis)) ** 2, axis=1)
    seed_index = int(np.argmin(np.where(confined, distance2, np.inf)))
    seed = np.zeros(len(confined), dtype=bool)
    seed[seed_index] = bool(np.any(confined))
    labels = np.asarray(
        label_saddle_aware_hex_connected_components(
            jnp.asarray(confined),
            jnp.asarray(rings),
            jnp.asarray(link),
            confined.size,
        )
    )
    private = np.asarray(
        private_flux_mask(jnp.asarray(labels), jnp.asarray(seed)), dtype=bool
    )
    np.testing.assert_array_equal(private, np.asarray(raw.masks.private_flux))
    return labels


def _distance_to_wall(points: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance from points to a closed wall polyline."""
    start = wall
    delta = np.roll(wall, -1, axis=0) - start
    relative = points[:, None, :] - start[None, :, :]
    length2 = np.sum(delta**2, axis=1)
    fraction = np.divide(
        np.sum(relative * delta[None, :, :], axis=-1),
        length2[None, :],
        out=np.zeros((len(points), len(wall))),
        where=length2[None, :] > 0.0,
    )
    projection = start[None, :, :] + np.clip(fraction, 0.0, 1.0)[..., None] * delta
    return np.min(np.linalg.norm(points[:, None, :] - projection, axis=-1), axis=1)


def _state_census(operator, machine, state: np.ndarray) -> dict[str, Any]:
    physical = np.asarray(state[: operator.physical_node_number], dtype=np.float64)
    raw = _raw_read(operator, jnp.asarray(physical))
    labels = _component_labels(operator, physical, raw)
    raw_private = np.asarray(raw.masks.private_flux, dtype=bool)
    qualified_masks, topology = operator.read(jnp.asarray(state))
    shadow = np.asarray(operator.residual_shadow_mask(jnp.asarray(state)), dtype=bool)
    topology_record = certificate._topology(operator, state)
    grid_coordinate = np.asarray(machine.node, dtype=np.float64)
    wall = np.asarray(machine.wall_node, dtype=np.float64)
    distance = _distance_to_wall(grid_coordinate, wall)
    private_rows = []
    for index in np.flatnonzero(raw_private):
        private_rows.append(
            {
                "index": int(index),
                "radius_m": float(grid_coordinate[index, 0]),
                "height_m": float(grid_coordinate[index, 1]),
                "domain_label": PlasmaDomain(int(raw.masks.label[index])).name,
                "psi_norm": float(raw.masks.psi_norm[index]),
                "component_id": int(labels[index]),
                "distance_to_wall_polygon_m": float(distance[index]),
            }
        )
    return {
        "raw_connectivity_rule": (
            "classify_domains marks closed and disconnected carriers PRIVATE_FLUX; "
            "nova/equilibrium/domain.py:206-212 classify_domains"
        ),
        "physical_rule": (
            "saddle_qualified_domains retains PRIVATE_FLUX only behind a finite "
            "admitted saddle; limited reads relabel provisional carriers CORE at "
            "nova/equilibrium/domain.py:219-240 and apply the rule to the forward "
            "read at nova/equilibrium/forward_operator.py:1892-1905"
        ),
        "topology": topology_record,
        "raw_private_count": int(np.count_nonzero(raw_private)),
        "raw_private_carriers": private_rows,
        "qualified_private_count": int(np.count_nonzero(qualified_masks.private_flux)),
        "flood_shadow_count": int(
            np.count_nonzero(shadow[: operator.grid.node_number])
        ),
        "wall_shadow_count": int(
            np.count_nonzero(
                shadow[operator.grid.node_number : operator.physical_node_number]
            )
        ),
        "all_physical_carriers_enter_residual": bool(
            not np.any(shadow[: operator.physical_node_number])
        ),
        "selected_axis_rz_m": np.asarray(topology.axis).tolist(),
        "selected_x_point_rz_m": (
            np.asarray(topology.x_point).tolist()
            if np.all(np.isfinite(np.asarray(topology.x_point)))
            else None
        ),
    }


def _render(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.4), constrained_layout=True)
    for row_index, row in enumerate(rows):
        shared = np.concatenate((row["terminal"], row["analytic"]))
        levels = poloidal.contour_levels(shared, count=10)
        for column, state_name in enumerate(("terminal", "analytic")):
            axis = axes[row_index, column]
            radial, height, field = certificate._raster_field(
                row["coordinates"], row[state_name], row["wall"]
            )
            poloidal.draw_flux_contours(axis, radial, height, field, levels)
            poloidal.draw_wall(axis, units=(row["wall"],))
            poloidal.draw_boundary(axis, row["boundary"][:, 0], row["boundary"][:, 1])
            topology = row[f"{state_name}_census"]["topology"]
            poloidal.draw_nulls(
                axis,
                magnetic_axis=topology["axis_rz_m"],
                x_points=topology["x_point_rz_m"],
                contain=(row["wall"],),
            )
            provisional = row[f"{state_name}_census"]["raw_private_carriers"]
            if provisional:
                points = np.asarray(
                    [[item["radius_m"], item["height_m"]] for item in provisional]
                )
                axis.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=12,
                    facecolors="none",
                    edgecolors="#b13f8c",
                    linewidths=0.8,
                )
            poloidal_axes(axis)
            axis.set_title(
                f"{abs(row['requested_cells'])} cells · {state_name} · "
                f"provisional {len(provisional)} · shadow 0",
                fontsize=9,
            )
    path = output_root / "limited-shadow-census.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "path": str(path.relative_to(ROOT)),
        "src": (
            "/nova/figures/cut-cell-current-attribution/limited-shadow/"
            "limited-shadow-census.png"
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _run(output_root: Path) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the limited-row shadow census requires binary64")
    original_mode = support_clip_mode()
    rows = []
    render_rows = []
    try:
        set_support_clip_mode("chord")
        for requested_cells in REQUESTED_CELLS:
            machine, operator, coordinates, analytic, exterior_cache = _rebuild(
                requested_cells
            )
            part = json.loads(_part_path(requested_cells).read_text(encoding="utf-8"))
            terminal = np.asarray(part["render_data"]["terminal_flux_wb"])
            persisted_coordinates = np.asarray(part["render_data"]["coordinates_rz_m"])
            np.testing.assert_array_equal(coordinates, persisted_coordinates)
            terminal_census = _state_census(operator, machine, terminal)
            analytic_census = _state_census(operator, machine, analytic)
            row = {
                "requested_cells": requested_cells,
                "realised_cells": len(machine.node),
                "source_part": str(_part_path(requested_cells).relative_to(ROOT)),
                "exterior_cache": exterior_cache,
                "terminal": terminal_census,
                "analytic": analytic_census,
            }
            rows.append(row)
            _write_json(
                output_root / "parts" / f"weak-cells-{abs(requested_cells)}.json",
                row,
            )
            carrier_case, _source_case, exact = certificate._case(CASE_NAME)
            render_rows.append(
                {
                    "requested_cells": requested_cells,
                    "coordinates": coordinates,
                    "wall": np.asarray(machine.wall_node),
                    "boundary": certificate._boundary(CASE_NAME, exact),
                    "terminal": terminal,
                    "analytic": analytic,
                    "terminal_census": terminal_census,
                    "analytic_census": analytic_census,
                }
            )
            _write_json(
                output_root / "receipt.json",
                {
                    "schema": "nova.limited-row-shadow-census",
                    "source_revision": _source_revision(),
                    "completed": False,
                    "rows": rows,
                },
            )
        figure = _render(output_root, render_rows)
    finally:
        set_support_clip_mode(original_mode)
    receipt = {
        "schema": "nova.limited-row-shadow-census",
        "source_revision": _source_revision(),
        "completed": True,
        "rows": rows,
        "figure": figure,
        "acceptance": {
            "limited_terminal_flood_shadow_empty": all(
                row["terminal"]["flood_shadow_count"] == 0 for row in rows
            ),
            "limited_analytic_flood_shadow_empty": all(
                row["analytic"]["flood_shadow_count"] == 0 for row in rows
            ),
            "limited_terminal_null_census": all(
                row["terminal"]["topology"]["o_candidate_count"] == 1
                and row["terminal"]["topology"]["x_candidate_count"] == 0
                for row in rows
            ),
            "limited_analytic_null_census": all(
                row["analytic"]["topology"]["o_candidate_count"] == 1
                and row["analytic"]["topology"]["x_candidate_count"] == 0
                for row in rows
            ),
        },
    }
    _write_json(output_root / "receipt.json", receipt)
    if not all(receipt["acceptance"].values()):
        raise RuntimeError(f"limited shadow acceptance failed: {receipt['acceptance']}")
    print(
        "LIMITED_SHADOW_CENSUS_EXIT=0 "
        + json.dumps(receipt["acceptance"], sort_keys=True),
        flush=True,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--solve-gate", action="store_true")
    parser.add_argument("--cpu-delta-arm", choices=("base", "after"))
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument(
        "--cpu-report-root",
        type=Path,
        default=Path(
            "/home/ITER/mcintos/.config/reckon/crew/reports/nova/"
            "s19-handoff/limited-shadow/cpu-delta"
        ),
    )
    arguments = parser.parse_args()
    if arguments.dry_run:
        print(
            "LIMITED_SHADOW_CENSUS_DRY_RUN rows="
            + ",".join(str(abs(value)) for value in REQUESTED_CELLS)
        )
        return
    if arguments.solve_gate:
        _solve_gate(arguments.output_root)
        return
    if arguments.cpu_delta_arm:
        _cpu_delta_arm(
            arguments.cpu_delta_arm,
            arguments.cpu_report_root,
            arguments.base_revision,
        )
        return
    _run(arguments.output_root)


if __name__ == "__main__":
    main()
