"""Render line-contour evidence from the retained analytic certificates.

The established evidence renderer is loaded as a read-only implementation
source.  This wrapper selects one durable figure at a time so each completed
render can be committed independently.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
# The committed renderer in the same figure tree, never an unmerged
# worktree's copy by absolute path: the source must be reachable from this
# repository alone.
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "mechanism-evidence"
    / "render_mechanism_evidence.py"
)
RECEIPT = Path(
    os.environ.get("CONTOUR_RECEIPT_PATH", OUTPUT / "contour-evidence-receipt.json")
)


def _renderer() -> Any:
    spec = importlib.util.spec_from_file_location("contour_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer source: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.OUTPUT = OUTPUT
    return module


def _build_diverted_terminal(source: Any, cells: int) -> dict[str, Any]:
    """Build the exact diverted state across receipt-shape revisions."""
    case_name = "diverted-jump-bearing"
    carrier_case, source_case, exact = source.certificate._case(case_name)
    machine = source.oracle_fixture.cached_machine(
        carrier_case,
        cells,
        wall_nodes=source.oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic_state = source.certificate._exact_state(case_name, exact, coordinates)
    empty_operator = source.oracle_fixture.forward_operator(source_case, machine)
    exact_physical = source.oracle_fixture.exact_current_moments(
        source_case, empty_operator, analytic_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = source.oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = source.oracle_fixture.forward_operator(
        source_case, machine, analytic_state - exact_internal
    )
    profile = source.ForwardProfile(
        operator,
        source.StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=source.recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = (
        source.certificate._closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
    )
    seed, _requested_class, _seed_receipt = source.certificate._production_seed(
        profile,
        case_name,
        source.COLD_START_TARGET_CURRENT_A,
        centroid,
        current_receipt,
    )
    request = source.certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"contour-evidence-clip:{cells}",
    )
    started = source.perf_counter()
    receipt = profile.solve(request)
    state = np.asarray(receipt.equilibrium.flux, dtype=np.float64)
    jax.block_until_ready(state)
    fixed_point = receipt.equilibrium.fixed_point
    return {
        "machine": machine,
        "operator": operator,
        "state": state,
        "target_current": float(target_current),
        "residual": float(fixed_point.residual),
        "refusals": int(getattr(fixed_point, "topology_trial_refusals", 0)),
        "boundary": source.certificate._boundary(case_name, exact),
        "wall": np.asarray(machine.wall_node, dtype=np.float64),
        "solve_seconds": source.perf_counter() - started,
    }


def _read_receipt() -> dict[str, Any]:
    if not RECEIPT.exists():
        return {"certificate": [], "clipped_cells": []}
    payload = json.loads(RECEIPT.read_text())
    if not isinstance(payload, dict):
        raise ValueError("contour evidence receipt must be an object")
    return payload


def _write_receipt(payload: dict[str, Any]) -> None:
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _base_payload() -> dict[str, Any]:
    payload = _read_receipt()
    payload.update(
        {
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_platform": jax.default_backend(),
            "renderer_source": str(SOURCE),
        }
    )
    payload.setdefault("certificate", [])
    payload.setdefault("clipped_cells", [])
    return payload


def _section_records(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The records list of a receipt section, tolerating a curated dict shape.

    The curated contour receipt stores each section as an object with a
    ``records`` list (carrying provenance keys like ``record``/``figure``);
    the incremental wrapper stores a bare list.  Both are updated in place.
    """
    section = payload.get(key)
    if isinstance(section, dict):
        records = section.setdefault("records", [])
        if not isinstance(records, list):
            raise ValueError(f"receipt {key}.records must be a list")
        return records
    if section is None:
        section = payload[key] = []
    if not isinstance(section, list):
        raise ValueError(f"receipt {key} must be a list or an object with records")
    return section


def _render_certificate(source: Any, case: str, cells: int) -> None:
    record = source._build_certificate_case(case, cells)
    rendered = source._draw_certificate(record)
    rendered["retained_h200_terminal_residual"] = {
        ("weak-rotation-reactor-static", -110): 7.447526861632619e-3,
        ("weak-rotation-reactor-static", -300): 5.5589300334190185e-3,
        ("weak-rotation-reactor-static", -500): 1.2464509730622084e-3,
        ("weak-rotation-reactor-static", -1000): 5.4014429054834836e-3,
        ("moderate-rotation-conventional-static", -110): 1.2457640060526081e-2,
        ("moderate-rotation-conventional-static", -300): 8.455791545012673e-3,
    }[(case, cells)]
    payload = _base_payload()
    records = _section_records(payload, "certificate")
    records[:] = [
        item
        for item in records
        if (item.get("case"), item.get("requested_cells")) != (case, cells)
    ]
    records.append({"case": case, "requested_cells": cells, **rendered})
    _write_receipt(payload)
    print(f"PERSISTED certificate {case} {cells}", flush=True)


def _render_clips(source: Any, cells: int) -> None:
    rendered = source._draw_clipped_cells(cells)
    payload = _base_payload()
    records = _section_records(payload, "clipped_cells")
    records[:] = [
        item
        for item in records
        if item.get("requested_cells") != cells
    ]
    records.append({"requested_cells": cells, **rendered})
    _write_receipt(payload)
    print(f"PERSISTED clipped cells {cells}", flush=True)


def _clip_area_differences(source: Any, cells: int) -> None:
    """Compare first-order support polygons against the analytic separatrix."""
    from shapely import Polygon

    record = source._build_diverted_terminal(cells)
    operator = record["operator"]
    state = record["state"]
    masks, topology, sample_psi_norm, _profile_support = operator._support_partition(
        state
    )
    shared_psi_norm = (
        operator.shared_node_flux(state) - topology.axis_flux
    ) / topology.flux_span
    atomic_mesh = operator.moment_geometry.atomic_mesh
    support = jax.lax.cond(
        jnp.all(jnp.isfinite(topology.x_point)),
        lambda x_point: atomic_mesh.traced_clip(
            1.0 - shared_psi_norm, saddle_vertex=x_point
        ),
        lambda _x_point: atomic_mesh.traced_clip(1.0 - shared_psi_norm),
        topology.x_point,
    )
    area_fraction = np.asarray(support.area / support.full_area, dtype=float)
    cut_indices = np.flatnonzero(
        (area_fraction > 1.0e-12) & (area_fraction < 1.0 - 1.0e-12)
    )
    boundary = Polygon(np.asarray(record["boundary"], dtype=float))
    vertices = np.asarray(support.support_vertices, dtype=float)
    counts = np.asarray(support.vertex_count, dtype=int)
    raw_polygons = np.asarray(operator.moment_geometry.polygons, dtype=float)
    rows = []
    for index in cut_indices:
        raw = Polygon(raw_polygons[index])
        straight = Polygon(vertices[index, : counts[index]])
        traced = raw.intersection(boundary)
        difference = abs(float(straight.area) - float(traced.area))
        rows.append(
            {
                "cell_index": int(index),
                "cell_area_m2": float(raw.area),
                "straight_clip_area_m2": float(straight.area),
                "traced_separatrix_area_m2": float(traced.area),
                "absolute_area_difference_m2": difference,
                "cell_area_fraction": difference / float(raw.area),
            }
        )
    payload = _base_payload()
    payload["clip_area_differences"] = {
        "requested_cells": cells,
        "terminal_residual": record["residual"],
        "topology_trial_refusals": record["refusals"],
        "rows": rows,
    }
    _write_receipt(payload)
    print(f"PERSISTED clip area differences {cells}", flush=True)


def _render_cold(source: Any) -> None:
    payload = _base_payload()
    payload["cold_start"] = source._draw_cold_start()
    payload["legend"] = source._draw_legend()
    _write_receipt(payload)
    print("PERSISTED cold start and marker legend", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("certificate", "clip", "clip-metrics", "cold")
    )
    parser.add_argument("--case")
    parser.add_argument("--cells", type=int)
    arguments = parser.parse_args()
    configure_dtypes()
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("jax_enable_x64 is false")
    if np.dtype(jax.numpy.asarray(1.0).dtype) != np.dtype(np.float64):
        raise RuntimeError("JAX default dtype is not float64")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = _renderer()
    source._build_diverted_terminal = lambda cells: _build_diverted_terminal(
        source, cells
    )
    if arguments.mode == "certificate":
        if arguments.case is None or arguments.cells is None:
            raise ValueError("certificate mode requires --case and --cells")
        _render_certificate(source, arguments.case, arguments.cells)
    elif arguments.mode == "clip":
        if arguments.cells is None:
            raise ValueError("clip mode requires --cells")
        _render_clips(source, arguments.cells)
    elif arguments.mode == "clip-metrics":
        if arguments.cells is None:
            raise ValueError("clip-metrics mode requires --cells")
        _clip_area_differences(source, arguments.cells)
    else:
        _render_cold(source)


if __name__ == "__main__":
    main()
