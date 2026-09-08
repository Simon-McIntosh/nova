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
SOURCE = Path(
    "/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/"
    "s19-labeller/nia-poloidal-mechanism-evidence/docs/figures/"
    "null-identification-authority/mechanism-evidence/"
    "render_mechanism_evidence.py"
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
    payload["certificate"] = [
        item
        for item in payload["certificate"]
        if (item["case"], item["requested_cells"]) != (case, cells)
    ]
    payload["certificate"].append(
        {"case": case, "requested_cells": cells, **rendered}
    )
    _write_receipt(payload)
    print(f"PERSISTED certificate {case} {cells}", flush=True)


def _render_clips(source: Any, cells: int) -> None:
    rendered = source._draw_clipped_cells(cells)
    payload = _base_payload()
    payload["clipped_cells"] = [
        item for item in payload["clipped_cells"] if item["requested_cells"] != cells
    ]
    payload["clipped_cells"].append({"requested_cells": cells, **rendered})
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
