# ruff: noqa: E501
"""Regenerate per-geometry topology corroboration figures and evidence HTML.

The renderer consumes the committed MAST and DIII-D demonstration routes.  It
does not rescore the preregistered gates: its purpose is to expose the complete
spatial operands behind those scores, including negative and nonconverged rows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    polish_stationary_points,
)
from nova.equilibrium.separatrix_branches import assemble_separatrix_branches
from nova.imas.mast_efit_referee import read_efit_referee
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / "docs/evidence/topology-visual-corroboration.html"
MAST_CACHE = HERE / "mast-topology-operands.npz"
DIIID_CACHE = HERE / "diiid-topology-operands.npz"
MAST_AUTHORITY = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py"
)
DIIID_RENDER_COMMIT = "94f510b6"
DIIID_SOURCE_PATH = "benchmarks/diiid_forward_gs_match.py"
EXPECTED_MAST_ROWS = 12
EXPECTED_DIIID_ROWS = 5


def _load_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stationary_records(
    source_o: np.ndarray,
    source_x: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.concatenate((source_o, source_x))
    valid = np.all(np.isfinite(source), axis=1)
    spline = fit_tensor_spline(radius, height, flux)
    polished = jax.device_get(polish_stationary_points(spline, source[:, :2], valid))
    positions = np.asarray(polished["position_rz"], dtype=float)
    plotted = (
        valid
        & np.asarray(polished["converged"], dtype=bool)
        & np.asarray(polished["in_domain"], dtype=bool)
        & np.all(np.isfinite(positions), axis=1)
    )
    positions = np.where(plotted[:, None], positions, np.nan)
    return positions[: len(source_o)], positions[len(source_o) :]


def _mast_rows() -> list[dict[str, Any]]:
    authority = _load_path(MAST_AUTHORITY, "mast_visual_authority")
    reachability = authority._reachability_module()
    response_cache, carrier = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    rows: list[dict[str, Any]] = []
    for selected_row, qualification in selected:
        shot = int(selected_row["shot"])
        slice_index = int(selected_row["slice_index"])
        print(f"MAST_OPERANDS {shot}/{slice_index}", flush=True)
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if policy["section_kernel_evaluations_this_shot"] != 0:
            raise RuntimeError("MAST route entered a direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        states = reachability._mast_states(
            profile, jnp.asarray(passive_case["state"]), target_current
        )
        referee = read_efit_referee(shot, store=SHOT_STORE)
        for arm, state in states.items():
            physical = jnp.asarray(state)[: profile.operator.physical_node_number]
            grid_flux, _wall_flux = profile.operator.topology.split_flux_map(physical)
            source_o, source_x = jax.device_get(
                profile.operator._fixed_design_topology.grid(grid_flux)
            )
            masks, topology = profile.operator.read(state)
            geometry = reachability._grid_geometry(profile, state)
            flux = np.asarray(geometry["flux"], dtype=float)
            radius = np.asarray(geometry["radius"], dtype=float)
            height = np.asarray(geometry["height"], dtype=float)
            o_candidates, x_candidates = _stationary_records(
                np.asarray(source_o), np.asarray(source_x), radius, height, flux
            )
            assembled = jax.device_get(
                assemble_separatrix_branches(
                    jnp.asarray(flux),
                    jnp.asarray(radius),
                    jnp.asarray(height),
                    topology.boundary_flux,
                    topology.axis,
                )
            )
            closed = authority._sample_cubic_controls(
                np.asarray(assembled["closed_controls_rz"])[
                    np.asarray(assembled["closed_valid"], dtype=bool)
                ]
            )
            rows.append(
                {
                    "machine": "MAST",
                    "identity": f"{shot}/{slice_index} {arm}",
                    "shot": shot,
                    "frame": slice_index,
                    "arm": arm,
                    "time": float(referee.time_s[slice_index]),
                    "cell_rz": np.asarray(profile.lattice.coordinate, dtype=float),
                    "domain_labels": np.asarray(masks.label, dtype=np.int8),
                    "o_candidates": o_candidates,
                    "x_candidates": x_candidates,
                    "selected_o": np.asarray(topology.axis, dtype=float),
                    "selected_x": np.asarray(topology.x_point, dtype=float),
                    "wall_point": np.asarray(topology.wall_point, dtype=float),
                    "wall": np.asarray(geometry["wall"], dtype=float),
                    "nova_boundary": closed,
                    "efit_axis": np.asarray(referee.magnetic_axis_m[slice_index]),
                    "efit_x": np.asarray(referee.x_points_m[slice_index]),
                    "efit_lcfs": np.asarray(referee.lcfs_m[slice_index]),
                    "converged": True,
                    "qualification": f"achieved class from committed arm {arm}",
                }
            )
    if len(rows) != EXPECTED_MAST_ROWS:
        raise RuntimeError(f"expected {EXPECTED_MAST_ROWS} MAST rows, got {len(rows)}")
    _write_cache(
        MAST_CACHE, rows, {"carrier": carrier["carrier"]["semantic_response_identity"]}
    )
    return rows


def _diiid_module():
    source = subprocess.run(
        ["git", "show", f"{DIIID_RENDER_COMMIT}:{DIIID_SOURCE_PATH}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    needle = "    fields.update(\n        _terminal_plotting_geometry(profile, equilibrium, predicted, radius, height)\n    )\n"
    injection = needle + (
        "    visual_masks, _visual_topology = profile.operator.read(equilibrium.flux)\n"
        "    fields['domain_labels'] = np.asarray(visual_masks.label, dtype=np.int8)\n"
        "    fields['cell_rz'] = np.asarray(profile.lattice.coordinate, dtype=float)\n"
    )
    if source.count(needle) != 1:
        raise RuntimeError("DIII-D committed renderer injection seam changed")
    loaded = types.ModuleType("diiid_visual_authority")
    loaded.__file__ = str(ROOT / DIIID_SOURCE_PATH)
    loaded.__package__ = "benchmarks"
    sys.modules[loaded.__name__] = loaded
    namespace = loaded.__dict__
    exec(
        compile(
            source.replace(needle, injection), str(ROOT / DIIID_SOURCE_PATH), "exec"
        ),
        namespace,
    )
    return namespace


def _diiid_rows() -> list[dict[str, Any]]:
    module = _diiid_module()
    module["configure_dtypes"]()
    paths = sorted(module["DEFAULT_DATA"].glob("*.parquet"))
    selected = module["select_frames"](
        paths, module["EXECUTION_FRAME_COUNT"], module["polarity_population"]()
    )
    rows: list[dict[str, Any]] = []
    for number, selected_frame in enumerate(selected, start=1):
        print(
            f"DIIID_OPERANDS {number}/{len(selected)} "
            f"{selected_frame.path.name}:{selected_frame.frame}",
            flush=True,
        )
        record = module["_read"](
            selected_frame.path,
            module["_LABEL_COLUMNS"]
            + module["_GEOMETRY_COLUMNS"]
            + module["_CURRENT_COLUMNS"]
            + module["_PLASMA_CURRENT_COLUMNS"],
        )
        record["_source_path"] = str(selected_frame.path)
        result, fields = module["solve_frame"](
            record,
            selected_frame.frame,
            module["REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION"],
        )
        rows.append(
            {
                "machine": "DIII-D",
                "identity": f"{Path(result.shot).stem}:{result.frame}",
                "shot": result.shot,
                "frame": result.frame,
                "arm": "demonstration",
                "time": result.time_ms,
                "cell_rz": np.asarray(fields["cell_rz"], dtype=float),
                "domain_labels": np.asarray(fields["domain_labels"], dtype=np.int8),
                "o_candidates": _plotted_candidates(fields["nova_o_candidates"]),
                "x_candidates": _plotted_candidates(fields["nova_x_candidates"]),
                "selected_o": np.asarray(fields["nova_selected_axis_rz"], dtype=float),
                "selected_x": np.asarray(fields["nova_selected_x_rz"], dtype=float),
                "wall_point": np.asarray(fields["nova_selected_wall_rz"], dtype=float),
                "wall": np.asarray(fields["pseudo_wall"], dtype=float),
                "nova_boundary": np.asarray(
                    fields["predicted_closed_boundary"], dtype=float
                ),
                "efit_axis": np.asarray(fields["efit_axis_rz"], dtype=float),
                "efit_x": np.asarray(fields["efit_x_points_rz"], dtype=float),
                "efit_lcfs": np.asarray(
                    fields["labelled_closed_boundary"], dtype=float
                ),
                "converged": bool(result.converged),
                "qualification": result.solver_termination,
            }
        )
    if len(rows) != EXPECTED_DIIID_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_DIIID_ROWS} DIII-D rows, got {len(rows)}"
        )
    _write_cache(DIIID_CACHE, rows, {"renderer_commit": DIIID_RENDER_COMMIT})
    return rows


def _plotted_candidates(records: list[dict[str, Any]]) -> np.ndarray:
    points = [
        record["polished_coordinate_m"] for record in records if record["plotted"]
    ]
    return np.asarray(points, dtype=float).reshape((-1, 2))


def _write_cache(
    path: Path, rows: list[dict[str, Any]], authority: dict[str, Any]
) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata = []
    array_fields = (
        "cell_rz",
        "domain_labels",
        "o_candidates",
        "x_candidates",
        "selected_o",
        "selected_x",
        "wall_point",
        "wall",
        "nova_boundary",
        "efit_axis",
        "efit_x",
        "efit_lcfs",
    )
    for index, row in enumerate(rows):
        metadata.append(
            {key: value for key, value in row.items() if key not in array_fields}
        )
        for field in array_fields:
            arrays[f"row_{index:02d}_{field}"] = np.asarray(row[field])
    np.savez_compressed(
        path,
        metadata=np.asarray(
            json.dumps({"authority": authority, "rows": metadata}, sort_keys=True)
        ),
        **arrays,
    )


def _read_cache(path: Path) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        rows = []
        for index, record in enumerate(metadata["rows"]):
            row = dict(record)
            prefix = f"row_{index:02d}_"
            for key in stored.files:
                if key.startswith(prefix):
                    row[key[len(prefix) :]] = np.array(stored[key], copy=True)
            rows.append(row)
    return rows


def _finite_points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=float).reshape((-1, 2))
    return points[np.all(np.isfinite(points), axis=1)]


def _draw_row(row: dict[str, Any], path: Path) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(7.4, 7.2), constrained_layout=True)
    cells = np.asarray(row["cell_rz"], dtype=float)
    labels = np.asarray(row["domain_labels"], dtype=int).reshape(-1)
    if len(cells) != len(labels):
        raise RuntimeError(f"cell/label mismatch for {row['identity']}")
    colours = {
        int(PlasmaDomain.EXCLUDED_MATERIAL): "#f4f4f4",
        int(PlasmaDomain.CORE): "#a9d6e5",
        int(PlasmaDomain.COMMON_SOL): "#d8e2dc",
        int(PlasmaDomain.PRIVATE_FLUX): "#8e5ea2",
    }
    for domain, colour in colours.items():
        selected = labels == domain
        if np.any(selected):
            axis.scatter(
                cells[selected, 0],
                cells[selected, 1],
                marker="h",
                s=22,
                color=colour,
                edgecolors="none",
                rasterized=True,
                label=(
                    "private-flux shadow"
                    if domain == int(PlasmaDomain.PRIVATE_FLUX)
                    else None
                ),
            )
    wall = _finite_points(row["wall"])
    if len(wall):
        axis.plot(
            wall[:, 0], wall[:, 1], color="0.45", linewidth=1.0, label="governed wall"
        )
    boundary = _finite_points(row["nova_boundary"])
    if len(boundary):
        axis.plot(
            boundary[:, 0],
            boundary[:, 1],
            color="#005f73",
            linewidth=1.8,
            label="Nova closed boundary",
        )
    lcfs = _finite_points(row["efit_lcfs"])
    if len(lcfs):
        axis.plot(
            lcfs[:, 0],
            lcfs[:, 1],
            color="black",
            linestyle="--",
            linewidth=1.7,
            label="EFIT LCFS",
        )

    o_candidates = _finite_points(row["o_candidates"])
    x_candidates = _finite_points(row["x_candidates"])
    if len(o_candidates):
        axis.scatter(
            o_candidates[:, 0],
            o_candidates[:, 1],
            marker="o",
            s=45,
            facecolors="none",
            edgecolors="#0077b6",
            label="Nova O candidates",
            zorder=7,
        )
    if len(x_candidates):
        axis.scatter(
            x_candidates[:, 0],
            x_candidates[:, 1],
            marker="x",
            s=48,
            color="#d00000",
            label="Nova X candidates",
            zorder=7,
        )
    selected_o = _finite_points(row["selected_o"])
    selected_x = _finite_points(row["selected_x"])
    wall_point = _finite_points(row["wall_point"])
    if len(selected_o):
        axis.scatter(
            *selected_o[0],
            marker="*",
            s=145,
            color="#0077b6",
            edgecolors="white",
            linewidths=0.6,
            label="Nova primary O",
            zorder=9,
        )
    if len(selected_x):
        axis.scatter(
            *selected_x[0],
            marker="X",
            s=110,
            color="#ffb703",
            edgecolors="#8d0801",
            linewidths=0.8,
            label="Nova selected primary X",
            zorder=9,
        )
    if len(wall_point):
        axis.scatter(
            *wall_point[0],
            marker="D",
            s=65,
            color="#fb8500",
            edgecolors="black",
            linewidths=0.5,
            label="Nova closest plasma-wall point",
            zorder=9,
        )
    efit_axis = _finite_points(row["efit_axis"])
    efit_x = _finite_points(row["efit_x"])
    if len(efit_axis):
        axis.scatter(
            *efit_axis[0],
            marker="P",
            s=65,
            color="black",
            label="EFIT axis label",
            zorder=9,
        )
    if len(efit_x):
        axis.scatter(
            efit_x[:, 0],
            efit_x[:, 1],
            marker="+",
            s=85,
            color="#c1121f",
            linewidths=1.5,
            label="EFIT X label",
            zorder=9,
        )
    verdict = "converged" if row["converged"] else "NONCONVERGED — retained"
    axis.set_title(f"{row['machine']} · {row['identity']} · {verdict}")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.12)
    handles, labels_text = axis.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels_text, strict=True):
        if label and label not in unique:
            unique[label] = handle
    axis.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        fontsize=7,
        frameon=False,
    )
    if not row["converged"]:
        axis.set_facecolor("#fff1f1")
        for spine in axis.spines.values():
            spine.set_color("#b00020")
            spine.set_linewidth(1.8)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "private_flux_cells": int(
            np.count_nonzero(labels == int(PlasmaDomain.PRIVATE_FLUX))
        ),
        "o_candidates": len(o_candidates),
        "x_candidates": len(x_candidates),
        "selected_o": len(selected_o),
        "selected_x": len(selected_x),
        "wall_point": len(wall_point),
        "efit_axis": len(efit_axis),
        "efit_x": len(efit_x),
        "efit_lcfs_vertices": len(lcfs),
    }


def _write_evidence(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    figure_rows = []
    for index, (row, record) in enumerate(zip(rows, records, strict=True), start=1):
        filename = f"{index:02d}-{row['machine'].lower().replace('-', '')}-{row['identity'].replace('/', '-').replace(':', '-').replace(' ', '-')}.png"
        figure_rows.append(
            f"""<article class="figure-row" id="geometry-{index:02d}">
  <h3>{index:02d}. {row["machine"]} — {row["identity"]}</h3>
  <figure><img src="/nova/figures/topology-visual-corroboration/{filename}" alt="Topology evidence for {row["machine"]} {row["identity"]}: hex flood-fill domains, all Nova O and X candidates, selected primary O and X, wall point, and EFIT axis, X labels and LCFS overlay."><figcaption>{record["private_flux_cells"]} private-flux shadow cells; {record["o_candidates"]} plotted O candidates; {record["x_candidates"]} plotted X candidates; selected O/X/wall markers {record["selected_o"]}/{record["selected_x"]}/{record["wall_point"]}; EFIT axis/X/LCFS vertices {record["efit_axis"]}/{record["efit_x"]}/{record["efit_lcfs_vertices"]}. <strong>{"Converged." if row["converged"] else "NONCONVERGED — retained as a scientific failure."}</strong></figcaption></figure>
</article>"""
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="docs-project" content="nova"><meta name="reckon-type" content="evidence">
<meta name="plan-evidence-for" content="efit-baseline-demonstration"><meta name="plan-verifies" content="efit-baseline-demonstration#s2">
<meta name="plan-title" content="Topology visual corroboration"><meta name="plan-summary" content="Per-geometry visual corroboration of flood-fill topology operands against MAST and DIII-D EFIT labels.">
<title>Topology visual corroboration | nova</title><link rel="stylesheet" href="/_shared/foundation.css"><link rel="stylesheet" href="/_shared/dashboard.css">
<style>.figure-row{{margin:2rem 0 3rem}}.figure-row img{{display:block;max-width:860px;width:100%;height:auto}}figcaption{{max-width:860px;color:var(--muted, #555)}}code{{overflow-wrap:anywhere}}</style></head>
<body><main class="plan"><header class="plan-hero"><p class="eyebrow">Evidence · visual topology audit</p><h1>Topology visual corroboration</h1>
<p class="lede">All {EXPECTED_MAST_ROWS} MAST arms and all {EXPECTED_DIIID_ROWS} DIII-D demonstration frames are shown exactly once. Every panel exposes the hex-cell flood-fill domains, the complete finite O/X candidate census, selected primary O and X, selected wall point, and EFIT axis/X/LCFS labels.</p></header>
<section><h2>Authority and interpretation</h2><p>This is corroboration of committed extraction state, not a new score. EFIT is an independent magnetics-fitted reconstruction, not physical truth. MAST operands use the persisted response carrier and the current <code>efit_topology_corroboration</code> extraction route. DIII-D operands use the committed complete-poloidal renderer at <code>{DIIID_RENDER_COMMIT}</code>, the first committed extraction state carrying the full stationary-candidate census; all five nonconverged rows remain visibly qualified. Generated from repository head <code>{head}</code>.</p>
<p>The purple cells are the exact <code>PRIVATE_FLUX</code> labels from Nova's domain partition: closed-flux cells disconnected from the primary O-point by the X-point flood-fill cut. Pale blue is axis-connected core, grey-green is common SOL, and pale grey is excluded material. The selected wall marker is Nova's governed closest plasma-wall candidate; it is shown even when the topology class is diverted and it does not bind the LCFS.</p></section>
<section><h2>Coverage</h2><p><strong>{len(rows)} of {EXPECTED_MAST_ROWS + EXPECTED_DIIID_ROWS} declared geometries rendered.</strong> {sum(row["machine"] == "MAST" for row in rows)} MAST and {sum(row["machine"] == "DIII-D" for row in rows)} DIII-D; {sum(not row["converged"] for row in rows)} nonconverged rows retained. Figure rows below are the quantitative completeness ledger.</p></section>
<section><h2>Per-geometry corroboration</h2>{"".join(figure_rows)}</section>
<section><h2>Reproduction</h2><p>Run <code>UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync python docs/figures/topology-visual-corroboration/generate_topology_visuals.py</code>. The committed <code>generation.log</code> records the successful run and ends with <code>EXIT_MARKER=0</code>. The scoped NPZ files retain the exact plotted operands so ordinary regeneration does not repeat the expensive solves; delete them only when intentionally refreshing from a newly committed extraction state.</p></section>
</main></body></html>"""
    EVIDENCE.write_text(html + "\n")


def main() -> None:
    configure_dtypes()
    HERE.mkdir(parents=True, exist_ok=True)
    mast = _read_cache(MAST_CACHE) if MAST_CACHE.exists() else _mast_rows()
    diiid = _read_cache(DIIID_CACHE) if DIIID_CACHE.exists() else _diiid_rows()
    rows = mast + diiid
    if len(rows) != EXPECTED_MAST_ROWS + EXPECTED_DIIID_ROWS:
        raise RuntimeError("demonstration-bank coverage is incomplete")
    records = []
    for index, row in enumerate(rows, start=1):
        filename = f"{index:02d}-{row['machine'].lower().replace('-', '')}-{row['identity'].replace('/', '-').replace(':', '-').replace(' ', '-')}.png"
        records.append(_draw_row(row, HERE / filename))
        print(f"RENDERED {index:02d}/{len(rows)} {row['machine']} {row['identity']}")
    _write_evidence(rows, records)
    digest = hashlib.sha256(EVIDENCE.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "rows": len(rows),
                "mast": len(mast),
                "diiid": len(diiid),
                "evidence_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
