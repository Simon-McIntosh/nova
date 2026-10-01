"""Name the clip stage that discards the exact polygon for eight cut cells.

The unit-amplitude census records eight cells whose analytic separatrix cuts a
real polygon out of the atomic mesh, yet whose exact-mode support carries zero
area at both fixed states.  This probe attributes that loss stage by stage.

Two instruments run over the same exact-mode partition:

* a recorder installed on ``separatrix_clip._pack_traced_vertices`` reports the
  per-cell vertex count emitted by each packing call site in the traced clip,
  so every stage of the chain is read from the production arithmetic rather
  than reconstructed beside it;
* the boundary-level sign and the curved-level sign are evaluated on the cell
  vertices directly, which gives the edge crossings and the participation gate
  that feed the candidate set.

The recorder is validated against the production support: for a cell that does
trace a polygon the final recorded count must equal
``profile_support.vertex_count`` for that cell, and the recorded crossing count
must equal the number of participating curved-level edge crossings computed
from the vertex signs.  A stage whose count falls below three is the first
stage that cannot produce a polygon, because the shoelace moment of fewer than
three vertices has zero area.

Run one job on an ``all_debug`` partition with the shared interpreter; the
run log is committed beside the report.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import benchmarks.unit_amplitude_current_census as census  # noqa: E402
from nova.equilibrium import separatrix_clip as clip_module  # noqa: E402
from nova.equilibrium.forward_operator import (  # noqa: E402
    _ExactClipLevel,
    set_support_clip_mode,
)

OUT = Path(__file__).resolve().parents[1] / (
    "docs/figures/cut-cell-current-attribution/empty-exact-polygon"
)
NAMED_CELLS = (2, 35, 75, 89, 114, 126, 128, 134)
STATES = ("seed", "terminal")

# Packing call sites of the traced clip, keyed by the line that calls them.
# The line is read from the calling frame at trace time, so a stage is named by
# where it is emitted rather than by the order the tracer happens to reach it.
STAGE_CHAIN = (
    ("crossing_vertices", 1135, "crossing points packed from the straddling edges"),
    ("candidate_vertices", 1207, "inside vertices, crossings and saddle candidate"),
    ("deduped_support_vertices", 1224, "candidates after the duplicate collapse"),
    ("post_arc_vertices", 1327, "support vertices after the spline-gap arc expansion"),
    ("branch_vertices", 1366, "support vertices split into the two saddle branches"),
    ("saddle_chain_vertices", 1904, "saddle-chain support after the chain expansion"),
)
STAGE_LINES = {line: name for name, line, _doc in STAGE_CHAIN}
MINIMUM_POLYGON_VERTICES = 3

_RECORD = [False]
_RECORDS: dict[str, dict[int, list[int]]] = {}
_ORIGINAL_PACK = clip_module._pack_traced_vertices


def _sink(stage: str):
    def record(block):
        array = np.asarray(block)
        rows = array.reshape(-1, array.shape[-1])
        for row in rows:
            for cell, value in enumerate(row):
                _RECORDS.setdefault(stage, {}).setdefault(int(cell), []).append(
                    int(value)
                )

    return record


def _install_recorder() -> None:
    def wrapped(vertices, valid, capacity):
        packed, count = _ORIGINAL_PACK(vertices, valid, capacity)
        if _RECORD[0]:
            line = inspect.currentframe().f_back.f_lineno
            stage = STAGE_LINES.get(line, f"line_{line}")
            cells = count.shape[-1]
            jax.debug.callback(_sink(stage), jnp.asarray(count).reshape(-1, cells))
        return packed, count

    clip_module._pack_traced_vertices = wrapped


def _reset_recorder() -> None:
    _RECORDS.clear()
    _RECORD[0] = True


def _stop_recorder() -> None:
    _RECORD[0] = False


def _stage_count(stage: str, cell: int):
    """Return the smallest count the stage emitted for one cell."""
    values = _RECORDS.get(stage, {}).get(int(cell))
    return None if not values else min(values)


def _stage_all(stage: str, cell: int) -> list[int]:
    return list(_RECORDS.get(stage, {}).get(int(cell), []))


def _full_probe(operator, state: np.ndarray) -> dict[str, Any]:
    """Trace the exact partition from the whole state vector.

    The fixed-design read consumes the authored direct-sampling rows alongside
    the grid flux, so the state is passed whole; the census's own partition
    probe slices it and is not usable here.
    """
    set_support_clip_mode("exact")
    physical = jnp.asarray(state)
    base_masks, topology, _connected, _admitted = operator._fixed_design_read(physical)
    sample_flux = operator.sample_node_flux(jnp.asarray(state))
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    _reset_recorder()
    profile_support = operator._profile_support(
        base_masks, topology, physical, sample_psi_norm
    )
    # Force the traced program to run so the recorder's callbacks fire.
    area = np.asarray(profile_support.area, dtype=np.float64)
    vertex_count = np.asarray(profile_support.vertex_count, dtype=np.int64)
    included = np.asarray(profile_support.included, dtype=bool)
    _stop_recorder()
    if area.shape[0] != vertex_count.shape[0]:
        raise RuntimeError(
            "support area and vertex count disagree on the cell axis: "
            f"{area.shape} versus {vertex_count.shape}"
        )
    moment_masks = operator._moment_support_masks(base_masks, profile_support)
    curve = census._curve_probe(operator, base_masks, topology, sample_psi_norm)
    shared_flux = operator.shared_node_flux(physical)
    inside_boundary = np.asarray(
        operator.polarity * (shared_flux - topology.boundary_flux), dtype=np.float64
    )
    level = _ExactClipLevel(
        curve["surface"],
        curve["inside_coefficient"],
        operator._support_curve_centre,
        operator._support_curve_scale,
    )
    atomic_mesh = operator.moment_geometry.atomic_mesh
    node_coordinates = np.asarray(atomic_mesh.node_coordinates, dtype=np.float64)
    # The level evaluator carries one local row per cell, so it is applied to a
    # cell-batched vertex array: axis 0 is the cell, exactly as the traced clip
    # calls it. A flat node array would broadcast the local rows against every
    # vertex and return one level per cell-vertex pair.
    cell_nodes = np.asarray(atomic_mesh.cell_nodes)
    cell_points = jnp.asarray(node_coordinates)[jnp.asarray(cell_nodes)]
    curved_cell = np.asarray(level(cell_points), dtype=np.float64)
    if curved_cell.shape[0] != cell_nodes.shape[0]:
        raise RuntimeError(
            "curved level and cell count disagree: "
            f"{curved_cell.shape} versus {cell_nodes.shape}"
        )
    participation = (
        np.asarray(moment_masks.profile_participation, dtype=bool)
        | curve["vertex_participation"]
    )
    return {
        "area": area,
        "vertex_count": vertex_count,
        "included": included,
        "records": {stage: dict(values) for stage, values in _RECORDS.items()},
        "profile_participation": np.asarray(
            moment_masks.profile_participation, dtype=bool
        ),
        "vertex_participation": curve["vertex_participation"],
        "participation": participation,
        "inside_boundary": inside_boundary,
        "curved_cell": curved_cell,
        "node_coordinates": node_coordinates,
        "curved_cell_inside": curved_cell > 0.0,
        "boundary_flux_wb": float(np.asarray(topology.boundary_flux)),
        "boundary": inside_boundary > 0.0,
    }


def _cell_geometry(probe: dict[str, Any], mesh, cell: int) -> dict[str, Any]:
    """Return one cell's vertex signs, edge crossings and candidate set."""
    count = int(np.asarray(mesh.cell_vertex_count)[cell])
    index = np.asarray(mesh.cell_nodes)[cell][:count]
    if count < 3:
        raise RuntimeError(f"cell {cell} carries only {count} vertices")
    curved = probe["curved_cell_inside"][cell][:count]
    boundary = probe["boundary"][index]
    following = np.roll(np.arange(count), -1)
    straddle = curved != curved[following]
    participates = participation_flag(probe, cell)
    edge_crossing_index = [
        int(edge) for edge in np.flatnonzero(straddle & participates)
    ]
    straddle_without_participation = [
        int(edge) for edge in np.flatnonzero(straddle & ~participates)
    ]
    unique_crossing = len(edge_crossing_index)
    inside_curved = int(np.count_nonzero(curved))
    inside_boundary = int(np.count_nonzero(boundary))
    return {
        "cell": int(cell),
        "cell_vertex_count": count,
        "vertices_inside_boundary_level": inside_boundary,
        "vertices_inside_curved_level": inside_curved,
        "vertices_outside_curved_level": count - inside_curved,
        "edges": count,
        "straddling_edges_curved_level": int(np.count_nonzero(straddle)),
        "crossing_edges_participating": edge_crossing_index,
        "straddling_edges_blocked_by_participation": straddle_without_participation,
        "participation": bool(participation_flag(probe, cell)),
        "profile_participation": bool(probe["profile_participation"][cell]),
        "vertex_level_participation": bool(probe["vertex_participation"][cell]),
        "unique_crossing_edges": unique_crossing,
        "inside_vertices_on_edges": inside_curved,
        "candidate_vertex_count_from_inputs": inside_curved + unique_crossing,
        "curved_level_per_vertex": [
            float(value) for value in probe["curved_cell"][cell][:count]
        ],
        "boundary_level_per_vertex": [
            float(value) for value in probe["inside_boundary"][index]
        ],
    }


def participation_flag(probe: dict[str, Any], cell: int) -> bool:
    return bool(probe["participation"][cell])


def _cell_stage_count(records: dict[str, Any], stage: str, cell: int):
    """Smallest count one stage emitted for one cell, or None if it emitted none."""
    values = records.get(stage, {}).get(cell)
    return None if not values else int(min(values))


def _stage_row(probe: dict[str, Any], cell: int) -> dict[str, Any]:
    """Return the smallest vertex count each packing stage emitted for a cell."""
    records = probe["records"]
    reported = {
        stage: _cell_stage_count(records, stage, cell) for stage in STAGE_LINES.values()
    }
    for stage in records:
        if stage not in reported:
            reported[stage] = _cell_stage_count(records, stage, cell)
    return reported


def _first_empty_stage(
    probe: dict[str, Any], cell: int, geometry: dict[str, Any]
) -> dict[str, Any]:
    """Return the first stage whose vertex count cannot form a polygon."""
    stages = _stage_row(probe, cell)
    ordered = [(name, line, stages.get(name)) for name, line, _doc in STAGE_CHAIN]
    for name, line, count in ordered:
        if count is None:
            continue
        if count < MINIMUM_POLYGON_VERTICES:
            return {
                "stage": name,
                "line": line,
                "vertex_count": int(count),
                "reason": (
                    f"fewer than {MINIMUM_POLYGON_VERTICES} vertices, so the "
                    "shoelace moment has zero area"
                ),
            }
    area = float(probe["area"][cell])
    if area <= 0.0:
        return {
            "stage": "traced_polygon_moments",
            "line": 1405,
            "vertex_count": int(stages.get("branch_vertices") or 0),
            "reason": (
                "the branch shoelace area is zero, so included = area > 0.0 "
                "is false at line 1408"
            ),
        }
    if not bool(probe["included"][cell]):
        return {
            "stage": "overflow_gate",
            "line": 1409,
            "vertex_count": int(stages.get("branch_vertices") or 0),
            "reason": (
                "live vertex count exceeded the polygon capacity, so "
                "included = included & ~overflow is false at line 1409"
            ),
        }
    return {
        "stage": "included",
        "line": 1408,
        "vertex_count": int(stages.get("branch_vertices") or 0),
        "reason": "the polygon is traced; the cell carries a non-zero area",
    }


def _control_check(probe: dict[str, Any], mesh, cell: int) -> dict[str, Any]:
    """Cross-check the recorded stages against the production support."""
    stages = _stage_row(probe, cell)
    production = int(probe["vertex_count"][cell])
    recorded_final = stages.get("post_arc_vertices")
    return {
        "cell": int(cell),
        "production_vertex_count": production,
        "recorded_post_arc_vertex_count": (
            int(recorded_final) if recorded_final is not None else None
        ),
        "agrees": recorded_final is not None and int(recorded_final) == production,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Empty exact polygon per cut cell: the clip stage that discards it")
    lines.append("")
    lines.append(
        "Eight cells carry an analytic separatrix cutting a real polygon out of "
        "the atomic mesh, yet the exact-mode support reports zero area for them "
        "at both fixed states. This report names, per cell and per state, the "
        "vertex signs, the level-set crossings on the cell edges, the vertex "
        "count emitted by each packing stage of the traced clip, and the first "
        "stage that cannot form a polygon."
    )
    lines.append("")
    lines.append(f"- revision: `{payload['revision']}`")
    lines.append(f"- case: `{payload['case']}`")
    lines.append(f"- requested cells: `{payload['requested_cells']}`")
    lines.append(f"- realised cells: `{payload['realised_cells']}`")
    lines.append("")
    lines.append("## Stage chain of the traced clip")
    lines.append("")
    lines.append("| stage | line | what it packs |")
    lines.append("| --- | --- | --- |")
    for stage, line, doc in STAGE_CHAIN:
        lines.append(f"| `{stage}` | {line} | {doc} |")
    lines.append("")
    lines.append(
        "A stage emitting fewer than three vertices cannot form a polygon and "
        "its shoelace area is zero, which is what makes `included = area > 0.0` "
        "false."
    )
    lines.append("")
    for state in STATES:
        block = payload["states"][state]
        lines.append(f"## {state} state")
        lines.append("")
        lines.append(
            f"boundary level `{block['boundary_flux_wb']:.7f}` Wb; "
            f"profile participation on {block['participation_count']} of "
            f"{block['cell_count']} cells; exact total area "
            f"`{block['exact_area_total_m2']:.6e}` m^2."
        )
        lines.append("")
        lines.append(
            "| cell | verts | inside boundary | inside curved | straddling edges "
            "| participating crossings | blocked by participation | candidate "
            "| crossing | deduped | post-arc | branch | area m^2 | included "
            "| first empty stage |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
            "| --- | --- | --- | --- |"
        )
        for row in block["cells"]:
            stage_row = row["stages"]
            first = row["first_empty_stage"]
            lines.append(
                "| {cell} | {verts} | {bnd} | {crv} | {strad} | {cross} | {blocked} "
                "| {cand} | {c1} | {c2} | {c3} | {c4} | {area:.6e} | {inc} "
                "| {stage} (line {line}) |".format(
                    cell=row["cell"],
                    verts=row["cell_vertex_count"],
                    bnd=row["vertices_inside_boundary_level"],
                    crv=row["vertices_inside_curved_level"],
                    strad=row["straddling_edges_curved_level"],
                    cross=len(row["crossing_edges_participating"]),
                    blocked=len(row["straddling_edges_blocked_by_participation"]),
                    cand=row["candidate_vertex_count_from_inputs"],
                    c1=_format_count(stage_row.get("crossing_vertices")),
                    c2=_format_count(stage_row.get("candidate_vertices")),
                    c3=_format_count(stage_row.get("deduped_support_vertices")),
                    c4=_format_count(stage_row.get("post_arc_vertices")),
                    area=row["area_m2"],
                    inc=row["included"],
                    stage=first["stage"],
                    line=first["line"],
                )
            )
        lines.append("")
        lines.append("### Per-edge crossings")
        lines.append("")
        for row in block["cells"]:
            lines.append(
                "- cell {cell}: participating crossings on edges {cross}; "
                "straddling edges blocked by participation: {blocked}; "
                "level per vertex [boundary] {bnd} / [curved] {crv}".format(
                    cell=row["cell"],
                    cross=row["crossing_edges_participating"],
                    blocked=row["straddling_edges_blocked_by_participation"],
                    bnd=[round(value, 5) for value in row["boundary_level_per_vertex"]],
                    crv=[round(value, 5) for value in row["curved_level_per_vertex"]],
                )
            )
        lines.append("")
        lines.append("### First empty stage")
        lines.append("")
        for row in block["cells"]:
            first = row["first_empty_stage"]
            lines.append(
                "- cell {cell}: `{stage}` at `nova/equilibrium/separatrix_clip.py:"
                "{line}` with {count} vertices — {reason}".format(
                    cell=row["cell"],
                    stage=first["stage"],
                    line=first["line"],
                    count=first["vertex_count"],
                    reason=first["reason"],
                )
            )
        lines.append("")
    lines.append("## Recorder control")
    lines.append("")
    lines.append(
        "The recorder is checked against the production support on every cell: "
        "the post-arc recorded count must equal `profile_support.vertex_count`, "
        "and the recorded crossing count must equal the participating "
        "curved-level edge crossings computed from the vertex signs."
    )
    lines.append("")
    lines.append(
        "| state | cells where recorded post-arc count equals production "
        "vertex count | cells where recorded crossings equal computed crossings |"
    )
    lines.append("| --- | --- | --- |")
    for state in STATES:
        block = payload["states"][state]
        lines.append(
            f"| {state} | {block['control']['post_arc_matches']} of "
            f"{block['control']['cells_checked']} | "
            f"{block['control']['crossing_matches']} of "
            f"{block['control']['cells_checked']} |"
        )
    lines.append("")
    lines.append(f"![empty exact polygon stage funnel]({payload['figure_src']})")
    lines.append("")
    return "\n".join(lines)


def _format_count(value) -> str:
    return "n/a" if value is None else str(int(value))


def _state_block(
    operator,
    mesh,
    state: np.ndarray,
    name: str,
) -> dict[str, Any]:
    probe = _full_probe(operator, state)
    cells = []
    for cell in NAMED_CELLS:
        geometry = _cell_geometry(probe, mesh, cell)
        stages = _stage_row(probe, cell)
        record = dict(geometry)
        record.update(
            {
                "stages": stages,
                "area_m2": float(probe["area"][cell]),
                "included": bool(probe["included"][cell]),
                "first_empty_stage": _first_empty_stage(probe, cell, geometry),
            }
        )
        cells.append(record)
    controls = [_control_check(probe, mesh, cell) for cell in NAMED_CELLS]
    crossing_agreement = 0
    for cell in NAMED_CELLS:
        geometry = _cell_geometry(probe, mesh, cell)
        recorded = _cell_stage_count(probe["records"], "crossing_vertices", cell)
        if recorded is not None and recorded == len(
            geometry["crossing_edges_participating"]
        ):
            crossing_agreement += 1
    return {
        "cells": cells,
        "boundary_flux_wb": probe["boundary_flux_wb"],
        "cell_count": int(np.asarray(mesh.cell_vertex_count).shape[0]),
        "participation_count": int(np.count_nonzero(probe["participation"])),
        "exact_area_total_m2": float(np.sum(probe["area"])),
        "control": {
            "cells_checked": len(NAMED_CELLS),
            "post_arc_matches": sum(1 for item in controls if item["agrees"]),
            "crossing_matches": crossing_agreement,
            "per_cell": controls,
        },
        "probe": probe,
    }


def _figure(payload: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_labels = [stage for stage, _line, _doc in STAGE_CHAIN]
    figure, axes = plt.subplots(1, 3, figsize=(19.0, 5.6))
    colours = plt.cm.tab10(np.linspace(0.0, 0.9, len(NAMED_CELLS)))
    for axis, state in zip(axes[:2], STATES, strict=True):
        block = payload["states"][state]
        for colour, row in zip(colours, block["cells"], strict=True):
            values = [row["stages"].get(stage) for stage in stage_labels]
            values = [np.nan if value is None else value for value in values]
            axis.plot(
                range(len(stage_labels)),
                values,
                marker="o",
                color=colour,
                label=f"cell {row['cell']}",
            )
        axis.axhline(
            MINIMUM_POLYGON_VERTICES - 0.5,
            color="k",
            linestyle=":",
            linewidth=1.0,
        )
        axis.set_title(f"{state}: traced polygon vertices per clip stage")
        axis.set_xticks(range(len(stage_labels)))
        axis.set_xticklabels(stage_labels, rotation=30, ha="right")
        axis.set_ylabel("packed vertex count")
        axis.set_ylim(-0.8, max(6.0, axis.get_ylim()[1]))
        axis.legend(fontsize=7, ncol=2)
    axis = axes[2]
    width = 0.2
    positions = np.arange(len(NAMED_CELLS))
    series = (
        ("seed", "vertices_inside_boundary_level", ""),
        ("seed", "vertices_inside_curved_level", "//"),
        ("terminal", "vertices_inside_boundary_level", ""),
        ("terminal", "vertices_inside_curved_level", "//"),
    )
    for offset, (state, field, hatch) in enumerate(series):
        values = [row[field] for row in payload["states"][state]["cells"]]
        axis.bar(
            positions + (offset - 1.5) * width,
            values,
            width,
            hatch=hatch,
            label=f"{state}: {field.replace('vertices_inside_', '')}",
        )
    axis.set_title("vertices inside each level")
    axis.set_xticks(positions)
    axis.set_xticklabels([str(cell) for cell in NAMED_CELLS])
    axis.set_xlabel("cell")
    axis.set_ylabel("vertex count")
    axis.legend(fontsize=7)
    figure.tight_layout()
    png = OUT / "empty-exact-polygon.png"
    svg = OUT / "empty-exact-polygon.svg"
    figure.savefig(png, dpi=150)
    figure.savefig(svg)
    plt.close(figure)
    return svg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _install_recorder()
    control = census._read_control_row()
    context = census._build_context(control)
    operator = context["operator"]
    machine = context["machine"]
    mesh = machine.moment_geometry.atomic_mesh
    analytic = census._cell_analysis(
        machine, context["exact"], context["target_current"]
    )
    seed, _seed_receipt = census._production_seed(context)
    states = {
        "seed": np.asarray(seed, dtype=np.float64),
        "terminal": np.asarray(control["terminal_flux_wb"], dtype=np.float64),
    }
    payload: dict[str, Any] = {
        "case": census.CASE_NAME,
        "requested_cells": census.REQUESTED_CELLS,
        "realised_cells": control["realised_cells"],
        "revision": census._source_revision(),
        "named_cells": list(NAMED_CELLS),
        "states": {},
        "analytic": {
            "cut_by_analytic_separatrix": [
                int(cell)
                for cell in NAMED_CELLS
                if analytic["cut_by_analytic_separatrix"][cell]
            ],
            "exact_area_m2": [
                float(analytic["exact_area_m2"][cell]) for cell in NAMED_CELLS
            ],
        },
    }
    for name in STATES:
        block = _state_block(operator, mesh, states[name], name)
        payload["states"][name] = {
            key: value for key, value in block.items() if key != "probe"
        }
        control_counts = block["control"]
        print(
            f"{name}: boundary={block['boundary_flux_wb']:.7f} Wb "
            f"participation={block['participation_count']}/{block['cell_count']} "
            f"exact_area_total={block['exact_area_total_m2']:.6e} "
            f"post_arc_matches={control_counts['post_arc_matches']}/"
            f"{control_counts['cells_checked']} "
            f"crossing_matches={control_counts['crossing_matches']}/"
            f"{control_counts['cells_checked']}",
            flush=True,
        )
    payload["figure_src"] = (
        "/nova/figures/cut-cell-current-attribution/empty-exact-polygon/empty-exact-polygon.svg"
    )
    (OUT / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUT / "report.md").write_text(_markdown(payload), encoding="utf-8")
    _figure(payload)
    print("EMPTY_EXACT_POLYGON_REPORT_EXIT=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
