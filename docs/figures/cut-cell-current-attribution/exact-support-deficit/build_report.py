"""Attribute the exact-mode current deficit to the clip step, per cut cell.

For every analytic cut cell of the weak 110-requested (135-realised)
certificate row this reports the analytic, chord and exact unit-amplitude
current, the clipped polygon area each clip mode produced, the per-vertex
signed flux against the boundary level, and the clip step that zeroed or
shrank the exact support.

The traced support is captured immediately before its participation
qualifying step, so a cell the participation gates refuse is distinguishable
from a cell the clip itself left empty.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import jax.numpy as jnp

from nova.equilibrium import clip_quadrature as quadrature_module
from nova.equilibrium import separatrix_clip as clip_module
from nova.equilibrium.forward_operator import (
    set_support_clip_mode,
)
from nova.equilibrium import forward_operator as forward_module

OUT = Path(__file__).resolve().parent
STATES = ("seed", "terminal")
MODES = ("chord", "exact")
NAMED_CELLS = (2, 35, 75, 89, 114, 126, 128, 134)
AREA_TOL = 1.0e-9

_RAW = {}
_ORIGINAL_QUALIFY = clip_module.TracedClippedSupports.qualify


def _recording_qualify(self, participation):
    _RAW["support"] = self
    _RAW["participation"] = np.asarray(participation, dtype=bool)
    return _ORIGINAL_QUALIFY(self, participation)


clip_module.TracedClippedSupports.qualify = _recording_qualify

_BANK_CALLS = []
_BANK_LOGGED = set()


def _install_bank_recorder():
    """Record the cut-cell bin count against its capacity at every entry.

    The integrator fails closed with a NaN field when a bin overflows, so the
    overflow has to be observed at its own call rather than inferred from the
    NaN afterwards: the NaN carries no bin index or capacity with it.
    """
    import jax

    original = forward_module.clipped_support_current_moments

    def recorded(support, selection, field, profile, *, cut_cell_capacity,
                 boundary_reduction=False):
        selected = jnp.asarray(selection, dtype=bool)
        boundary = selected & jnp.asarray(support.boundary, dtype=bool)
        key = (
            int(support.support_vertices.shape[0]),
            int(cut_cell_capacity),
            bool(boundary_reduction),
        )

        def report(cut_count, key=key):
            record = {
                "cells": key[0],
                "capacity": key[1],
                "boundary_reduction": key[2],
                "cut_count": int(cut_count),
                "overflow": int(cut_count) > key[1],
            }
            if record not in _BANK_CALLS:
                _BANK_CALLS.append(record)

        try:
            jax.debug.callback(report, jnp.sum(boundary.astype(jnp.int32)))
        except Exception:  # eager call path: the count is already concrete
            count = int(np.count_nonzero(np.asarray(boundary, dtype=bool)))
            report(count)
        return original(
            support,
            selection,
            field,
            profile,
            cut_cell_capacity=cut_cell_capacity,
            boundary_reduction=boundary_reduction,
        )

    forward_module.clipped_support_current_moments = recorded
    quadrature_module.clipped_support_current_moments = recorded
    return original


def _cause(row):
    """Name the clip step that zeroed or shrank the cell's exact support."""
    exact = row["exact_area_m2"]
    chord = row["chord_area_m2"]
    if exact > AREA_TOL and exact >= chord - AREA_TOL:
        return "carried-full"
    if exact > AREA_TOL:
        return "shrunk-partial-polygon"
    if not row["exact_participation"]:
        return "empty-participation-gate-refuses"
    total = row["vertex_total"]
    inside = row["inside_vertex_count"]
    if total and inside == 0:
        return "empty-no-vertex-inside-level"
    if total and inside == total:
        return "empty-no-boundary-crossing"
    if total and inside:
        return "empty-crossing-present-predicate-refuses"
    return "empty-no-vertex-recorded"


def _vertex_flux(cell_nodes, counts, shared, level, polarity):
    """Return per-cell signed-flux statistics against the boundary level."""
    stats = []
    for cell in range(len(counts)):
        count = int(counts[cell])
        index = np.asarray(cell_nodes[cell])[:count]
        signed = polarity * (shared[index] - level)
        inside = int(np.count_nonzero(signed >= 0.0))
        stats.append(
            {
                "vertex_total": count,
                "inside_vertex_count": inside,
                "outside_vertex_count": count - inside,
                "signed_flux_min_wb": float(np.min(signed)) if count else 0.0,
                "signed_flux_max_wb": float(np.max(signed)) if count else 0.0,
                "signed_flux_wb": [float(value) for value in signed],
            }
        )
    return stats


def _probe(operator, state, mode):
    """Trace one mode's partition from the full state vector.

    The state is passed whole: the fixed-design read consumes the authored
    direct-sampling rows alongside the grid flux, and the partition label and
    the sample normalisation both come from that same read.
    """
    physical = jnp.asarray(state)
    base_masks, topology, _connected, _admitted = operator._fixed_design_read(
        physical
    )
    sample_flux = operator.sample_node_flux(jnp.asarray(state))
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    profile_support = operator._profile_support(
        base_masks, topology, physical, sample_psi_norm
    )
    moment_masks = operator._moment_support_masks(base_masks, profile_support)
    return {
        "base_masks": base_masks,
        "topology": topology,
        "sample_psi_norm": sample_psi_norm,
        "profile_support": profile_support,
        "moment_masks": moment_masks,
    }


def _mode_geometry(census, operator, state, mode):
    """Return one mode's raw clip geometry and unit-amplitude current."""
    set_support_clip_mode(mode)
    _RAW.clear()
    probe = _probe(operator, state, mode)
    raw = _RAW["support"]
    return {
        "current_a": np.asarray(census._mode_current(operator, state, mode)),
        "area_m2": np.asarray(raw.area, dtype=np.float64),
        "included": np.asarray(raw.included, dtype=bool),
        "boundary": np.asarray(raw.boundary, dtype=bool),
        "full_area_m2": np.asarray(raw.full_area, dtype=np.float64),
        "vertex_count": np.asarray(raw.vertex_count, dtype=np.float64),
        "participation": np.asarray(_RAW["participation"], dtype=bool),
        "vertex_capacity": int(np.asarray(raw.vertex_capacity)),
        "refused_cell_count": int(np.asarray(raw.refused_cell_count)),
        "topology": probe["topology"],
    }


def _state_report(census, operator, analytic, state, name, target):
    """Return the per-cut-cell attribution for one flux state."""
    _BANK_CALLS.clear()
    geometry = {
        mode: _mode_geometry(census, operator, state, mode) for mode in MODES
    }
    bank_calls = list(_BANK_CALLS)
    overflow_calls = [call for call in bank_calls if call["overflow"]]
    physical = np.asarray(state)[: operator.physical_node_number]
    shared = np.asarray(operator.shared_node_flux(jnp.asarray(physical)))
    level = float(np.asarray(geometry["exact"]["topology"].boundary_flux))
    polarity = float(np.asarray(operator.polarity))
    mesh = operator.moment_geometry.atomic_mesh
    counts = np.asarray(mesh.cell_vertex_count)
    flux = _vertex_flux(
        np.asarray(mesh.cell_nodes), counts, shared, level, polarity
    )
    cut = np.asarray(analytic["cut_by_analytic_separatrix"], dtype=bool)
    rows = []
    for cell in range(len(counts)):
        if not cut[cell]:
            continue
        area = float(analytic["exact_area_m2"][cell])
        fraction = float(analytic["analytic_plasma_area_fraction"][cell])
        included = bool(geometry["exact"]["included"][cell])
        row = {
            "cell": int(cell),
            "centre_rz_m": analytic["centre_rz_m"][cell],
            "analytic_area_m2": area,
            "atomic_area_m2": area / fraction if fraction > 0.0 else 0.0,
            "analytic_current_a": float(analytic["exact_main_axis_current_a"][cell]),
            "chord_current_a": float(geometry["chord"]["current_a"][cell]),
            "exact_current_a": float(geometry["exact"]["current_a"][cell]),
            "chord_area_m2": float(geometry["chord"]["area_m2"][cell]),
            "exact_area_m2": float(geometry["exact"]["area_m2"][cell]),
            "exact_full_area_m2": float(geometry["exact"]["full_area_m2"][cell]),
            "chord_vertex_count": int(geometry["chord"]["vertex_count"][cell]),
            "exact_vertex_count": int(geometry["exact"]["vertex_count"][cell]),
            "exact_included": included,
            "exact_boundary_predicate": bool(geometry["exact"]["boundary"][cell]),
            "chord_participation": bool(geometry["chord"]["participation"][cell]),
            "exact_participation": bool(geometry["exact"]["participation"][cell]),
            "boundary_level_wb": level,
            "polarity": polarity,
        }
        row.update(flux[cell])
        row["exact_current_defined"] = bool(
            np.isfinite(row["exact_current_a"])
        )
        row["exact_cause"] = (
            _cause(row)
            if row["exact_current_defined"]
            else "exact-current-undefined-bank-overflow"
        )
        rows.append(row)
    return {
        "state": name,
        "target_current_a": target,
        "boundary_level_wb": level,
        "polarity": polarity,
        "cut_cell_count": len(rows),
        "refused_cell_count": geometry["exact"]["refused_cell_count"],
        "exact_current_defined": bool(
            all(r["exact_current_defined"] for r in rows)
        ),
        "exact_undefined_cell_count": sum(
            1 for r in rows if not r["exact_current_defined"]
        ),
        "cut_cell_bank_calls": bank_calls,
        "cut_cell_bank_overflows": overflow_calls,
        "cause_counts": {
            cause: sum(1 for r in rows if r["exact_cause"] == cause)
            for cause in sorted({r["exact_cause"] for r in rows})
        },
        "totals_a": {
            "analytic": float(sum(r["analytic_current_a"] for r in rows)),
            "chord": float(sum(r["chord_current_a"] for r in rows)),
            "exact": float(sum(r["exact_current_a"] for r in rows)),
        },
        "rows": rows,
    }


def _markdown(report):
    """Render the attribution as a markdown record."""
    lines = [
        "# Exact-support deficit per cut cell, weak 110 (135 realised) row",
        "",
        "Revision `%s`." % report["source_revision"],
        "Analytic oracle total over the cut cells is %.6f MA; the target"
        " current is %.6f MA." % (
            report["analytic_total_a"] / 1.0e6,
            report["target_current_a"] / 1.0e6,
        ),
        "",
    ]
    for entry in report["states"]:
        totals = entry["totals_a"]
        lines.append("## State `%s`" % entry["state"])
        lines.append("")
        lines.append(
            "Boundary level %.9g Wb, polarity %.4g, cut cells %d."
            % (entry["boundary_level_wb"],
               entry["polarity"], entry["cut_cell_count"])
        )
        lines.append("")
        lines.append(
            "Totals over the cut cells: analytic %.4f MA, chord %.4f MA,"
            " exact %.4f MA (defined: %s)."
            % (totals["analytic"] / 1.0e6, totals["chord"] / 1.0e6,
               totals["exact"] / 1.0e6, entry["exact_current_defined"])
        )
        lines.append("")
        lines.append(
            "Cut-cell moment bank: %d call signature(s), %d overflowing."
            % (len(entry["cut_cell_bank_calls"]),
               len(entry["cut_cell_bank_overflows"]))
        )
        lines.append("")
        lines.append("| cells | capacity | boundary reduction | cut count |"
                     " overflow |")
        lines.append("|---|---|---|---|---|")
        for call in entry["cut_cell_bank_calls"]:
            lines.append(
                "| %d | %d | %s | %d | %s |"
                % (call["cells"], call["capacity"],
                   call["boundary_reduction"], call["cut_count"],
                   call["overflow"])
            )
        lines.append("")
        lines.append("| cause | cells |")
        lines.append("|---|---|")
        for cause in sorted(entry["cause_counts"]):
            lines.append(
                "| `%s` | %d |" % (cause, entry["cause_counts"][cause])
            )
        lines.append("")
        lines.append("### Named cells")
        lines.append("")
        lines.append(
            "| cell | analytic A | chord A | exact A | exact area m2 |"
            " chord area m2 | vertices inside | cause |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in entry["rows"]:
            if row["cell"] not in NAMED_CELLS:
                continue
            lines.append(
                "| %d | %.3f | %.3f | %.3f | %.6e | %.6e | %d/%d | `%s` |"
                % (row["cell"], row["analytic_current_a"],
                   row["chord_current_a"], row["exact_current_a"],
                   row["exact_area_m2"], row["chord_area_m2"],
                   row["inside_vertex_count"], row["vertex_total"],
                   row["exact_cause"])
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _strict(value):
    """Return ``value`` with non-finite floats rendered as ``None``."""
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _strict(value.item())
    return value


def _figure(report, machine):
    """Draw the per-cell clip comparison and the cut-cell map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wall = np.asarray(machine.wall_node, dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    for row_index, entry in enumerate(report["states"]):
        rows = entry["rows"]
        ranked = sorted(rows, key=lambda item: item["analytic_area_m2"],
                        reverse=True)
        rank = np.arange(len(ranked))
        panel = axes[row_index][0]
        panel.step(rank, [item["analytic_area_m2"] for item in ranked],
                   where="mid", color="#111111", linewidth=1.4,
                   label="analytic")
        panel.step(rank, [item["chord_area_m2"] for item in ranked],
                   where="mid", color="#1f77b4", linewidth=1.2,
                   label="chord clip")
        defined = [item for item in ranked if item["exact_current_defined"]]
        if defined:
            panel.step(np.arange(len(defined)),
                       [item["exact_area_m2"] for item in defined],
                       where="mid", color="#d62728", linewidth=1.2,
                       label="exact clip")
        else:
            panel.axhline(0.0, color="#d62728", linewidth=1.2,
                          linestyle="--",
                          label="exact clip undefined (bank overflow)")
        panel.set_xlabel("cut cell, ranked by analytic plasma area")
        panel.set_ylabel("clipped plasma area [m$^2$]")
        panel.set_title("state `%s`, %d cut cells"
                        % (entry["state"], entry["cut_cell_count"]),
                        loc="left", fontsize=10)
        panel.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)

        panel = axes[row_index][1]
        panel.plot(wall[:, 0], wall[:, 1], color="#444444", linewidth=1.0)
        cut = np.asarray([item["centre_rz_m"] for item in rows],
                         dtype=np.float64)
        panel.scatter(cut[:, 0], cut[:, 1], s=14, facecolors="none",
                      edgecolors="#1f77b4", linewidths=0.8)
        closed = np.asarray(
            [item["centre_rz_m"] for item in rows
             if item["exact_area_m2"] > AREA_TOL], dtype=np.float64
        )
        if closed.size:
            panel.scatter(closed[:, 0], closed[:, 1], s=26,
                          color="#d62728", marker="^")
        named = np.asarray(
            [item["centre_rz_m"] for item in rows
             if item["cell"] in NAMED_CELLS], dtype=np.float64
        )
        panel.scatter(named[:, 0], named[:, 1], s=64, facecolors="none",
                      edgecolors="#111111", linewidths=1.4, marker="s")
        panel.set_aspect("equal")
        panel.set_axis_off()
        panel.set_title("cut cells with a traced exact polygon (triangle)",
                        loc="left", fontsize=10)
    figure.suptitle(
        "Exact-support deficit per cut cell, weak 110 (135 realised) row"
        " at revision %s" % report["source_revision"][:12],
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    for suffix in ("png", "svg"):
        figure.savefig(OUT / ("exact-support-deficit." + suffix), dpi=160)
    plt.close(figure)


def main():
    """Run the census internals and write the per-cell attribution report."""
    import benchmarks.unit_amplitude_current_census as census

    _install_bank_recorder()
    started = time.perf_counter()
    control = census._read_control_row()
    context = census._build_context(control)
    operator = context["operator"]
    machine = context["machine"]
    target = float(context["target_current"])
    seed, _ = census._production_seed(context)
    analytic = census._cell_analysis(machine, context["exact"], target)
    states = {
        "seed": np.asarray(seed),
        "terminal": np.asarray(control["terminal_flux_wb"]),
    }
    report = {
        "case": census.CASE_NAME,
        "requested_cells": census.REQUESTED_CELLS,
        "realised_cells": analytic["cell_count"],
        "source_revision": census._source_revision(),
        "lane": census._lane_receipt(),
        "target_current_a": target,
        "analytic_total_a": float(sum(analytic["exact_main_axis_current_a"])),
        "named_cells": list(NAMED_CELLS),
        "states": [
            _state_report(census, operator, analytic, states[name], name, target)
            for name in STATES
        ],
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    (OUT / "report.json").write_text(
        json.dumps(_strict(report), indent=2, sort_keys=True,
                   allow_nan=False) + "\n"
    )
    (OUT / "report.md").write_text(_markdown(report))
    _figure(report, machine)
    for entry in report["states"]:
        print(entry["state"], entry["cause_counts"], flush=True)
    print("EXACT_SUPPORT_REPORT_EXIT=0", flush=True)
    return report


if __name__ == "__main__":
    main()