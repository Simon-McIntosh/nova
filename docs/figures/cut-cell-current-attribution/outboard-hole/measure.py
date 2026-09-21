"""Attribute booked-current errors on a persisted terminal flux state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import types

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from benchmarks import unit_amplitude_current_census as census  # noqa: E402
import nova.equilibrium.forward_operator as forward  # noqa: E402
from nova.equilibrium.domain import DomainMasks, PlasmaDomain  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as oracle  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
FIXTURE_PATH = "scripts/analytic_oracle_fixtures/measure.py"
GATE_PATH = "tests/test_outboard_hole_census_gate.py"


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def _read_module(revision, path, module):
    """Evaluate an explicitly selected repository blob without editing its path."""
    source = _git("show", f"{revision}:{path}")
    exec(compile(source, str(ROOT / path), "exec"), module.__dict__)
    return {
        "path": path,
        "revision": revision,
        "blob": _git("rev-parse", f"{revision}:{path}").decode().strip(),
        "sha256": hashlib.sha256(source).hexdigest(),
    }


def _load_inputs(revision):
    """Read the repaired fixture and reusable gate from one immutable revision."""
    fixture = _read_module(revision, FIXTURE_PATH, oracle)
    gate = types.ModuleType("current_booking_gate")
    gate.__file__ = str(ROOT / GATE_PATH)
    gate_source = _read_module(revision, GATE_PATH, gate)
    driver = _git("show", f"{revision}:benchmarks/unit_amplitude_current_census.py")
    assert driver == (ROOT / "benchmarks/unit_amplitude_current_census.py").read_bytes()
    assert hasattr(oracle, "_analytic_topology")
    assert np.all(np.isnan(np.asarray(oracle._analytic_saddle(object()))))
    assert not getattr(
        forward.ForwardFluxOperator._profile_support, gate._BRIDGE_MARKER, False
    ), "the repaired fixture must run without the gate's absent-saddle bridge"
    return gate, [fixture, gate_source]


def _write(name, payload):
    (OUTPUT / name).write_text(
        json.dumps(census._strict(payload), indent=2, allow_nan=False) + "\n"
    )


def _total_record(rows, key, selection):
    selected = [row for row in rows if selection(row)]
    return {
        "cells": [row["cell"] for row in selected],
        "count": len(selected),
        "analytic_a": sum(row["analytic_exact_current_a"] for row in selected),
        "booked_a": sum(row[f"{key}_current_a"] for row in selected),
        "net_deficit_a": sum(row[f"{key}_deficit_a"] for row in selected),
    }


def measure(revision):
    gate, inputs = _load_inputs(revision)
    captured = {}
    original = census._state_census

    def capture(operator, machine, state, state_name, target, analytic, curve):
        row = original(operator, machine, state, state_name, target, analytic, curve)
        _write("booked-census.json", row)
        captured.update(
            operator=operator, machine=machine, state=state, analytic=analytic, row=row
        )
        return row

    census._state_census = capture
    try:
        booked = gate.terminal_booked_currents()
    finally:
        census._state_census = original
    checks = []
    for name in (
        "test_booked_totals_reproduce_the_committed_state",
        "test_outboard_cells_book_the_committed_current",
        "test_outboard_cells_book_nonzero_in_both_clip_modes",
    ):
        getattr(gate, name)(booked)
        checks.append({"name": name, "passed": True})
        print(f"PASS {name}", flush=True)
    operator, state = captured["operator"], captured["state"]
    rows = captured["row"]["per_cell"]
    analytic = captured["analytic"]
    mode_details = {}
    for mode in ("chord", "exact"):
        probe = census._partition_probe(operator, state, mode)
        masks = probe["base_masks"]
        support = probe["profile_support"]
        field = forward.flux_field_polynomial(
            operator._support_moment_stencils, masks.psi_norm, probe["sample_psi_norm"]
        )
        for i, row in enumerate(rows):
            row[f"{mode}_label"] = int(masks.label[i])
            row[f"{mode}_label_eligible"] = bool(masks.profile_participation[i])
            row[f"{mode}_included"] = bool(support.included[i])
            row[f"{mode}_field_active"] = bool(field.active[i])
            row[f"{mode}_support_area_m2"] = float(support.area[i])
            row[f"{mode}_psi_norm"] = float(masks.psi_norm[i])
            row[f"{mode}_deficit_a"] = (
                row["analytic_exact_current_a"] - row[f"{mode}_current_a"]
            )
        mode_details[mode] = {
            "label_excluded": _total_record(
                rows, mode, lambda row: not row[f"{mode}_label_eligible"]
            ),
            "support_excluded": _total_record(
                rows,
                mode,
                lambda row: (
                    row[f"{mode}_label_eligible"] and not row[f"{mode}_included"]
                ),
            ),
            "inactive_field": _total_record(
                rows, mode, lambda row: not row[f"{mode}_field_active"]
            ),
            "zero_booking": _total_record(
                rows, mode, lambda row: row[f"{mode}_current_a"] == 0
            ),
            "underbooked": _total_record(
                rows, mode, lambda row: row[f"{mode}_deficit_a"] > 1.0e-6
            ),
            "overbooked": _total_record(
                rows, mode, lambda row: row[f"{mode}_deficit_a"] < -1.0e-6
            ),
            "by_analytic_region": {
                region: _total_record(
                    rows,
                    mode,
                    lambda row, region=region: row["analytic_surface_class"] == region,
                )
                for region in ("interior", "cut", "exterior")
            },
            "topology": {
                name: np.asarray(value)
                for name, value in probe["topology"]._asdict().items()
            },
        }
    analytic_total = sum(row["analytic_exact_current_a"] for row in rows)
    payload = {
        "source_revision": _git("rev-parse", "HEAD").decode().strip(),
        "loaded_inputs": inputs,
        "lane": census._lane_receipt(),
        "state_sha256": census._state_digest(state),
        "target_current_a": gate.TARGET_CURRENT_A,
        "analytic_cell_sum_a": analytic_total,
        "analytic_cell_sum_minus_target_a": analytic_total - gate.TARGET_CURRENT_A,
        "totals_a": captured["row"]["unit_amplitude_totals_a"],
        "target_deficit_a": {
            mode: gate.TARGET_CURRENT_A - value
            for mode, value in captured["row"]["unit_amplitude_totals_a"].items()
        },
        "gate_assertions": checks,
        "gate_bridge_installed": False,
        "mode_details": mode_details,
        "per_cell": rows,
        "polygons": analytic["polygons"],
        "analytic_integration_warning_count": analytic["integration_warning_count"],
    }
    _write("terminal-attribution.json", payload)
    print(
        json.dumps(
            census._strict(
                {
                    key: value
                    for key, value in payload.items()
                    if key not in ("per_cell", "polygons", "mode_details")
                }
            )
        ),
        flush=True,
    )
    for mode in ("chord", "exact"):
        ranking = sorted(rows, key=lambda row: row[f"{mode}_deficit_a"], reverse=True)
        print(mode, json.dumps(census._strict(mode_details[mode])), flush=True)
        print(
            "ranked",
            mode,
            [(row["cell"], row[f"{mode}_deficit_a"]) for row in ranking[:20]],
            flush=True,
        )


def controls(revision):
    _, inputs = _load_inputs(revision)
    context = census._build_context(census._read_control_row())
    operator = context["operator"]
    state = census._read_control_row()["terminal_flux_wb"]
    results = {}
    for mode in ("chord", "exact"):
        probe = census._partition_probe(operator, state, mode)
        masks, topology = probe["base_masks"], probe["topology"]
        physical = jnp.asarray(state)[: operator.physical_node_number]
        sample = probe["sample_psi_norm"]
        currents = {}
        for name, labels in (
            ("measured_labels", masks.label),
            ("all_labels_core", jnp.full_like(masks.label, int(PlasmaDomain.CORE))),
            (
                "exclude_known_cell",
                masks.label.at[106].set(int(PlasmaDomain.EXCLUDED_MATERIAL)),
            ),
        ):
            selected = DomainMasks(labels, masks.psi_norm)
            support = operator._profile_support(selected, topology, physical, sample)
            selected = operator._moment_support_masks(selected, support)
            moments = operator.source.current_moments(
                selected, operator.support_current_moments, support, sample_flux=sample
            )
            currents[name] = np.asarray(moments.cell_current)
        reference = currents["measured_labels"]
        neutral = currents["all_labels_core"]
        excluded = currents["exclude_known_cell"]
        assert reference[106] > 190_000.0
        assert excluded[106] == 0.0
        results[mode] = {
            "current_a": {name: value.tolist() for name, value in currents.items()},
            "totals_a": {name: float(value.sum()) for name, value in currents.items()},
            "label_neutralization_max_abs_delta_a": float(
                np.max(np.abs(neutral - reference))
            ),
            "label_neutralization_changed_cells": np.flatnonzero(
                neutral != reference
            ).tolist(),
            "known_cell_current_a": float(reference[106]),
            "known_cell_current_after_exclusion_a": float(excluded[106]),
            "known_exclusion_total_delta_a": float(reference.sum() - excluded.sum()),
        }
        print(
            mode,
            json.dumps({k: v for k, v in results[mode].items() if k != "current_a"}),
            flush=True,
        )
    _write(
        "label-controls.json",
        {"loaded_inputs": inputs, "lane": census._lane_receipt(), "results": results},
    )


def render():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from benchmarks import solovev_certificate as certificate
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    data = json.loads((OUTPUT / "terminal-attribution.json").read_text())
    control = json.loads(census.CONTROL_PART.read_text())
    fields = control["render_data"]
    coordinates = np.asarray(fields["coordinates_rz_m"])
    wall_units = tuple(np.asarray(unit) for unit in fields["wall_units_rz_m"])
    wall = np.concatenate(wall_units)
    r, z, reference = certificate._raster_field(
        coordinates, fields["analytic_flux_wb"], wall
    )
    _, _, solved = certificate._raster_field(
        coordinates, fields["terminal_flux_wb"], wall
    )
    levels = np.linspace(5.0, 45.0, 9)
    rows = data["per_cell"]
    centres = np.asarray([row["centre_rz_m"] for row in rows])
    panels = (
        ("Analytic per-cell current", "analytic_exact_current_a"),
        ("Booked current: chord mode", "chord_current_a"),
        ("Booked current: exact mode", "exact_current_a"),
    )
    maximum = max(row["analytic_exact_current_a"] for row in rows)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 8.7))
    null_tallies = []
    for ax, (title, key) in zip(axes, panels, strict=True):
        poloidal.draw_flux_contours(
            ax,
            r,
            z,
            reference,
            levels,
            color=certificate.ANALYTIC_INK_COLOR,
            linewidth=0.65,
        )
        poloidal.draw_flux_contours(
            ax, r, z, solved, levels, color=certificate.SOLVED_INK_COLOR, linewidth=0.65
        )
        poloidal.draw_wall(ax, units=wall_units)
        for name, color in (
            ("analytic_topology", certificate.ANALYTIC_INK_COLOR),
            ("terminal_topology", certificate.SOLVED_INK_COLOR),
        ):
            null = fields[name]
            null_tallies.append(
                poloidal.draw_nulls(
                    ax,
                    magnetic_axis=null["axis_rz_m"],
                    x_points=null["x_point_rz_m"],
                    style=DEFAULT_INK.variant(
                        axis_color=color,
                        xpoint_color=color,
                        axis_marker="^",
                        axis_markersize=7.0 if name == "analytic_topology" else 5.0,
                    ),
                    contain=wall_units,
                )
            )
        boundaries = [
            np.vstack((polygon, polygon[:1]))
            for polygon in map(np.asarray, data["polygons"])
        ]
        ax.add_collection(
            LineCollection(
                boundaries, colors="0.65", linewidths=0.35, alpha=0.65, zorder=2
            )
        )
        current = np.asarray([row[key] for row in rows])
        active = current > 1.0e-6
        ax.scatter(
            centres[active, 0],
            centres[active, 1],
            s=150.0 * current[active] / maximum,
            facecolors="none",
            edgecolors="#a92336",
            linewidths=0.85,
            zorder=5,
        )
        ax.scatter(
            centres[~active, 0],
            centres[~active, 1],
            s=15.0,
            marker="x",
            color="0.4",
            linewidths=0.7,
            zorder=5,
        )
        selected = sorted(rows, key=lambda row: row["exact_deficit_a"], reverse=True)[
            :8
        ]
        for row in selected:
            point = centres[row["cell"]]
            ax.annotate(
                str(row["cell"]),
                point,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="black",
                zorder=7,
            )
        poloidal_axes(ax)
        ax.set_xlim(wall[:, 0].min() - 0.12, wall[:, 0].max() + 0.12)
        ax.set_ylim(wall[:, 1].min() - 0.12, wall[:, 1].max() + 0.12)
        ax.set_title(f"{title}\n{current.sum() / 1e6:.6f} MA", fontsize=12)
    residual = control["solver"]["terminal_fixed_point_residual"]
    converged = control["solver"]["production_telemetry"]["converged"]
    figure.suptitle(
        "Remaining current deficit on the persisted weak-rotation terminal state",
        fontsize=15,
        y=0.97,
    )
    caption = (
        "Hollow red circle area = per-cell current (same scale in every panel); "
        "grey cross = zero. Labels identify the eight largest exact-mode deficits.\n"
        "Analytic blue and terminal ochre flux contours share 5, 10, ..., 45 Wb; "
        "both axes are solid triangles and the unit-faithful wall is drawn; "
        "neither state admits an X-point.\n"
        f"Persisted terminal residual {residual:.6g}; converged={converged}. "
        "Current totals are unscaled; no solve or candidacy change was made."
    )
    figure.text(0.5, 0.025, caption, ha="center", fontsize=9, linespacing=1.5)
    figure.subplots_adjust(left=0.025, right=0.975, top=0.88, bottom=0.13, wspace=0.08)
    for extension in ("png", "svg"):
        figure.savefig(OUTPUT / f"booked-versus-analytic.{extension}", dpi=170)
    plt.close(figure)
    _write(
        "figure-receipt.json",
        {
            "caption": caption,
            "shared_flux_levels_wb": levels.tolist(),
            "null_tallies": null_tallies,
            "figures": [
                str(OUTPUT / f"booked-versus-analytic.{ext}") for ext in ("png", "svg")
            ],
        },
    )
    print(caption, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--task", choices=("census", "controls", "render"), required=True
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "cpu"
    if args.task == "census":
        measure(args.revision)
    elif args.task == "controls":
        controls(args.revision)
    else:
        render()


if __name__ == "__main__":
    main()
