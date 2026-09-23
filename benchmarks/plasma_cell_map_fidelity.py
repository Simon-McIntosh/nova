"""Measure the certificate map at analytic flux, without taking solver steps.

Relative sup and RMS use the corresponding analytic-state norms, on every state
node, in the fixture's common Biot gauge. Span-normalized errors are also retained.
The fixture exterior is analytic total minus the analytic moment image; this
experiment checks composition and support fidelity, not an independent Green kernel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

NEGATIVE_CONTROL = (
    "the same evaluation at the analytic state shifted vertically by one pitch "
    "must raise the recorded mismatch; if it does not, the measure cannot see a "
    "displaced plasma and the receipt says so"
)
CASES = ("diverted-single-null", "weak-rotation-reactor-static")
MODES = ("exact", "chord")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/plasma-cell-read-fidelity/map-fidelity"


def write(path, value):
    from benchmarks.plasma_cell_first_step_directions import clean

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def norms(value, reference):
    import numpy as np

    value, reference = np.asarray(value), np.asarray(reference)
    finite = bool(np.all(np.isfinite(value)))
    if not finite:
        return {"sup_relative": None, "rms_relative": None, "finite": False}
    sup = float(np.max(np.abs(value)))
    rms = float(np.sqrt(np.mean(value**2)))
    ref_sup = float(np.max(np.abs(reference)))
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    span = float(np.ptp(reference))
    return {
        "sup_relative": sup / ref_sup if ref_sup else None,
        "rms_relative": rms / ref_rms if ref_rms else None,
        "sup_wb": sup,
        "rms_wb": rms,
        "sup_over_reference_span": sup / span if span else None,
        "rms_over_reference_span": rms / span if span else None,
        "reference_sup_wb": ref_sup,
        "reference_rms_wb": ref_rms,
        "finite": True,
    }


def passes(measure):
    return bool(
        measure["finite"]
        and measure["sup_relative"] is not None
        and measure["rms_relative"] is not None
        and max(measure["sup_relative"], measure["rms_relative"]) < 1e-2
    )


def instrument_controls():
    import numpy as np

    reference = np.array([1.0, -2.0, 3.0])
    assert passes(norms(np.zeros(3), reference))
    assert not passes(norms(0.02 * reference, reference))
    assert not passes(norms(np.array([np.nan, 0.0, 0.0]), reference))
    return {
        "identity_accepted": True,
        "two_percent_refused": True,
        "nonfinite_refused": True,
        "nonfinite_counter_sees_injected_nan": int(
            np.count_nonzero(~np.isfinite([1.0, np.nan]))
        )
        == 1,
    }


def nulls(operator, state):
    from benchmarks.plasma_cell_terminal_state import _nulls
    from nova.equilibrium.topology import NoQualifiedAxisError

    try:
        return _nulls(operator, state)[0]
    except NoQualifiedAxisError as error:
        return {
            "axis_rz_m": None,
            "x_point_rz_m": None,
            "qualified_saddles_rz_m": [],
            "unavailable": str(error),
        }


def measure_pair(case_name, requested, output):
    import jax
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    import jax.numpy as jnp
    import numpy as np
    from matplotlib.path import Path as PolygonPath
    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium.forward_operator import set_support_clip_mode
    from scripts.analytic_oracle_fixtures import measure as fixture

    assert jax.default_backend() == "gpu", "physical gate requires a GPU"
    carrier, source, exact = certificate._case(case_name)
    print(f"BUILD case={case_name} requested={requested}", flush=True)
    machine = certificate._case_machine(case_name, carrier, exact, requested)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)
    empty = fixture.forward_operator(source, machine)
    physical, fixture_external, cache = fixture.cached_fixture_exterior(
        source, exact, machine, empty, analytic
    )
    operator = fixture.forward_operator(source, machine, fixture_external)
    target, _, current_receipt = certificate._closed_form_current_target(
        case_name, source, operator, physical
    )
    # The certificate's explicit seed supplies no requested topology class.
    requested_class = None
    analytic_coefficients = operator.coupling_current_moments(physical)
    analytic_plasma = np.asarray(operator.current_moment_image(analytic_coefficients))
    analytic_external = analytic - analytic_plasma
    external = np.asarray(operator.external())
    assert np.all(np.isfinite(analytic_plasma)) and np.ptp(analytic_plasma) > 0
    boundary = fixture._analytic_separatrix(exact)
    analytic_centres = PolygonPath(boundary).contains_points(machine.node)
    analytic_support = np.asarray(physical.cell_current) != 0
    pitch = float(np.sqrt(np.median(machine.area)))
    shifted_coordinates = coordinates.copy()
    shifted_coordinates[:, 1] -= pitch
    shifted = certificate._exact_state(case_name, exact, shifted_coordinates)
    assert np.max(np.abs(shifted - analytic)) > 0, "displacement control is unchanged"
    reference_nulls = nulls(operator, analytic)
    offsets = np.asarray(operator.wall_unit_offsets)
    wall_units = [
        [int(start), int(stop), bool(closed), kind]
        for start, stop, closed, kind in zip(
            offsets[:-1],
            offsets[1:],
            operator.wall_unit_closed,
            operator.wall_unit_kinds,
            strict=True,
        )
    ]
    for mode in MODES:
        started = time.monotonic()
        set_support_clip_mode(mode)
        jax.clear_caches()
        print(f"MAP case={case_name} requested={requested} mode={mode}", flush=True)
        traced = operator.traced_flux_map(requested_class, target)
        evaluate = jax.jit(traced)
        mapped = np.asarray(
            jax.block_until_ready(
                evaluate(
                    jnp.asarray(analytic),
                    jnp.asarray(external),
                    operator,
                    jnp.asarray(target),
                )
            )
        )

        @jax.jit
        def support(state, active):
            partition = active._support_partition(state, requested_class)
            moments = active._partitioned_current_moments(partition)
            current = jnp.sum(moments.cell_current)
            amplitude = active.current_normalisation_amplitude(target, current)
            scaled = active.scaled_current_moments(moments, amplitude)
            return (
                partition[0].core,
                partition[3].area,
                moments,
                amplitude,
                active.current_moment_image(scaled),
                active.residual_shadow_mask(state, requested_class),
            )

        core, area, moments, amplitude, plasma, shadow = jax.device_get(
            support(jnp.asarray(analytic), operator)
        )
        raw = external + plasma
        mismatch = norms(mapped - analytic, analytic)
        raw_mismatch = norms(raw - analytic, analytic)
        shifted_map = np.asarray(
            jax.block_until_ready(
                evaluate(
                    jnp.asarray(shifted),
                    jnp.asarray(external),
                    operator,
                    jnp.asarray(target),
                )
            )
        )
        control = norms(shifted_map - analytic, analytic)
        shifted_self = norms(shifted_map - shifted, shifted)
        delta = {
            key: control[key] - mismatch[key]
            if control[key] is not None and mismatch[key] is not None
            else None
            for key in ("sup_relative", "rms_relative")
        }
        detected = all(value is not None and value > 0 for value in delta.values())
        moment_array = np.asarray(moments)
        physical_array = np.asarray(physical)
        external_error = norms(external - analytic_external, analytic_external)
        plasma_error = norms(plasma - analytic_plasma, analytic_plasma)
        support_cells = np.asarray(area) > 0
        label = f"{case_name}-cells-{abs(requested)}-{mode}"
        row = {
            "status": "measured",
            "case": case_name,
            "requested_cells": requested,
            "clip_mode": mode,
            "realised_cells": len(machine.node),
            "state_nodes": len(analytic),
            "pitch_m": pitch,
            "mismatch": mismatch,
            "raw_composition_mismatch": raw_mismatch,
            "passes": passes(mismatch),
            "core_cell_count": int(np.count_nonzero(core)),
            "analytic_centre_inside_cell_count": int(analytic_centres.sum()),
            "analytic_plasma_support_cell_count": int(analytic_support.sum()),
            "positive_area_support_cell_count": int(support_cells.sum()),
            "missing_analytic_support_cells": int(
                np.count_nonzero(analytic_support & ~support_cells)
            ),
            "extra_support_cells": int(
                np.count_nonzero(~analytic_support & support_cells)
            ),
            "unscaled_support_current_a": float(np.sum(moments.cell_current)),
            "analytic_plasma_current_a": float(target),
            "analytic_discrete_moment_current_a": float(np.sum(physical.cell_current)),
            "unscaled_over_analytic_current": float(
                np.sum(moments.cell_current) / target
            ),
            "lambda": float(amplitude),
            "current_target_provenance": current_receipt,
            "nonfinite_support_moments": int(
                np.count_nonzero(~np.isfinite(moment_array))
            ),
            "nonfinite_support_areas": int(np.count_nonzero(~np.isfinite(area))),
            "nonzero_support_moments": int(np.count_nonzero(moment_array)),
            "support_moment_elements": int(moment_array.size),
            "shadow_copied_state_nodes": int(np.count_nonzero(shadow)),
            "exterior_mismatch": external_error,
            "plasma_response_mismatch": plasma_error,
            "scaled_current_l1_relative_error": float(
                np.sum(
                    np.abs(
                        float(amplitude) * np.asarray(moments.cell_current)
                        - physical_array[0]
                    )
                )
                / np.sum(np.abs(physical_array[0]))
            ),
            "composition_closure_sup_wb": float(
                np.max(np.abs(np.where(shadow, analytic, raw) - mapped))
            ),
            "fixture_closure_sup_wb": float(
                np.max(np.abs(analytic_external + analytic_plasma - analytic))
            ),
            "exterior_counterfactual_two_percent_detected": not passes(
                norms(0.02 * analytic_external, analytic_external)
            ),
            "failure_component": "none"
            if passes(mismatch)
            else "support/profile moment mismatch carried by plasma response"
            if external_error["finite"] and external_error["sup_relative"] < 1e-10
            else "exterior and/or plasma response; inspect component norms",
            "component_attribution_limit": (
                "Analytic plasma and map share the Green operator; "
                "no independent kernel-accuracy claim."
            ),
            "negative_control": {
                "vertical_shift_m": pitch,
                "comparison": "shifted-input map against unshifted analytic reference",
                "mismatch": control,
                "shifted_self_mismatch": shifted_self,
                "increase": delta,
                "detected_in_both_norms": detected,
                "verdict": "displacement detected"
                if detected
                else "not-supported: declared measure does not raise both mismatches",
            },
            "reference_nulls": reference_nulls,
            "mapped_nulls": nulls(operator, mapped),
            "wall_units": wall_units,
            "fixture_cache": cache,
            "jax_backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "x64": bool(jax.config.jax_enable_x64),
            "source_revision": certificate._source_revision(),
            "wall_seconds": time.monotonic() - started,
            "state_archive": label + ".npz",
        }
        np.savez_compressed(
            output / row["state_archive"],
            coordinates=coordinates,
            wall=machine.wall_node,
            analytic=analytic,
            mapped=mapped,
            raw=raw,
            external=external,
            analytic_external=analytic_external,
            plasma=plasma,
            analytic_plasma=analytic_plasma,
            moments=moment_array,
            analytic_physical_moments=physical_array,
            shadow=shadow,
            shifted=shifted,
            shifted_map=shifted_map,
            core=core,
            area=area,
        )
        write(output / (label + ".json"), row)
        print(f"ROW {label} mismatch={mismatch} control={delta}", flush=True)


def render(row, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from benchmarks import solovev_certificate as certificate
    from benchmarks.plasma_cell_terminal_state import _draw_nulls
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes
    from nova.media.sources.frame import WallUnit

    with np.load(output / row["state_archive"]) as data:
        reference, mapped = data["analytic"], data["mapped"]
        coordinates, wall = data["coordinates"], data["wall"]
        units = tuple(
            WallUnit(wall[a:b, 0], wall[a:b, 1], closed=closed, kind=kind)
            for a, b, closed, kind in row["wall_units"]
        )
        levels = poloidal.contour_levels(
            reference,
            count=15,
            axis=row["reference_nulls"]["axis_flux_wb"],
            boundary=row["reference_nulls"]["boundary_flux_wb"],
        )
        figure, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
        blue = DEFAULT_INK.variant(
            axis_color="#3366cc",
            xpoint_color="#3366cc",
            axis_markersize=9,
            xpoint_markersize=11,
        )
        red = DEFAULT_INK.variant(axis_markersize=5, xpoint_markersize=6)
        for index, field in enumerate((reference, mapped, mapped - reference)):
            axis = axes[index]
            local_levels = (
                levels
                if index < 2
                else np.linspace(
                    -max(float(np.max(np.abs(field))), 1e-15),
                    max(float(np.max(np.abs(field))), 1e-15),
                    17,
                )
            )
            radial, height, raster = certificate._raster_field(coordinates, field, wall)
            contours = poloidal.draw_flux_contours(
                axis, radial, height, raster, local_levels, color="#444444"
            )
            assert any(len(part) > 1 for group in contours.allsegs for part in group)
            poloidal.draw_wall(axis, units=units)
            if index == 1:
                _draw_nulls(axis, row["mapped_nulls"], units, red)
            if index < 2:
                _draw_nulls(axis, row["reference_nulls"], units, blue)
            poloidal_axes(axis)
            axis.set_title(
                ("Analytic reference", "One map application", "Mapped minus analytic")[
                    index
                ]
            )
        score = row["mismatch"]
        figure.suptitle(
            f"{row['case']} / {row['clip_mode']} / "
            f"{abs(row['requested_cells'])} requested\n"
            f"sup={score['sup_relative']:.6g}; RMS={score['rms_relative']:.6g}; "
            "no solve; convergence=n/a",
            fontsize=11,
        )
        figure.supxlabel(
            "Shared physical levels in flux panels; mismatch on its own levels. "
            "Blue: analytic nulls; red: mapped nulls.",
            fontsize=9,
        )
        base = Path(row["state_archive"]).stem
        for suffix in ("png", "svg"):
            figure.savefig(output / f"{base}.{suffix}", dpi=150)
        plt.close(figure)
        row["panels"] = [f"{base}.png", f"{base}.svg"]
        row["shared_levels_wb"] = levels.tolist()


def run(output):
    from benchmarks import solovev_certificate as certificate

    output.mkdir(parents=True, exist_ok=True)
    revision = certificate._source_revision()
    requested = list(certificate.MEASUREMENT_REQUESTS)
    receipt = {
        "source_revision": revision,
        "worktree": str(ROOT),
        "command": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "requested_rungs": requested,
        "cases": list(CASES),
        "clip_modes": list(MODES),
        "completed": False,
        "rows": [],
        "instrument_controls": instrument_controls(),
        "metric": (
            "sup(error)/sup(analytic), RMS(error)/RMS(analytic), "
            "all state nodes; both <1e-2"
        ),
        "gauge": "same analytic-clipped fixture exterior; no flux re-zeroing",
        "negative_control_declaration": NEGATIVE_CONTROL,
    }
    write(output / "map-fidelity.json", receipt)
    with (output / "negative-control.log").open("w") as control_log:
        control_log.write(NEGATIVE_CONTROL + "\n")
        control_log.flush()
        for cells in requested:
            for case in CASES:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--case",
                    case,
                    "--cells",
                    str(cells),
                    "--output",
                    str(output),
                ]
                log = output / f"{case}-cells-{abs(cells)}.log"
                with log.open("w") as stream:
                    stream.write(
                        f"revision={revision} tree={ROOT} command={command!r}\n"
                    )
                    stream.flush()
                    result = subprocess.run(
                        command, stdout=stream, stderr=subprocess.STDOUT
                    )
                for mode in MODES:
                    part = output / f"{case}-cells-{abs(cells)}-{mode}.json"
                    if part.exists():
                        row = json.loads(part.read_text())
                        assert row["source_revision"] == revision
                        control_log.write(
                            json.dumps(
                                {
                                    "case": case,
                                    "requested_cells": cells,
                                    "clip_mode": mode,
                                    **row["negative_control"],
                                }
                            )
                            + "\n"
                        )
                        control_log.flush()
                    else:
                        row = {
                            "status": "not-measured",
                            "case": case,
                            "requested_cells": cells,
                            "clip_mode": mode,
                            "process_exit_status": result.returncode,
                            "log": str(log),
                        }
                    receipt["rows"].append(row)
                write(output / "map-fidelity.json", receipt)
                print(
                    f"PAIR_DONE case={case} requested={cells} exit={result.returncode}",
                    flush=True,
                )
    receipt["first_passing_rung"] = {}
    for case in CASES:
        receipt["first_passing_rung"][case] = {}
        for mode in MODES:
            rows = [
                row
                for row in receipt["rows"]
                if row["case"] == case and row["clip_mode"] == mode
            ]
            passing = [row for row in rows if row.get("passes")]
            first = (
                min(passing, key=lambda row: abs(row["requested_cells"]))
                if passing
                else None
            )
            receipt["first_passing_rung"][case][mode] = (
                abs(first["requested_cells"]) if first else None
            )
            for row in rows:
                if row["status"] == "measured" and (
                    row["requested_cells"] == requested[0] or row is first
                ):
                    render(row, output)
    receipt["completed"] = all(row["status"] == "measured" for row in receipt["rows"])
    receipt["negative_control_all_rows_detected"] = all(
        row.get("negative_control", {}).get("detected_in_both_norms", False)
        for row in receipt["rows"]
    )
    write(output / "map-fidelity.json", receipt)
    print(
        "MAP_FIDELITY_COMPLETE " + json.dumps(receipt["first_passing_rung"]), flush=True
    )
    if not receipt["completed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cells", type=int)
    args = parser.parse_args()
    if args.case:
        measure_pair(args.case, args.cells, args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
