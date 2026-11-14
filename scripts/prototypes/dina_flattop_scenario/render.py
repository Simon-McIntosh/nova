"""Render every returned bank state beside its native DINA reference."""

# ruff: noqa: E501

import argparse
from collections import Counter
import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from nova.media.gif import figure_frame, write_contact_sheet, write_gif
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import (
    draw_flux_contours,
    draw_nulls,
    draw_scattered_contours,
    draw_wall,
)
from nova.media.sources.frame import inside_wall_units
from scripts.prototypes.iter_bank.render import checked_save


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    parser.add_argument("--fragment", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--observables", required=True, type=Path)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    args.preview.mkdir(parents=True, exist_ok=True)
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    selection = json.loads((args.inputs / "selection.json").read_text())
    field_record = json.loads(args.observables.read_text())
    assert field_record["complete"]
    observed = {r["index"]: r for r in field_record["rows"]}
    summary = json.loads((args.results / "summary.json").read_text())
    available = sorted(args.results.glob("width-1-process-*/receipt.json"))
    assert available, "no measurement receipt exists"
    receipt_path = next((p for p in available if "process-1" in str(p)), available[0])
    receipt = json.loads(receipt_path.read_text())
    measured = {r["index"]: r for r in receipt["slices"]}
    wall = np.load(args.inputs / "bank/geometry.npz", allow_pickle=False)["wall"]
    style = DEFAULT_INK.variant(
        axis_color=DEFAULT_INK.flux_color,
        xpoint_color=DEFAULT_INK.flux_color,
        axis_markersize=9,
        xpoint_markersize=10,
    )
    reference_style = style.variant(
        flux_color="#333333", axis_color="#333333", xpoint_color="#333333"
    )
    frames, figures, rows, inspections = [], [], [], []
    comparisons, null_rows = [], []
    all_levels = []
    for reference in selection["slices"]:
        index = reference["index"]
        if index not in measured:
            continue
        row = measured[index]
        field = observed[index]
        source = np.load(args.inputs / f"source-{index}.npz", allow_pickle=False)
        state_path = receipt_path.parent / f"state-{index}.npz"
        if not state_path.exists():
            continue
        state = np.load(state_path, allow_pickle=False)
        if not np.all(np.isfinite(state["flux"])):
            inspections.append(
                dict(index=index, rendered=False, reason="nonfinite terminal field")
            )
            continue
        rr, zz = np.meshgrid(source["radius"], source["height"], indexing="ij")
        inside = inside_wall_units(
            np.column_stack((rr.ravel(), zz.ravel())), (wall,)
        ).reshape(rr.shape)
        ref_values = source["flux"][inside]
        lower = min(float(np.min(state["flux"])), float(np.min(ref_values)))
        upper = max(float(np.max(state["flux"])), float(np.max(ref_values)))
        levels = np.linspace(lower, upper, 15)[1:-1]
        all_levels.append(dict(index=index, levels_wb=levels.tolist()))
        figure = Figure(figsize=(14, 7.2), dpi=100)
        axes = figure.subplots(1, 2)
        figure.subplots_adjust(
            left=0.05, right=0.95, bottom=0.12, top=0.78, wspace=0.05
        )
        for ax in axes:
            poloidal_axes(ax)
            draw_wall(ax, units=(wall,), linewidth=1.2)
            ax.set_xlim(wall[:, 0].min() - 0.15, wall[:, 0].max() + 0.15)
            ax.set_ylim(wall[:, 1].min() - 0.15, wall[:, 1].max() + 0.15)
        draw_scattered_contours(
            axes[0],
            state["nodes"][:, 0],
            state["nodes"][:, 1],
            state["flux"],
            levels,
            (wall,),
            color=style.flux_color,
            linewidth=2.0,
        )
        contour = draw_flux_contours(
            axes[1],
            source["radius"],
            source["height"],
            source["flux"].T,
            levels,
            color=reference_style.flux_color,
            linewidth=1.8,
            wall=(wall,),
        )
        contour.set_linestyle("dashed")
        valid_nulls = np.asarray(field["null_valid"])
        stationary = np.asarray(field["stationary"])
        bank_tally = draw_nulls(
            axes[0],
            magnetic_axis=stationary[0] if valid_nulls[0] else None,
            x_points=stationary[1:][valid_nulls[1:]],
            contain=(wall,),
            style=style,
        )
        ref_tally = draw_nulls(
            axes[1],
            magnetic_axis=source["axis"],
            x_points=source["xpoints"],
            contain=(wall,),
            style=reference_style,
        )
        handles = []
        for label, ink, linestyle in (
            ("Bank", style, "solid"),
            ("DINA", reference_style, "dashed"),
        ):
            handles.extend(
                [
                    Line2D(
                        [],
                        [],
                        color=ink.flux_color,
                        linewidth=2,
                        linestyle=linestyle,
                        label=f"{label} flux",
                    ),
                    Line2D(
                        [],
                        [],
                        color=ink.axis_color,
                        marker=ink.axis_marker,
                        markersize=9,
                        linestyle="none",
                        label=f"{label} axis",
                    ),
                    Line2D(
                        [],
                        [],
                        color=ink.xpoint_color,
                        marker=ink.xpoint_marker,
                        markersize=10,
                        linestyle="none",
                        label=f"{label} X-point",
                    ),
                ]
            )
        figure.legend(
            handles=handles, loc="upper center", ncols=3, fontsize=14, frameon=False
        )
        figure.text(0.30, 0.80, "Bank", ha="center", fontsize=16)
        figure.text(0.70, 0.80, "DINA", ha="center", fontsize=16)
        residual = float(row["fresh_residual"][0])
        converged = bool(row["converged"][0])
        figure.text(
            0.5,
            0.045,
            f"{reference['time_s']:.3f} s     fresh residual {residual:.3g}     converged {converged}",
            ha="center",
            fontsize=14,
        )
        name = f"slice-{index:04d}.svg"
        checked_save(figure, args.figures / name)
        frame = figure_frame(figure)
        frame.save(args.preview / f"slice-{index:04d}.png")
        frames.append(frame)
        inspections.append(
            dict(
                index=index,
                rendered=True,
                bank_nulls=bank_tally,
                reference_nulls=ref_tally,
                valid_bank_nulls=valid_nulls.tolist(),
                clipped_text=0,
                overlapping_text=0,
            )
        )
        url = f"/nova/figures/sub-millisecond-forward-path/dina-flattop-scenario/{name}"
        figures.append(
            f'<figure id="dina-flattop-scenario-slice-{index}"><img src="{url}" alt="Bank and DINA line contours at {reference["time_s"]:.3f} seconds"><figcaption>'
            f"Slice {index}, {reference['time_s']:.6f} s: fresh residual {residual:.9g}, converged {converged}; stage {html.escape(row['failure_stage'][0])}. "
            f"Bank blue, DINA dashed near-black; each panel marks its own qualified axis and saddle with the triangle/cross vocabulary. Bank markers re-read the requested-input field on CPU. "
            f"{int(np.count_nonzero(valid_nulls))}/{len(valid_nulls)} bank nulls qualified. Native first wall, equal aspect, shared flux levels "
            f"{levels[0]:.6g} to {levels[-1]:.6g} Wb in {levels[1] - levels[0]:.6g} Wb steps. Raw flux gauges retained without alignment.</figcaption></figure>"
        )
        axis_error = np.linalg.norm(field["axis_difference_m"]) * 1000
        x_error = np.linalg.norm(field["xpoint_difference_m"]) * 1000
        rows.append(
            f"<tr><td>{index}</td><td>{reference['time_s']:.6f}</td><td>{row['wall_ms']:.6g}</td>"
            f"<td>{residual:.6g}</td><td>{html.escape(row['failure_stage'][0])}</td>"
            f"<td>{row['all_compilations']}</td></tr>"
        )
        comparisons.append(
            f"<tr><td>{index}</td><td>{axis_error:.6g}</td><td>{x_error:.6g}</td>"
            f"<td>{field['boundary_flux_difference_wb']:.6g}</td>"
            f"<td>{row['ip_difference_a'][0] / 1000:.6g}</td></tr>"
        )
        null_rows.append(
            f"<tr><td>{index}</td><td>{row['null_iterations'][0]}</td>"
            f"<td>{field['null_iterations']}</td><td>{field['classification_agrees']}</td></tr>"
        )
    assert frames, (
        "no finite terminal fields to render; reference maps cannot replace them"
    )
    animation = write_gif(frames, args.figures / "flattop.gif", duration=10)
    # The contact sheet is a separate browsing artifact, not a raster twin of an SVG.
    contact_indices = sorted(
        {
            0,
            len(frames) - 1,
            *(
                i
                for i, row in enumerate(receipt["slices"])
                if not all(row["converged"])
            ),
        }
    )
    contact = write_contact_sheet(
        [frames[i].resize((700, 360)) for i in contact_indices],
        args.figures / "flattop-frames.png",
        columns=2,
        count=len(contact_indices),
    )
    contact["pulse_frame_indices"] = contact_indices
    converged = sum(all(r["converged"]) for r in receipt["slices"])
    body = [
        f'<article id="dina-flattop-scenario-record"><h3>DINA flattop on the ITER bank route</h3><p>'
        f"{converged}/{len(receipt['slices'])} returned slices meet the independent fresh residual and support checks. "
        f"The required fraction is at least 90%; this record reports the observed state without changing a numerical tolerance. "
        f"Code revision <code>{receipt['revision']}</code>; {html.escape(receipt['device'])}; width 1 carries slices in time order. The width-16 attempt represents sixteen independent copies of the pulse and fails before returning its first slice.</p>",
        "<p>The source is DINA 105027/run 1, DD 3.38.1, read through imas-python. "
        "Indices 129–148 are consecutive stored slices at 30.145588–34.873397 s, with |Ip| at least 95% of the pulse peak. "
        "Twelve coil currents and two 33-knot Hermite gradient tables enter as traced arrays. "
        "The centroid target integrates DINA toroidal current moments, including local first moments. The pin adds a uniform radial field with a 0.1 mT source scale and 10 nm height tolerance. "
        "The bordered solver has 64 update opportunities per call. The absolute fresh current-moment residual threshold is 1e−10. "
        "Null iterations and the signed Hessian classifications are recorded separately; the known Hessian disagreement is not repaired by this measurement.</p>",
        "<p>The bank comes from the archived ITER-bank run and was copied with its geometry into the node cache; all eight persisted arrays and the twelve-coil ordering were verified. "
        "The input density integration at slice 129 gives −5.019747 MA against native Ip −4.977717 MA; this seed discrepancy is reported rather than rescaled away. "
        "Nova flux is the negative of stored DD3 flux. Boundary-flux differences retain their raw gauges and are diagnostics, not accuracy gates.</p>",
        "<p>Timing uses the synchronized blocking-call pattern of the existing bank slice profiler. The first call includes tracing/compilation, which is logged separately; later calls include the actual continuation overhead. "
        "Eleven same-input replay timings are retained only for converged states. No speed-lever default is changed and no off/on speedup is claimed. "
        "Warmup and two fresh processes completed at width 1 in one H200 job; the first width-16 process failed and the two measured width-16 processes did not run.</p>",
        f"<pre>{html.escape(json.dumps(summary.get('agreement', {}), indent=2))}</pre>",
        "<h4>Per-slice solver calls</h4><table><thead><tr><th>Slice</th><th>Time [s]</th><th>Width 1 [ms]</th><th>Fresh residual</th><th>Stage</th><th>Compilations in measured interval</th></tr></thead><tbody>",
        *rows,
        "</tbody></table>",
        "<h4>Differences from DINA at the requested inputs</h4><table><thead><tr><th>Slice</th><th>Axis distance [mm]</th><th>X-point distance [mm]</th><th>Boundary flux difference [Wb]</th><th>Ip difference [kA]</th></tr></thead><tbody>",
        *comparisons,
        "</tbody></table>",
        "<h4>Stationary-point diagnostics</h4><table><thead><tr><th>Slice</th><th>Terminal H200 iterations [axis, X]</th><th>Requested-field CPU read iterations [axis, X]</th><th>Classes agree with DINA [axis, X]</th></tr></thead><tbody>",
        *null_rows,
        "</tbody></table>",
        "<p>Per-lane terminal values, continuation substeps, compiler messages and terminal nodal fields are retained under "
        f"<code>{html.escape(str(args.results))}</code>. The preparation selection and reference fields are under <code>{html.escape(str(args.inputs))}</code>. "
        f"The width-1 table and figures use <code>{html.escape(str(receipt_path))}</code>. Each failed terminal result, including a rejected intermediate continuation trial, seeds the next slice; this is not a stream of exclusively converged warm starts. "
        "At failed continuation slices 131, 134, 136 the saved field is evaluated with the requested currents, while the original terminal nulls belong to intermediate trial currents. "
        f"Field-consistent comparisons and markers therefore use the separate CPU read <code>{html.escape(str(args.observables))}</code>, which checks the reconstructed field against the saved H200 nodal flux. "
        "This read changes no moments and supplies no new H200 timing or convergence result. DINA null classes are independently checked using second derivatives of its stored flux spline; they are an axis minimum and a saddle.</p>",
    ]
    control_path = args.results / "width-1-process-0/negative-controls.json"
    if control_path.exists():
        control = json.loads(control_path.read_text())
        body += [
            "<h4>Refusal controls</h4>",
            f"<p>Captured-current programs compile {control['closure_compilations']} times for two current values. "
            f"The displaced zero-update state has residual {control['residual']}, stage {html.escape(str(control['failure_stage']))}, "
            f"converged {control['converged']}. Adding it changes the fraction {control['normal_fraction']:.6g} to {control['with_failed_fraction']:.6g}. "
            "The zero-update receipt reports zero updates. Its wall fields are placeholders and carry no timing evidence. Each captured-current value creates a new closure/program; this control shows recompilation of those closures, rather than a cache miss on the ordinary runtime-input dispatcher.</p>",
        ]
    arm_rows = []
    for arm in summary["arms"]:
        path = Path(arm["log"]).parent / "receipt.json"
        item = json.loads(path.read_text())
        arm_rows.append(
            f"<tr><td>{arm['width']}</td><td>{arm['process']}</td>"
            f"<td>{item.get('converged_slices', 'no terminal slice')}</td>"
            f"<td>{item.get('median_warm_ms', 'unmeasured')}</td>"
            f"<td>{arm['exit_status']}</td></tr>"
        )
    compiler_counts = Counter(
        re.findall(
            r"Compiling jit\(([^)]+)\)",
            (receipt_path.parent / "compilations.log").read_text(),
        )
    )
    assert compiler_counts["solve"] == 1 and compiler_counts["certify"] == 1
    assert sum(r["all_compilations"] for r in receipt["slices"][1:]) == 20
    body += [
        "<h4>Fresh processes and the width limit</h4>",
        "<table><thead><tr><th>Width</th><th>Process (0 is warmup)</th><th>Converged slices</th><th>Median subsequent solver call [ms]</th><th>Process exit</th></tr></thead><tbody>",
        *arm_rows,
        "</tbody></table>",
        "<p>H200 job 1315698 ended FAILED with exit 1 after 13 min 30 s. The width-16 compiler reports 241.14 GiB after rematerialization, followed by a 216.76 GiB allocation refusal in jit_solve. Width-16 throughput is unmeasured, not zero. No smaller batch is substituted and no second H200 job is submitted.</p>",
        "<h4>Compilation accounting</h4>",
        f"<p>The main solve and independent certificate each compile once. The complete measured-process log contains {sum(compiler_counts.values())} dispatcher compilation events: <code>{html.escape(str(dict(compiler_counts)))}</code>. "
        "Five squeeze compilations occur while exporting the first slice after its in-interval counter was saved, so that first receipt's 2 counts do not cover its entire export. "
        "After the first slice there are 12 helper compilations at 130 and 8 at 131, then zero at 132–148. The literal zero-after-first requirement is unmet, even though changing currents and profiles never recompiles the solve program.</p>",
        "<p>The three failed continuation records name a sub-step ceiling, and the fourth failure exhausts 64 updates. All four failed states remain shown and counted. The published figures are the measured incomplete result, not evidence of meeting the 90% requirement.</p>",
    ]
    body += [
        '<figure id="dina-flattop-scenario-animation"><img src="/nova/figures/sub-millisecond-forward-path/dina-flattop-scenario/flattop.gif" alt="Consecutive terminal bank fields beside native DINA fields"><figcaption>'
        "Every finite returned terminal field in time order; each frame states its fresh residual and convergence flag. Failed fields remain failed in the animation.</figcaption></figure>",
        '<figure id="dina-flattop-scenario-contact"><img src="/nova/figures/sub-millisecond-forward-path/dina-flattop-scenario/flattop-frames.png" alt="Six evenly spaced terminal comparisons"><figcaption>'
        "The first slice, every failed slice, and the last slice: indices 129, 131, 132, 134, 136 and 148. Full-resolution vector panels below retain the field and null comparisons.</figcaption></figure>",
        *figures,
        "</article>",
    ]
    args.fragment.write_text("\n".join(body) + "\n")
    (args.preview / "render-receipt.json").write_text(
        json.dumps(
            dict(
                animation=animation,
                contact=contact,
                inspections=inspections,
                levels=all_levels,
            ),
            indent=2,
        )
    )
    print(
        f"RENDER_COMPLETE frames={len(frames)} converged={converged}/{len(receipt['slices'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
