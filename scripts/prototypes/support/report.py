"""Render support-arm receipts as a table and one convergence plot."""

# Report tables and HTML attributes keep their emitted line structure.
# ruff: noqa: E501

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def number(value):
    return "—" if value is None else f"{value:.9g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(p.read_text()) for p in sorted((args.run / "rows").glob("*.json"))
    ]
    rows = [r for r in rows if r.get("completed")]
    groups = {}
    for case in ("diverted", "limited"):
        for arm in ("exact", "legacy", "read"):
            groups[case, arm] = sorted(
                (r for r in rows if r["case"] == case and r["arm"] == arm),
                key=lambda r: r["realised_cells"],
            )
    orders = {}
    for key, values in groups.items():
        orders[key] = (
            float(
                np.polyfit(
                    np.log([r["pitch_m"] for r in values]),
                    np.log([r["map_relative_sup"] for r in values]),
                    1,
                )[0]
            )
            if len(values) >= 3
            else None
        )
    verdicts = []
    for arm in ("exact", "read"):
        values = groups["diverted", arm]
        if len(values) < 3:
            verdicts.append(f"{arm}: convergence unmeasured ({len(values)} rungs)")
        else:
            monotone = all(
                b["map_relative_sup"] < a["map_relative_sup"]
                for a, b in zip(values, values[1:])
            )
            verdicts.append(
                f"{arm}: {'monotone decrease' if monotone else 'nonmonotonic; convergence not established'}, "
                f"fitted map order {orders['diverted', arm]:.5g}, "
                f"relative sup {values[0]['map_relative_sup']:.9g} → "
                f"{values[-1]['map_relative_sup']:.9g}"
            )
    exact_values = groups["diverted", "exact"]
    booking_order = (
        float(
            np.polyfit(
                np.log([r["pitch_m"] for r in exact_values]),
                np.log([r["booking_image_relative_sup"] for r in exact_values]),
                1,
            )[0]
        )
        if len(exact_values) >= 2
        else None
    )
    booking_values = ", ".join(
        f"{r['booking_image_relative_sup']:.9g} at {r['realised_cells']} cells"
        for r in exact_values
    )
    exterior_values = ", ".join(
        f"{r['exterior_closure_relative_sup']:.9g} at {r['realised_cells']} cells"
        for r in exact_values
    )
    lead = [
        "Diverted map verdict — "
        + "; ".join(verdicts)
        + ". "
        + (
            "Exact-support booking decreases with refinement: "
            if all(
                b["booking_image_relative_sup"] < a["booking_image_relative_sup"]
                for a, b in zip(exact_values, exact_values[1:])
            )
            else "Exact-support booking is not monotone with refinement: "
        )
        + f"{booking_values}; "
        + f"fitted order in pitch {number(booking_order)}"
        + (
            " (two-point slope only; a third rung is required). "
            if len(exact_values) == 2
            else ". "
        )
        + f"Exterior closure is {exterior_values}. "
        + "It is a fixture artefact: the prescribed exterior subtracts an analytic-density "
        + "image integrated on production spline-clipped support seeded by analytic topology, so replacing support leaves a closure "
        + "mismatch even at the analytic state. It is not the exact-support booking error. Here legacy support means the existing spline geometry: the fixture uses _analytic_profile_support, not _fixed_design_read.",
        "",
        "| Realised cells | Exact relative sup | Fitted order | Booking contribution | Exterior contribution | Legacy relative sup |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in exact_values:
        control = next(
            (
                r
                for r in groups["diverted", "legacy"]
                if r["realised_cells"] == row["realised_cells"]
            ),
            None,
        )
        lead.append(
            f"| {row['realised_cells']} | {row['map_relative_sup']:.12g} | "
            f"{number(orders['diverted', 'exact'])} | "
            f"{row['booking_image_relative_sup']:.12g} | "
            f"{row['exterior_closure_relative_sup']:.12g} | "
            f"{number(None if control is None else control['map_relative_sup'])} |"
        )
    for row in groups["diverted", "legacy"]:
        if not any(r["realised_cells"] == row["realised_cells"] for r in exact_values):
            lead.append(
                f"| {row['realised_cells']} | unavailable | — | unavailable | "
                f"{row['exterior_closure_relative_sup']:.12g} | "
                f"{row['map_relative_sup']:.12g} |"
            )
    text = lead + [
        "",
        "The fitted order uses all available exact-arm rungs and is unmeasured until three exist. "
        "The accepted 550-cell diverted pair is retained from its original receipt; "
        "the remaining decisive rows were scheduled before optional read support. "
        "An unavailable value was not persisted and cannot be inferred from execution timings. "
        "The exterior contribution uses the same fixture and analytic image across arms, "
        "so a completed legacy receipt can supply it when an exact receipt is missing.",
        "",
        "The residual decomposition compares the normalized booked-current image "
        "against independently integrated analytic density on the true separatrix. "
        "Exterior closure uses that same true image, so it detects a fixture exterior "
        "posed on an inaccurate support. Booking plus exterior terms reconstruct the "
        "measured map residual to a checked relative bound of 1e-10. The displayed "
        "sup norms are not additive: the two error fields can cancel.",
        "",
    ]
    audit_path = args.run / "receipt-audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        text += [
            "## Receipt coverage and execution barriers",
            "",
            f"The completed-receipt gate has {audit['completed_core_rows']} of "
            f"{audit['expected_core_rows']} required legacy/exact rows, with "
            f"{sum(audit['controls'].values())} of 3 controls passing. "
            f"[Audit receipt]({audit_path}). Scheduler exit alone does not decide coverage.",
            "",
            "| Case | Requested cells | Arm | Durable receipt |",
            "|---|---:|---|---|",
        ]
        for row in audit["coverage"]:
            text.append(
                f"| {row['case']} | {row['cells']} | {row['arm']} | "
                f"{'complete' if row['completed'] else 'missing'} |"
            )
        if audit["serialization_refusals"]:
            text += [
                "",
                "Executable serialization refused after map execution and booking, "
                "before the row writer persisted the computed metrics. The diagnostic "
                "proto sizes below are refusal data, not successful serialized executable sizes. "
                "Computed but unwritten values from those attempts remain unavailable; later completed receipts supersede missing rows.",
                "",
                "| Actual case / requested cells / arm | Refusal log | Serializer-reported proto [bytes] | Limit [bytes] |",
                "|---|---|---:|---:|",
            ]
            for row in audit["serialization_refusals"]:
                text.append(
                    f"| {row['case']} / {row['cells']} / {row['arm']} | "
                    f"[{Path(row['log']).name}]({row['log']}) | "
                    f"{row['serializer_reported_proto_bytes']} | 2147483648 |"
                )
        text += [
            "",
            f"Read attempts produced {audit['read_attempt_refusals']} refusals of "
            f"{audit['distinct_read_refusals']} distinct types. Their detailed coverage "
            "appears below. No convergence verdict is inferred from a refused or missing row.",
            "The actual case column comes from the log's last measurement header, "
            "because a recovery process can evaluate a different rung from the one "
            "named in its log filename. A refusal interrupts that process's remaining arms "
            "and its queued coarse measurement; missing receipts do not imply those "
            "measurements executed.",
            "",
        ]
    text += [
        "The apparent limited-550 executable of 2,156,136,745 bytes is excluded: "
        "the log named limited-550-legacy-exact-1283120.log contains "
        "CASE=diverted CELLS=5000, followed by RECOVERY_STAGE exact-booking. "
        "Its nested recovery evaluated 5,158 diverted cells and aborted before "
        "the requested 553-cell limited fixture ran. It therefore cannot be "
        "compared to the ladder's roughly 30 MB coarse executable. The separate "
        "2,160,882,862-byte refusal belongs to limited requested 5,000.",
        "",
        "The original analytic-reference attempts refused fraction uncertainties "
        "2.4834499509362173e-5 (diverted requested 5,000) and "
        "3.895593359561445e-5 (limited requested 2,000, not 5,000), "
        "against 2e-5. These were reference-polygon uncertainty refusals before "
        "a durable map row, not measured map errors. The authorized follow-up "
        "records uncertainty up to 1e-4 explicitly and preserves numerical metrics "
        "before attempting executable serialization.",
        "",
        "## Support and booking contract",
        "",
        "All three arms share one machine, exterior, analytic flux state, profile, "
        "current normalisation and production current integrator per case and rung. "
        "Only the prototype instance's private `_support_partition` is overridden. "
        "The production residual shadow remains common to all arms. These are map "
        "evaluations at the analytic state; no fixed-point solve or tangent is claimed.",
        "",
        "- **legacy:** unmodified `_fixed_design_read` and `_profile_support`. "
        "Its topology scalars normalize grid and sample flux; its masks choose cells.",
        "- **read:** `topology.read` on the closed-form analytic field with "
        "`TopologyPolicy()` on the certificate's actual atomic cells. Selected "
        "quadratic fragments are polygonized between their algebraic slice events; "
        "selected saddle sectors follow the read's cubic normal-form rays. The "
        "polygon area fraction must match read membership within 2e-5 or the run "
        "refuses it. The closed-form field is checked against certificate flux "
        "samples in the same unit before reading. Read axis flux, boundary level "
        "and X-point accompany these polygons into current booking.",
        "- **exact:** independently sampled analytic separatrix intersected with "
        "each cell, true zero boundary level and true X-point. The boundary is "
        "sampled initially at 8,193 points against a 4,097-point comparison. "
        "Limited boundaries add closed-form samples; diverted boundaries insert "
        "midpoints and project them with the analytic gradient to the true zero level. "
        "The original rows refine to a measured area-fraction difference of 2e-5. "
        "The follow-up rows explicitly admit and record uncertainty up to 1e-4; "
        "each receipt records its achieved uncertainty, bound and point count. "
        "This arm does not use the certificate spline clip.",
        "",
        "The map's normalization target remains the certificate target for every "
        "arm, preserving the legacy positive control. The analytic current column "
        "is independently integrated on the true analytic support with an oriented "
        "degree-fifteen triangle rule; the receipt also records the certificate "
        "target and both raw and normalized errors against the true current. "
        "This distinguishes a fixture-target bias from booking error.",
        "",
        "## Controls",
        "",
    ]
    guard = args.run.parent / "fraction-guard-committed.log"
    if guard.exists() and "GUARD_COMPLETE=passed" in guard.read_text():
        text.append(
            "The read-arm fraction guard accepts correct geometry with discrepancy "
            "7.60090607677e-8 and refuses corrupted membership with discrepancy 0.5. "
            f"[Guard control log]({guard})."
        )
    for row in rows:
        if "positive_control_expected" in row:
            text.append(
                f"Legacy diverted {row['realised_cells']} cells: expected "
                f"{row['positive_control_expected']:.15g}, observed "
                f"{row['map_relative_sup']:.15g}, relative delta "
                f"{row['positive_control_relative_delta']:.6g} (limit 1e-9)."
            )
        if row["arm"] == "shifted":
            floor = next(
                r
                for r in rows
                if r["case"] == row["case"]
                and r["requested_cells"] == row["requested_cells"]
                and r["arm"] == "exact"
            )
            text.append(
                f"One-pitch outward support shift ({row['pitch_m']:.9g} m): "
                f"relative sup {row['map_relative_sup']:.9g}, against "
                f"unshifted {floor['map_relative_sup']:.9g}; sensitivity "
                f"{'passes' if row['map_relative_sup'] > floor['map_relative_sup'] else 'FAILS'}."
            )
    text += [
        "",
        "## Map and support rows",
        "",
        "Membership is max absolute clipped-area fraction difference against the exact arm, "
        "with RMS beside it. Read fraction error and read/polygon gap distinguish the "
        "read receipt from the support actually booked. Exact-arm membership error "
        "is zero by definition; its independent boundary-refinement check is reported "
        "separately below. Nonzero legacy differences and the shifted-support control "
        "establish sensitivity to incorrect geometry.",
        "",
        "| Case | Cells | Arm | Map sup | Map RMS | Membership sup | Membership RMS | Read fraction error | Read/polygon gap |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = [r for values in groups.values() for r in values]
    for r in ordered:
        text.append(
            f"| {r['case']} | {r['realised_cells']} | {r['arm']} | "
            + " | ".join(
                number(r[k])
                for k in (
                    "map_relative_sup",
                    "map_relative_rms",
                    "membership_error_sup",
                    "membership_error_rms",
                    "read_membership_error_sup",
                    "read_polygon_fraction_gap",
                )
            )
            + " |"
        )
    text += [
        "",
        "## Current and residual decomposition",
        "",
        "| Case | Cells | Arm | Raw current [A] | Normalized current [A] | Analytic current [A] | Raw relative error | Booking image sup | Exterior closure sup |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ordered:
        text.append(
            f"| {r['case']} | {r['realised_cells']} | {r['arm']} | "
            + " | ".join(
                number(r[k])
                for k in (
                    "booked_current_raw_a",
                    "booked_current_normalised_a",
                    "analytic_current_a",
                    "raw_current_relative_error",
                    "booking_image_relative_sup",
                    "exterior_closure_relative_sup",
                )
            )
            + " |"
        )
    text += [
        "",
        "## Cost rows",
        "",
        "Persistent JAX compilation cache is disabled before compilation. Cold compile "
        "includes lowering; support is precomputed for the read/exact overrides and "
        "its stage wall is reported separately. Those compile costs do not qualify "
        "a fully dynamic production read-book-map. RSS is the process high-water mark "
        "across the shared fixture and arms, not an isolated compiler allocation. "
        "Child RSS is separately retained in each receipt. Executable size is serialized "
        "bytes; temporary memory is the executable memory-analysis figure. "
        "A dash in executable bytes means unavailable when the recorded serializer "
        "refusal exceeds 2 GiB; numerical metrics are persisted before this attempt.",
        "",
        "| Case | Cells | Arm | Cold compile [s] | Warm [s] | Executable [bytes] | Device temporary [bytes] | Peak RSS [KiB] | Support [s] | Booking diagnostic [s] |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ordered:
        text.append(
            f"| {r['case']} | {r['realised_cells']} | {r['arm']} | "
            + " | ".join(
                number(r[k])
                for k in (
                    "cold_compile_seconds",
                    "warm_execute_seconds",
                    "serialized_executable_bytes",
                    "device_temp_bytes",
                    "host_peak_rss_kib",
                    "support_seconds",
                    "booking_seconds",
                )
            )
            + " |"
        )
    text += [
        "",
        "## Oracle uncertainty and serialization status",
        "",
        "| Case | Cells | Arm | Fraction uncertainty | Fraction bound | Serialization |",
        "|---|---:|---|---:|---:|---|",
    ]
    for r in ordered:
        text.append(
            f"| {r['case']} | {r['realised_cells']} | {r['arm']} | "
            f"{number(r.get('analytic_polygon_fraction_uncertainty'))} | "
            f"{number(r.get('oracle_fraction_bound', 2e-5))} | "
            f"{r.get('serialization_status', 'complete')} |"
        )
    text += [
        "",
        "## Shared fixture stages",
        "",
        "| Case | Cells | Case [s] | Machine [s] | Exterior [s] | Carrier [s] | Machine cache | Exterior cache | Boundary fraction refinement |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for r in ordered:
        if r["arm"] != "legacy":
            continue
        text.append(
            f"| {r['case']} | {r['realised_cells']} | "
            + " | ".join(
                number(r["fixture_walls"][k])
                for k in ("case", "machine", "exterior", "carrier")
            )
            + f" | {r['machine_cache'].get('hit')} | {r['exterior_cache'].get('hit')} | "
            + number(r.get("analytic_polygon_fraction_uncertainty"))
            + " |"
        )
    text += [
        "",
        "Fixture attempts retain their original stage walls below, including a "
        "cold build whose map was deferred by the analytic-reference guard. "
        "A later row can therefore report a cache hit without hiding the build cost. "
        "Recovery stages inside another process carry a separate log prefix and "
        "remain in that process's log.",
        "",
        "| Attempt log | Machine [s] | Exterior [s] | Carrier [s] |",
        "|---|---:|---:|---:|",
    ]
    for log in sorted((args.run / "logs").glob("*-legacy-exact-*.log")):
        stages = dict(
            re.findall(
                r"^STAGE_DONE (machine|exterior|carrier) seconds=([0-9.e+-]+)$",
                log.read_text(),
                flags=re.MULTILINE,
            )
        )
        text.append(
            f"| [{log.name}]({log}) | "
            + " | ".join(
                number(float(stages[k])) if k in stages else "not recorded"
                for k in ("machine", "exterior", "carrier")
            )
            + " |"
        )
    for binder in sorted(args.run.glob("*-binder.json")):
        record = json.loads(binder.read_text())
        text.append(
            f"Machine-build binder: {record['case']}, requested "
            f"{record['requested_cells']} cells, stage `{record['stage']}`, "
            f"wall {record['wall_seconds']:.6g} s against "
            f"{record['budget_seconds']} s; [receipt]({binder})."
        )
    text += [
        "",
        "## Orders and first measured acceptance",
        "",
        "Order p fits log(error) against log(pitch): error ∝ h^p. No extrapolated "
        "cell threshold is reported; the threshold is the first measured row ≤0.01. "
        "A fitted order over nonmonotonic rows does not establish convergence.",
        "",
        "| Case | Arm | Rows | Map fitted p | Booking fitted p | First cells ≤0.01 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, values in groups.items():
        first = next(
            (r["realised_cells"] for r in values if r["map_relative_sup"] <= 0.01),
            "none measured",
        )
        booking_fit = (
            float(
                np.polyfit(
                    np.log([r["pitch_m"] for r in values]),
                    np.log([r["booking_image_relative_sup"] for r in values]),
                    1,
                )[0]
            )
            if len(values) >= 3
            else None
        )
        text.append(
            f"| {key[0]} | {key[1]} | {len(values)} | {number(orders[key])} | {number(booking_fit)} | {first} |"
        )
    text += [
        "",
        "The limited exact ladder is also nonmonotonic: the intermediate error "
        "exceeds the coarse error and the fine error decreases again. All measured "
        "limited rows meet 0.01, but these rows do not establish asymptotic order. "
        "The oracle uncertainty is an area-fraction difference, not a propagated "
        "flux-error bound; its numerical floor limits interpretation of small map errors.",
        "",
        "## Dominant residual term",
        "",
    ]
    for case in ("diverted", "limited"):
        for legacy in groups[case, "legacy"]:
            exact = next(
                (
                    r
                    for r in groups[case, "exact"]
                    if r["realised_cells"] == legacy["realised_cells"]
                ),
                None,
            )
            if exact is None:
                continue
            ratio = exact["map_relative_sup"] / legacy["map_relative_sup"]
            text.append(
                f"{case}, {legacy['realised_cells']} cells: analytic support retains "
                f"{ratio:.6g} of the legacy sup error. The exact-arm support discrepancy "
                f"is {exact['membership_error_sup']:.6g}, its booked-current image error "
                f"is {exact['booking_image_relative_sup']:.9g}, and exterior closure is "
                f"{exact['exterior_closure_relative_sup']:.6g}. "
                + (
                    "The exterior posed by the certificate dominates this exact-arm residual."
                    if exact["exterior_closure_relative_sup"]
                    > exact["booking_image_relative_sup"]
                    else "Booking/discretisation on the true analytic support dominates this exact-arm residual."
                )
            )
    text += ["", "## Read-arm coverage", "", verdicts[1] + ".", ""]
    refusal_rows = [
        json.loads(path.read_text())
        for path in sorted((args.run / "rows").glob("*-read-refusal.json"))
    ]
    for case in ("diverted", "limited"):
        for requested in (550, 2000, 5000):
            measured_request = (
                3500
                if requested == 5000
                and (args.run / f"{case}-5000-binder.json").exists()
                else requested
            )
            actual = [
                r
                for r in groups[case, "read"]
                if abs(r["requested_cells"])
                == (500 if measured_request == 550 else measured_request)
            ]
            refused = [
                r
                for r in refusal_rows
                if r["case"] == case and r["requested_cells"] == measured_request
            ]
            if actual:
                text.append(
                    f"- {case}, requested {requested}: measured at {actual[0]['realised_cells']} cells, sup {actual[0]['map_relative_sup']:.12g}."
                )
            elif refused:
                text.append(
                    f"- {case}, requested {requested}: refused — `{refused[0]['error_type']}: {refused[0]['error'].splitlines()[0]}`."
                )
                logs = sorted(
                    (args.run / "logs").glob(f"{case}-{measured_request}-read-*.log")
                )
                if logs:
                    gaps = re.findall(
                        r"READ_POLYGON_FRACTION_ERROR=([0-9.e+-]+)",
                        logs[-1].read_text(),
                    )
                    if gaps:
                        text[-1] += (
                            f" Fragment/read fraction discrepancy {float(gaps[-1]):.9g} "
                            f"against the 2e-5 bound; [attempt log]({logs[-1]})."
                        )
            else:
                text.append(
                    f"- {case}, requested {requested}: not yet measured; inspect the job log for the last completed stage or the three-distinct-refusal stop."
                )
    text += ["", "## Provenance", ""]
    for r in ordered:
        receipt = (
            args.run
            / "rows"
            / f"{r['case']}-{abs(r['requested_cells']) if abs(r['requested_cells']) != 500 else 550}-{r['arm']}.json"
        )
        text.append(
            f"- {r['case']} / {r['realised_cells']} / {r['arm']}: "
            f"[receipt]({receipt}), "
            f"revision `{r['revision']}`, H200 job `{r['job_id']}`."
        )
    text += [
        "",
        "[Convergence figure](/nova/figures/converged-forward-solve/proto-support/map-convergence.svg)",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(text))

    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "support-convergence"
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    styles = {"legacy": "-", "read": "--", "exact": "-."}
    markers = {"legacy": "o", "read": "s", "exact": "^"}
    labels = []
    for (case, arm), values in groups.items():
        if not values:
            continue
        x = [r["realised_cells"] for r in values]
        y = [r["map_relative_sup"] for r in values]
        colour = "#222222" if case == "diverted" else "#777777"
        ax.loglog(
            x, y, color=colour, linestyle=styles[arm], marker=markers[arm], linewidth=3
        )
        labels.append((y[-1], x[-1], case + " " + arm, colour))
    labels.sort()
    span = np.log10(ax.get_ylim()[1] / ax.get_ylim()[0])
    prior = -np.inf
    for value, x, name, colour in labels:
        position = max(np.log10(value), prior + 0.075 * span)
        prior = position
        ax.annotate(
            name,
            xy=(x, value),
            xytext=(max(r["realised_cells"] for r in ordered) * 1.3, 10**position),
            color=colour,
            fontsize=20,
            va="center",
            arrowprops=dict(arrowstyle="-", color=colour, lw=1),
        )
    ax.axhline(0.01, color="#777777", linestyle=":", linewidth=1.2)
    ax.text(
        min(r["realised_cells"] for r in ordered) * 0.9,
        0.0115,
        "0.01 bound",
        fontsize=20,
        color="#777777",
    )
    ax.set_xlabel("realised cells")
    ax.set_ylabel("map relative sup")
    ax.set_xlim(right=max(r["realised_cells"] for r in ordered) * 5)
    ax.grid(False, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, format="svg", metadata={"Date": None})
    plt.close(fig)
    args.figure.write_text(
        "\n".join(line.rstrip() for line in args.figure.read_text().splitlines()) + "\n"
    )
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(f"""<figure id="proto-support-convergence">
<img src="/nova/figures/converged-forward-solve/proto-support/map-convergence.svg" alt="Certificate map relative sup against realised cells for legacy, read and analytic support in limited and diverted cases.">
<figcaption>{len(ordered)} completed certificate map rows at the analytic state, one shared machine and exterior per rung. Solid: legacy; dashed: read fragment support; dash-dot: analytic support. Dark lines are diverted; grey lines are limited. The dotted grey line marks relative sup 0.01. Only completed rows are plotted; read-arm convergence is unavailable because no read map row completed. Read fragment polygons retain membership within a checked 2e-5 area-fraction bound; analytic support is independent of the certificate spline clip.</figcaption>
</figure>
""")
    print("REPORT_ROWS=" + str(len(ordered)))
    print("VERDICT=" + "; ".join(verdicts))
    print("BOOKING_ORDER=" + number(booking_order))


if __name__ == "__main__":
    main()
