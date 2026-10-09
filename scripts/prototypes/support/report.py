"""Render support-arm receipts as a table and one convergence plot."""

# Report tables and HTML attributes keep their emitted line structure.
# ruff: noqa: E501

import argparse
import json
from pathlib import Path

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
            verdicts.append(
                f"{arm}: {'decreases' if orders['diverted', arm] > 0 else 'does not decrease'} "
                f"with refinement, fitted order {orders['diverted', arm]:.5g}, "
                f"relative sup {values[0]['map_relative_sup']:.9g} → "
                f"{values[-1]['map_relative_sup']:.9g}"
            )
    exact_values = groups["diverted", "exact"]
    lead = [
        "Exact-arm diverted map — " + verdicts[0] + ".",
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
    text = lead + [
        "",
        "The fitted order uses all available exact-arm rungs and is unmeasured until three exist. "
        "The accepted 550-cell diverted pair is retained from its original receipt; "
        "the remaining decisive rows run before optional read support in the resumed allocation.",
        "",
        "The residual decomposition compares the normalized booked-current image "
        "against independently integrated analytic density on the true separatrix. "
        "Exterior closure uses that same true image, so it detects a fixture exterior "
        "posed on an inaccurate support. Booking plus exterior terms reconstruct the "
        "measured map residual to a checked relative bound of 1e-10.",
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
        "sampled at 8,193 points; a 4,097-point comparison bounds the area-fraction "
        "uncertainty to 2e-5. This arm does not use the certificate spline clip.",
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
        "read receipt from the support actually booked.",
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
        "bytes; temporary memory is the executable memory-analysis figure.",
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
        "## Shared fixture stages",
        "",
        "| Case | Cells | Case [s] | Machine [s] | Exterior [s] | Carrier [s] | Machine cache | Exterior cache |",
        "|---|---:|---:|---:|---:|---:|---|---|",
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
            + f" | {r['machine_cache'].get('hit')} | {r['exterior_cache'].get('hit')} |"
        )
    text += [
        "",
        "## Orders and first measured acceptance",
        "",
        "Order p fits log(error) against log(pitch): error ∝ h^p. No extrapolated "
        "cell threshold is reported; the threshold is the first measured row ≤0.01.",
        "",
        "| Case | Arm | Rows | Fitted p | First cells ≤0.01 |",
        "|---|---|---:|---:|---:|",
    ]
    for key, values in groups.items():
        first = next(
            (r["realised_cells"] for r in values if r["map_relative_sup"] <= 0.01),
            "none measured",
        )
        text.append(
            f"| {key[0]} | {key[1]} | {len(values)} | {number(orders[key])} | {first} |"
        )
    text += ["", "## Dominant residual term", ""]
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
    text += ["", "## Read-arm coverage", ""]
    refusal_rows = [
        json.loads(path.read_text())
        for path in sorted((args.run / "rows").glob("*-read-refusal.json"))
    ]
    for case in ("diverted", "limited"):
        for requested in (550, 2000, 5000):
            actual = [
                r
                for r in groups[case, "read"]
                if abs(r["requested_cells"]) == (500 if requested == 550 else requested)
            ]
            refused = [
                r
                for r in refusal_rows
                if r["case"] == case and r["requested_cells"] == requested
            ]
            if actual:
                text.append(
                    f"- {case}, requested {requested}: measured at {actual[0]['realised_cells']} cells, sup {actual[0]['map_relative_sup']:.12g}."
                )
            elif refused:
                text.append(
                    f"- {case}, requested {requested}: refused — `{refused[0]['error_type']}: {refused[0]['error']}`."
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
    text += ["", f"[Convergence figure]({args.figure})", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(text))

    plt.style.use("data-ink")
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
    fig.savefig(args.figure, format="svg")
    plt.close(fig)
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text("""<figure id="proto-support-convergence">
<img src="/nova/figures/converged-forward-solve/proto-support/map-convergence.svg" alt="Certificate map relative sup against realised cells for legacy, read and analytic support in limited and diverted cases.">
<figcaption>Certificate map at the analytic state, one shared machine and exterior per rung. Solid: legacy; dashed: read fragment support; dash-dot: analytic support. Dark lines are diverted; grey lines are limited. The dotted grey line marks relative sup 0.01. The read fragment polygons retain the read membership within a checked 2e-5 area-fraction bound; the analytic support is independent of the certificate spline clip.</figcaption>
</figure>
""")
    print("REPORT_ROWS=" + str(len(ordered)))
    print("VERDICT=" + "; ".join(verdicts))


if __name__ == "__main__":
    main()
