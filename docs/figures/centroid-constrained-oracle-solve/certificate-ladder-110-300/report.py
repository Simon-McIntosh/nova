"""Summarise persisted constrained-certificate receipts without tuning failures."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
)
CELLS = (110, 300)
SUPPORTS = ("exact", "whole-cell")
FIGURE_URL = (
    "/nova/figures/centroid-constrained-oracle-solve/cco-certificate-ladder-110-300/"
)


def _number(value: float) -> str:
    return f"{value:.3g}"


def _clause(value: float, limit: float, *, absolute: bool = False) -> str:
    passed = abs(value) <= limit if absolute else value <= limit
    return f"{'PASS' if passed else 'FAIL'} ({_number(value)})"


def _centroid(receipt: dict) -> tuple[bool, str]:
    pitch = receipt["characteristic_pitch_m"]
    errors = [abs(value) / pitch for value in receipt["solve"]["centroid_error_m"]]
    limits = receipt["row_tolerance_pitches"]
    passed = all(error <= limit for error, limit in zip(errors, limits, strict=True))
    reading = ", ".join(
        f"{_number(error)}/{_number(limit)}"
        for error, limit in zip(errors, limits, strict=True)
    )
    return passed, f"{'PASS' if passed else 'FAIL'} ({reading})"


def _row(receipt: dict, partner: dict | None) -> tuple[list[str], list[str]]:
    centroid_pass, centroid = _centroid(receipt)
    support = receipt["support"]
    same_state = (
        partner is not None
        and receipt["solve"]["state_sha256_binary64"]
        == partner["solve"]["state_sha256_binary64"]
    )
    controls = (
        "—"
        if support == "exact"
        else ("FAIL (same state)" if same_state else "unresolved (different state)")
    )
    verdict = [
        receipt["case"],
        str(abs(receipt["requested_cells"])),
        support,
        str(receipt["realised_cells"]),
        _clause(receipt["maximum_difference_of_span"], 1.1e-3),
        _clause(receipt["axis_error_pitches"], 0.1),
        _clause(receipt["boundary_level_offset_of_span"], 1e-3, absolute=True),
        _clause(receipt["terminal_global_residual"], 1e-12),
        _clause(receipt["terminal_row_residual"], 1e-12),
        centroid,
        controls,
    ]
    fields = receipt["compensating_field_t"]
    measure = [
        receipt["case"],
        str(abs(receipt["requested_cells"])),
        support,
        _number(fields[0]),
        _number(fields[1]),
        _number(receipt["compensating_field_t_abs_sup"]),
        _number(receipt["level_amplitude_wb"]),
        _number(receipt["terminal_row_residual_pitches"]),
        _number(receipt["backend_compile_or_cache_seconds"]),
        _number(receipt["solve_other_seconds"]),
        "yes" if receipt["converged"] and centroid_pass else "no",
    ]
    return verdict, measure


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(item)}</th>" for item in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(item)}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for case in CASES:
        for cells in CELLS:
            for support in SUPPORTS:
                stem = f"{case}-{cells}-{support}"
                records[stem] = json.loads((args.receipts / f"{stem}.json").read_text())
                if not (args.figures / f"{stem}.png").is_file():
                    raise FileNotFoundError(args.figures / f"{stem}.png")

    verdict_rows = []
    measure_rows = []
    figure_html = []
    for case in CASES:
        for cells in CELLS:
            exact = records[f"{case}-{cells}-exact"]
            for support, partner in (("exact", None), ("whole-cell", exact)):
                stem = f"{case}-{cells}-{support}"
                receipt = records[stem]
                verdict, measure = _row(receipt, partner)
                verdict_rows.append(verdict)
                measure_rows.append(measure)
                caption = (
                    f"{case}, requested {cells}, realised {receipt['realised_cells']} "
                    f"cells, {support} support. Analytic blue and solved orange "
                    "flux contours use shared physical levels; the right panel "
                    "shows their difference. Both panels show the wall and both "
                    "null sets. "
                    f"Global residual {_number(receipt['terminal_global_residual'])}; "
                    f"row residual {_number(receipt['terminal_row_residual'])}; "
                    f"converged {str(receipt['converged']).lower()}."
                )
                figure_html.append(
                    '<figure><img src="'
                    + FIGURE_URL
                    + escape(f"{stem}.png")
                    + '" alt="Shared-level analytic and solved flux contours, '
                    'difference contours, both null sets and the wall">'
                    + f"<figcaption>{escape(caption)}</figcaption></figure>"
                )

    verdict_headers = [
        "Case",
        "Cells",
        "Support",
        "Realised",
        "Max/span ≤0.0011",
        "Axis/pitch ≤0.1",
        "Net boundary/span ≤0.001",
        "Global ≤1e-12",
        "Row ≤1e-12",
        "Centroid |r|,|z|/derived limits [pitches]",
        "Whole-cell control",
    ]
    measure_headers = [
        "Case",
        "Cells",
        "Support",
        "Vertical field [T]",
        "Radial field [T]",
        "Field sup [T]",
        "Level [Wb]",
        "Centroid error [pitches]",
        "Backend compile/cache [s]",
        "Other solve [s]",
        "Converged",
    ]
    report = "\n".join(
        (
            "# Constrained certificate at 110 and 300 requested cells",
            "",
            "Titan job 1280637 completed in 01:57:58 with sacct exit 0:0, "
            "MEASUREMENT_COMPLETE and final EXIT=0. H200 job 1280636 stayed "
            "pending on Resources beyond one minute, so it was cancelled by ID "
            "before the prescribed titan fallback. The device was a Tesla P100, "
            "not an H200. All twelve receipts and terminal states landed under "
            f"`{args.receipts}` at source revision "
            "`f379a5c97e903f2b347d89fdaae109c6d31adf17`.",
            "",
            "No exact-support row met all clauses. The prior qualifying weak-110 "
            "receipt used a displaced analytic seed; this measurement used the "
            "production seed required here. Every whole-cell terminal state is "
            "bit-identical to its exact-support partner, so the intended "
            "centroid-versus-booking separation was not demonstrated. These are "
            "failed measurements, not qualified certificate rows.",
            "",
            "## Clause verdicts",
            "",
            _markdown_table(verdict_headers, verdict_rows),
            "",
            "The centroid column reports radial and vertical absolute errors "
            "in pitches, each divided by its own analytic-state-derived "
            "tolerance. The boundary value is net of the compensator flux. "
            "Whole-cell control requires movement to the oracle centroid "
            "alongside a retained booking error; the matching terminal hashes "
            "and failed centroid clauses mean it did not fire.",
            "",
            "## Amplitudes and timing",
            "",
            _markdown_table(measure_headers, measure_rows),
            "",
            "The field amplitude does not fall from 110 to 300 cells: weak and "
            "moderate stay at zero because no Newton promotion was accepted, "
            "while strong rises from 0.0130 T to 0.113 T. The timing instrument "
            "measures backend compile-or-cache calls separately from the rest "
            "of the solve wall. It does not isolate pure compilation from cache "
            "lookup, or pure execution from tracing and lowering; titan timing "
            "must not be quoted as H200 throughput.",
            "",
            "## Interpretation and follow-on",
            "",
            "The weak-110 production-seed row had global residual 1.764e-2, "
            "row residual 63.30, centroid error 0.2794 pitches and zero accepted "
            "Newton promotions. Its qualifying displaced-analytic-seed "
            "predecessor measured global 2.681e-14, row 9.529e-15 pitches, "
            "derived radial tolerance 1.660e-4 pitches and net span offset "
            "−3.860e-4. The seed difference is material. Strong-300 accepted "
            "four promotions but still ended at global residual 0.506 and "
            "centroid error 9.064 pitches. No failed row was tuned or rerun.",
            "",
            "A banked analytic-state support control at 1000 cells shows the "
            "instrument can see a whole-cell change: with an analytic-clipped "
            "exterior, weak chord support had map-floor RMS 0.129 of span "
            "against 3.58e-5 for exact support. The present terminal states "
            "do not reproduce the requested roughly 0.2-of-span separated "
            "booking error. A follow-on must investigate why the production "
            "seed stalls and verify that each solve actually exercises its "
            "declared support mode before treating the paired states as a "
            "negative control.",
            "",
            f"Log: `{args.receipts.parent / 'certificate.log'}`. "
            "Banked support control: "
            "`docs/figures/cut-cell-current-attribution/reposed-fixture/map-floor.json`.",
            "",
        )
    )
    fragment = "\n".join(
        (
            '<section id="constrained-certificate-ladder" '
            'data-source-job="1280637" '
            'data-source-revision="f379a5c97e903f2b347d89fdaae109c6d31adf17">',
            "<p>Titan P100 fallback: twelve rows landed, none qualified. "
            "The H200 submission remained pending on resources. All six "
            "whole-cell terminal states match their exact-support partners "
            "bit for bit, so the required booking-error negative control is "
            "unresolved. No failed row was tuned.</p>",
            "<h3>Clause verdicts</h3>",
            _html_table(verdict_headers, verdict_rows),
            "<h3>Compensators and timing</h3>",
            _html_table(measure_headers, measure_rows),
            "<p>The field amplitude is 0 at both weak and moderate resolutions "
            "and rises from 0.0130 T to 0.113 T in strong. Backend compile/cache "
            "time is reported separately from the remaining solve wall, and "
            "the P100 walls are not H200 timings. The weak-110 production seed "
            "stalled at global residual 1.764e-2 and row residual 63.30; the "
            "previous qualifying receipt started from a displaced analytic "
            "seed, a materially different start. A banked analytic-state "
            "control sees whole-cell booking at 1000 cells, but these "
            "nonconverged terminal states do not establish it.</p>",
            "<h3>Terminal flux panels</h3>",
            *figure_html,
            "</section>",
            "",
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    args.fragment.write_text(fragment)
    print(f"REPORT {args.report}")
    print(f"FRAGMENT {args.fragment}")


if __name__ == "__main__":
    main()
