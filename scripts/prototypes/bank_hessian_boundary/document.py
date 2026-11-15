"""Fill the evidence fragment from the completed numerical receipt."""

import argparse
import hashlib
import json
from pathlib import Path


def pair(values):
    return " / ".join(f"{x:.6g}" for x in values)


def render(data):
    assert all(row["control"]["accepted"] for row in data["states"])
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    interior = data["interior_count"]
    boundary = len(data["nodes"])
    parts = [
        f'<section id="bank-hessian-boundary-result" data-receipt-sha256="{digest}">',
        "<h3>Interior and one-sided boundary checks of the retained ITER Hessians</h3>",
        f"<p><strong>The boundary-only explanation is refuted.</strong> The same "
        f"{interior + boundary} sampled nodes contain {interior} interior nodes "
        f"more than 8 mm from every source edge, and {boundary} boundary-group "
        "nodes. Both retained routes have large interior discrepancies against "
        "the independent estimate. The preceding all-node discrepancies are "
        "reproduced before interpreting this split.</p>",
        "<table><thead><tr><th>State</th><th>Component</th><th>Bank error</th>"
        "<th>Direct error</th><th>Reference envelope</th></tr></thead><tbody>",
    ]
    for row in data["states"]:
        for column, component in enumerate(("RZ", "ZZ")):
            item = row["interior"]
            parts.append(
                f"<tr><td>{row['state']}</td><td>ψ<sub>{component}</sub></td>"
                + "".join(
                    f"<td>{item[key][column]:.9g}</td>"
                    for key in ("bank", "direct", "reference_bound")
                )
                + "</tr>"
            )
    parts += [
        "</tbody></table>",
        "<p>Errors are Euclidean discrepancy norms divided by the independent "
        "total-field component norm, including coils. The numerical envelopes "
        "are empirical indicators, not interval enclosures. The corner correction "
        "in the retained reference already uses arbitrary-precision differences "
        "bounded by the stored coordinate’s actual boundary clearance.</p>",
        '<div id="bank-hessian-boundary-limits">',
        "<p>The boundary table compares each route with the lower-side estimate, "
        "upper-side estimate and their average at the smallest retained distance. "
        "Each entry is the relative norm error for ψ<sub>RZ</sub> / ψ<sub>ZZ</sub>. "
        "At active corners these are finite-distance comparisons, not finite "
        "limiting values. Neither route implements one consistent side or average.</p>",
        "<table><thead><tr><th>State</th><th>Route</th><th>Lower</th><th>Upper</th>"
        "<th>Average</th></tr></thead><tbody>",
    ]
    for row in data["states"]:
        for route in ("bank", "direct"):
            entry = row["boundary"][route]
            parts.append(
                f"<tr><td>{row['state']}</td><td>{route}</td>"
                + "".join(
                    f"<td>{pair(entry[k])}</td>" for k in ("lower", "upper", "average")
                )
                + "</tr>"
            )
    parts += [
        "</tbody></table>",
        "<p>For the nodes without a current-carrying corner in each state ("
        + ", ".join(str(r["smooth_boundary"]["count"]) for r in data["states"])
        + " nodes respectively), quadratic extrapolation "
        "of the three smallest approach distances estimates finite limits. The "
        "table reports errors against their average, the lower-versus-upper "
        "limit discrepancy, and an empirical envelope from extrapolation order "
        "and the finite-difference estimate. The receipt retains all three "
        "comparisons separately for each route.</p>",
        "<p>The observed side discrepancy is larger than the extrapolation-only "
        "envelope in some entries. These limits are therefore resolved to only "
        "a few parts in a million, not 1e-8; the envelope is an empirical "
        "indicator, not an enclosure. The route discrepancies remain much larger.</p>",
        "<table><thead><tr><th>State</th><th>Bank</th><th>Direct</th>"
        "<th>Side discrepancy</th><th>Envelope</th></tr></thead><tbody>",
    ]
    for row in data["states"]:
        entry = row["smooth_boundary"]
        parts.append(
            f"<tr><td>{row['state']}</td>"
            + "".join(
                f"<td>{pair(v)}</td>"
                for v in (
                    entry["bank"]["average"],
                    entry["direct"]["average"],
                    entry["side_difference"],
                    entry["envelope"],
                )
            )
            + "</tr>"
        )
    parts += [
        "</tbody></table>",
        "<p>The source-wedge calculation finds a nonzero logarithmic curvature "
        "coefficient at "
        + ", ".join(str(x) for x in data["corner_prediction"]["nonzero_per_state"])
        + " nodes in states 220, 333, 336 respectively. Such nominal corners have "
        "no finite one-sided Hessian limit or finite average. The jump in the "
        "trace, however, can be checked against the density forcing. The following "
        "relative discrepancies use the 1 mm approach, where the derivative "
        "calculation is best conditioned; all five distances are in the receipt.</p>",
        "<table><thead><tr><th>State</th><th>Trace jump versus density</th>"
        "<th>ψ<sub>ZZ</sub> jump versus density</th></tr></thead><tbody>",
    ]
    for row in data["states"]:
        entry = row["active_corner_jump"]
        parts.append(
            f"<tr><td>{row['state']}</td><td>{entry['trace_relative'][0]:.6g}</td><td>{entry['zz_relative'][0]:.6g}</td></tr>"
        )
    control = data["uniform_edge_control"]
    parts += [
        "</tbody></table>",
        f"<p>A uniform rectangular source at a horizontal edge recovers the density "
        f"jump to {control['relative']:.6g} relative at 100 nm, and "
        f"{control['extrapolated_relative']:.6g} after distance extrapolation. "
        "This is the positive control for the jump instrument.</p></div>",
    ]
    parts += [
        "<p>The Grad–Shafranov equation constrains ψ<sub>RR</sub> + ψ<sub>ZZ</sub> "
        "− ψ<sub>R</sub>/R = −2πμ<sub>0</sub>RJ. Across a smooth edge the Hessian "
        "jump is the forcing jump times n⊗n. ψ<sub>ZZ</sub> therefore carries "
        "the n<sub>Z</sub>² fraction, and ψ<sub>RZ</sub> the "
        "n<sub>R</sub>n<sub>Z</sub> "
        "fraction. At corners, opposing logarithmic terms cancel in the trace.</p>",
        "<p>The nominal corner differs from the rounded stored coordinate. The "
        "approach sequence diagnoses the nominal corner behaviour down to 100 nm; "
        "the retained arbitrary-precision correction estimates the derivative at "
        "the stored coordinate. The interior discrepancy survives both readings.</p>",
        '<p id="bank-hessian-boundary-controls">The sensitivity control multiplies '
        "the <em>measured bank and direct-route</em> interior ψ<sub>ZZ</sub> values "
        "by 1.000001 before recomputing their errors against the unchanged independent "
        "estimate. The table records norms of the changes in per-node residuals, "
        "normalised by the independent component norm. Comparing vectors before "
        "reducing them avoids cancellation between scalar error norms. A "
        "positive arm reproduces the retained unperturbed measurements; the "
        "perturbed arm must be refused in every state and route. These controls "
        "detect measurement-path regressions and do not certify either route’s "
        "physical accuracy.</p>",
        "<table><thead><tr><th>State</th><th>Bank change</th><th>Direct change</th>"
        "<th>Threshold</th></tr></thead><tbody>",
    ]
    for row in data["states"]:
        control = row["sensitivity"]
        parts.append(
            f"<tr><td>{row['state']}</td><td>{control['changes']['bank']:.9g}</td>"
            f"<td>{control['changes']['direct']:.9g}</td><td>{control['threshold']:.9g}</td></tr>"
        )
    parts += [
        "</tbody></table>",
        '<p id="bank-hessian-boundary-method">Each side uses cell-contained '
        "central differences of analytic first-derivative fields with four steps "
        "and eighth-order Richardson extrapolation. The split and the derivative "
        "stencils use one shared point-to-edge clearance primitive. At active "
        "corners only nearby source cells are reevaluated, with the independently "
        "measured smooth remainder held at the target. Finite-distance totals "
        "therefore omit that remainder’s O(distance) variation. Elsewhere the "
        "complete active-source field is reevaluated. Coil Hessians stay at the "
        "target. No terminal solve or timing claim is produced.</p>",
        '<p id="bank-hessian-boundary-provenance">Owned receipts, input copies '
        "and per-file digests: "
        "<code>/home/ITER/mcintos/.cache/nova-hessian-boundary/</code>. "
        "The numerical receipt is <code>receipt.json</code>; <code>split.json</code>, "
        "<code>mutation.json</code>, <code>retained-inputs.json</code> and "
        "<code>nodes/</code> retain the split, refusals, provenance and full "
        "one-sided records. The receipt fingerprint on this section binds every "
        "rendered table to that receipt. The gate regenerates this entire fragment "
        "and then checks it for exact agreement.</p>",
        f"<p>Reporting revision: <code>{data['revision']}</code>. Numerical "
        "record revisions: "
        + ", ".join(
            f"<code>{r}</code>"
            for r in sorted({n.get("revision", "unspecified") for n in data["nodes"]})
        )
        + ". The bank budget is 1e-8; this diagnostic does not hold bank or "
        "scenario work. Product source is unchanged.</p>",
        "</section>",
    ]
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fragment", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path.home() / ".cache/nova-hessian-boundary"
    data = json.loads((root / "receipt.json").read_text())
    text = render(data)
    if args.check:
        if not args.fragment.exists() or args.fragment.read_text() != text:
            print(f"stale evidence fragment: {args.fragment}")
            raise SystemExit(1)
        print("DOCUMENT_CHECK_COMPLETE")
    else:
        args.fragment.write_text(text)
        print(
            f"DOCUMENT_COMPLETE: {len(data['nodes'])} boundary nodes, "
            f"{data['interior_count']} interior nodes"
        )


if __name__ == "__main__":
    main()
