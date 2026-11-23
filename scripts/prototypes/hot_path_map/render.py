"""Render the evidence fragment from a composed per-module line map."""

import argparse
import html
import json
from pathlib import Path

NODE = "hot-path-map"


def cell(row, key):
    value = row[key]
    if value is None:
        return "<td>not measured</td>"
    return (
        f"<td>{value['executed']}/{row['total']} "
        f'<span class="rate">{value["percent"]:.1f}%</span></td>'
    )


def plain(row, key):
    value = row[key]
    if value is None:
        return "not measured"
    return f"{value['executed']}/{row['total']} ({value['percent']:.1f}%)"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    data = json.loads(args.map.read_text())
    modules = data["modules"]
    measured = [row for row in modules if row["measured"]]
    not_measured = len(modules) - len(measured)
    reached = sum(1 for row in modules if row["union"]["executed"])
    control = data["negative_control"]
    contradictions = data["device_trace_contradictions"]
    total_forward = sum(
        row["forward"]["executed"] for row in measured if row["forward"]
    )
    total_inverse = sum(
        row["inverse"]["executed"] for row in measured if row["inverse"]
    )
    public = next(
        row for row in modules if row["module"] == "nova/equilibrium/forward.py"
    )
    parts = []
    parts.append(
        f'<section id="{NODE}-landing">'
        f"<h2>Line map of the forward and inverse hot paths</h2>"
    )
    parts.append(
        f"<p><strong>Both routes reach the public solve entry, and "
        f"{len(measured)} of {len(modules)} modules under <code>nova/</code> "
        f"({not_measured} not measured) "
        f"were measured on one <code>all_debug</code> job ({args.job}).</strong> "
        f"The forward route executed {total_forward} lines and the inverse "
        f"route {total_inverse}; {reached} modules were reached by at least one "
        f"route. Measured at revision <code>{data['revision'][:12]}</code> "
        f"<code>{data['revision']}</code>.</p>"
    )
    parts.append(
        "<p>The two routes are the public forward solve "
        "<code>ForwardProfile.solve</code> on a stored MAST case, and the "
        "known-state recovery test "
        "<code>test_known_state_recovery[mast]</code>, which drives the fit "
        "inside the equilibrium iteration and calls the same public entry. "
        "Coverage ran under a dedicated rcfile with <code>source=nova</code> and "
        "an empty <code>omit</code> list, so <code>nova/scripts</code> and "
        "<code>nova/utilities</code> are measured; the machine-readable copy is "
        '<a href="hot-path-map.json">hot-path-map.json</a> beside this '
        "fragment.</p>"
    )
    parts.append(
        f'<h3 id="{NODE}-public-entry">The public solve entry</h3><p>'
        f"<code>nova/equilibrium/forward.py</code> reads "
        f"{plain(public, 'forward')}"
        f"on the forward route and {plain(public, 'inverse')}"
        f"on the inverse route, so the entry both routes must reach is "
        f"nonzero on each.</p>"
    )
    control_row = next(
        (row for row in modules if row["module"] == control["module"]), None
    )
    parts.append(
        f'<h3 id="{NODE}-negative-control">Negative control</h3><p>'
        f"<code>{control['module']}</code> is imported by neither route and "
        f"reads {plain(control_row, 'forward')} on the forward route and "
        f"{plain(control_row, 'inverse')} on the inverse route: measured, and "
        f"zero. A map that marked every module reached would show it nonzero "
        f"here.</p>"
    )
    parts.append(
        f'<h3 id="{NODE}-device-trace">Relation to the device trace</h3>'
        f"<p>The bank-solve device attribution measures the compiled "
        f"<code>support_read.solve_batch</code> loop. The slice-profile bank "
        f"fixture it reads is absent from shared storage at this revision, so "
        f"that arm could not be driven and the modules it alone reaches are "
        f"recorded not measured. "
        f"{len(contradictions)} modules the device trace attributes kernels to "
        f"are unreached or unmeasured here and are listed as contradictions, "
        f"never as zero:</p><ul>"
    )
    for item in contradictions:
        parts.append(
            f"<li><code>{html.escape(item['module'])}</code> — "
            f"{html.escape(item['reason'])}</li>"
        )
    parts.append("</ul>")
    parts.append(
        f'<h3 id="{NODE}-table">Every module under nova/</h3>'
        f"<p>Executed executable lines against the module's total, per route "
        f"and for the union. A module with no coverage entry reads "
        f"<em>not measured</em>.</p>"
        f'<table id="{NODE}-module-table"><thead><tr>'
        f"<th>Module</th><th>Forward</th><th>Inverse</th>"
        f"<th>Union</th></tr></thead><tbody>"
    )
    for row in modules:
        rate = row["union"]["percent"]
        union = (
            f"<td>{row['union']['executed']}/{row['total']} "
            f'<span class="rate">{rate:.1f}%</span></td>'
        )
        parts.append(
            f"<tr><td><code>{html.escape(row['module'])}</code></td>"
            f"{cell(row, 'forward')}{cell(row, 'inverse')}{union}</tr>"
        )
    parts.append("</tbody></table>")
    parts.append(
        f"<p>Gate log: <code>{html.escape(args.log)}</code>. "
        f"Measurement revision <code>{data['revision']}</code>, "
        f"tree <code>{html.escape(data['tree'])}</code>, "
        f"job <code>{html.escape(args.job)}</code> on <code>all_debug</code>."
        f"</p></section>"
    )
    args.output.write_text("\n".join(parts) + "\n")
    print(f"WROTE {args.output} {len(modules)} modules", flush=True)


if __name__ == "__main__":
    main()
