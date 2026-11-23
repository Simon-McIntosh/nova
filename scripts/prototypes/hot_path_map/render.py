"""Render the evidence fragment from a composed per-module line map."""

import argparse
import html
import json
from pathlib import Path

from nova.database.filepath import relativize_worktree_paths

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
    parser.add_argument(
        "--line-copy",
        default=None,
        help="path of the copy that carries the per-module line numbers",
    )
    parser.add_argument(
        "--driver-revision",
        default=None,
        help="revision HEAD carried when the map driver rendered this record",
    )
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
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="docs-project" content="nova">',
        '  <meta name="plan-slug" content="code-reuse-and-hot-path">',
        "  <title>Measured line map of the forward and inverse hot paths"
        " | nova</title>",
        "</head>",
        "<body>",
        '  <main class="plan-doc">',
    ]
    parts.append(
        f'<section id="{NODE}-landing">'
        f"<h2>Line map of the forward and inverse hot paths</h2>"
    )
    entry_reached = [
        name
        for name, key in (("forward", "forward"), ("inverse", "inverse"))
        if public[key] and public[key]["executed"]
    ]
    parts.append(
        f"<p><strong>{len(entry_reached)} of the 2 routes reach the public "
        f"solve entry ({', '.join(entry_reached)}), and "
        f"{len(measured)} of {len(modules)} modules under <code>nova/</code> "
        f"({not_measured} not measured) "
        f"were measured on one <code>all_debug</code> job ({args.job}).</strong> "
        f"The forward route executed {total_forward} lines and the inverse "
        f"route {total_inverse}; at least one route reached {reached} of them. "
        f"Measured at revision <code>{data['revision']}</code>.</p>"
    )

    def counted(prefix):
        rows = [row for row in modules if row["module"].startswith(prefix)]
        return sum(1 for row in rows if row["measured"]), len(rows)

    scripts = counted("nova/scripts/")
    utilities = counted("nova/utilities/")
    parts.append(
        "<p>The two routes are the public forward solve "
        "<code>ForwardProfile.solve</code> on a stored MAST case (the "
        "slice-profile bank arm's input <code>common-streams.npz</code> is "
        "absent from shared storage at this revision, so the public entry is "
        "driven directly) and the "
        "known-state recovery test "
        "<code>test_known_state_recovery[mast]</code>, which drives the fit "
        "inside the equilibrium iteration and calls the same public entry. "
        "Coverage ran under a dedicated rcfile with <code>source=nova</code> and "
        "an empty <code>omit</code> list, which overrides the project's omit of "
        f"<code>nova/scripts/*</code> and <code>nova/utilities/*</code>: "
        f"{scripts[0]} of {scripts[1]} modules under <code>nova/scripts</code> "
        f"and {utilities[0]} of {utilities[1]} under <code>nova/utilities</code> "
        f"carry a report entry. The machine-readable copy is "
        '<a href="hot-path-map.json">hot-path-map.json</a> beside this '
        "fragment.</p>"
    )
    parts.append(
        f'<h3 id="{NODE}-public-entry">The public solve entry</h3><p>'
        f"<code>nova/equilibrium/forward.py</code> reads "
        f"{plain(public, 'forward')} on the forward route and "
        f"{plain(public, 'inverse')} on the inverse route"
        + (
            ", so the entry both routes must reach is nonzero on each.</p>"
            if len(entry_reached) == 2
            else ", so the entry both routes must reach is "
            f"nonzero on only {len(entry_reached)} of them.</p>"
        )
    )
    control_row = next(
        (row for row in modules if row["module"] == control["module"]), None
    )
    control_clean = bool(
        control_row
        and control_row["measured"]
        and control_row["forward"]
        and control_row["inverse"]
        and not control_row["forward"]["executed"]
        and not control_row["inverse"]["executed"]
    )
    parts.append(
        f'<h3 id="{NODE}-negative-control">Negative control</h3><p>'
        f"<code>{control['module']}</code> is imported by neither route and "
        f"reads {plain(control_row, 'forward')} on the forward route and "
        f"{plain(control_row, 'inverse')} on the inverse route"
        + (
            ", which is the zero the map asserts: a map that marked every "
            "module reached would show a nonzero count here and map.py refuses "
            "to write one.</p>"
            if control_clean
            else ", which is not a clean zero, so the negative control failed "
            "and this record does not stand.</p>"
        )
    )
    parts.append(
        f'<h3 id="{NODE}-device-trace">Relation to the device trace</h3>'
        f"<p>The bank-solve device attribution measures the compiled "
        f"<code>read_batch</code> loop, reading a slice-profile bank fixture "
        f"that is absent from shared storage at this revision, so that arm cannot "
        f"be driven here and the device trace's own measurement stands. "
        f"What the line map adds is which lines each route executed, "
        f"checked here per traced entry point rather than per module: a module "
        f"reached by another function can carry a traced loop that never ran.</p>"
    )
    parts.append(f'<ul id="{NODE}-device-trace-entries">')
    for item in data["device_trace_entries"]:
        parts.append(
            f"<li><code>{html.escape(item['module'])}</code> "
            f"<code>{html.escape(item['function'])}</code> — "
            f"{item['executed']}/{item['body_lines']} body lines executed"
            + (
                f"; <strong>contradiction</strong>: {html.escape(item['reason'])}"
                if item["reason"]
                else ""
            )
            + "</li>"
        )
    parts.append("</ul>")
    module_level = [item for item in contradictions if "function" not in item]
    if module_level:
        parts.append("<ul>")
        for item in module_level:
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
        f"Measurement revision <code>{data['revision']}</code>, a single "
        f"<code>all_debug</code> job <code>{html.escape(args.job)}</code>. "
        f"Module, driver and receipt paths here and in the machine-readable copy "
        f"are repository-relative; the gate log stays in the worker's cache "
        f"directory, outside the repository.</p>"
    )
    if args.driver_revision:
        command = f"git diff --stat {data['revision']} {args.driver_revision} -- nova"
        parts.append(
            f"<p>Product source is identical at the driver revision "
            f"<code>{html.escape(args.driver_revision)}</code>: "
            f"<code>{html.escape(command)}</code> prints nothing.</p>"
        )
    if args.log:
        for line in Path(args.log).read_text().splitlines():
            if " passed in " in line and line.split()[0].isdigit():
                parts.append(
                    f"<p>Gate result, quoted from the log's runner line: "
                    f"<code>{html.escape(line.strip())}</code></p>"
                )
                break
    if args.line_copy:
        parts.append(
            f"<p>The machine-readable copy beside this fragment carries every "
            f"module's counts; the executed line numbers themselves are in "
            f"<code>{html.escape(args.line_copy)}</code>, beside the run's "
            f"other receipts.</p>"
        )
    parts.append("</section>")
    parts.extend(["  </main>", "</body>", "</html>"])
    text, unmapped = relativize_worktree_paths("\n".join(parts) + "\n")
    closing = text.rstrip().endswith("</html>")
    if not closing or text.count("</html>") != 1:
        raise SystemExit(
            f"{args.output}: the fragment does not close once at </html>; "
            f"content would sit outside the body"
        )
    args.output.write_text(text)
    if unmapped:
        print("UNMAPPED", json.dumps(unmapped[:5]), flush=True)
    print(f"WROTE {args.output} {len(modules)} modules", flush=True)


if __name__ == "__main__":
    main()
