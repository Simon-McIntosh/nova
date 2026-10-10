"""Render measured booking errors and their certificate receipts."""

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(rows_directory, figure_path, fragment_path, commits):
    rows_directory, figure_path, fragment_path = map(
        Path, (rows_directory, figure_path, fragment_path)
    )
    rows = [json.loads(path.read_text()) for path in rows_directory.glob("*.json")]
    rows.sort(key=lambda row: (row["case"], row["realised_cells"]))
    if not rows:
        raise ValueError("at least one measured certificate row is required")
    cases = sorted({row["case"] for row in rows})
    plt.style.use("data-ink")
    figure, axes = plt.subplots(1, len(cases), figsize=(14, 6), dpi=100, squeeze=False)
    axes = axes[0]
    styles = [
        ("read", "-", "o", "#222222"),
        ("exact", "--", "s", "#555555"),
        ("legacy", ":", "^", "#777777"),
    ]
    for ax, kind in zip(axes, cases, strict=True):
        selected = [row for row in rows if row["case"] == kind]
        for arm, dash, marker, colour in styles:
            ax.loglog(
                [row["realised_cells"] for row in selected],
                [row[f"{arm}_image_error"] for row in selected],
                linestyle=dash,
                marker=marker,
                color=colour,
                linewidth=3 if arm == "read" else 2.6,
                label=arm,
            )
        ax.set_xlabel("Realised cells")
        ax.text(
            0.02, 0.97, kind, transform=ax.transAxes, ha="left", va="top", fontsize=20
        )
        ax.legend(frameon=False, loc="lower left", fontsize=20)
        ax.grid(False, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Booking image error [relative sup]")
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, format="svg")
    plt.close(figure)
    table = []
    for row in rows:
        table.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    row["case"],
                    row["realised_cells"],
                    f"{row['read_image_error']:.9g}",
                    f"{row['exact_image_error']:.9g}",
                    f"{row['legacy_image_error']:.9g}",
                    f"{row['net_current_relative_error']:.3g}",
                    f"{row['private_current']:.3g}",
                    f"{row['smooth_current_budget_ratio']:.4g}",
                    f"{row['saddle_current_budget_ratio']:.4g}",
                    f"{row['oracle_area_fraction_uncertainty']:.3g}",
                    row["job_id"],
                )
            )
            + "</tr>"
        )
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text(
        '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="docs-project" content="nova">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="cfs-current-book">
<meta name="plan-title" content="Connected fragment current booking">
<meta name="plan-summary" content="Traced current moments on connected conic and
normal-form fragments">
<meta name="plan-evidence-for" content="converged-forward-solve">
<meta name="plan-verifies" content="converged-forward-solve#s3">
<meta name="plan-commits" content="'''
        + html.escape(commits)
        + """">
<meta name="plan-recorded-at" content="2026-10-10">
<meta name="plan-verdict" content="failed">
<meta name="plan-environment" content="H200, binary64; CPU regression delta">
<title>Connected fragment current booking | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css">
<link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body>
<main class="plan-doc">
<h1 id="cfs-current-book-fragments-booking">Connected fragment current booking</h1>
<table>
<thead>
<tr>
<th>Case</th>
<th>Cells</th>
<th>Read image</th>
<th>Exact image</th>
<th>Legacy image</th>
<th>Net current relative error</th>
<th>Private current [A]</th>
<th>Smooth current / budget</th>
<th>Saddle current / budget</th>
<th>Oracle area uncertainty</th>
<th>H200 job</th>
</tr>
</thead>
<tbody>
"""
        + "\n".join(table)
        + """
</tbody>
</table>
<figure>
<img
src="/nova/figures/converged-forward-solve/cfs-current-book/booking-image-error.svg"
alt="Log-log booking image error against realised cell count, comparing read, exact
and legacy supports for limited and diverted states">
<figcaption>Current moments from the connected topology read, exact analytic support
and legacy support, on the same cached certificate carriers. Errors use the analytic
density image and fixed-cell moment basis. Only measured rows are plotted. The first
limited row fails image dominance; the other five certificate rows were not run after
the fail-fast refusal, and no convergence rate is inferred. The read books all
selected pieces; holes subtract through ring winding. The prescribed total is the
independently integrated analytic-density total on the refined analytic support. The
oracle area-refinement change is reported separately in the table.</figcaption>
</figure>
<table>
<thead>
<tr>
<th>Contract</th>
<th>Measured result</th>
</tr>
</thead>
<tbody>
<tr>
<td>Negative control: level-test membership</td>
<td>test_private_fragment_current_is_zero fails, private current 1.6000000000000001;
connected arm 0</td>
</tr>
<tr>
<td>JVP scaled discrepancy at step 1e-5</td>
<td>Smooth 5.13e-12; axis 9.22e-12; boundary 8.05e-9; support 1.97e-11</td>
</tr>
<tr>
<td>Regression delta</td><td>42 passed on baseline; 42 passed on head; zero added failures across six modules.</td></tr>
<tr>
<td>Independent support attribution</td>
<td>Analytic density on the read polygons gives image error 1.7777786713e-5; booked
minus read-polygon image is 5.13e-13. Improving this fixed-support result requires the
topology owner or clarification of the limited-state acceptance.</td>
</tr>
<tr>
<td>Raw rows and moment arrays</td>
<td>
<code>"""
        + html.escape(str(rows_directory))
        + """</code>
</td>
</tr>
</tbody>
</table>
</main>
</body>
</html>
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--fragment", required=True)
    parser.add_argument("--commits", required=True)
    args = parser.parse_args()
    render(args.rows, args.figure, args.fragment, args.commits)
