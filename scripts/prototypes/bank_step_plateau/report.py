"""Write the paired P100 derivative ladder as one evidence fragment."""

# ruff: noqa: E501

import argparse
import json
from pathlib import Path


BLOCKS = (
    ("full", "full residual"),
    ("booking", "held-read booking"),
    ("geometry", "read geometry increment"),
    ("flux", "bank flux"),
)
LADDER = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def scientific(value):
    return f"{value:.3e}"


def main(head_path, base_path, refine_path, output):
    arms = {
        "head": json.loads(head_path.read_text()),
        "base": json.loads(base_path.read_text()),
    }
    assert all(len(rows) == 60 for rows in arms.values())
    assert all(
        row["plus_valid"] and row["minus_valid"]
        for rows in arms.values()
        for row in rows
    )
    refined = json.loads(refine_path.read_text())
    assert len(refined) == 50
    assert all(item["plus_valid"] and item["minus_valid"] for item in refined)
    refined_best = {
        iterate: min(
            (item for item in refined if item["iterate"] == iterate),
            key=lambda item: item["full"],
        )
        for iterate in (3, 4)
    }
    assert all(item["full"] < 1e-6 for item in refined_best.values())

    def refined_at(iterate, spacing):
        return min(
            (item for item in refined if item["iterate"] == iterate),
            key=lambda item: abs(item["h_over_x"] - spacing),
        )

    indexed = {
        arm: {(row["iterate"], row["h_over_x"]): row for row in rows}
        for arm, rows in arms.items()
    }

    def row(arm, iterate, spacing):
        return indexed[arm][iterate, spacing]

    table = []
    for arm in ("head", "base"):
        for iterate in range(10):
            for block, label in BLOCKS:
                errors = "".join(
                    f"<td>{scientific(row(arm, iterate, spacing)[block])}</td>"
                    for spacing in LADDER
                )
                table.append(
                    f"<tr><td>{arm} P100</td><td>{iterate}</td>"
                    f"<td>{scientific(row(arm, iterate, 1e-5)['fresh_residual'])}</td>"
                    f"<td>{label}</td>{errors}</tr>"
                )
    at_plateau = row("head", 4, 1e-6)
    at_wide = row("head", 4, 1e-4)
    at_mid = row("head", 4, 1e-5)
    best = min(row("head", 4, spacing)["full"] for spacing in LADDER)
    base = row("base", 4, 1e-6)
    assert abs(at_plateau["full"] - base["full"]) < 1e-7
    assert at_plateau["booking"] < 1e-6
    assert at_plateau["full"] > 1e-4
    header = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="docs-project" content="nova">
<meta name="reckon-type" content="evidence">
<meta name="plan-evidence-for" content="sub-millisecond-forward-path">
<meta name="plan-verifies" content="sub-millisecond-forward-path#s3">
<meta name="plan-verdict" content="partial">
<title>Bank Newton plateau derivative diagnosis | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css">
<link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body><main class="plan-doc">
<h1 id="bank-step-plateau-landing">Bank Newton plateau derivative diagnosis</h1>
"""
    narrative = f"""<p id="bank-step-plateau-summary">On titan P100, the diverted 550-cell booked equation reproduced the ten-step solve: after two full steps its fresh residual was 1.644203872e-7, three more full steps left it at 1.644200690e-7, and step ten reached 7.970e-11. All ten accepted states and Newton directions were retained. At those identical states and directions, the base and head Jacobians have nearly identical directional-error ladders. No newly enlarged booking, geometry or flux error explains the extra steps. A tighter head spacing probe finds full fresh-read agreement of {scientific(refined_best[3]["full"])} and {scientific(refined_best[4]["full"])} at the two hardest plateau iterates. Large errors at nearby spacings identify the clipped fresh read's h-sensitive numerical response as the source of the apparent plateau; this measure does not claim a topological branch switch.</p>
<figure id="bank-step-plateau-directional-figure"><img src="/nova/figures/sub-millisecond-forward-path/bank-step-plateau/directional-error.svg" alt="P100 relative directional derivative error by accepted iterate for full residual, held booking, read geometry increment and bank flux, head and base"><figcaption>Titan P100, one diverted 550-cell trajectory, head and archived base evaluated at the same states and directions. Shown at perturbation displacement 1e-5 times the norm of each state. Blue circles are head; open grey squares are base. Their overlap is the result. The full six-spacing ladder is tabled below. The SVG was rendered to PNG, checked for clipped and colliding text, and visually inspected.</figcaption></figure>
<section id="bank-step-plateau-method"><h2>Same-state measurement</h2>
<p>At every accepted iterate, the head supplies x and its fresh-factor Newton direction d. For both revisions, a fresh <code>support_read.read</code> evaluates the residual at x plus and minus h times d. Each h is chosen so that the displacement norm is the table's fraction of the state norm. The full column compares the assembled fresh-geometry Jacobian action with that central difference. Held-read booking compares the assembled Jacobian without moving geometry against a held-read central difference. The read geometry increment subtracts the held response from the fresh response on both sides. Bank flux compares the linear bank rows against a central difference of sampled nodal flux; its fixed exterior field is excluded from the action. Each error is the norm of the action minus its difference divided by the norm of that difference. Every plus and minus read in all 120 rows was valid. The base code was imported from an archive of revision <code>731c57cae</code>, as the base receipt's module path confirms; both arms used the head trajectory on the same P100.</p>
<p>The read-sensitive split changes with h. At accepted iterate 4, with a 1e-6 state-relative displacement, held booking error is {scientific(at_plateau["booking"])} while full fresh error is {scientific(at_plateau["full"])} and the read geometry increment error is {scientific(at_plateau["geometry"])}. At 1e-4 they are {scientific(at_wide["booking"])}, {scientific(at_wide["full"])} and {scientific(at_wide["geometry"])}; at 1e-5 the geometry increment falls to {scientific(at_mid["geometry"])} while booking reads {scientific(at_mid["booking"])}. The base full error at the 1e-6 point is {scientific(base["full"])}. The best full error on the initial six-point ladder at iterate 4 is {scientific(best)}, but the tighter probe below resolves a sub-1e-6 agreement window. The high error at adjacent spacings is a property of the fresh-read finite-difference instrument at this tiny Newton direction; no one booking or geometry block error grows between base and head. No product change is proposed from this measure alone.</p>
</section>
<section id="bank-step-plateau-refinement"><h2>P100 plateau spacing refinement</h2>
<p>Full fresh-read relative error from 25 logarithmically spaced perturbations between 1e-4 and 1e-7 of the state norm at the two hardest plateau iterates. Both best values are below 1e-6, while the neighboring values move nonmonotonically. The refinement checked valid plus and minus reads at all 50 points. Its zero fragment-selection-change count was not used as an absence claim because this run did not include a positive control for that counter.</p>
<table id="bank-step-plateau-refinement-table"><thead><tr><th>Iterate</th><th>At 1e-5</th><th>At 7.5e-6</th><th>At 5.62e-6</th><th>At 3.16e-6</th><th>Best h/|x|</th><th>Best error</th></tr></thead><tbody>
<tr><td>3</td><td>{scientific(refined_at(3, 1e-5)["full"])}</td><td>{scientific(refined_at(3, 7.5e-6)["full"])}</td><td>{scientific(refined_at(3, 5.62e-6)["full"])}</td><td>{scientific(refined_at(3, 3.16e-6)["full"])}</td><td>{scientific(refined_best[3]["h_over_x"])}</td><td>{scientific(refined_best[3]["full"])}</td></tr>
<tr><td>4</td><td>{scientific(refined_at(4, 1e-5)["full"])}</td><td>{scientific(refined_at(4, 7.5e-6)["full"])}</td><td>{scientific(refined_at(4, 5.62e-6)["full"])}</td><td>{scientific(refined_at(4, 3.16e-6)["full"])}</td><td>{scientific(refined_best[4]["h_over_x"])}</td><td>{scientific(refined_best[4]["full"])}</td></tr>
</tbody></table></section>
<section id="bank-step-plateau-ladder"><h2>P100 relative directional-error ladder</h2>
<p>Columns are the norm of the perturbation divided by the norm of x. Full and component errors use their own finite-difference norm as denominator. The geometry increment is a difference of two responses, so its relative error can amplify cancellation; compare it with the full column and its h dependence.</p>
<table id="bank-step-plateau-ladder-table"><thead><tr><th>Arm</th><th>Iterate</th><th>Fresh residual</th><th>Response</th><th>1e-3</th><th>1e-4</th><th>1e-5</th><th>1e-6</th><th>1e-7</th><th>1e-8</th></tr></thead><tbody>
"""
    footer = """</tbody></table></section>
<section id="bank-step-plateau-control"><h2>Positive and negative controls</h2>
<p>The original two-step diagnostic was rejected before comparison because it enabled target-current normalization and therefore changed the equation. With the booked quadrature restored, the ten-step history and 1.6442e-7 plateau reproduced. A separate control adds 1e-3 to one booking Jacobian column in an immutable matrix copy at plateau iterate 2. At h/|x|=1e-5 the booking error rises from 7.180e-6 to 7.864e-4, so the table can detect a material block error. The flux check first exposed an instrument error: evaluating nodal flux on d also includes the fixed exterior field. The independent linear-row check corrected that column; at the plotted spacing its plateau flux error is about 1e-11. These receipts prevent the initially uniform flux result from masquerading as evidence about the code.</p>
<p>Durable inputs are <code>head-trajectory.log</code>, <code>head-difference.log</code>, <code>base-difference.log</code>, <code>negative-control.log</code>, <code>head-flux.log</code>, <code>refine.log</code>, <code>trajectory.npz</code>, <code>head-checked.json</code>, <code>base-difference.json</code> and <code>refine.json</code> under this run's crew directory. Each measurement log ends with <code>EXIT=0</code>. The figure's rendered PNG is in that same directory. No product source was changed.</p>
</section></main></body></html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + narrative + "\n".join(table) + "\n" + footer)
    print(
        f"FRAGMENT path={output} rows={len(table)} "
        f"refined_error_3={refined_best[3]['full']:.9g} "
        f"refined_error_4={refined_best[4]['full']:.9g}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("head", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("refine", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.head, args.base, args.refine, args.output)
