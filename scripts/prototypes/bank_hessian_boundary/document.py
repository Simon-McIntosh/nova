"""Fill the evidence fragment from the completed numerical receipt."""

import argparse
import json
from pathlib import Path
import re


def pair(values):
    return " / ".join(f"{x:.6g}" for x in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fragment", type=Path)
    args = parser.parse_args()
    root = Path.home() / ".cache/nova-hessian-boundary"
    data = json.loads((root / "receipt.json").read_text())
    assert len(data["nodes"]) == 92
    assert all(row["control"]["accepted"] for row in data["states"])
    parts = [
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
    text = args.fragment.read_text()
    text, count = re.subn(
        r'<div id="bank-hessian-boundary-limits">.*?</div>',
        "\n".join(parts),
        text,
        count=1,
        flags=re.S,
    )
    assert count == 1
    args.fragment.write_text(text)
    print("DOCUMENT_COMPLETE: 92 boundary nodes, 64 interior nodes, three states")


if __name__ == "__main__":
    main()
