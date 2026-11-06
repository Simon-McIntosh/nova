"""Check the sweep receipt against the section's acceptance conditions.

Fails (exit 1) if any target's reference did not converge, if any slope's curve
does not turn over strictly inside the sweep, or if a slope's best attainable
error is not recorded with its width and rounding term.

    python scripts/edge_sheet/verify.py --json <receipt>
"""

from __future__ import annotations

import argparse
import json

TOLERANCE = 1e-10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    arguments = parser.parse_args()
    with open(arguments.json) as handle:
        receipt = json.load(handle)

    failures = []
    targets = 0
    for slope in receipt["slopes"]:
        for row in slope["targets"]:
            targets += 1
            convergence = row["reference_convergence"]
            if not convergence["converged"]:
                failures.append(
                    "reference did not converge: slope %.4f station %s d=%.3f "
                    "(rounds %d, achieved %.2e)"
                    % (
                        slope["slope"],
                        row["station"],
                        row["distance"],
                        convergence["rounds"],
                        convergence["achieved"],
                    )
                )
        curve = slope["curve"]
        best = min(range(len(curve)), key=lambda index: curve[index])
        interior = 0 < best < len(curve) - 1
        if not interior:
            failures.append(
                "slope %.4f curve does not turn over strictly inside the sweep "
                "(minimum at index %d of %d)" % (slope["slope"], best, len(curve))
            )
        elif not (curve[best] < curve[0] and curve[best] < curve[-1]):
            failures.append(
                "slope %.4f minimum is not strictly below both ends" % slope["slope"]
            )
        fit = slope.get("fit", {})
        if not fit.get("rounding_term_at_best"):
            failures.append("slope %.4f has no rounding term recorded" % slope["slope"])
        print(
            "slope %.4f  best %.3e at eps/L %.2e  model %.3e  rounding %.3e  %s"
            % (
                slope["slope"],
                slope["best_error"],
                slope["best_fraction"],
                fit.get("model_term_at_best", float("nan")),
                fit.get("rounding_term_at_best", float("nan")),
                "reaches 1e-10" if slope["best_error"] <= TOLERANCE else "above 1e-10",
            )
        )

    combined = receipt["combined"]
    print(
        "targets %d  combined best %.3e at eps/L %.2e  fractions meeting 1e-10: %s"
        % (
            targets,
            combined["best_error"],
            combined["best_fraction"],
            combined["fractions_meeting_tolerance"] or "none",
        )
    )
    for failure in failures:
        print("FAIL:", failure)
    print("EXIT=%d" % (1 if failures else 0))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
