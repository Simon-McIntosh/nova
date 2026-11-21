"""Negative control for the psi-identity guard.

The guard under test is ``compare.psi_identity``: it must examine every ladder
record and refuse when it examines none.  This control rebuilds the loop the
finding named -- ``break`` on the arm test, which stops on the plan's first
direct record -- and runs it and the shipped form over the same synthetic
table.  A control passes when the guard separates the two: it then prints the
refusal and exits 0.  If the two forms do not separate it prints CONTROL FAILED
and exits 1, so the process status carries the verdict and a gate reading the
log cannot score a silent failure as a pass.  ``--repaired`` swaps the loop's
``break`` for ``continue`` to exercise that failing path.
"""

from __future__ import annotations

import argparse

import numpy as np

from scripts.prototypes.polygon_symmetry_violation.compare import psi_identity

STEPS = (1e-3, 1e-6)
AMOUNT = 3e-6


def records():
    """Direct record first, then one psi ladder, as the plan emits them."""
    plan = [dict(arm="direct", entry=0, index=0, step=0.0)]
    for step in STEPS:
        for sign in (1.0, -1.0):
            plan.append(dict(arm="psi", entry=0, index=len(plan), step=sign * step))
    return plan


def table():
    """Row archives that differ by one known amount at every ladder record."""
    plan = records()
    shape = (len(plan), 1, 3, 3)
    current = np.zeros(shape)
    base = np.zeros(shape)
    for record in plan:
        if record["arm"] == "psi":
            current[record["index"], 0, 1, 0] = 1.0
            base[record["index"], 0, 1, 0] = 1.0 - AMOUNT
    return dict(current=current, base=base)


def unfixed_gap(values, *, repaired):
    """The loop the finding describes: break on the arm test.

    ``repaired`` swaps ``break`` for ``continue``, which is the separation the
    control has to see; it exists so the control's own failing path can be
    exercised with this same file.
    """
    gap = 0.0
    for record in records():
        if record["arm"] != "psi":
            if not repaired:
                break
            continue
        gap = max(
            gap,
            abs(
                values["current"][record["index"], 0, 1, 0]
                - values["base"][record["index"], 0, 1, 0]
            ),
        )
    return gap


def run(argv=None):
    """Run the control and return its status: 0 when it caught the mutant."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repaired",
        action="store_true",
        help="run the loop with continue, so the two forms do not separate",
    )
    args = parser.parse_args(argv)

    values = table()
    gap, examined = psi_identity(values, records(), [dict(cell=0, moment=1)], {0: 0})
    broken = unfixed_gap(values, repaired=args.repaired)
    separated = broken == 0.0 and abs(gap - AMOUNT) <= 2 * np.finfo(float).eps
    print("injected difference %.3e" % AMOUNT)
    print("shipped psi_identity gap %.3e examined %d" % (gap, examined))
    print("unfixed loop gap %.3e" % broken)
    if separated:
        print(
            "REFUSED: with break the identity check reports 0.000e+00 for a "
            "%.3e difference" % AMOUNT
        )
        return 0
    print("CONTROL FAILED: the two forms did not separate")
    return 1


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
