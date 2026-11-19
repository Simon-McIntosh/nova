"""Negative control for the psi-identity guard.

The guard under test is ``compare.psi_identity``: it must examine every ladder
record and refuse when it examines none.  This control rebuilds the loop the
finding named -- ``break`` on the arm test, which stops on the plan's first
direct record -- and runs it and the shipped form over the same synthetic
table.  It prints both results and exits nonzero after the refusal, so the log
ends ``EXIT=1`` and records that the check fails without the repair.
"""

from __future__ import annotations

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


def unfixed_gap(values):
    """The loop the finding describes: break on the arm test."""
    gap = 0.0
    for record in records():
        if record["arm"] != "psi" or "base" not in values:
            break
        gap = max(
            gap,
            abs(
                values["current"][record["index"], 0, 1, 0]
                - values["base"][record["index"], 0, 1, 0]
            ),
        )
    return gap


def main():
    values = table()
    gap, examined = psi_identity(values, records(), [dict(cell=0, moment=1)], {0: 0})
    broken = unfixed_gap(values)
    print("injected difference %.3e" % AMOUNT)
    print("shipped psi_identity gap %.3e examined %d" % (gap, examined))
    print("unfixed loop gap %.3e" % broken)
    if broken != 0.0 or abs(gap - AMOUNT) > 1e-12:
        print("CONTROL FAILED: the two forms did not separate")
        raise SystemExit(0)
    print(
        "REFUSED: with break the identity check reports 0.000e+00 for a %.3e difference"
        % AMOUNT
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
