"""The stage split and its closure ratio, read from a receipt log.

The evidence table, the figure and the negative control all read the split
through this module, so a control exercises the code the evidence uses rather
than a second copy of the same arithmetic.
"""

import json
from pathlib import Path

NAMES = [
    "read census",
    "read decision",
    "read geometry",
    "step residual",
    "step assembly",
    "factor",
    "back-solve",
]
REQUIRED = (
    "census",
    "fit",
    "topology",
    "support_geometry",
    "residual",
    "assembly",
    "factor",
    "backsolve",
    "whole_solve",
)


def read(path):
    """The stage medians keyed by (lanes, name), and the solve receipts by lanes."""
    medians = {}
    solves = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("STAGE "):
            row = json.loads(line.partition(" ")[2])
            medians[(row["lanes"], row["name"])] = row["median_ms"]
        elif line.startswith("SOLVE "):
            row = json.loads(line.partition(" ")[2])
            solves[row["lanes"]] = row
    return medians, solves


def stages_at(medians, lanes):
    """The per-iterate stage values at one batch size, in ms.

    Raises when a required record is absent rather than treating the stage as
    zero: a split missing a stage is not a split, and a closure ratio computed
    without it would report a fit it did not measure.
    """
    missing = [(lanes, name) for name in REQUIRED if (lanes, name) not in medians]
    if missing:
        raise ValueError(f"receipt has no record for {missing}")
    return {
        "read census": medians[(lanes, "census")],
        "read decision": medians[(lanes, "topology")]
        - medians[(lanes, "support_geometry")],
        "read geometry": medians[(lanes, "fit")] + medians[(lanes, "support_geometry")],
        "step residual": 2.0 * medians[(lanes, "residual")],
        "step assembly": medians[(lanes, "assembly")],
        "factor": medians[(lanes, "factor")],
        "back-solve": medians[(lanes, "backsolve")],
    }


def iterate(medians, solves, lanes):
    """The whole solve's per-iterate wall: its own wall over the steps it ran."""
    return medians[(lanes, "whole_solve")] / solves[lanes]["max_steps"]


def closure(values, iterate_ms):
    """The stage sum as a multiple of the wall it is meant to close on."""
    return sum(values[name] for name in NAMES) / iterate_ms
