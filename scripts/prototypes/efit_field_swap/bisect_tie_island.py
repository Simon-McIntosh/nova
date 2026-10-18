"""Bisect the offset at which a tie-island carrier's read decision changes.

The tie-island states exported for imas-efit are the moving field at an offset
where something about nova's read changes -- the admitted saddle identity or the
limited-or-diverted class -- so the exported record is a state on a transition
rather than an arbitrary point of the sweep. That offset has to be measured, not
assumed: the offset that bisects one cell count's arm is not the offset that
bisects another's, because the carrier geometry and the field both move.

This driver does the measurement with the probe that owns the carriers. It builds
the carrier exactly as ``saddle_admission_probe`` does,
``tie_probe.build(kind, requested, cache)``, scans a window of offsets to find the
brackets in which the decision key changes, and bisects each bracket to adjacent
binary64 with the probe's own ``measure`` and ``bisect``. The key is the probe's
default ``both``: the pair (boundary class, admitted saddle identity), so an
offset that moves the saddle without changing the class is a bracket too.

The receipt is a JSON file listing every scan row and every bisected bracket, so
the offset an export uses can be checked against the measurement rather than
taken on trust.

Usage: bisect_tie_island.py <cache> <output.json> <requested-cells> [kind] [lo hi count]
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import sys
from pathlib import Path

import jax
import numpy as np

from scripts.prototypes.fast_map_production.saddle_admission_probe import (
    bisect,
    build,
    key_of,
    measure,
    sampler_with_enclosure,
)

KEY = "both"


def scan(offsets, state, sampled, prior=None):
    """One measure per offset, in offset units rather than shifted units."""
    return [measure(float(offset), state, sampled, prior) for offset in offsets]


def brackets(offsets, rows):
    """Offset intervals over which the decision key changes."""
    found = []
    for index in range(len(rows) - 1):
        if key_of(rows[index], KEY) != key_of(rows[index + 1], KEY):
            found.append((float(offsets[index]), float(offsets[index + 1])))
    return found


def main(cache, out_path, requested, kind="diverted", window=(0.0, 0.3), count=31):
    assert jax.config.jax_enable_x64 is True
    state = build(kind, requested, cache)
    sampled = sampler_with_enclosure(state)
    offsets = np.linspace(window[0], window[1], count)
    rows = scan(offsets, state, sampled)
    for offset, row in zip(offsets, rows):
        print(
            "scan offset=%.6f class=%d saddle=%d live=%d valid=%s envelope=%.6f"
            % (
                offset,
                row["boundary_class"],
                row["saddle_identity"],
                row["live_saddle_count"],
                row["valid"],
                row["psi_enclosure"],
            ),
            flush=True,
        )
    found = brackets(offsets, rows)
    print("BRACKETS", found, flush=True)
    results = []
    for left, right in found:
        outcome = bisect(left, right, state, sampled, None, KEY)
        results.append(outcome)
        print(
            "BISECTED key=%s bracket=%r steps=%d adjacent=%s left=%s right=%s"
            % (
                KEY,
                outcome["bracket"],
                outcome["steps"],
                outcome["adjacent"],
                key_of(outcome["left"], KEY),
                key_of(outcome["right"], KEY),
            ),
            flush=True,
        )
    payload = {
        "kind": kind,
        "requested_cells": requested,
        "achieved_cells": int(len(np.asarray(state["geometry"].centre))),
        "key": KEY,
        "scan": {
            "window": [float(window[0]), float(window[1])],
            "count": int(count),
            "rows": rows,
        },
        "brackets": found,
        "bisected": results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", out_path, "brackets", len(found), flush=True)
    return 0 if found else 1


if __name__ == "__main__":
    cache, out_path = sys.argv[1], sys.argv[2]
    requested = int(sys.argv[3])
    kind = sys.argv[4] if len(sys.argv) > 4 else "diverted"
    if len(sys.argv) > 7:
        window = (float(sys.argv[5]), float(sys.argv[6]))
        count = int(sys.argv[7])
    else:
        window, count = (0.0, 0.3), 31
    sys.exit(main(cache, out_path, requested, kind, window, count))
