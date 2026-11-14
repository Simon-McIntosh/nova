"""Test whether the solved operator's external field responds to the seam.

The corrected arms came back bit-identical to the measured ones while the
recorded inputs showed a thirty percent change in every conductor. This probe
builds the same row twice with two different current vectors at the documented
source seam and compares the external field the operator holds, which is the
field the residual is formed against.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np

    from benchmarks import diiid_forward_gs_match as bench

    paths = sorted(bench.DEFAULT_DATA.glob("*.parquet"))
    selected = bench.select_frames(
        paths, bench.EXECUTION_FRAME_COUNT, bench.polarity_population()
    )
    match = [item for item in selected if item.frame == 137]
    columns = (
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
    )
    row = bench._read(match[0].path, columns)
    original = bench._source_current
    captured = {}

    def currents(base):
        def read(row, description, names, time_ms):
            values = original(row, description, names, time_ms)
            captured["base"] = [float(item) for item in values]
            return base(values)

        return read

    bench._source_current = currents(lambda values: values)
    built = bench._build_profile(row, 137)
    plain = np.asarray(built.profile.operator.external())

    bench._source_current = currents(lambda values: values + 54058.0)
    built = bench._build_profile(row, 137)
    shifted = np.asarray(built.profile.operator.external())

    report = {
        "conductors": len(captured["base"]),
        "base_current_sup_a_turn": max(abs(item) for item in captured["base"]),
        "external_base_sup_wb": float(np.max(np.abs(plain))),
        "external_shifted_sup_wb": float(np.max(np.abs(shifted))),
        "external_difference_sup_wb": float(np.max(np.abs(shifted - plain))),
        "external_identical": bool(np.array_equal(plain, shifted)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
