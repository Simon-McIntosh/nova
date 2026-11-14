"""Compare the solved operator under the measured and the fitted-corrected currents.

Two arms that solved bit-identically while their recorded inputs differed were
either handed the same field or handed a difference the solve does not consume.
This probe rebuilds frame 137's profile twice through the documented source
seam, once with the measured current vector and once with the fitted equivalent
error removed, and reports what the operator holds in each case.
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

    basis = Path(sys.argv[1])
    correction = json.loads(Path(sys.argv[2]).read_text())
    recorded = json.loads((basis / "current-inputs.json").read_text())
    names = recorded["names"]
    delta = [float(correction.get(name, 0.0)) for name in names]

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
    seen = {}

    def currents(shift):
        def read(source_row, description, labels, time_ms):
            values = original(source_row, description, labels, time_ms)
            seen["measured"] = [float(item) for item in values]
            return [float(item) - amount for item, amount in zip(values, shift(labels))]

        return read

    bench._source_current = currents(
        lambda labels: [float(correction.get(name, 0.0)) for name in labels]
    )
    corrected = np.asarray(bench._build_profile(row, 137).profile.operator.external())

    bench._source_current = currents(lambda labels: [0.0] * len(labels))
    measured = np.asarray(bench._build_profile(row, 137).profile.operator.external())

    report = {
        "conductors": len(names),
        "recorded_measured_matches_seam": bool(
            seen["measured"] == [float(item) for item in recorded["measured_a_turn"]]
        ),
        "correction_sup_a_turn": float(max(abs(item) for item in delta)),
        "measured_sup_wb": float(np.max(np.abs(measured))),
        "corrected_sup_wb": float(np.max(np.abs(corrected))),
        "difference_sup_wb": float(np.max(np.abs(corrected - measured))),
        "relative_difference": float(
            np.max(np.abs(corrected - measured)) / np.max(np.abs(measured))
        ),
        "identical": bool(np.array_equal(corrected, measured)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
