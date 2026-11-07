"""Save the flux map and read arguments at the moment axis identification refuses.

Frames the limited seed of the frame 144 benchmark, runs the same solve the
branch record runs, and records, on refusal, the field and the arguments
handed to the qualification read that raised.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
for extra in (
    REPOSITORY,
    REPOSITORY / "scripts" / "prototypes" / "diiid_axis_identification",
    REPOSITORY / "scripts" / "prototypes" / "diiid_forward_convergence",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import measure_branches  # noqa: E402
import benchmarks.diiid_forward_gs_match as bench  # noqa: E402
from nova.equilibrium.topology import TopologyClass  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    expansion = float(bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION)
    prepared = measure_branches._prepare_frame(144, expansion)
    profile = prepared["profile"]
    topology = profile.operator.topology
    kind = type(topology)
    original_read = kind.read_qualification
    calls: list[dict] = []

    def wrapper(self, psi, polarity, inside_material, requested_class=None, *rest):
        calls.append(
            {
                "psi": np.asarray(psi),
                "polarity": int(np.asarray(polarity)),
                "inside_material": np.asarray(inside_material, dtype=bool),
                "requested_class": None
                if requested_class is None
                else int(np.asarray(requested_class)),
                "site": "qualified_o_candidates -> read_qualification",
            }
        )
        return original_read(
            self, psi, polarity, inside_material, requested_class, *rest
        )

    kind.read_qualification = wrapper
    seed = np.asarray(
        prepared["cold"].branches.flux[int(TopologyClass.LIMITED)], dtype=float
    )
    status = "solved"
    detail = ""
    try:
        receipt, seconds = measure_branches._solve(
            profile,
            seed,
            prepared["current"],
            prepared["target_current_a"],
            "diiid-frame-144-axis-capture:limited",
        )
        detail = f"solve returned, residual {float(receipt.equilibrium.fixed_point.residual)}"
    except Exception as error:
        status = f"refused: {type(error).__name__}"
        detail = traceback.format_exc()[-3000:]
    finally:
        kind.read_qualification = original_read

    record = {
        "frame": 144,
        "status": status,
        "detail": detail,
        "read_call_count": len(calls),
        "calls": [
            {
                "call": index,
                "polarity": call["polarity"],
                "requested_class": call["requested_class"],
                "inside_material_cells": int(call["inside_material"].sum()),
                "psi_shape": list(call["psi"].shape),
            }
            for index, call in enumerate(calls)
        ],
    }
    arguments.output.with_suffix(".json").write_text(json.dumps(record, indent=2))
    if calls:
        last = calls[-1]
        np.savez(
            arguments.output,
            psi=last["psi"],
            inside_material=last["inside_material"],
            polarity=np.asarray(last["polarity"]),
            requested_class=np.asarray(
                -1 if last["requested_class"] is None else last["requested_class"]
            ),
            seed=seed,
        )
    print(json.dumps(record, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
