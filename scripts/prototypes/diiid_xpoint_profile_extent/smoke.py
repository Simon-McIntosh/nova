"""Check the arm reaches the profile an actual solve would be handed.

Each arm is measured on the same frame, and the comparison is against what the
built profile's source core carries, not against the extraction's own return.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_xpoint_profile_extent"))
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_forward_convergence"))
import arms  # noqa: E402
import measure_branches as probe  # noqa: E402


def tables(frame, arm):
    arms.install(probe.bench, arm)
    prepared = probe._prepare_frame(frame, None)
    core = prepared["profile"].operator.source.core
    return {
        name: dict(
            coordinate=np.asarray(getattr(core, name).coordinate).tolist(),
            values=np.asarray(getattr(core, name).values).tolist(),
        )
        for name in ("p_prime", "ff_prime")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=137)
    args = parser.parse_args()
    record = {"frame": args.frame}
    for arm in arms.ARMS:
        record[arm] = tables(args.frame, arm)
    base = record["base"]["p_prime"]["coordinate"]
    edge = record["edge"]["p_prime"]["coordinate"]
    assert len(base) == 19, len(base)
    assert len(edge) > len(base), len(edge)
    for name in ("p_prime", "ff_prime"):
        assert record["base"][name]["coordinate"] == edge[: len(base)]
        assert record["base"][name]["values"] == record["edge"][name]["values"][
            : len(base)
        ]
    record["interior_ok"] = True
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"BASE_KNOTS {len(base)} EDGE_KNOTS {len(edge)}")
    print("INTERIOR_IDENTICAL True")
    print("EDGE_P_PRIME " + json.dumps(record["edge"]["p_prime"]["values"][-10:]))
    print("EDGE_FF_PRIME " + json.dumps(record["edge"]["ff_prime"]["values"][-10:]))


if __name__ == "__main__":
    main()