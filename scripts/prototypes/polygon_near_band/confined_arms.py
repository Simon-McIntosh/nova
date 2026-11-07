"""Compare the baseline, the unconfined head and the confined near-band bucket.

The three arms differ only in where the graded quadrature is evaluated: the
baseline revision's kernel, this worktree's kernel with the whole-array
evaluation, and this worktree's kernel with the static near-band bucket. Each
arm is compiled and timed on the same packed inputs for each coil set.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure  # noqa: E402


def _wall(coilset, device, kernel) -> float:
    return measure._run_case(coilset, device, "batched", kernel)["wall_ms"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    coilsets = (measure._mast_coils(), measure._diiid_coils())
    record = {
        "device": f"{measure.jax.devices()[0]}",
        "backend": measure.jax.default_backend(),
        "baseline_revision": args.baseline_revision,
        "head_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "rows": {},
    }
    with measure._baseline_kernel(args.baseline_revision) as baseline:
        for coilset in coilsets:
            base = _wall(coilset, args.device, baseline)
            os.environ["NOVA_NEAR_BUCKET"] = "off"
            whole = _wall(coilset, args.device, measure.polygonanalytic)
            os.environ.pop("NOVA_NEAR_BUCKET", None)
            confined = _wall(coilset, args.device, measure.polygonanalytic)
            record["rows"][coilset.name] = {
                "baseline_ms": base,
                "head_whole_ms": whole,
                "head_confined_ms": confined,
            }
            print(
                f"CONFINED machine={coilset.name} device={record['device']} "
                f"baseline_ms={base:.4f} head_whole_ms={whole:.4f} "
                f"head_confined_ms={confined:.4f}",
                flush=True,
            )
    encoded = json.dumps(record, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
