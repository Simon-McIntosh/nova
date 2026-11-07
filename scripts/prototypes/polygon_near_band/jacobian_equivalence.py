"""Compare the packed analytic moments at the base and head revisions.

The weighted quadrature reaches its Jacobian through ``sqrt(1 + s^2)`` at the
head revision, where the deleted weighted residual used ``cosh(stretch)``.  The
two are equal analytically and need not be equal in floating point, so this
prints the bitwise verdict and the maximum relative difference of every output
leaf over both packaged coil sets on CPU x64.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure  # noqa: E402


def _moments(kernel, coilset):
    polygons = coilset.polygons
    targets = measure._target_cloud(polygons)
    pair_fraction, _, selected, total = measure._near_band_fraction(polygons, targets)
    packed = measure._pack(polygons, targets)
    inputs = tuple(jax.device_put(value) for value in packed)
    evaluate = jax.jit(
        lambda *values: kernel.packed_analytic_moments(jax.numpy, *values)[:3]
    )
    outputs = jax.block_until_ready(evaluate(*inputs))
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(outputs)]
    return leaves, {
        "near_band_target_fraction": pair_fraction,
        "selected_pairs": selected,
        "pair_count": total,
        "target_count": len(targets),
    }


def _compare(base, head):
    leaves = []
    for index, (left, right) in enumerate(zip(base, head, strict=True)):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        scale = np.maximum(np.abs(left), np.abs(right))
        scale[scale == 0.0] = 1.0
        leaves.append(
            {
                "leaf": index,
                "shape": list(left.shape),
                "bitwise_equal": left.tobytes() == right.tobytes(),
                "max_abs_difference": float(np.max(np.abs(left - right))),
                "max_relative_difference": float(np.max(np.abs(left - right) / scale)),
                "elements": int(left.size),
            }
        )
    return leaves


def main() -> int:
    record = {
        "device": str(jax.devices()[0]),
        "backend": jax.default_backend(),
        "x64": bool(jax.config.jax_enable_x64),
        "coil_sets": {},
    }
    coilsets = (measure._mast_coils(), measure._diiid_coils())
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with measure._baseline_kernel(args.base_revision) as base_kernel:
        for coilset in coilsets:
            base_leaves, selection = _moments(base_kernel, coilset)
            head_leaves, _ = _moments(measure.polygonanalytic, coilset)
            rows = _compare(base_leaves, head_leaves)
            record["coil_sets"][coilset.name] = {
                "selection": selection,
                "leaves": rows,
                "bitwise_equal": all(row["bitwise_equal"] for row in rows),
                "max_relative_difference": max(
                    row["max_relative_difference"] for row in rows
                ),
            }
            print(
                f"EQUIVALENCE machine={coilset.name} "
                f"bitwise_equal={record['coil_sets'][coilset.name]['bitwise_equal']} "
                f"max_relative="
                f"{record['coil_sets'][coilset.name]['max_relative_difference']:.3e} "
                f"selected={selection['selected_pairs']}/"
                f"{selection['pair_count']}",
                flush=True,
            )
    encoded = json.dumps(record, indent=2, sort_keys=True)
    print(encoded, flush=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
