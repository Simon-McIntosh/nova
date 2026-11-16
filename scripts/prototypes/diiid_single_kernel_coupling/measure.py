"""Measure uniform coupling differences alongside the governed-wall branch solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", choices=("limited", "diverted"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.tree))

    import jax
    import numpy as np

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "gpu", jax.devices()

    from benchmarks import diiid_forward_gs_match as bench
    from scripts.prototypes.diiid_diverted_branch import measure as cohort

    assert Path(bench.__file__).resolve().is_relative_to(args.tree.resolve())
    print(f"MODULE={bench.__file__}; CWD={Path.cwd().resolve()}", flush=True)
    print(f"DEVICES={jax.devices()}", flush=True)
    original = bench._couplings
    measured = False

    def observed(row, machine, radius, height, *rest):
        nonlocal measured
        result = original(row, machine, radius, height, *rest)
        if measured:
            return result
        measured = True
        lattice = bench.FluxLattice(radius, height)
        centres = np.asarray(lattice.coordinate)
        width, extent = np.diff(radius).mean(), np.diff(height).mean()
        maps = dict(radius=radius, height=height, wall=result[1])
        records = {}
        for name, targets, blocks, attached in (
            ("grid", centres, result[7], result[5]),
            ("wall", result[1], result[8], result[6]),
            ("sample", result[9], result[12], result[11]),
        ):
            hybrid = bench._green_block(targets, centres, width, extent)
            polygon = blocks[0]
            difference = hybrid - polygon
            # One value per source cell, with all target rows in its norm.
            relative = np.linalg.norm(difference, axis=0) / np.linalg.norm(
                polygon, axis=0
            )
            assert np.isfinite(relative).all() and relative.max() > 1e-5
            maps[name] = relative.reshape(lattice.shape)
            records[name] = dict(
                target_rows=len(targets),
                source_cells=len(centres),
                relative_min=float(relative.min()),
                relative_median=float(np.median(relative)),
                relative_max=float(relative.max()),
                relative_frobenius=float(
                    np.linalg.norm(difference) / np.linalg.norm(polygon)
                ),
                attached_equals_polygon=bool(np.array_equal(attached, polygon)),
                attached_equals_hybrid=bool(np.array_equal(attached, hybrid)),
            )
        np.savez_compressed(args.output / "coupling-differences.npz", **maps)
        (args.output / "coupling-differences.json").write_text(
            json.dumps(
                dict(
                    revision=args.revision,
                    targets=records,
                    definition=(
                        "norm(hybrid[:, cell] - polygon[:, cell]) "
                        "/ norm(polygon[:, cell])"
                    ),
                ),
                indent=2,
            )
            + "\n"
        )
        print("COUPLING_DIFFERENCE " + json.dumps(records), flush=True)
        return result

    bench._couplings = observed
    cohort.row(args)
    assert measured, "the branch solve did not reach the DIII-D coupling builder"


if __name__ == "__main__":
    main()
