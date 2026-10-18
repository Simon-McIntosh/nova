"""Falsify the census admission repair by reinstating the dropping step.

The control rebuilds the census with its exact-zero cell-membership step
restored from the live source, bisects the adjacent binary64 pair at which that
step loses a root, and runs both the restored and the repaired census at the
pair under the same moving field and prior. It exits nonzero while the restored
step reproduces the loss the repair removes, so removing the repair is what
makes this check fail; a zero exit means the excess-one admission step no longer
changes the census count at the pair, and the repair would then be guarding
nothing.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.fast_map_production import tie_probe
from scripts.prototypes.read_robustness import census_base_step


def count_roots(census_function, state, offset, prior):
    """Census and saddle count under a census chosen by the caller."""

    def run(shifted):
        varied = replace(
            state["jets"],
            exterior=state["jets"].exterior + shifted * state["jet_slope"],
        )
        field = varied.field(state["reference"])
        return census_function(
            field, state["geometry"], state["convention"], state["policy"], prior
        ).roots

    return jax.jit(run)(jnp.asarray(offset * state["scale"]))


def summary(state, roots):
    eigenvalues = np.linalg.eigvalsh(
        np.asarray(state["convention"].sigma) * np.asarray(roots.jet.hessian)
    )
    valid = np.asarray(roots.valid)
    saddles = valid & (eigenvalues[:, 0] < 0) & (eigenvalues[:, 1] > 0)
    determinant = np.linalg.det(np.asarray(roots.jet.hessian))
    return {
        "valid": int(valid.sum()),
        "saddles": int(saddles.sum()),
        "positions": np.asarray(roots.position)[valid].tolist(),
        "determinants": determinant[valid].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low", type=float, default=0.17)
    parser.add_argument("--high", type=float, default=0.18)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True

    restored = census_base_step.base_admission_census()
    from scripts.prototypes.read_robustness import census as repaired_module

    repaired = repaired_module.census

    state = tie_probe.build("diverted", 250, args.cache)
    sampled = tie_probe.make_sampler(state)
    prior, warm_count = tie_probe.warm(state, sampled, -0.2, 0.16, 0.01)
    print(
        f"cells {len(state['geometry'].centre)} warm {warm_count} "
        f"backend {jax.default_backend()}",
        flush=True,
    )

    low, high = args.low, args.high
    reference = int(np.asarray(count_roots(restored, state, low, prior).valid).sum())
    if (
        int(np.asarray(count_roots(restored, state, high, prior).valid).sum())
        >= reference
    ):
        raise RuntimeError(
            f"the restored dropping step does not depart over {low}..{high}; "
            "there is no pair for this control to sit on"
        )
    while np.nextafter(low, high) < high:
        middle = (low + high) / 2
        if middle in (low, high):
            break
        if (
            int(np.asarray(count_roots(restored, state, middle, prior).valid).sum())
            == reference
        ):
            low = middle
        else:
            high = middle

    rows = []
    lost = False
    for offset in (low, high):
        restored_summary = summary(state, count_roots(restored, state, offset, prior))
        repaired_summary = summary(state, count_roots(repaired, state, offset, prior))
        rows.append(
            {
                "offset": offset,
                "restored": restored_summary,
                "repaired": repaired_summary,
            }
        )
        print(
            f"offset {offset!r} restored {restored_summary['valid']} roots / "
            f"{restored_summary['saddles']} saddles  repaired "
            f"{repaired_summary['valid']} roots / "
            f"{repaired_summary['saddles']} saddles",
            flush=True,
        )
        if restored_summary["valid"] < repaired_summary["valid"]:
            lost = True

    Path(args.output).write_text(
        json.dumps(
            {
                "cells": len(state["geometry"].centre),
                "warm_count": warm_count,
                "adjacent": bool(np.nextafter(low, high) == high),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    if lost:
        print(
            "CONTROL FAILS AS DECLARED: restoring the exact-zero admission step "
            "loses the hyperbolic root again at the adjacent pair",
            flush=True,
        )
        return 1
    print(
        "CONTROL PASSES UNEXPECTEDLY: the restored step keeps every root the "
        "repair keeps, so the repair is unexercised",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
