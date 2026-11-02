"""Count the cut cells the fused booking tangent evaluates and the moving subset."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


def main():
    root = Path(__file__).resolve().parents[3]
    print(
        "HEADER",
        json.dumps(
            {
                "revision": subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "tree": str(root),
                "cwd": str(Path.cwd().resolve()),
                "nova": nova.__file__,
                "module": support_read.__file__,
                "backend": jax.default_backend(),
            }
        ),
        flush=True,
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, _, _, moments, _ = machine("diverted")
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    read = jax.jit(
        lambda state: support_read.read(
            bank, cache, state, seeds, roots, convention, policy
        )
    )(moments).reading
    membership = np.asarray(read.membership)
    normal = np.asarray(read.normal_form_cells)
    cut = (membership > 0) & (membership < 1 - 1e-14)
    samples = moments[0].size and bank.sample_index.shape[1]
    print(
        "CUT_COUNTS",
        json.dumps(
            {
                "cells": int(len(membership)),
                "cut": int(np.count_nonzero(cut)),
                "moving_edge": int(np.count_nonzero(cut & ~normal)),
                "normal_form": int(np.count_nonzero(cut & normal)),
                "channels": int(samples + 8),
                "samples": int(samples),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
