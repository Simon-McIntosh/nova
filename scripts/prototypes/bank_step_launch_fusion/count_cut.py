"""Count live cut cells at the diverted bank seed before sizing a device batch."""

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
    )(moments)
    membership = np.asarray(read.reading.membership)
    normal = np.asarray(read.reading.normal_form_cells)
    cut = (membership > 0) & (membership < 1 - 1e-14)
    print(
        "CUT_COUNTS",
        json.dumps(
            {
                "cells": len(membership),
                "cut": int(np.count_nonzero(cut)),
                "normal": int(np.count_nonzero(normal & cut)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
