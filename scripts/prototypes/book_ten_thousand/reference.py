"""Compare the split-read exact-support fixture with an independent integral."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import nova

from tests.equilibrium.test_topology_read import (
    _analytic_inputs,
    _kernel_backed_field,
    _realised_hex_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"REVISION={revision} TREE={Path.cwd().resolve()} "
        f"COMMAND={sys.executable} {' '.join(sys.argv)}",
        flush=True,
    )
    print(f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()}", flush=True)
    assert jax.config.jax_enable_x64 is True
    oracle, field, wall, _, _ = _analytic_inputs("diverted")
    geometry = _realised_hex_geometry(wall, args.cells)
    print(f"GEOMETRY_READY {len(geometry.centre)}", flush=True)
    fixture = _kernel_backed_field("diverted", oracle, field, geometry)
    reference = np.stack(tuple(map(np.asarray, fixture.moments)))
    direct = np.load(args.out / f"diverted-{args.cells}-cells.npz")
    np.testing.assert_allclose(
        np.asarray(geometry.centre), direct["centre"], rtol=0, atol=1e-11
    )
    exact = direct["exact"]
    booked = direct["booked"]
    scale = np.max(np.abs(reference))
    discrepancy = np.max(np.abs(booked - reference), axis=0)
    ranked = np.argsort(discrepancy)[::-1]
    records = []
    for index in ranked[:20]:
        records.append(
            {
                "cell": int(index),
                "centre": direct["centre"][index].tolist(),
                "membership": float(direct["membership"][index]),
                "exact_area_fraction": float(
                    direct["exact_area"][index] / direct["full_area"][index]
                ),
                "booked": booked[:, index].tolist(),
                "reference": reference[:, index].tolist(),
                "independent": exact[:, index].tolist(),
            }
        )
    result = {
        "cells": args.cells,
        "booked_vs_reference": float(np.max(np.abs(booked - reference)) / scale),
        "independent_vs_reference": float(np.max(np.abs(exact - reference)) / scale),
        "booked_vs_independent": float(
            np.max(np.abs(booked - exact)) / np.max(np.abs(exact))
        ),
        "reference_scale": float(scale),
        "top_twenty_l1_share": float(
            discrepancy[ranked[:20]].sum() / discrepancy.sum()
        ),
        "records": records,
    }
    np.savez_compressed(
        args.out / f"diverted-{args.cells}-reference.npz", reference=reference
    )
    (args.out / f"diverted-{args.cells}-reference.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
