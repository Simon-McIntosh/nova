"""Measure the limited rows' oracle ratios and their fresh-residual floor.

A limited row has no magnetic island, so ``row_oracle`` judges it against its
limiter contact.  The residual floor is measured by moving each converged
moment by one to four representable steps and reading the fresh residual back,
using the landed noise-floor instrument's ``nudged`` unchanged.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_residual_noise_floor.measure import nudged
from scripts.prototypes.bank_route_rows import measure as rows

PERTURBATIONS = 16


def noise_floor(built, moments, cells):
    """Fresh residual under ulp-scale moves of each converged moment."""
    operands = built["operands"]
    cache = built["cache"]
    roots = built["roots"]
    convention = built["convention"]
    policy = built["policy"]
    profile = built["profile"]
    quadrature = built["quadrature"]
    seeds = roots + 0.03 * jnp.median(operands.geometry.pitch)

    @jax.jit
    def read(bank, window, state):
        return support_read.read(bank, window, state, seeds, roots, convention, policy)

    @jax.jit
    def residual(bank, base, candidate, support):
        return support_read.step_residual(
            bank, base, candidate, support, profile, quadrature
        )

    def measure(state):
        support = read(operands, cache, state)
        result = residual(operands, state, state, support)
        value = jax.block_until_ready(result.residual)
        return float(jnp.max(jnp.abs(jnp.concatenate(value)))), bool(support.valid)

    baseline, valid = measure(moments)
    records = []
    for index in range(PERTURBATIONS):
        steps = 1 + index % 4
        shifted = jax.tree.map(
            lambda value, seed=index + 1, count=steps: nudged(value, seed, count),
            moments,
        )
        value, ok = measure(shifted)
        records.append(
            {"seed": index + 1, "steps": steps, "fresh_residual": value, "valid": ok}
        )
        rows.emit(
            "PERTURB", cells=cells, seed=index + 1, steps=steps, fresh_residual=value
        )
    norms = [record["fresh_residual"] for record in records]
    return {
        "baseline": baseline,
        "baseline_valid": valid,
        "perturbations": records,
        "minimum": min(norms),
        "median": float(np.median(norms)),
        "maximum": max(norms),
        "all_valid": all(record["valid"] for record in records),
    }


def run_row(cells, output_dir):
    from tests.equilibrium import test_topology_read as fixture

    built = rows.build("limited", cells)
    fast = rows.fast_route(built)
    _, _, wall, _, _ = fixture._analytic_inputs("limited")
    oracle, analytic_roots = rows.row_oracle(
        "limited", fast["analytic"], built["roots"], fast["reading"], wall
    )
    rows.emit("ORACLE", kind="limited", cells=cells, **oracle)
    floor = noise_floor(built, fast["moments"], cells)
    rows.emit(
        "NOISE_FLOOR",
        cells=cells,
        **{
            key: floor[key]
            for key in ("baseline", "minimum", "median", "maximum", "all_valid")
        },
    )
    archive = output_dir / f"limited-{cells}-oracle.npz"
    np.savez_compressed(
        archive,
        nodes=fast["nodes"],
        solved_flux=fast["solved_flux"],
        analytic_roots=analytic_roots,
        solved_roots=np.asarray(
            jnp.stack((fast["reading"].axis, fast["reading"].boundary))
        ),
        wall=np.asarray(wall),
        converged=np.asarray(fast["converged"]),
        residual_sup=np.asarray(fast["fresh_residual"]),
    )
    return {
        "kind": "limited",
        "cells": cells,
        "realised_cells": int(len(np.asarray(built["operands"].geometry.vertices))),
        "bank_seconds": built["bank_seconds"],
        "fast": {
            "converged": fast["converged"],
            "updates": fast["steps"],
            "fresh_residual": fast["fresh_residual"],
            "residual_trace": fast["trace"],
        },
        "oracle": oracle,
        "budget": {
            "axis": oracle["axis_ratio"] <= 5e-4,
            "xpoint": None,
            "boundary": oracle["boundary_ratio"] <= 1e-4,
            "residual": fast["fresh_residual"] <= 1e-10,
        },
        "noise_floor": floor,
        "metric_path": str(archive),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=[550, 2000])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.emit(
        "HEADER",
        nova=nova.__file__,
        module=support_read.__file__,
        cells=",".join(str(cells) for cells in args.cells),
        device=jax.devices()[0].device_kind,
        x64=jax.config.jax_enable_x64,
    )
    report = {"rows": []}
    for cells in args.cells:
        report["rows"].append(run_row(cells, args.output.parent))
        args.output.write_text(json.dumps(report, indent=2, default=float) + "\n")
    rows.emit("EXIT_ROW", rows=len(report["rows"]))


if __name__ == "__main__":
    main()
