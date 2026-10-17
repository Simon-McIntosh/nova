"""Build shared-level solved and analytic flux panels for terminal states."""

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
from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.fine_grid_convergence.runtime_solve import without_profile
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=(
            "halfplane",
            "bell",
            "saddle",
            "pinned",
            "anchored",
            "shifted",
            "ordinary",
            "oriented",
        ),
        default="halfplane",
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() in ("gpu", "cpu")
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    saved = json.loads(args.start_json.read_text())
    if "terminal_state" not in saved or not saved.get("history"):
        raise ValueError("terminal panel requires a completed cold receipt")
    context = build(args.cells, args.cache)
    moments = jnp.asarray(saved["terminal_state"]["moments"])
    operands = without_profile(context.operands)
    profile = context.profile
    machine = context.machine
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(
        certificate._exact_state("diverted-single-null", context.exact, coordinates)
    )
    exterior, _ = _exterior(context.cache, context.carrier_key)
    operator = fixture.forward_operator(context.source, machine)
    terminal = jax.jit(
        lambda point, carrier: map_moments(
            point, *carrier[:7], profile, *carrier[7:], component=args.component
        )
    )(moments, operands)
    solved_flux = exterior + _image(operator, moments)
    result = {
        "cells": len(context.reference),
        "panel": {
            "coordinates": np.asarray(machine.node).tolist(),
            "wall": np.asarray(machine.wall_node).tolist(),
            "solved_flux": np.asarray(solved_flux)[: len(machine.node)].tolist(),
            "analytic_flux": np.asarray(analytic)[: len(machine.node)].tolist(),
            "solved_roots": np.asarray(terminal.roots).tolist(),
            "analytic_roots": np.asarray(context.seeds).tolist(),
            "terminal_residual": saved["history"][-1]["residual"],
            "converged": bool(saved.get("converged")),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    residual = result["panel"]["terminal_residual"]
    converged = result["panel"]["converged"]
    print(
        f"PANEL cells={result['cells']} residual={residual} converged={converged}",
        flush=True,
    )


if __name__ == "__main__":
    main()
