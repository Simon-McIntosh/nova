"""Build shared-level solved and analytic flux panels for terminal states."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium.constraint import uniform_radial_field_jet
from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.fine_grid_convergence.runtime_solve import without_profile
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments
from scripts.prototypes.xpoint_cell_owner.render import terminal_panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
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
            "lobe",
        ),
        default="halfplane",
    )
    parser.add_argument(
        "--density-mode",
        choices=("fitted", "profile"),
        default="fitted",
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() in ("gpu", "cpu")
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    saved = json.loads(args.start_json.read_text())
    terminal_saved = saved.get("comparison", {}).get("pinned", saved)
    if "terminal_state" not in terminal_saved or not terminal_saved.get("history"):
        raise ValueError("terminal panel requires a completed cold receipt")
    context = build(args.cells, args.cache)
    moments = jnp.asarray(terminal_saved["terminal_state"]["moments"])
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
    field_amplitude = jnp.asarray(
        terminal_saved["terminal_state"].get("field_amplitude_tesla", 0.0)
    )
    if field_amplitude != 0:
        values = replace(
            operands[0],
            exterior=operands[0].exterior
            + field_amplitude
            * uniform_radial_field_jet(context.rule.points.reshape((-1, 2)))[:, 0],
        )
        window = replace(
            operands[2],
            exterior=operands[2].exterior
            + field_amplitude * uniform_radial_field_jet(context.window.nodes),
        )
        operands = (values, operands[1], window, *operands[3:])
    terminal = jax.jit(
        lambda point, carrier: map_moments(
            point,
            *carrier[:7],
            profile,
            *carrier[7:],
            component=args.component,
            density_mode=args.density_mode,
        )
    )(moments, operands)
    solved_flux = (
        exterior
        + _image(operator, moments)
        + field_amplitude * uniform_radial_field_jet(coordinates)[:, 0]
    )
    result = {
        "cells": len(context.reference),
        "panel": {
            "coordinates": np.asarray(machine.node).tolist(),
            "wall": np.asarray(machine.wall_node).tolist(),
            "solved_flux": np.asarray(solved_flux)[: len(machine.node)].tolist(),
            "analytic_flux": np.asarray(analytic)[: len(machine.node)].tolist(),
            "solved_roots": np.asarray(terminal.roots).tolist(),
            "analytic_roots": np.asarray(context.seeds).tolist(),
            "terminal_residual": terminal_saved["history"][-1]["residual"],
            "converged": bool(terminal_saved.get("converged")),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        terminal_panel(result, args.svg)
        lines = args.svg.read_text().splitlines()
        args.svg.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    residual = result["panel"]["terminal_residual"]
    converged = result["panel"]["converged"]
    print(
        f"PANEL cells={result['cells']} residual={residual} converged={converged}",
        flush=True,
    )


if __name__ == "__main__":
    main()
