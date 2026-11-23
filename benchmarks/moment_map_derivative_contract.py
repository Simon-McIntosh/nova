"""Measure the direct moment-map derivative against finite differences.

The script constructs the fixture used by the forward-solve derivative test and
compares reverse mode, forward JVPs, and central differences through each
observable stage.  Its report identifies the earliest stage whose analytic
derivative disagrees with the finite-difference contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium.observation import MomentTargets
from nova.equilibrium.flux_surface_connectivity import fit_tensor_spline
from nova.jax.config import configure_dtypes


STEPS = (1.0e-3, 3.0e-5, 1.0e-5)


def _load_fixture_module(path: Path):
    """Load the test module so this measurement uses its declared fixture."""
    spec = importlib.util.spec_from_file_location("forward_solve_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(test_path: Path):
    """Build the exact module-scoped test fixture without invoking pytest."""
    module = _load_fixture_module(test_path)
    machine = module.machine.__wrapped__()
    return machine[0], module.converged.__wrapped__(machine)


def _flatten(value) -> jax.Array:
    """Concatenate array leaves into one derivative-comparison output vector."""
    leaves = jax.tree.leaves(value)
    return jnp.concatenate([jnp.ravel(jnp.asarray(leaf)) for leaf in leaves])


def _relative_error(estimate: jax.Array, reference: jax.Array) -> float:
    """Return the max-norm discrepancy scaled by the analytic signal."""
    denominator = jnp.maximum(jnp.max(jnp.abs(estimate)), 1.0)
    return float(jnp.max(jnp.abs(estimate - reference)) / denominator)


def _measure_stage(
    name: str, function, flux: jax.Array, direction: jax.Array, reference=None
):
    """Measure one stage with reverse, forward, and finite-difference arms."""
    primal, forward = jax.jvp(function, (flux,), (direction,))
    reverse = jax.jacrev(function)(flux) @ direction
    steps = {}
    central_function = function if reference is None else reference
    for step in STEPS:
        positive = central_function(flux + step * direction)
        negative = central_function(flux - step * direction)
        central = (positive - negative) / (2.0 * step)
        steps[str(step)] = {
            "central_max_abs": float(jnp.max(jnp.abs(central))),
            "forward_relative_error": _relative_error(forward, central),
            "reverse_relative_error": _relative_error(reverse, central),
        }
    return {
        "name": name,
        "output_size": int(primal.size),
        "forward_max_abs": float(jnp.max(jnp.abs(forward))),
        "reverse_max_abs": float(jnp.max(jnp.abs(reverse))),
        "reverse_forward_relative_error": _relative_error(reverse, forward),
        "steps": steps,
    }


def _scalar_control() -> dict[str, float]:
    """Verify the differencing instrument on a smooth no-selection scalar map."""
    point = jnp.asarray(0.37, dtype=jnp.float64)
    tangent = jnp.asarray(1.0, dtype=jnp.float64)

    def smooth(value):
        return value**3 - 0.4 * value + jnp.exp(0.3 * value)

    _value, jvp = jax.jvp(smooth, (point,), (tangent,))
    reverse = jax.grad(smooth)(point) * tangent
    step = jnp.asarray(1.0e-5, dtype=jnp.float64)
    central = (smooth(point + step) - smooth(point - step)) / (2.0 * step)
    return {
        "forward_relative_error": _relative_error(jvp, central),
        "reverse_relative_error": _relative_error(reverse, central),
        "reverse_forward_relative_error": _relative_error(reverse, jvp),
    }


def _render(report: dict, output: Path) -> None:
    """Render the central-difference discrepancy for every measured stage."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 7), dpi=100)
    colours = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(report["stages"])))
    for colour, stage in zip(colours, report["stages"], strict=True):
        errors = [stage["steps"][str(step)]["forward_relative_error"] for step in STEPS]
        linewidth = 3.0 if stage["name"] in {"axis flux", "boundary flux"} else 2.6
        axis.loglog(
            STEPS,
            errors,
            marker="o",
            linewidth=linewidth,
            color=colour,
        )
        axis.annotate(
            stage["name"],
            (STEPS[-1], errors[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=colour,
            fontsize=20,
            va="center",
        )
    axis.axhline(1.0e-10, color="0.45", linestyle=":", linewidth=1.2)
    axis.annotate(
        "no-selection control requirement",
        (STEPS[0], 1.0e-10),
        color="0.35",
        fontsize=20,
    )
    axis.set_xlabel("central-difference step [Wb]", fontsize=22)
    axis.set_ylabel("forward JVP vs central relative error", fontsize=22)
    axis.tick_params(labelsize=20, width=1.2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def measure(test_path: Path, figure_path: Path) -> dict:
    """Return the fixture-specific staged derivative comparison receipt."""
    configure_dtypes()
    profile, converged = _fixture(test_path)
    targets = MomentTargets(
        plasma_current=0.9 * float(converged.moments.plasma_current),
        poloidal_beta=0.5,
        internal_inductance=0.8,
    )
    flux = jnp.asarray(converged.flux)
    rng = np.random.default_rng(11)
    direction = jnp.asarray(rng.standard_normal(flux.shape), dtype=flux.dtype)
    direction = direction / jnp.max(jnp.abs(direction))

    def integral_state(state):
        return profile._integral_state(state)

    def topology_values(state):
        _current, _integrals, _masks, topology, _amplitude = integral_state(state)
        return jnp.stack(
            (topology.axis_flux, topology.boundary_flux, topology.flux_span)
        )

    def axis_flux(state):
        _current, _integrals, _masks, topology, _amplitude = integral_state(state)
        return jnp.atleast_1d(topology.axis_flux)

    def boundary_flux(state):
        _current, _integrals, _masks, topology, _amplitude = integral_state(state)
        return jnp.atleast_1d(topology.boundary_flux)

    _current, _integrals, _masks, terminal_topology, _amplitude = integral_state(flux)
    boundary_point = jax.lax.stop_gradient(terminal_topology.boundary)
    fixed_topology = profile.operator._fixed_design_topology

    def fixed_boundary_flux(state):
        grid_flux, _wall_flux = fixed_topology.split_flux_map(state)
        values = grid_flux.reshape(
            (
                fixed_topology.connectivity_radius.size,
                fixed_topology.connectivity_height.size,
            )
        ).T
        surface = fit_tensor_spline(
            fixed_topology.connectivity_radius,
            fixed_topology.connectivity_height,
            values,
        )
        return jnp.atleast_1d(surface(boundary_point[0], boundary_point[1]))

    def support_partition(state):
        _current, _integrals, masks, _topology, _amplitude = integral_state(state)
        return _flatten((masks.core, masks.profile_participation, masks.psi_norm))

    def cell_moments(state):
        current, _integrals, _masks, _topology, _amplitude = integral_state(state)
        return _flatten(current)

    def measure_terms(state):
        _current, integrals, _masks, _topology, _amplitude = integral_state(state)
        return _flatten(integrals[:-1])

    def observation(state):
        return profile.integral_observation(state).stack()

    def residual(state):
        return profile.moment_residual(state, targets)

    stages = [
        _measure_stage("topology flux levels", topology_values, flux, direction),
        _measure_stage("axis flux", axis_flux, flux, direction),
        _measure_stage(
            "boundary flux", boundary_flux, flux, direction, fixed_boundary_flux
        ),
        _measure_stage(
            "support partition and clip weights", support_partition, flux, direction
        ),
        _measure_stage("cell current moments", cell_moments, flux, direction),
        _measure_stage("cell integral measure", measure_terms, flux, direction),
        _measure_stage("normalised moment observations", observation, flux, direction),
        _measure_stage("target residual normalisation", residual, flux, direction),
    ]
    report = {
        "fixture": str(test_path),
        "forward_module": __import__(
            "nova.equilibrium.forward", fromlist=["ForwardProfile"]
        ).__file__,
        "flux_size": int(flux.size),
        "steps": list(STEPS),
        "scalar_control": _scalar_control(),
        "stages": stages,
    }
    _render(report, figure_path)
    return report


def main() -> None:
    """Run the derivative measurement and emit one JSON receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    report = measure(args.test_path.resolve(), args.figure.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
