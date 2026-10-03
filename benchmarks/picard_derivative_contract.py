"""Measure derivative consistency for the accelerated fixed-point route."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium import fixed_point
from nova.equilibrium import stencil_nulls
from nova.jax.config import configure_dtypes


STEPS = (1.0e-3, 3.0e-4, 1.0e-4)


class _LaxWithLiveSelection:
    def __init__(self, lax):
        self._lax = lax

    def __getattr__(self, name):
        return getattr(self._lax, name)

    @staticmethod
    def stop_gradient(value):
        return value


class _JaxWithLiveSelection:
    def __init__(self, module):
        self._module = module
        self.lax = _LaxWithLiveSelection(module.lax)

    def __getattr__(self, name):
        return getattr(self._module, name)


@contextmanager
def live_null_coordinates(enabled: bool):
    """Expose null-coordinate tangents without changing discrete selection."""
    original = stencil_nulls.jax
    if enabled:
        stencil_nulls.jax = _JaxWithLiveSelection(original)
    try:
        yield
    finally:
        stencil_nulls.jax = original


def _fixture():
    path = Path(__file__).parents[1] / "tests" / "test_equilibrium_forward_solve.py"
    specification = importlib.util.spec_from_file_location(
        "forward_solve_fixture", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load fixture from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.machine.__wrapped__()


def _relative_error(left, right) -> float:
    scale = max(float(np.max(np.abs(np.asarray(right)))), 1.0e-30)
    return float(np.max(np.abs(np.asarray(left - right))) / scale)


def _terminal_map(profile, terminal_flux, current):
    operator = profile.operator
    shadow = operator.residual_shadow_mask(terminal_flux)
    traced = operator.traced_flux_map_with_shadow()

    def mapping(state, conductor):
        return traced(state, shadow, operator.external(conductor), operator, None)

    return mapping


def _fixed_point_tangent(mapping, terminal_flux, conductor, direction):
    """Solve (I - dG/dpsi) x = dG/dtheta v with matrix-free GMRES."""
    _, theta_tangent = jax.jvp(
        lambda value: mapping(terminal_flux, value), (conductor,), (direction,)
    )
    _, state_linear = jax.linearize(
        lambda value: mapping(value, conductor), terminal_flux
    )
    solution, information = jax.scipy.sparse.linalg.gmres(
        lambda value: value - state_linear(value),
        theta_tangent,
        tol=1.0e-10,
        atol=1.0e-12,
        restart=80,
        maxiter=20,
    )
    residual = solution - state_linear(solution) - theta_tangent
    return solution, int(np.asarray(information)), float(jnp.max(jnp.abs(residual)))


def _scalar_methods(
    profile,
    seed,
    conductor,
    *,
    live_selection: bool,
    probe: int | None = None,
    steps: tuple[float, ...] | None = None,
):
    """Compare reverse, forward, fixed-point and finite-difference responses."""
    with live_null_coordinates(live_selection):

        def flux_function(value):
            return jnp.sum(
                profile.solve(
                    seed, route="picard", current=value, evaluations=80, relaxation=0.7
                ).flux
                ** 2
            )

        def current_function(value):
            return profile.solve(
                seed, route="picard", current=value, evaluations=80, relaxation=0.7
            ).moments.plasma_current

        reverse = jax.grad(flux_function)(conductor)
        if probe is None:
            probe = int(np.argmax(np.abs(np.asarray(reverse))))
        if not 0 <= probe < conductor.size:
            raise ValueError(
                f"probe {probe} is outside conductor width {conductor.size}"
            )
        direction = jnp.zeros_like(conductor).at[probe].set(1.0)
        _, forward = jax.jvp(flux_function, (conductor,), (direction,))
        _, current_forward = jax.jvp(current_function, (conductor,), (direction,))
        terminal = profile.solve(
            seed, route="picard", current=conductor, evaluations=80, relaxation=0.7
        )
        mapping = _terminal_map(profile, terminal.flux, conductor)
        fixed_tangent, information, linear_residual = _fixed_point_tangent(
            mapping, terminal.flux, conductor, direction
        )
        flux_ift = jnp.vdot(2.0 * terminal.flux, fixed_tangent)
        current_ift = jax.jvp(
            lambda state: profile.integral_observation(state).plasma_current,
            (terminal.flux,),
            (fixed_tangent,),
        )[1]
        rows = []
        deltas = (
            tuple(fraction * float(jnp.abs(conductor[probe])) for fraction in STEPS)
            if steps is None
            else steps
        )
        for delta in deltas:
            flux_fd = (
                flux_function(conductor + delta * direction)
                - flux_function(conductor - delta * direction)
            ) / (2.0 * delta)
            current_fd = (
                current_function(conductor + delta * direction)
                - current_function(conductor - delta * direction)
            ) / (2.0 * delta)
            rows.append(
                {
                    "step_fraction": delta / float(jnp.abs(conductor[probe])),
                    "step": delta,
                    "flux_fd": float(flux_fd),
                    "current_fd": float(current_fd),
                    "flux_reverse_error": _relative_error(forward, flux_fd),
                    "flux_ift_error": _relative_error(flux_ift, flux_fd),
                    "current_forward_error": _relative_error(
                        current_forward, current_fd
                    ),
                    "current_ift_error": _relative_error(current_ift, current_fd),
                }
            )
        return {
            "probe": probe,
            "flux_reverse": float(reverse[probe]),
            "flux_forward": float(forward),
            "flux_ift": float(flux_ift),
            "current_forward": float(current_forward),
            "current_ift": float(current_ift),
            "gmres_information": information,
            "gmres_residual": linear_residual,
            "rows": rows,
        }


def _map_tangent_contract(profile, terminal_flux, conductor):
    """Measure a held-selection map tangent against its central difference."""
    direction = jnp.asarray(
        np.random.default_rng(7).standard_normal(terminal_flux.shape)
    )
    direction = direction / jnp.max(jnp.abs(direction))
    results = {}
    for label, live_selection in (("held", False), ("live_null", True)):
        with live_null_coordinates(live_selection):
            mapping = _terminal_map(profile, terminal_flux, conductor)
            _, tangent = jax.jvp(
                lambda state: mapping(state, conductor),
                (terminal_flux,),
                (direction,),
            )
            rows = []
            for step in (1.0e-3, 3.0e-5, 1.0e-5):
                numeric = (
                    mapping(terminal_flux + step * direction, conductor)
                    - mapping(terminal_flux - step * direction, conductor)
                ) / (2.0 * step)
                rows.append(
                    {"step": step, "relative_error": _relative_error(tangent, numeric)}
                )
            results[label] = rows
    return results


def _moment_contract(profile, terminal_flux):
    """Reproduce the moment-Jacobian ladder with a fixed random direction."""
    from nova.equilibrium.observation import MomentTargets

    targets = MomentTargets(
        plasma_current=0.9
        * float(profile.integral_observation(terminal_flux).plasma_current),
        poloidal_beta=0.5,
        internal_inductance=0.8,
    )
    direction = jnp.asarray(
        np.random.default_rng(11).standard_normal(terminal_flux.shape)
    )
    direction = direction / jnp.max(jnp.abs(direction))
    results = {}
    for label, live_selection in (("held", False), ("live_null", True)):
        with live_null_coordinates(live_selection):
            analytic = jax.jvp(
                lambda state: profile.moment_residual(state, targets),
                (terminal_flux,),
                (direction,),
            )[1]
            rows = []
            for step in (1.0e-3, 3.0e-5, 1.0e-5):
                numeric = (
                    profile.moment_residual(terminal_flux + step * direction, targets)
                    - profile.moment_residual(terminal_flux - step * direction, targets)
                ) / (2.0 * step)
                rows.append(
                    {
                        "step": step,
                        "relative_error": float(
                            jnp.max(jnp.abs(numeric - analytic))
                            / jnp.max(jnp.abs(analytic))
                        ),
                    }
                )
            results[label] = rows
    return results


def _smooth_control():
    """Show that the measuring chain agrees when no selection enters the map."""
    initial = jnp.asarray([0.0])
    parameter = jnp.asarray([2.0])
    direction = jnp.asarray([1.0])

    def solve(value):
        return fixed_point.picard(
            lambda state, control: 0.2 * state + control,
            initial,
            evaluations=80,
            relaxation=0.7,
            map_arguments=(value,),
        ).state[0]

    reverse = jax.grad(solve)(parameter)[0]
    _, forward = jax.jvp(solve, (parameter,), (direction,))
    fixed = 1.0 / 0.8
    numeric = (
        solve(parameter + 1.0e-4 * direction) - solve(parameter - 1.0e-4 * direction)
    ) / 2.0e-4
    return {
        "reverse": float(reverse),
        "forward": float(forward),
        "implicit": float(fixed),
        "finite_difference": float(numeric),
        "maximum_error": max(
            _relative_error(reverse, numeric),
            _relative_error(forward, numeric),
            _relative_error(jnp.asarray(fixed), numeric),
        ),
    }


def _write_figure(report, output):
    plt.style.use("data-ink")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    held = report["held"]
    steps = [row["step"] for row in held["rows"]]
    for label, values, colour in (
        ("reverse / forward", [held["flux_forward"]] * len(steps), "#386cb0"),
        ("fixed-point", [held["flux_ift"]] * len(steps), "#f0027f"),
        ("central difference", [row["flux_fd"] for row in held["rows"]], "#6a3d9a"),
    ):
        axes[0].plot(
            steps,
            values,
            marker="o",
            linewidth=2.6,
            color=colour,
            label=label,
        )
    for label, values, colour in (
        ("reverse / forward", [held["current_forward"]] * len(steps), "#386cb0"),
        ("fixed-point", [held["current_ift"]] * len(steps), "#f0027f"),
        ("central difference", [row["current_fd"] for row in held["rows"]], "#6a3d9a"),
    ):
        axes[1].plot(
            steps,
            values,
            marker="o",
            linewidth=2.6,
            color=colour,
            label=label,
        )
    for axis, quantity in zip(axes, ("linked flux", "plasma current"), strict=True):
        axis.set_xlabel("central-difference step [A]")
        axis.set_ylabel(f"{quantity} derivative [per A]")
        axis.set_xticks(steps, [f"{step:.1f}" for step in steps])
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="lower right")
    figure.savefig(output, dpi=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument(
        "--probe",
        type=int,
        help=(
            "conductor component for central differences "
            "(default: largest reverse tangent)"
        ),
    )
    parser.add_argument(
        "--steps",
        type=float,
        nargs=3,
        metavar="A",
        help=(
            "three absolute central-difference steps in A (default: existing fractions)"
        ),
    )
    arguments = parser.parse_args()
    configure_dtypes()
    profile, seed, _vacuum = _fixture()
    conductor = profile.operator.external_current
    default = _scalar_methods(profile, seed, conductor, live_selection=False)
    held = (
        default
        if arguments.probe is None and arguments.steps is None
        else _scalar_methods(
            profile,
            seed,
            conductor,
            live_selection=False,
            probe=arguments.probe,
            steps=None if arguments.steps is None else tuple(arguments.steps),
        )
    )
    terminal = profile.solve(
        seed, route="picard", current=conductor, evaluations=80, relaxation=0.7
    )
    report = {
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "default_selection": default,
        "held": held,
        "map_tangent": _map_tangent_contract(profile, terminal.flux, conductor),
        "moment": _moment_contract(profile, terminal.flux),
        "smooth_control": _smooth_control(),
    }
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_figure(report, arguments.figure)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
