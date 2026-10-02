"""Measure Picard finite differences with a terminal residual shadow held fixed."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium import fixed_point
from nova.equilibrium.forward import MomentTargets
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "test_equilibrium_forward_solve.py"
MOMENT_STEPS = (1.0e-3, 3.0e-5, 1.0e-5)
CURRENT_STEP = 1.0e-3


def _fixture_module():
    spec = importlib.util.spec_from_file_location("forward_solve_fixture", FIXTURE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module at {FIXTURE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixed_shadow_history(
    profile, initial, current, shadow, *, evaluations, relaxation
):
    """Run the public Picard map while using one constant residual shadow."""
    operator = profile.operator
    mapped = operator.traced_flux_map()
    shadowed = operator.traced_flux_map_with_shadow()
    external = operator.external(current)

    def fixed_shadow(_state, _operator):
        return shadow

    def promoted_shadow(_state, _previous, _operator):
        return shadow

    return fixed_point.picard(
        mapped,
        initial,
        evaluations=evaluations,
        relaxation=relaxation,
        shadow_mask_fn=fixed_shadow,
        promoted_shadow_mask_fn=promoted_shadow,
        shadowed_map_fn=shadowed,
        map_arguments=(external, operator, None),
        callback_arguments=(operator,),
    )


def _live_state(profile, seed, current, *, evaluations, relaxation):
    return profile.solve(
        seed,
        route="picard",
        current=current,
        evaluations=evaluations,
        relaxation=relaxation,
    )


def _current_arm(profile, seed, current, *, observable, label):
    """Measure live and frozen-shadow current differences at the test probe."""
    evaluations, relaxation = 80, 0.7
    unperturbed = _live_state(
        profile, seed, current, evaluations=evaluations, relaxation=relaxation
    )
    shadow = profile.operator.residual_shadow_mask(unperturbed.flux)
    probe = int(np.argmax(np.abs(np.asarray(jax.grad(observable)(current)))))
    delta = CURRENT_STEP * float(jnp.abs(current[probe]))

    def live_scalar(value):
        return observable(value)

    def frozen_scalar(value):
        history = _fixed_shadow_history(
            profile,
            seed,
            value,
            shadow,
            evaluations=evaluations,
            relaxation=relaxation,
        )
        if label == "flux-current":
            return jnp.sum(history.state**2)
        receipt = profile._receipt(history.state, history, None, None, value, None)
        return receipt.moments.plasma_current

    plus = current.at[probe].add(delta)
    minus = current.at[probe].add(-delta)
    live_plus = _live_state(
        profile, seed, plus, evaluations=evaluations, relaxation=relaxation
    )
    live_minus = _live_state(
        profile, seed, minus, evaluations=evaluations, relaxation=relaxation
    )
    plus_shadow = profile.operator.residual_shadow_mask(live_plus.flux)
    minus_shadow = profile.operator.residual_shadow_mask(live_minus.flux)
    live_value = float((live_scalar(plus) - live_scalar(minus)) / (2.0 * delta))
    held_value = float((frozen_scalar(plus) - frozen_scalar(minus)) / (2.0 * delta))
    live_reverse = float(jax.grad(live_scalar)(current)[probe])
    held_reverse = float(jax.grad(frozen_scalar)(current)[probe])
    rows = []
    for sign, state, mask in (
        ("+", live_plus, plus_shadow),
        ("-", live_minus, minus_shadow),
    ):
        rows.append(
            {
                "sign": sign,
                "step": delta,
                "terminal_shadow_hamming": int(jnp.sum(mask != shadow)),
                "picard_trips": evaluations,
                "terminal_residual": float(state.fixed_point.residual),
            }
        )
    return {
        "label": label,
        "probe": probe,
        "reverse_live": live_reverse,
        "reverse_mask_held": held_reverse,
        "finite_difference_live": live_value,
        "finite_difference_mask_held": held_value,
        "mask_change_component": live_value - held_value,
        "reverse_held_relative_error": abs(held_reverse - held_value)
        / max(abs(held_value), 1.0e-30),
        "perturbations": rows,
    }


def _moment_arm(profile, flux, targets):
    """Measure the direct moment map, which does not execute the Picard residual map."""
    jacobian = profile.moment_jacobian(flux, targets)
    rng = np.random.default_rng(11)
    direction = jnp.asarray(rng.standard_normal(flux.shape))
    direction = direction / jnp.max(jnp.abs(direction))
    reverse = np.asarray(jacobian @ direction)
    baseline_shadow = profile.operator.residual_shadow_mask(flux)
    rows, values = [], []
    for step in MOMENT_STEPS:
        plus = flux + step * direction
        minus = flux - step * direction
        plus_shadow = profile.operator.residual_shadow_mask(plus)
        minus_shadow = profile.operator.residual_shadow_mask(minus)
        live = np.asarray(
            (
                profile.moment_residual(plus, targets)
                - profile.moment_residual(minus, targets)
            )
            / (2.0 * step)
        )
        values.append(
            {
                "step": step,
                "finite_difference_live": live.tolist(),
                "finite_difference_mask_held": live.tolist(),
                "mask_change_component": [0.0] * len(live),
                "reverse": reverse.tolist(),
                "relative_error": float(
                    np.max(np.abs(live - reverse)) / np.max(np.abs(reverse))
                ),
            }
        )
        for sign, mask in (("+", plus_shadow), ("-", minus_shadow)):
            rows.append(
                {
                    "sign": sign,
                    "step": step,
                    "terminal_shadow_hamming": int(jnp.sum(mask != baseline_shadow)),
                    "picard_trips": 0,
                    "terminal_residual": None,
                }
            )
    return {
        "label": "moment-map",
        "perturbations": rows,
        "steps": values,
        "note": (
            "The direct moment map does not enter the Picard residual-shadow branch; "
            "holding that branch changes no direct-map value."
        ),
    }


def _draw(report, output):
    """Render the comparison with direct labels and no decorative ink."""
    plt.style.use("data-ink")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    panels = (
        (report["moment-map"], "moment-map", "max |directional derivative|"),
        (report["flux-current"], "flux-current", "d(sum flux²)/dI"),
        (report["plasma-current"], "plasma-current", "d(plasma current)/dI"),
    )
    for axis, (arm, name, unit) in zip(axes, panels, strict=True):
        if name == "moment-map":
            x = np.asarray([entry["step"] for entry in arm["steps"]])
            live = np.asarray(
                [
                    np.max(np.abs(entry["finite_difference_live"]))
                    for entry in arm["steps"]
                ]
            )
            held = np.asarray(
                [
                    np.max(np.abs(entry["finite_difference_mask_held"]))
                    for entry in arm["steps"]
                ]
            )
            reverse = np.max(np.abs(np.asarray(arm["steps"][0]["reverse"])))
        else:
            x = np.asarray([arm["perturbations"][0]["step"]])
            live = np.asarray([arm["finite_difference_live"]])
            held = np.asarray([arm["finite_difference_mask_held"]])
            reverse = arm["reverse_mask_held"]
        axis.plot(x, live, "o-", color="#456990", label="live")
        axis.plot(x, held, "s--", color="#7a7a7a", label="mask-held")
        axis.axhline(reverse, color="#202020", linestyle="--", linewidth=1.2)
        axis.text(x[-1], live[-1], " live", color="#456990", va="bottom")
        axis.text(x[-1], held[-1], " held", color="#7a7a7a", va="top")
        axis.text(x[0], reverse, " reverse", color="#202020", va="bottom")
        axis.set_xscale("log")
        axis.set_xlabel("central-difference step")
        axis.set_ylabel(unit)
        axis.set_title(name)
    figure.savefig(output, dpi=100)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    arguments = parser.parse_args()
    configure_dtypes()
    fixture = _fixture_module()
    profile, seed, _vacuum = fixture.machine.__wrapped__()
    converged = profile.solve(seed, route="anderson", evaluations=fixture.EVALUATIONS)
    targets = MomentTargets(
        plasma_current=0.9 * float(converged.moments.plasma_current),
        poloidal_beta=0.5,
        internal_inductance=0.8,
    )
    current = profile.operator.external_current

    def flux_observable(value):
        return jnp.sum(
            _live_state(profile, seed, value, evaluations=80, relaxation=0.7).flux ** 2
        )

    def plasma_observable(value):
        return _live_state(
            profile, seed, value, evaluations=80, relaxation=0.7
        ).moments.plasma_current

    report = {
        "moment-map": _moment_arm(profile, converged.flux, targets),
        "flux-current": _current_arm(
            profile, seed, current, observable=flux_observable, label="flux-current"
        ),
        "plasma-current": _current_arm(
            profile,
            seed,
            current,
            observable=plasma_observable,
            label="plasma-current",
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _draw(report, arguments.figure)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
