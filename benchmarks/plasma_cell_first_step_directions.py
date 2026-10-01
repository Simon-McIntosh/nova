"""Measure current-support directions and every physical trial in one native trip.

The normalized map and its Newton residual have different null spaces. Both are
reported on the unshadowed coordinates actually advanced by GMRES. A frozen
lambda control changes the operator, not the measured physical displacement.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

NEGATIVE_CONTROL = (
    "the same singular-direction analysis repeated with lambda held fixed at its "
    "seed value must change the projection of the accepted step onto the "
    "support-shrink direction; if the projection reading does not change, the "
    "degenerate-direction reading is not supported and the receipt says so"
)


def clean(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def install_observers(fp, events, output):
    """Observe native trial evaluations and decisions without modifying carries."""
    import jax
    import numpy as np

    originals = {}

    def emit(kind, *values):
        arrays = [np.asarray(value).copy() for value in values]
        index = len(events)
        events.append((kind, arrays))
        np.savez(
            output / f"event-{index:05d}.npz",
            **{f"value_{i}": value for i, value in enumerate(arrays)},
        )
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": index, "kind": kind}) + "\n")
        if kind != "trial":
            print(f"EVENT {index} {kind}", flush=True)

    def callback(kind, *values):
        jax.debug.callback(functools.partial(emit, kind), *values, ordered=True)

    families = (
        "_backtracking_scores",
        "_backtracked_promotion",
        "_rebuilt_model_promotion",
        "_steepest_descent_promotion",
    )
    for name in families:
        original = getattr(fp, name)
        originals[name] = original

        def instrument(*args, _original=original, _name=name, **kwargs):
            bound = inspect.signature(_original).bind(*args, **kwargs)
            bound.apply_defaults()
            acceptance = (
                bound.arguments["acceptance_map_fn"] or bound.arguments["map_fn"]
            )

            def observed(candidate):
                mapped = acceptance(candidate)
                callback("trial", candidate, mapped)
                return mapped

            bound.arguments["acceptance_map_fn"] = observed
            result = _original(*bound.args, **bound.kwargs)
            if _name == "_backtracking_scores":
                callback("ladder", *result)
            else:
                callback(_name, result.state, result.accepted)
            return result

        setattr(fp, name, instrument)

    original_complete = fp._complete_newton_promotion
    originals["_complete_newton_promotion"] = original_complete

    def complete(*args, **kwargs):
        bound = inspect.signature(original_complete).bind(*args, **kwargs)
        values = bound.arguments
        result = original_complete(*args, **kwargs)
        state, mapped = values["state"], values["mapped"]
        candidate = result.state
        predicted = mapped + values["tangent"](candidate - state)
        actual = values["frozen_map"](candidate)
        callback(
            "promotion",
            state,
            candidate,
            fp._smooth_relative_sup_merit(mapped, state),
            fp._smooth_relative_sup_merit(predicted, candidate),
            fp._smooth_relative_sup_merit(actual, candidate),
            result.accepted > values["measured"].accepted,
            values["promotion"].model_distrusted,
            values["qualified_step"].step,
        )
        return result

    fp._complete_newton_promotion = complete
    original_carry = fp._ActiveSetIterationState
    originals["_ActiveSetIterationState"] = original_carry

    def carry(*args, **kwargs):
        result = original_carry(*args, **kwargs)
        callback("trip", result.state, result.live_residual, result.iterations)
        return result

    fp._ActiveSetIterationState = carry
    return originals


def spectrum(matrix, step, shrink, coordinates, name):
    import numpy as np

    _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
    order = np.argsort(singular)
    threshold = float(singular.max() * 1e-6)
    near = singular <= threshold
    norm = float(np.linalg.norm(step))
    coefficients = vectors @ step
    fractions = coefficients**2 / max(norm**2, np.finfo(float).tiny)
    rows = []
    for index in order[:12]:
        rows.append(
            {
                "singular_value": singular[index],
                "relative_singular_value": singular[index] / singular.max(),
                "right_singular_vector_active_coordinates": vectors[index],
                "support_shrink_cosine": np.dot(vectors[index], shrink),
                "step_projection_wb": coefficients[index],
                "step_energy_fraction": fractions[index],
            }
        )
    return clean(
        {
            "operator": name,
            "shape": matrix.shape,
            "active_coordinate_indices": coordinates,
            "all_singular_values_descending": singular,
            "smallest_directions": rows,
            "near_null_relative_threshold": 1e-6,
            "near_null_absolute_threshold": threshold,
            "near_null_dimension": int(near.sum()),
            "step_energy_fraction_in_near_null_space": fractions[near].sum(),
            "mainly_near_null": bool(fractions[near].sum() > 0.5),
            "support_shrink_directional_gain": np.linalg.norm(matrix @ shrink),
            "step_norm_wb": norm,
        }
    )


def render(driver, built, states, residual, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from benchmarks.plasma_cell_trip_panels import wall_units

    reference, _ = driver._nulls(built["operator"], built["analytic"])
    assert reference["axis_rz_m"] is not None
    assert reference["x_point_rz_m"] is not None
    wall = built["machine"].wall_node
    panels = []
    for name, state in states.items():
        nulls, _ = driver._nulls(built["operator"], state)
        levels = np.linspace(np.min(state), np.max(state), 17)[1:-1]
        fig, ax = plt.subplots(figsize=(6, 7), constrained_layout=True)
        counts = []
        for flux, color in ((built["analytic"], "#84a7dc"), (state, "#303030")):
            r, z, field = driver.certificate._raster_field(
                built["coordinates"], flux, wall
            )
            contours = driver.poloidal.draw_flux_contours(
                ax, r, z, field, levels, color=color
            )
            counts.append(
                sum(len(segment) > 1 for group in contours.allsegs for segment in group)
            )
        assert counts[-1] > 0
        driver.poloidal.draw_wall(ax, units=wall_units(driver, built["operator"], wall))
        for found, color, size in ((reference, "#3366cc", 10), (nulls, "#d52d28", 5)):
            others = np.asarray(found["qualified_saddles_rz_m"]).reshape(-1, 2)
            admitted = found["x_point_rz_m"]
            if admitted is not None:
                others = others[np.linalg.norm(others - admitted, axis=1) > 1e-10]
            driver.poloidal.draw_nulls(
                ax,
                magnetic_axis=found["axis_rz_m"],
                x_points=admitted,
                other_x_points=others,
                style=driver.DEFAULT_INK.variant(
                    axis_color=color,
                    xpoint_color=color,
                    axis_markersize=size,
                    xpoint_markersize=size + 2,
                ),
            )
        driver.poloidal_axes(ax)
        score = residual[name]
        ax.set_title(
            f"{name}: own levels\nresidual={score:.8g}; converged={score <= 1e-12}"
        )
        fig.supxlabel(
            "Blue: analytic contours and large nulls. Gray/red: measured state.\n"
            "Triangles: axes; filled crosses: admitted saddles; hollow: other saddles.",
            fontsize=8,
        )
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=170)
        plt.close(fig)
        panels.append(
            {
                "state": name,
                "levels_wb": levels.tolist(),
                "contour_segments": counts,
                "nulls": nulls,
                "reference_nulls": reference,
            }
        )
    return panels


def measure(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import nova
    from benchmarks import plasma_cell_terminal_state as driver
    from benchmarks.plasma_cell_fixed_point_attribution import build_construction
    from nova.equilibrium import fixed_point as fp
    from nova.jax.config import (
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    started = time.monotonic()
    driver.configure_dtypes()
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "gpu"
    assert Path(nova.__file__).resolve().is_relative_to(Path.cwd())
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt = {
        "status": "in-progress",
        "source_revision": revision,
        "dispatch_base": "b0a432d184a747cf3dbed176dc354507c6af0b03",
        "worktree": str(Path.cwd()),
        "nova_file": nova.__file__,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "job": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "devices": [str(device) for device in jax.devices()],
        "requested_cells": 110,
        "clip_mode": "exact",
        "case": "diverted-single-null",
    }
    write(output / "first-step.json", receipt)
    observed = []
    jax.jit(
        lambda x: (
            jax.debug.callback(lambda y: observed.append(float(y)), x, ordered=True),
            x + 1,
        )[1]
    )(jnp.asarray(2.0)).block_until_ready()
    jax.effects_barrier()
    assert observed == [2.0], observed
    receipt["gpu_callback_positive_control"] = observed
    driver.set_support_clip_mode("exact")
    print("BUILD production", flush=True)
    built = build_construction(driver, "diverted-single-null", "production")
    operator, request = built["operator"], built["request"]
    seed = jnp.asarray(request.seed_policy.state)
    target = request.target_current
    assert request.policy.route == "newton_krylov" and request.policy.current_pin
    receipt.update(
        construction=built["snapshot"],
        provenance=built["provenance"],
        realised_cells=len(built["machine"].node),
    )
    write(output / "first-step.json", receipt)

    @jax.jit
    def health(state):
        partition = operator._support_partition(state)
        moments = operator._partitioned_current_moments(partition)
        values = jnp.stack(moments)
        current = jnp.sum(moments.cell_current)
        amplitude = operator.current_normalisation_amplitude(target, current)
        return (
            jnp.sum(partition[0].label == 1),
            current,
            amplitude,
            jnp.sum(~jnp.isfinite(values)),
            jnp.count_nonzero(values),
            values.size,
        )

    def read_health(state):
        values = [float(x) for x in jax.block_until_ready(health(jnp.asarray(state)))]
        return clean(
            dict(
                zip(
                    (
                        "core_cells",
                        "unscaled_support_current_a",
                        "lambda",
                        "nonfinite_support_count",
                        "nonzero_moment_count",
                        "moment_elements",
                    ),
                    values,
                    strict=True,
                )
            )
        )

    seed_health = read_health(seed)
    assert seed_health["core_cells"] > 0 and seed_health["nonzero_moment_count"] > 0
    assert np.count_nonzero(~np.isfinite([1.0, np.nan])) == 1
    receipt["seed"] = seed_health
    print(f"SEED {seed_health}", flush=True)
    write(output / "first-step.json", receipt)
    events = []
    originals = install_observers(fp, events, output)
    try:
        print("SOLVE one native active-set trip", flush=True)
        solved = built["profile"].solve(
            replace(request, policy=replace(request.policy, active_set_steps=1))
        )
        jax.block_until_ready(solved.equilibrium.flux)
        jax.effects_barrier()
    finally:
        for name, original in originals.items():
            setattr(fp, name, original)
    terminal = np.asarray(solved.equilibrium.flux)
    trips = [
        values for kind, values in events if kind == "trip" and int(values[2]) == 1
    ]
    assert len(trips) == 1
    np.testing.assert_array_equal(terminal, trips[0][0])
    assert any(kind == "ladder" for kind, _ in events), (
        "trial instrument saw no native ladder"
    )
    residual = float(solved.equilibrium.fixed_point.residual)
    receipt.update(
        trip_residual=residual,
        converged=bool(solved.equilibrium.fixed_point.converged),
        resolved_defaults=solved.resolved_defaults.to_dict(),
        accepted_trip=read_health(terminal),
        raw_event_count=len(events),
    )
    np.savez(
        output / "states.npz",
        seed=np.asarray(seed),
        accepted_trip=terminal,
        coordinates=built["coordinates"],
    )
    write(output / "first-step.json", receipt)
    print(f"TRIP residual={residual} health={receipt['accepted_trip']}", flush=True)
    promotions = []
    trials = []
    pending = []
    for event_id, (kind, values) in enumerate(events):
        if kind == "trial":
            state, mapped = values
            entry = {
                "event": event_id,
                "inner_attempt": len(promotions) + 1,
                "acceptance": False,
                "role": "physical merit evaluation",
                **read_health(state),
                "merit": float(
                    fp._smooth_relative_sup_merit(
                        jnp.asarray(mapped), jnp.asarray(state)
                    )
                ),
            }
            trials.append(entry)
            pending.append((entry, state))
        elif kind == "promotion":
            (
                origin,
                candidate,
                before,
                predicted,
                actual,
                accepted,
                distrusted,
                proposed,
            ) = values
            for entry, state in pending:
                entry["acceptance"] = bool(accepted) and bool(
                    np.array_equal(state, candidate)
                )
                if np.array_equal(state, origin):
                    entry["role"] = "incumbent evaluation"
            pending.clear()
            promotions.append(
                {
                    "event": event_id,
                    "accepted": bool(accepted),
                    "model_distrusted": bool(distrusted),
                    "norm_wb": np.linalg.norm(candidate - origin),
                    "predicted_merit_decrease": float(before - predicted),
                    "actual_merit_decrease": float(before - actual),
                    "merit_before": float(before),
                    "predicted_merit": float(predicted),
                    "actual_merit": float(actual),
                    "candidate_health": read_health(candidate),
                }
            )
    receipt["trials"] = trials
    receipt["promotions"] = promotions
    receipt["trial_semantics"] = (
        "Every native nonlinear trial-map evaluation, including refused and "
        "duplicate evaluations; incumbent evaluations are labelled. GMRES "
        "basis vectors are linear algebra directions, not physical trial "
        "states."
    )
    receipt["collapse_definition"] = (
        "First physical candidate below half the seed core-cell count or half "
        "its absolute unscaled current; diagnostic threshold, not a "
        "production guard."
    )
    collapsed = [
        row
        for row in trials
        if row["role"] != "incumbent evaluation"
        and (
            row["core_cells"] < seed_health["core_cells"] / 2
            or (
                row["unscaled_support_current_a"] is not None
                and abs(row["unscaled_support_current_a"])
                < abs(seed_health["unscaled_support_current_a"]) / 2
            )
        )
    ]
    receipt["first_collapse_trial"] = collapsed[0] if collapsed else None
    receipt["any_nonfinite_trial"] = any(
        row["nonfinite_support_count"] > 0 for row in trials
    )
    write(output / "first-step.json", receipt)

    print("JACOBIAN seed raw image and current", flush=True)
    shadow = operator.residual_shadow_mask(seed)
    active = np.flatnonzero(~np.asarray(shadow))
    seed_moments = operator.cell_current_moments(seed)
    seed_image = np.asarray(operator.current_moment_image(seed_moments))
    seed_current = float(jnp.sum(seed_moments.cell_current))
    amplitude = float(operator.current_normalisation_amplitude(target, seed_current))

    def raw_map(state, op):
        moments = op.cell_current_moments(state)
        return jnp.concatenate(
            (op.current_moment_image(moments), jnp.sum(moments.cell_current)[None])
        )

    @jax.jit
    def columns(state, directions, op):
        return jax.vmap(
            lambda direction: jax.jvp(
                lambda flux: raw_map(flux, op), (state,), (direction,)
            )[1]
        )(directions)

    blocks = []
    for start in range(0, len(active), 4):
        indices = active[start : start + 4]
        directions = jnp.eye(seed.size, dtype=seed.dtype)[indices]
        blocks.append(
            np.asarray(jax.block_until_ready(columns(seed, directions, operator)))
        )
        if start % 80 == 0:
            print(f"JACOBIAN columns {start}/{len(active)}", flush=True)
    derivative = np.concatenate(blocks).T
    gradient = derivative[-1]
    assert np.all(np.isfinite(derivative)), "seed derivative nonfinite"
    assert np.linalg.norm(gradient) > 0, "support-shrink gradient absent"
    shrink = -np.sign(seed_current) * gradient / np.linalg.norm(gradient)
    raw_jacobian = derivative[:-1][active]
    fixed_jacobian = amplitude * raw_jacobian
    normalized_jacobian = fixed_jacobian - (amplitude / seed_current) * np.outer(
        seed_image[active], gradient
    )
    step = (terminal - np.asarray(seed))[active]
    identity = np.eye(len(active))
    receipt["support_shrink"] = {
        "definition": (
            "negative Euclidean gradient of absolute unscaled current at seed, "
            "restricted to unshadowed state coordinates; lambda renormalization "
            "keeps scaled current fixed"
        ),
        "active_coordinate_indices": active,
        "unit_direction": shrink,
        "current_gradient_a_per_wb": gradient,
        "step_cosine": np.dot(step, shrink) / np.linalg.norm(step),
        "step_projection_wb": np.dot(step, shrink),
        "step_norm_wb": np.linalg.norm(step),
        "shadow_step_norm_wb": np.linalg.norm(
            (terminal - np.asarray(seed))[np.asarray(shadow)]
        ),
    }
    receipt["spectra"] = {}
    for name, matrix in (
        ("normalized_map", normalized_jacobian),
        ("fixed_lambda_map", fixed_jacobian),
        ("normalized_newton_residual", identity - normalized_jacobian),
        ("fixed_lambda_newton_residual", identity - fixed_jacobian),
    ):
        print(f"SVD {name}", flush=True)
        receipt["spectra"][name] = spectrum(matrix, step, shrink, active, name)
        write(output / "first-step.json", receipt)
    np.savez(
        output / "jacobians.npz",
        normalized=normalized_jacobian,
        fixed_lambda=fixed_jacobian,
        current_gradient=gradient,
        active=active,
    )
    # A central difference proves the local support direction lowers current.
    epsilon = 1e-6 * max(float(np.linalg.norm(seed)), 1e-12)
    perturbation = np.zeros(seed.size)
    perturbation[active] = shrink * epsilon
    minus = read_health(np.asarray(seed) - perturbation)
    plus = read_health(np.asarray(seed) + perturbation)
    receipt["support_direction_check"] = {
        "epsilon_wb": epsilon,
        "minus": minus,
        "plus": plus,
        "lowers_current": abs(plus["unscaled_support_current_a"])
        < abs(minus["unscaled_support_current_a"]),
    }
    projection = float(np.dot(step, shrink))
    control_projection = float(np.dot(step, shrink))
    changed = not np.isclose(projection, control_projection, rtol=1e-12, atol=1e-15)
    receipt["negative_control"] = {
        "mutation": NEGATIVE_CONTROL,
        "fixed_lambda": amplitude,
        "normalized_step_projection_wb": projection,
        "fixed_lambda_step_projection_wb": control_projection,
        "projection_changed": changed,
        "verdict": "not-supported" if not changed else "changed",
        "interpretation": (
            "The accepted physical displacement and unscaled-current gradient are "
            "held fixed, so their geometric projection is invariant under "
            "changing lambda in the map. The separately measured singular "
            "subspaces may change; this declared control does not establish a "
            "causal degenerate-direction explanation."
        ),
    }
    with (output / "negative-control.log").open("w") as stream:
        stream.write(NEGATIVE_CONTROL + "\n")
        stream.write(json.dumps(clean(receipt["negative_control"]), indent=2) + "\n")
    receipt["near_null_verdict"] = {
        name: {
            "mainly": data["mainly_near_null"],
            "energy_fraction": data["step_energy_fraction_in_near_null_space"],
        }
        for name, data in receipt["spectra"].items()
    }
    receipt["interpretation_supported_by_declared_control"] = changed
    write(output / "first-step.json", receipt)
    print("RENDER own-level panels", flush=True)
    live_map = operator.flux_map(target_current=target)
    seed_residual = float(fp._relative_residual(live_map(seed), seed))
    receipt["panels"] = render(
        driver,
        built,
        {"seed": np.asarray(seed), "accepted-trip": terminal},
        {"seed": seed_residual, "accepted-trip": residual},
        output,
    )
    receipt.update(status="measured", wall_seconds=time.monotonic() - started)
    write(output / "first-step.json", receipt)
    print("MEASURED " + json.dumps(clean(receipt["near_null_verdict"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measure(args.output)


if __name__ == "__main__":
    main()
