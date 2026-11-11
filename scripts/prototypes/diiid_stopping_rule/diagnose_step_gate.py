"""Print the inputs of the reduced-route step-admission predicate.

The reduced Newton route proposes a step only when one of its backtracking
candidates strictly lowers the merit: ``found = jnp.any(valid)`` where
``valid = isfinite(merit) & (merit < current_merit)``.  A report of zero
attempted promotions therefore means either the direction is zero or no
damped candidate improves, and the two need different repairs.  This script
wraps the direction and score kernels to print the quantities the predicate
compares, at every trip, while the frame is solved through the public route.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))

import measure_branches  # noqa: E402

import nova.equilibrium.reduced_newton as reduced_newton  # noqa: E402
from nova.equilibrium.topology import TopologyClass  # noqa: E402


def _instrument() -> None:
    original = reduced_newton._reduced_kernels

    def wrapped(*arguments, **keywords):
        kernels = original(*arguments, **keywords)
        direction = kernels["direction"]
        scores = kernels["step_scores"]

        def traced_direction(*args, **kwargs):
            value = direction(*args, **kwargs)
            jax.debug.print(
                "*",
                "DIRECTION",
                "norm={n}",
                "max={m}",
                n=jnp.linalg.norm(value),
                m=jnp.max(jnp.abs(value)),
            )
            return value

        def traced_scores(reduced, shadow, base_state, *args, **kwargs):
            result = scores(reduced, shadow, base_state, *args, **kwargs)
            jax.debug.print(
                "*",
                "SCORES",
                "merit={merit}",
                "flux={flux}",
                "mask_difference={difference}",
                merit=result.merit,
                flux=result.flux_residual,
                difference=result.mask_difference,
            )
            return result

        kernels["direction"] = traced_direction
        kernels["step_scores"] = traced_scores
        return kernels

    reduced_newton._reduced_kernels = wrapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", choices=("limited", "diverted"), required=True)
    parser.add_argument("--trips", type=int, default=2)
    parser.add_argument(
        "--pseudo-wall-expansion",
        type=float,
        default=None,
        help="explicit pseudo-wall control; omit to use the governed limiter",
    )
    arguments = parser.parse_args()

    _instrument()
    prepared = measure_branches._prepare_frame(
        arguments.frame, arguments.pseudo_wall_expansion
    )
    topology_class = TopologyClass[arguments.branch.upper()]
    seed = prepared["cold"].branches.flux[int(topology_class)]
    receipt, seconds = measure_branches._solve(
        prepared["profile"],
        seed,
        prepared["current"],
        prepared["target_current_a"],
        f"step-gate-diagnosis:{arguments.frame}:{arguments.branch}",
        active_set_steps=arguments.trips,
        stagnation_stop=False,
    )
    receipt.equilibrium.flux.block_until_ready()
    print(
        "FRAME",
        arguments.frame,
        "BRANCH",
        arguments.branch,
        "SECONDS",
        round(seconds, 1),
    )
    print("residual_history", measure_branches._finite_list(receipt.residual_history))
    print("mask_history", measure_branches._int_list(receipt.changed_mask_entries))
    print("attempted", int(receipt.attempted_newton_promotions))
    print("accepted", int(receipt.accepted_newton_promotions))
    print("terminal_residual", float(receipt.equilibrium.fixed_point.residual))


if __name__ == "__main__":
    main()
