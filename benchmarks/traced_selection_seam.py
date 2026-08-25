"""Prototype continuous traced selection between qualified forward branches.

The production selector is deliberately unchanged.  This benchmark composes
the existing smooth ``p_diverted`` observable with the exact signed
``class_margin`` comparator, then applies the convergence and topology
qualification already carried by ``ForwardBranchReceipt``.  Continuity is
claimed only while both qualification masks stay fixed; loss of qualification
intentionally clamps the result to the sole usable branch.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
from typing import Any, NamedTuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.connectivity_boundary import traced_smooth_boundary_read
from nova.equilibrium.forward import ForwardBranchReceipt, ForwardEquilibrium
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from tests import test_connectivity_boundary as classified_fixtures


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    HERE / "docs/figures/dual-branch-selection/traced-selection-continuity.json"
)
TOPOLOGY_TEMPERATURE = 1.0e-2
INTERPOLATION_POINTS = 257


class TracedBranchSelection(NamedTuple):
    """Continuous flux selection plus its device-visible qualification."""

    flux: jax.Array
    diverted_weight: jax.Array
    selected_class: jax.Array
    qualified: jax.Array


def traced_select_forward_branch(
    branches: ForwardBranchReceipt,
    p_diverted: jax.Array,
    class_margin: jax.Array,
) -> TracedBranchSelection:
    """Blend two branch flux maps without host control flow on traced values.

    The branch axis is ordered limited, diverted.  When both branches qualify,
    the two existing continuous class observables contribute equally; their
    equivalence is measured independently below.  A sole qualified branch is
    selected exactly.  With no qualified branch, the flux and weight are NaN
    and ``qualified`` is false, preventing an unqualified state from looking
    usable.
    """

    p_weight = jnp.clip(jnp.asarray(p_diverted), 0.0, 1.0)
    margin_weight = jax.nn.sigmoid(jnp.asarray(class_margin) / TOPOLOGY_TEMPERATURE)
    smooth_weight = 0.5 * (p_weight + margin_weight)
    qualifies = jnp.asarray(branches.converged) & jnp.asarray(
        branches.topology_consistent
    )
    limited_qualifies = qualifies[int(TopologyClass.LIMITED)]
    diverted_qualifies = qualifies[int(TopologyClass.DIVERTED)]
    both_qualify = limited_qualifies & diverted_qualifies
    exactly_limited = limited_qualifies & ~diverted_qualifies
    exactly_diverted = diverted_qualifies & ~limited_qualifies
    any_qualifies = limited_qualifies | diverted_qualifies
    weight = jnp.where(
        both_qualify,
        smooth_weight,
        jnp.where(
            exactly_limited,
            0.0,
            jnp.where(exactly_diverted, 1.0, jnp.nan),
        ),
    )
    limited_flux = branches.equilibrium.flux[int(TopologyClass.LIMITED)]
    diverted_flux = branches.equilibrium.flux[int(TopologyClass.DIVERTED)]
    flux = limited_flux + weight * (diverted_flux - limited_flux)
    selected_class = jnp.where(
        any_qualifies,
        jnp.asarray(weight >= 0.5, dtype=jnp.int8),
        jnp.asarray(-1, dtype=jnp.int8),
    )
    return TracedBranchSelection(flux, weight, selected_class, any_qualifies)


def _placeholder_equilibrium(flux: jax.Array) -> ForwardEquilibrium:
    """Build a fixed-shape equilibrium payload for selector properties.

    Only ``flux`` participates in this seam.  The remaining array leaves keep
    the real ``ForwardEquilibrium`` container traceable without claiming that
    this prototype performed a forward solve.
    """

    scalar = jnp.asarray(0.0, dtype=flux.dtype)
    return ForwardEquilibrium(
        flux=flux,
        cell_current=scalar[None],
        domains=scalar[None],
        topology=scalar[None],
        fixed_point=scalar[None],
        moments=scalar[None],
        ledger=scalar[None],
        conservation=scalar[None],
        normalisation=scalar[None],
        rotation=scalar[None],
        continuation=scalar[None],
        finite=scalar[None],
    )


def _branch_pair() -> ForwardBranchReceipt:
    """Return a limited/diverted pair with distinct continuous flux states."""

    limited = ForwardBranchReceipt(
        equilibrium=_placeholder_equilibrium(jnp.asarray([-1.0, 0.0, 1.0])),
        requested_class=jnp.asarray(int(TopologyClass.LIMITED), dtype=jnp.int8),
        achieved_class=jnp.asarray(int(TopologyClass.LIMITED), dtype=jnp.int8),
        converged=jnp.asarray(True),
        residual=jnp.asarray(1.0e-12),
        iterations=jnp.asarray(4, dtype=jnp.int32),
        topology_consistent=jnp.asarray(True),
    )
    diverted = ForwardBranchReceipt(
        equilibrium=_placeholder_equilibrium(jnp.asarray([1.0, 2.0, 3.0])),
        requested_class=jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
        achieved_class=jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
        converged=jnp.asarray(True),
        residual=jnp.asarray(2.0e-12),
        iterations=jnp.asarray(5, dtype=jnp.int32),
        topology_consistent=jnp.asarray(True),
    )
    return jax.tree.map(lambda left, right: jnp.stack((left, right)), limited, diverted)


def _class_observables(operator: Any, state: jax.Array) -> tuple[jax.Array, ...]:
    """Read the exact class and its smooth probability from one flux state."""

    physical = jnp.asarray(state)[: operator.physical_node_number]
    grid_flux, wall_flux = operator.topology.split_flux_map(physical)
    _masks, topology = operator._fixed_design_topology.read(
        physical,
        operator.polarity,
        operator.inside_material,
        None,
    )
    _axis_candidates, x_candidates = operator._fixed_design_topology.grid(grid_flux)
    classification_wall = jnp.concatenate(
        (topology.wall_point, topology.wall_point_flux[None])
    )
    radius, height, connectivity_shape = operator.connectivity_grid_axes()
    radial_count, vertical_count = connectivity_shape
    smooth = traced_smooth_boundary_read(
        grid_flux.reshape((radial_count, vertical_count)).T,
        radius,
        height,
        operator.inside_material.reshape((radial_count, vertical_count)).T,
        topology.axis[0],
        topology.axis[1],
        96,
        18,
        2,
        jnp.empty((0,), dtype=radius.dtype),
        jnp.asarray(1.0, dtype=grid_flux.dtype),
        operator.wall.coordinate[:, 0],
        operator.wall.coordinate[:, 1],
        wall_flux,
        jnp.asarray(TOPOLOGY_TEMPERATURE, dtype=grid_flux.dtype),
        classification_x=x_candidates,
        classification_wall=classification_wall,
    )
    return (
        smooth["p_diverted"],
        operator._connectivity_class_margin(physical, topology),
        topology.diverted,
    )


def _observe_family(operator: Any, states: jax.Array) -> tuple[np.ndarray, ...]:
    """Batch the paired class reads for one fixed-shape fixture family."""

    observed = jax.jit(jax.vmap(lambda state: _class_observables(operator, state)))(
        states
    )
    return tuple(np.asarray(value) for value in observed)


def _fixture_families() -> list[dict[str, Any]]:
    """Reconstruct the three classified fixture rails pinned by the test suite."""

    amplitudes = np.linspace(0.0, 0.9, 19)
    _psi, rg, zg, _axis, limiter_r, limiter_z, inside = (
        classified_fixtures._sweep_field(float(amplitudes[0]), 61, 61)
    )
    wall_r, wall_z = classified_fixtures._dense_wall(limiter_r, limiter_z, m=160)
    growing_operator, _lattice = classified_fixtures._forward_operator(
        rg, zg, inside, wall_r, wall_z
    )
    growing_states = jnp.stack(
        [
            classified_fixtures._forward_state(
                classified_fixtures._psi_sweep(*np.meshgrid(rg, zg), float(amplitude)),
                classified_fixtures._psi_sweep(wall_r, wall_z, float(amplitude)),
            )
            for amplitude in amplitudes
        ]
    )

    psi, rg, zg, _axis, limiter_r, limiter_z, inside = (
        classified_fixtures._persistent_saddle_field()
    )
    wall_r, wall_z = classified_fixtures._dense_wall(limiter_r, limiter_z, m=160)
    persistent_operator, _lattice = classified_fixtures._forward_operator(
        rg, zg, inside, wall_r, wall_z
    )
    wall_psi = classified_fixtures._persistent_saddle_psi(wall_r, wall_z)
    base_state = classified_fixtures._forward_state(psi, wall_psi)
    _masks, base = persistent_operator.read(base_state)
    crossing_shift = float(base.x_point_flux) - float(np.max(wall_psi))
    scale = abs(float(base.axis_flux) - float(base.x_point_flux))
    shifts = crossing_shift + scale * np.linspace(-0.75, 0.75, 17)
    persistent_states = jnp.stack(
        [
            classified_fixtures._forward_state(psi, wall_psi + float(shift))
            for shift in shifts
        ]
    )
    growing_observables = _observe_family(growing_operator, growing_states)
    persistent_observables = _observe_family(persistent_operator, persistent_states)
    persistent_classes = persistent_observables[2]
    persistent_flips = np.flatnonzero(persistent_classes[1:] != persistent_classes[:-1])
    if persistent_flips.size != 1:
        raise AssertionError("the persistent-saddle fixture must transition once")
    transition_index = int(persistent_flips[0])
    transition_coordinates = np.linspace(
        shifts[transition_index],
        shifts[transition_index + 1],
        INTERPOLATION_POINTS,
    )
    transition_states = jnp.stack(
        [
            classified_fixtures._forward_state(psi, wall_psi + float(shift))
            for shift in transition_coordinates
        ]
    )
    positive_transition_observables = _observe_family(
        persistent_operator, transition_states
    )
    negative_operator, _lattice = classified_fixtures._forward_operator(
        rg, zg, inside, wall_r, wall_z, polarity=-1
    )
    negative_observables = _observe_family(negative_operator, -persistent_states)
    negative_transition_observables = _observe_family(
        negative_operator, -transition_states
    )
    return [
        {
            "name": "growing_saddle",
            "coordinate_name": "blob_amplitude",
            "coordinates": amplitudes,
            "observables": growing_observables,
            "transition_observables": None,
        },
        {
            "name": "persistent_saddle_positive_polarity",
            "coordinate_name": "wall_flux_shift",
            "coordinates": shifts,
            "observables": persistent_observables,
            "transition_coordinates": transition_coordinates,
            "transition_observables": positive_transition_observables,
        },
        {
            "name": "persistent_saddle_negative_polarity",
            "coordinate_name": "wall_flux_shift",
            "coordinates": shifts,
            "observables": negative_observables,
            "transition_coordinates": transition_coordinates,
            "transition_observables": negative_transition_observables,
        },
    ]


def _finite_number(value: float) -> float | None:
    """Return a strict-JSON finite number or ``None`` for an infinity."""

    return value if np.isfinite(value) else None


def _infinity_sign(value: float) -> str | None:
    """Return a stable description for a non-finite class margin."""

    if np.isposinf(value):
        return "positive"
    if np.isneginf(value):
        return "negative"
    return None


def _transition_interpolation(
    branches: ForwardBranchReceipt,
    family: dict[str, Any],
) -> dict[str, Any] | None:
    """Interpolate the observable bracket around one exact class transition."""

    p_diverted, margins, exact_class = family["observables"]
    flips = np.flatnonzero(exact_class[1:] != exact_class[:-1])
    if flips.size == 0:
        return None
    if flips.size != 1:
        raise AssertionError(
            f"{family['name']} must carry exactly one class transition"
        )
    lower_index = int(flips[0])
    margin_endpoints = margins[lower_index : lower_index + 2].astype(float)
    transition_observables = family["transition_observables"]
    if transition_observables is None:
        return {
            "fixture_indices": [lower_index, lower_index + 1],
            "measured": False,
            "reason": (
                "the exact class transition is an operand-localisation event, "
                "not a persistent two-operand wall-to-saddle hand-off"
            ),
            "fixture_class_margin": [
                _finite_number(float(value)) for value in margin_endpoints
            ],
            "fixture_class_margin_infinity_sign": [
                _infinity_sign(float(value)) for value in margin_endpoints
            ],
        }
    probability_rail, margin_rail, exact_class_rail = transition_observables
    assert len(probability_rail) == INTERPOLATION_POINTS
    assert np.all(np.isfinite(probability_rail))
    assert np.all(np.isfinite(margin_rail))
    probability_from_margin = np.asarray(
        jax.nn.sigmoid(jnp.asarray(margin_rail) / TOPOLOGY_TEMPERATURE)
    )
    probability_difference = float(
        np.max(np.abs(probability_rail - probability_from_margin))
    )
    assert probability_difference <= 64.0 * np.finfo(float).eps
    selection = jax.jit(jax.vmap(traced_select_forward_branch, in_axes=(None, 0, 0)))(
        branches,
        jnp.asarray(probability_rail),
        jnp.asarray(margin_rail),
    )
    weights = np.asarray(selection.diverted_weight, dtype=float)
    flux = np.asarray(selection.flux, dtype=float)
    selected_class = np.asarray(selection.selected_class, dtype=np.int8)
    np.testing.assert_array_equal(
        selected_class, np.asarray(exact_class_rail, dtype=np.int8)
    )
    assert int(np.count_nonzero(selected_class[1:] != selected_class[:-1])) == 1
    margin_step = float(np.max(np.abs(np.diff(margin_rail))))
    weight_step = float(np.max(np.abs(np.diff(weights))))
    lipschitz_bound = margin_step / (4.0 * TOPOLOGY_TEMPERATURE)
    assert np.all(np.isfinite(weights))
    assert np.all(np.isfinite(flux))
    assert weight_step <= lipschitz_bound + 32.0 * np.finfo(float).eps
    return {
        "fixture_indices": [lower_index, lower_index + 1],
        "fixture_bracket_coordinates": [
            float(family["coordinates"][lower_index]),
            float(family["coordinates"][lower_index + 1]),
        ],
        "fixture_p_diverted": [
            float(p_diverted[lower_index]),
            float(p_diverted[lower_index + 1]),
        ],
        "fixture_class_margin": margin_endpoints.tolist(),
        "measured": True,
        "interpolation_points": INTERPOLATION_POINTS,
        "measurement": "dense flux-map reads inside the classified bracket",
        "dense_coordinate_endpoints": [
            float(family["transition_coordinates"][0]),
            float(family["transition_coordinates"][-1]),
        ],
        "dense_exact_class_transition_count": 1,
        "maximum_p_diverted_margin_probability_difference": (probability_difference),
        "maximum_adjacent_weight_step": weight_step,
        "sigmoid_lipschitz_step_bound": lipschitz_bound,
        "maximum_adjacent_flux_step": float(
            np.max(np.linalg.norm(np.diff(flux, axis=0), axis=1))
        ),
        "finite_weights": True,
        "finite_selected_flux": True,
    }


def _qualification_properties(branches: ForwardBranchReceipt) -> dict[str, Any]:
    """Exercise every convergence-mask combination through one vmap call."""

    masks = jnp.asarray([[True, True], [True, False], [False, True], [False, False]])
    batch_size = masks.shape[0]
    batched = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size, *value.shape)), branches
    )
    batched = batched._replace(
        converged=masks,
        topology_consistent=jnp.ones_like(masks),
    )
    selection = jax.jit(jax.vmap(traced_select_forward_branch))(
        batched,
        jnp.full((batch_size,), 0.8),
        jnp.full((batch_size,), TOPOLOGY_TEMPERATURE),
    )
    weights = np.asarray(selection.diverted_weight, dtype=float)
    qualified = np.asarray(selection.qualified, dtype=bool)
    selected_class = np.asarray(selection.selected_class, dtype=np.int8)
    expected_weights = np.asarray(
        [0.5 * (0.8 + float(jax.nn.sigmoid(1.0))), 0.0, 1.0, np.nan]
    )
    np.testing.assert_allclose(weights[:3], expected_weights[:3], rtol=0.0, atol=0.0)
    assert np.isnan(weights[3])
    np.testing.assert_array_equal(qualified, [True, True, True, False])
    np.testing.assert_array_equal(selected_class, [1, 0, 1, -1])
    return {
        "mask_order": [
            "both_qualify",
            "limited_only",
            "diverted_only",
            "neither_qualifies",
        ],
        "diverted_weights": [
            float(weights[0]),
            float(weights[1]),
            float(weights[2]),
            None,
        ],
        "limited_only_exact": bool(weights[1] == 0.0),
        "diverted_only_exact": bool(weights[2] == 1.0),
        "selected_class_code": selected_class.tolist(),
        "neither_qualified_is_explicit": bool(
            np.isnan(weights[3]) and not qualified[3] and selected_class[3] == -1
        ),
    }


def _source_identity() -> dict[str, str]:
    """Return the source commit and tree measured by this prototype."""

    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=HERE, text=True
        ).strip(),
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Measure selector tracing, batching, qualification, and continuity."""

    configure_dtypes()
    branches = _branch_pair()
    compiled = jax.jit(traced_select_forward_branch)
    sample = compiled(
        branches,
        jnp.asarray(0.6),
        jnp.asarray(0.4 * TOPOLOGY_TEMPERATURE),
    )
    jax.block_until_ready(sample)
    jaxpr = jax.make_jaxpr(traced_select_forward_branch)(
        branches,
        jnp.asarray(0.6),
        jnp.asarray(0.4 * TOPOLOGY_TEMPERATURE),
    )
    selection_gradient = jax.grad(
        lambda probability, margin: jnp.sum(
            traced_select_forward_branch(branches, probability, margin).flux
        ),
        argnums=(0, 1),
    )(
        jnp.asarray(0.5),
        jnp.asarray(0.0),
    )
    gradient_values = np.asarray(selection_gradient, dtype=float)
    assert np.all(np.isfinite(gradient_values))
    assert np.all(np.abs(gradient_values) > 0.0)

    families = _fixture_families()
    rows: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []
    total_agreements = 0
    maximum_observable_disagreement = 0.0
    interpolations = []
    for family in families:
        p_diverted, margins, exact_class = family["observables"]
        selection = jax.jit(
            jax.vmap(traced_select_forward_branch, in_axes=(None, 0, 0))
        )(
            branches,
            jnp.asarray(p_diverted),
            jnp.asarray(margins),
        )
        weights = np.asarray(selection.diverted_weight, dtype=float)
        selected_class = np.asarray(selection.selected_class, dtype=np.int8)
        expected_class = np.asarray(exact_class, dtype=np.int8)
        agreement = selected_class == expected_class
        assert np.all(np.asarray(selection.qualified))
        assert np.all(np.isfinite(weights))
        assert np.all((weights >= 0.0) & (weights <= 1.0))
        assert np.all(agreement)
        margin_probability = np.asarray(
            jax.nn.sigmoid(jnp.asarray(margins) / TOPOLOGY_TEMPERATURE),
            dtype=float,
        )
        observable_disagreement = np.abs(p_diverted - margin_probability)
        family_maximum_disagreement = float(np.max(observable_disagreement))
        maximum_observable_disagreement = max(
            maximum_observable_disagreement, family_maximum_disagreement
        )
        total_agreements += int(np.count_nonzero(agreement))
        transitions = int(np.count_nonzero(expected_class[1:] != expected_class[:-1]))
        family_summaries.append(
            {
                "name": family["name"],
                "continuity_role": (
                    "localisation_boundary"
                    if family["transition_observables"] is None
                    else "continuous_transition_rail"
                ),
                "classified_fixtures": int(len(weights)),
                "exact_class_agreements": int(np.count_nonzero(agreement)),
                "class_transitions": transitions,
                "maximum_p_diverted_margin_probability_difference": (
                    family_maximum_disagreement
                ),
                "maximum_adjacent_fixture_weight_step": float(
                    np.max(np.abs(np.diff(weights)))
                ),
            }
        )
        interpolation = _transition_interpolation(branches, family)
        if interpolation is not None:
            interpolations.append({"family": family["name"], **interpolation})
        for index, (coordinate, probability, margin, exact, weight) in enumerate(
            zip(
                family["coordinates"],
                p_diverted,
                margins,
                exact_class,
                weights,
                strict=True,
            )
        ):
            rows.append(
                {
                    "family": family["name"],
                    "fixture_index": index,
                    family["coordinate_name"]: float(coordinate),
                    "p_diverted": float(probability),
                    "class_margin": _finite_number(float(margin)),
                    "class_margin_infinity_sign": _infinity_sign(float(margin)),
                    "exact_class": "diverted" if bool(exact) else "limited",
                    "selected_class": (
                        "diverted" if bool(weight >= 0.5) else "limited"
                    ),
                    "diverted_weight": float(weight),
                    "both_branches_qualified": True,
                }
            )

    classified_count = sum(item["classified_fixtures"] for item in family_summaries)
    assert classified_count == 53
    assert total_agreements == classified_count
    assert maximum_observable_disagreement <= 64.0 * np.finfo(float).eps
    measured_interpolations = [item for item in interpolations if item["measured"]]
    assert measured_interpolations

    qualification = _qualification_properties(branches)
    receipt = {
        "artifact": "traced_selection_continuity",
        "created_at": datetime.now(UTC).isoformat(),
        "source": _source_identity(),
        "scope": {
            "verdict": "qualified_prototype",
            "prototype_only": True,
            "production_selector_changed": False,
            "forward_solve_performed": False,
            "claim_boundary": (
                "selection continuity is conditional on fixed convergence and "
                "topology-qualification masks; a mask change deliberately clamps "
                "to the sole qualified branch"
            ),
            "localisation_limitation": (
                "the growing-saddle fixture changes class when the saddle operand "
                "appears from an infinite sentinel, so it is retained as an "
                "observable-localisation boundary rather than claimed as a "
                "continuous two-operand transition"
            ),
        },
        "selector": {
            "branch_axis": ["limited", "diverted"],
            "both_qualified_weight": (
                "0.5 * (p_diverted + sigmoid(class_margin / temperature))"
            ),
            "temperature_normalised_flux": TOPOLOGY_TEMPERATURE,
            "state_rule": "limited_flux + weight * (diverted_flux - limited_flux)",
            "class_rule": (
                "diverted when weight >= 0.5; minus one when neither qualifies"
            ),
            "qualification_rule": (
                "ForwardBranchReceipt.converged and topology_consistent"
            ),
        },
        "properties": {
            "jax_jit_traceable": True,
            "jaxpr_equation_count": len(jaxpr.jaxpr.eqns),
            "jax_vmap_batchable": True,
            "finite_nonzero_observable_gradients": True,
            "flux_sum_gradient_by_observable": {
                "p_diverted": float(gradient_values[0]),
                "class_margin": float(gradient_values[1]),
            },
            "classified_fixture_count": classified_count,
            "exact_boolean_class_agreement_count": total_agreements,
            "exact_boolean_class_disagreement_count": (
                classified_count - total_agreements
            ),
            "maximum_p_diverted_margin_probability_difference": (
                maximum_observable_disagreement
            ),
            "sole_qualified_branch_degradation": qualification,
            "continuous_transition_interpolations_measured": len(
                measured_interpolations
            ),
            "localisation_transition_limitation_count": len(interpolations)
            - len(measured_interpolations),
            "all_property_assertions_passed": True,
        },
        "fixture_source": {
            "path": "tests/test_connectivity_boundary.py",
            "families": family_summaries,
        },
        "transition_interpolations": interpolations,
        "fixtures": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    """Run the prototype and print its compact property summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.output)
    print(json.dumps(receipt["properties"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
