"""Cold production-seed recovery of limited and diverted branch fixtures."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import (
        ColdSeedConstruction,
        ForwardProfile,
        PerturbedSeedPolicy,
        SaddleSeedGeometry,
    )
    from nova.equilibrium.convention import toroidal_current_density
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.equilibrium.stencil_mesh import StencilMesh
    from nova.equilibrium.topology import TopologyClass
    from nova.jax.config import configure_dtypes
    from scripts.analytic_oracle_fixtures.measure import (
        FIXTURE_REQUESTS,
        WALL_POINT_COUNT,
        analytic_case,
        cached_machine,
        exact_current_moments,
        exact_state,
        forward_operator,
    )
    from scripts.dual_basin_fixtures.qualify_diverted_root import (
        ROOT_RECEIPT_PATH,
        qualify,
    )


BANK = Path("scripts/oracle_rebaseline")
DIVERTED_BANK = Path("scripts/dual_basin_fixtures")
NEWTON_STEPS = 10
KRYLOV_ITERATIONS = 30
RECOVERY_CRITERION = 1.0e-10
ROOT_PARITY = 1.0e-10
ROUND_OFF_ABSOLUTE_TOLERANCE = np.finfo(np.float64).eps
DIVERTED_STATE_DIGEST = (
    "11a7e9d00556e91a6d76a69212107592501e1e8cedae60fd17e9e8032ff14801"
)


def _digest(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _profile(operator, machine) -> ForwardProfile:
    lattice = StencilMesh(machine.node, machine.stencil, machine.area)
    return ForwardProfile(operator, lattice, newton_steps=NEWTON_STEPS)


def _solve(profile, seeds):
    def portfolio(states):
        return profile.solve_portfolio(
            states,
            route="newton_krylov",
            tolerance=RECOVERY_CRITERION,
            warmup=0,
            gmres_iterations=KRYLOV_ITERATIONS,
        )

    return jax.jit(portfolio)(seeds)


def _flat(value):
    def profile(psi_norm):
        return jnp.full_like(jnp.asarray(psi_norm), value)

    return profile


def _limited_problem(resolution: str):
    case = analytic_case()
    machine = cached_machine(
        case,
        FIXTURE_REQUESTS[resolution],
        wall_nodes=WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle = exact_state(case, coordinates)
    empty = forward_operator(case, machine)
    physical = exact_current_moments(case, empty, oracle)
    coefficients = empty.coupling_current_moments(physical)
    exterior = oracle - np.asarray(empty.current_moment_image(coefficients))
    profile = _profile(forward_operator(case, machine, exterior), machine)
    receipt = json.loads((BANK / f"receipt-{resolution}.json").read_text())
    root = np.load(BANK / f"root-{resolution}.npz")
    aggregate = receipt["seed"]["aggregate_moment"]
    seeds = profile.cold_seed_portfolio(
        aggregate["declared_current_a"],
        aggregate["current_centroid_m"],
    )
    return profile, seeds, receipt, root


def _diverted_problem():
    case = analytic_case()
    machine = cached_machine(
        case,
        FIXTURE_REQUESTS["fine"],
        wall_nodes=WALL_POINT_COUNT,
    )
    fixture = json.loads((DIVERTED_BANK / "diverted-receipt.json").read_text())
    bank = np.load(DIVERTED_BANK / "diverted-state.npz")
    state = np.asarray(bank["state"])
    gradients = fixture["closed_form"]["constant_flux_functions"]
    source = ForwardSource(
        core=DomainProfile(
            p_prime=_flat(gradients["p_prime_pa_per_wb"]),
            ff_prime=_flat(gradients["ff_prime_t2_m2_per_wb"]),
        ),
        boundary_pressure=0.0,
        boundary_field_function=5.0,
    )
    empty = replace(forward_operator(case, machine), source=source)
    exterior = state - np.asarray(empty.internal(state, TopologyClass.DIVERTED))
    operator = replace(forward_operator(case, machine, exterior), source=source)
    profile = _profile(operator, machine)
    axis = fixture["analytic_stationary_points"]["axis"]["coordinate_m"]
    saddle = fixture["analytic_stationary_points"]["x_point"]["coordinate_m"]
    geometry = SaddleSeedGeometry(tuple(axis), tuple(saddle))
    seed_radius = 0.9 * np.linalg.norm(np.asarray(saddle) - axis)
    supported = np.linalg.norm(machine.node - axis, axis=1) < seed_radius
    cell_current = (
        toroidal_current_density(
            machine.node[:, 0],
            gradients["p_prime_pa_per_wb"],
            gradients["ff_prime_t2_m2_per_wb"],
        )
        * machine.area
        * supported
    )
    total_current = float(cell_current.sum())
    centroid = np.sum(machine.node * cell_current[:, None], axis=0) / total_current
    seeds = profile.cold_seed_portfolio(
        total_current,
        centroid,
        diverted_geometry=geometry,
    )
    return profile, seeds, fixture, state, geometry


@pytest.mark.slow
@pytest.mark.parametrize("resolution", ("coarse", "fine"))
def test_cold_limited_branch_recovers_each_banked_root(resolution):
    configure_dtypes()
    profile, seeds, receipt, root = _limited_problem(resolution)
    banked_seed = np.asarray(root["seed_state"])
    np.testing.assert_array_equal(np.asarray(seeds.branches.flux[0]), banked_seed)
    assert int(seeds.branches.construction[0]) == int(
        ColdSeedConstruction.CURRENT_CENTROID_DISC
    )
    assert not bool(seeds.branches.stored_flux_samples_used[0])
    assert bool(seeds.branches.anchor_available[0])
    assert not bool(seeds.branches.anchor_available[1])

    portfolio = _solve(profile, seeds.branches.flux)
    limited = int(TopologyClass.LIMITED)
    branch = jax.tree.map(lambda value: value[limited], portfolio.branches)
    assert branch.equilibrium.flux.shape == banked_seed.shape
    assert int(branch.requested_class) == limited
    assert int(branch.achieved_class) == limited
    assert bool(branch.topology_consistent)
    assert bool(branch.converged)
    assert float(branch.residual) <= receipt["solver"]["criterion"]
    scale = max(float(np.max(np.abs(root["root_state"]))), np.finfo(float).tiny)
    difference = np.max(
        np.abs(np.asarray(branch.equilibrium.flux) - root["root_state"])
    )
    assert difference / scale <= ROOT_PARITY

    diverted = int(TopologyClass.DIVERTED)
    assert int(portfolio.branches.requested_class[diverted]) == diverted
    assert not bool(portfolio.branches.topology_consistent[diverted])
    assert not bool(portfolio.branches.converged[diverted])


def test_cold_seed_receipts_keep_one_fixed_branch_axis_under_vmap():
    configure_dtypes()
    profile, seeds, _fixture, state, geometry = _diverted_problem()

    def solve(states):
        return profile.solve_portfolio(
            states,
            route="newton_krylov",
            tolerance=RECOVERY_CRITERION,
            warmup=0,
            gmres_iterations=KRYLOV_ITERATIONS,
        )

    batch = jnp.stack((seeds.branches.flux, seeds.branches.flux))
    portfolios = jax.jit(jax.vmap(solve))(batch)
    assert seeds.branches.flux.shape == (2, state.size)
    assert seeds.branches.anchor.shape == (2, 2)
    assert int(seeds.branches.construction[1]) == int(
        ColdSeedConstruction.AXIS_SADDLE_GEOMETRY
    )
    np.testing.assert_array_equal(seeds.branches.declared_axis[1], geometry.axis)
    np.testing.assert_array_equal(seeds.branches.declared_boundary[1], geometry.saddle)
    assert not bool(seeds.branches.stored_flux_samples_used[1])
    assert (
        np.linalg.norm(np.asarray(seeds.branches.anchor[1]) - geometry.saddle) < 1.0e-2
    )
    assert portfolios.branches.equilibrium.flux.shape == (2, 2, state.size)
    np.testing.assert_array_equal(
        np.asarray(portfolios.branches.requested_class[0]),
        (int(TopologyClass.LIMITED), int(TopologyClass.DIVERTED)),
    )
    np.testing.assert_array_equal(
        np.asarray(portfolios.branches.achieved_class[0]),
        (int(TopologyClass.LIMITED), int(TopologyClass.DIVERTED)),
    )
    np.testing.assert_array_equal(
        np.asarray(portfolios.branches.topology_consistent[0]),
        (True, True),
    )
    np.testing.assert_array_equal(
        np.asarray(portfolios.branches.converged[0]),
        (False, False),
    )
    diverted = int(TopologyClass.DIVERTED)
    cold_diverted = jax.tree.map(lambda value: value[0, diverted], portfolios.branches)
    assert int(cold_diverted.requested_class) == diverted
    assert int(cold_diverted.achieved_class) == diverted
    assert bool(cold_diverted.topology_consistent)
    assert not bool(cold_diverted.converged)
    np.testing.assert_allclose(
        np.asarray(portfolios.branches.equilibrium.flux[0]),
        np.asarray(portfolios.branches.equilibrium.flux[1]),
        equal_nan=True,
    )


@pytest.mark.slow
def test_diverted_near_basin_perturbation_ladder_recovers_banked_root():
    configure_dtypes()
    profile, seeds, _fixture, state, _geometry = _diverted_problem()
    diverted = int(TopologyClass.DIVERTED)
    cold_diverted = np.asarray(seeds.branches.flux[diverted])
    direction = cold_diverted - state
    policy = PerturbedSeedPolicy()

    references = jnp.stack((jnp.asarray(state), jnp.asarray(state)))
    directions = jnp.stack((jnp.asarray(direction), jnp.asarray(direction)))
    receipts = jax.jit(
        jax.vmap(
            lambda reference, perturbation: profile.solve_diverted_perturbations(
                reference,
                perturbation,
                policy,
            )
        )
    )(references, directions)
    receipt = jax.tree.map(lambda value: value[0], receipts)

    assert _digest(state) == DIVERTED_STATE_DIGEST
    np.testing.assert_array_equal(
        np.asarray(receipt.relative_amplitude),
        np.asarray(policy.relative_amplitudes),
    )
    actual_amplitude = np.max(
        np.abs(np.asarray(receipt.seed_flux) - state), axis=1
    ) / float(receipt.reference_flux_span)
    np.testing.assert_allclose(actual_amplitude, policy.relative_amplitudes)
    np.testing.assert_array_equal(
        np.asarray(receipt.rungs.requested_class),
        np.full(len(policy.relative_amplitudes), diverted),
    )
    banked_amplitude = 1.0e-2
    banked_rungs = np.asarray(receipt.relative_amplitude) <= banked_amplitude
    assert np.any(np.asarray(receipt.relative_amplitude) > banked_amplitude)
    assert np.all(np.asarray(receipt.passed)[banked_rungs])
    np.testing.assert_array_equal(
        np.asarray(receipt.rungs.achieved_class)[banked_rungs],
        np.full(np.count_nonzero(banked_rungs), diverted),
    )
    assert np.all(np.asarray(receipt.rungs.topology_consistent)[banked_rungs])
    assert np.all(np.asarray(receipt.rungs.converged)[banked_rungs])
    passing = np.asarray(receipt.relative_amplitude)[np.asarray(receipt.passed)]
    assert float(receipt.largest_passing_amplitude) >= banked_amplitude
    assert float(receipt.largest_passing_amplitude) == float(np.max(passing))
    assert np.all(
        np.asarray(receipt.rungs.residual)[np.asarray(receipt.passed)]
        <= RECOVERY_CRITERION
    )
    assert np.all(
        np.asarray(receipt.root_relative_error)[np.asarray(receipt.passed)]
        <= ROOT_PARITY
    )
    np.testing.assert_allclose(
        np.asarray(receipts.rungs.residual[0]),
        np.asarray(receipts.rungs.residual[1]),
        equal_nan=True,
    )
    print(
        "DIVERTED_PERTURBATIONS "
        f"amplitudes={np.asarray(receipt.relative_amplitude).tolist()} "
        f"passed={np.asarray(receipt.passed).tolist()} "
        f"residuals={np.asarray(receipt.rungs.residual).tolist()} "
        f"root_parity={np.asarray(receipt.root_relative_error).tolist()} "
        f"largest_passing={float(receipt.largest_passing_amplitude)}"
    )


@pytest.mark.slow
def test_banked_diverted_state_is_a_machine_precision_pinned_root():
    banked = json.loads(ROOT_RECEIPT_PATH.read_text(encoding="utf-8"))
    measured = qualify(write=False)
    assert measured.keys() == banked.keys()
    assert {key: value for key, value in measured.items() if key != "composition"} == {
        key: value for key, value in banked.items() if key != "composition"
    }

    composition = measured["composition"]
    banked_composition = banked["composition"]
    assert composition.keys() == banked_composition.keys()

    external = composition["external_field"]
    banked_external = banked_composition["external_field"]
    assert external.keys() == banked_external.keys()
    assert (
        external["maximum_absolute_flux_wb"]
        == banked_external["maximum_absolute_flux_wb"]
    )
    assert (
        external["reconstruction_difference_wb"]
        == banked_external["reconstruction_difference_wb"]
    )

    source = composition["source_forcing"]
    banked_source = banked_composition["source_forcing"]
    assert source.keys() == banked_source.keys()
    assert {key: value for key, value in source.items() if key != "sha256"} == {
        key: value for key, value in banked_source.items() if key != "sha256"
    }

    anchor = composition["normalization_anchor"]
    banked_anchor = banked_composition["normalization_anchor"]
    assert anchor.keys() == banked_anchor.keys()
    physical_anchor_fields = (
        "pinned_axis_m",
        "unpinned_axis_m",
        "pinned_boundary_m",
        "unpinned_boundary_m",
        "pinned_axis_flux_wb",
        "unpinned_axis_flux_wb",
        "pinned_boundary_flux_wb",
        "unpinned_boundary_flux_wb",
    )
    for field in physical_anchor_fields:
        np.testing.assert_allclose(
            anchor[field],
            banked_anchor[field],
            rtol=0.0,
            atol=ROUND_OFF_ABSOLUTE_TOLERANCE,
        )
    assert {
        key: value for key, value in anchor.items() if key not in physical_anchor_fields
    } == {
        key: value
        for key, value in banked_anchor.items()
        if key not in physical_anchor_fields
    }
    assert (
        composition["closure_absolute_residual_wb"]
        == banked_composition["closure_absolute_residual_wb"]
    )

    state = measured["state"]
    mapped = measured["map"]
    composition = measured["composition"]
    assert state["sha256"] == DIVERTED_STATE_DIGEST
    assert mapped["requested_class"] == int(TopologyClass.DIVERTED)
    assert mapped["achieved_class"] == int(TopologyClass.DIVERTED)
    assert mapped["topology_consistent"]
    assert mapped["converged"]
    assert mapped["iterations"] == 1
    assert mapped["relative_residual"] <= mapped["machine_precision_floor"]
    assert composition["external_field"]["reconstruction_difference_wb"] == 0.0
    assert composition["source_forcing"]["repeat_difference_wb"] == 0.0
    anchor = composition["normalization_anchor"]
    assert anchor["axis_distance_m"] == 0.0
    assert anchor["boundary_distance_m"] == 0.0
    assert anchor["axis_flux_difference_wb"] == 0.0
    assert anchor["boundary_flux_difference_wb"] == 0.0
    assert anchor["domain_label_difference_count"] == 0
    assert measured["evidence"]["verdict"] == "genuine_machine_precision_root"
