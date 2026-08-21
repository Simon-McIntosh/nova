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
DIVERTED_STATE_DIGEST = (
    "1e6e1043b84a7833adeb51916d3dca36c92d291a4a40dd924ce9b7cae87e7a8d"
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

    def one_step(states):
        return profile.solve_portfolio(
            states,
            route="picard",
            evaluations=1,
            tolerance=np.inf,
        )

    batch = jnp.stack((seeds.branches.flux, seeds.branches.flux))
    portfolios = jax.jit(jax.vmap(one_step))(batch)
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
    np.testing.assert_allclose(
        np.asarray(portfolios.branches.equilibrium.flux[0]),
        np.asarray(portfolios.branches.equilibrium.flux[1]),
        equal_nan=True,
    )


@pytest.mark.slow
def test_banked_diverted_state_is_a_machine_precision_pinned_root():
    banked = json.loads(ROOT_RECEIPT_PATH.read_text(encoding="utf-8"))
    measured = qualify(write=False)
    assert measured == banked

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
