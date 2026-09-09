"""Host-geometry and member-data boundaries for heterogeneous forward maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from benchmarks.mast_response_carrier_warm import (
    DEFAULT_CARRIER,
    SEMANTIC_RESPONSE_IDENTITY,
)
from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward_operator import (
    ForwardFluxOperator,
    PrescribedCurrentField,
    stack_forward_operators,
)
from nova.equilibrium.solve_request import (
    ForwardSolveMemberData,
    SampledFluxFunction,
)
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


OPERAND_CARRIER = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20260901T143942750055-sto-mast-bank-telemetry-relaunch/"
    "logs/exact-operand-cache.npz"
)
WIDTH = 12
DIIID_MACHINE_ARTIFACT_CACHE = Path(
    "/work/projects/imas_gpu/sophelio/diiid_machine_artifact_cache"
)


@dataclass
class _ReferenceAnchoredOperator(ForwardFluxOperator):
    """Read current on member-supplied flux anchors over one fixed support."""

    declared_axis_flux: float = 0.0
    declared_boundary_flux: float = 1.0
    declared_support: np.ndarray | None = None

    def __post_init__(self, prescribed_current_field):
        super().__post_init__(prescribed_current_field)
        if self.declared_axis_flux == self.declared_boundary_flux:
            raise ValueError("reference flux anchors need a nonzero span")
        if self.declared_support is None:
            raise ValueError("reference support is required")
        self.declared_support = jnp.asarray(self.declared_support, dtype=bool)
        if self.declared_support.shape != (self.grid.node_number,):
            raise ValueError("reference support must match the grid")
        self.use_linear_moments = False

    def cell_current_moments(self, psi, requested_class=None) -> CellCurrentMoments:
        del requested_class
        grid_flux = jnp.asarray(psi)[: self.grid.node_number]
        normalized = (grid_flux - self.declared_axis_flux) / (
            self.declared_boundary_flux - self.declared_axis_flux
        )
        density = self.source.core.current_density(self.radius, normalized)
        current = jnp.where(self.declared_support, density * self.area, 0.0)
        zero = jnp.zeros_like(current)
        return CellCurrentMoments(current, zero, zero)


def _read_carriers() -> tuple[
    np.ndarray, np.ndarray, list[dict], dict[str, np.ndarray]
]:
    if not DEFAULT_CARRIER.exists() or not OPERAND_CARRIER.exists():
        pytest.skip("persisted MAST carriers are unavailable")
    with np.load(DEFAULT_CARRIER, allow_pickle=False) as stored:
        identity = str(stored["semantic_response_identity"].item())
        if identity != SEMANTIC_RESPONSE_IDENTITY:
            raise AssertionError("response carrier semantic identity changed")
        response = np.array(stored["response"], copy=True)
        targets = np.array(stored["resolved_targets"], copy=True)
    with np.load(OPERAND_CARRIER, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        if metadata["response_carrier_semantic_identity"] != identity:
            raise AssertionError("operand and response carriers disagree")
        arrays = {name: np.array(stored[name], copy=True) for name in stored.files}
    if int(metadata["arm_count"]) != WIDTH:
        raise AssertionError("operand carrier must contain twelve members")
    return response, targets, metadata["rows"], arrays


def _synthetic_profile_batch():
    """Build MAST-shaped members with independently varying numerical inputs.

    The persisted bank has two arms for each of six source profiles.  This
    fixture retains its MAST geometry, responses, and cached seeds while
    synthesizing the member data that must vary across all twelve slices.
    """
    response, targets, rows, arrays = _read_carriers()
    full_count = 33
    selection = np.arange(0, full_count, 4)
    full_indices = np.asarray(
        [
            radial * full_count + vertical
            for radial in selection
            for vertical in selection
        ]
    )
    grid_response = response[full_indices]
    wall_response = response[full_count * full_count :]
    grid_coordinate = targets[full_indices]
    wall_coordinate = targets[full_count * full_count :]
    radius = np.unique(grid_coordinate[:, 0])
    height = np.unique(grid_coordinate[:, 1])
    lattice = FluxLattice(radius, height)
    np.testing.assert_array_equal(lattice.coordinate, grid_coordinate)
    support = np.ones(lattice.node_count, dtype=bool)
    profile_coordinate = np.linspace(0.0, 1.0, 65)
    operators = []
    member_data = []
    for index, row in enumerate(rows):
        source_scale = 1.0 + 0.125 * index
        source = ForwardSource(
            core=DomainProfile(
                p_prime=SampledFluxFunction(
                    profile_coordinate,
                    source_scale * 2.0e-7 * (1.0 - profile_coordinate),
                ),
                ff_prime=SampledFluxFunction(
                    profile_coordinate,
                    source_scale * 1.0e-8 * (1.0 - profile_coordinate**2),
                ),
            ),
            boundary_pressure=0.25 * source_scale,
            boundary_field_function=0.5 * source_scale,
        )
        response_scale = 1.0 + index * np.finfo(np.float64).eps
        plasma_grid = response_scale * 1.0e-12 * np.eye(lattice.node_count)
        plasma_wall = (
            response_scale
            * 1.0e-14
            * np.outer(
                np.linspace(1.0, 2.0, len(wall_coordinate)),
                np.linspace(1.0, 2.0, lattice.node_count),
            )
        )
        ordinary_grid = grid_response[:, :1] * response_scale
        ordinary_wall = wall_response[:, :1] * response_scale
        prescribed = PrescribedCurrentField(
            response=np.vstack((grid_response, wall_response)) * response_scale,
            current=np.linspace(-2.0e-3, 2.0e-3, response.shape[1]) + index * 1.0e-6,
        )
        pair = 2 * (index // 2)
        source_radius = arrays[f"arm_{pair:02d}_radius"]
        source_height = arrays[f"arm_{pair:02d}_height"]
        source_flux = arrays[f"arm_{pair:02d}_flux"]
        if source_flux.shape != (full_count, full_count):
            raise AssertionError("each MAST pair needs one persisted flux map")
        grid_flux = source_flux[np.ix_(selection, selection)].T.reshape(-1)
        spline = RectBivariateSpline(
            source_radius, source_height, source_flux.T, kx=3, ky=3, s=0.0
        )
        wall_flux = spline.ev(wall_coordinate[:, 0], wall_coordinate[:, 1])
        seed_offset = (index + 1) * 1.0e-8
        seed = np.r_[grid_flux, wall_flux] + seed_offset
        axis_flux = float(np.min(grid_flux) + seed_offset)
        boundary_flux = float(
            arrays[f"arm_{index:02d}_binding_flux"] + (index + 1) * 1.0e-7
        )
        if boundary_flux == axis_flux:
            boundary_flux = float(np.max(grid_flux) + (index + 1) * 1.0e-7)
        operator = _ReferenceAnchoredOperator(
            grid=FluxTarget(
                source_target=jnp.asarray(ordinary_grid),
                plasma_target=jnp.asarray(plasma_grid),
                null=Null2D.from_coordinates(
                    np.asarray(lattice.coordinate), hex_stencil(lattice.shape)
                ),
            ),
            wall=FluxTarget(
                source_target=jnp.asarray(ordinary_wall),
                plasma_target=jnp.asarray(plasma_wall),
                null=Null1D(np.asarray(wall_coordinate)),
            ),
            source=source,
            external_current=jnp.asarray([1.0e-3 + index * 1.0e-6]),
            area=np.asarray(lattice.cell_area),
            polarity=-1,
            inside_material=support,
            use_linear_moments=False,
            prescribed_current_field=prescribed,
            declared_axis_flux=axis_flux,
            declared_boundary_flux=boundary_flux,
            declared_support=support,
        )
        operators.append(operator)
        member_data.append(
            ForwardSolveMemberData(
                seed_state=jnp.asarray(seed),
                target_current=jnp.asarray(float(index + 1)),
                current=jnp.asarray([1.0e-3 + index * 1.0e-6]),
                prescribed_current=prescribed.current,
            )
        )
    stacked_data = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values), *member_data
    )
    return operators, stacked_data


def _array_bytes(values) -> set[bytes]:
    """Return the exact byte identities of a collection of numerical leaves."""
    return {np.asarray(value).tobytes() for value in values}


def _assert_member_data_varies(operators, member_data) -> None:
    """Prove each contract-defined member input differs across the batch."""
    assert (
        len(_array_bytes([item.source.core.p_prime.values for item in operators]))
        == WIDTH
    )
    assert (
        len(_array_bytes([item.source.core.ff_prime.values for item in operators]))
        == WIDTH
    )
    for blocks in (
        [item.grid.source_target for item in operators],
        [item.grid.plasma_target for item in operators],
        [item.wall.source_target for item in operators],
        [item.wall.plasma_target for item in operators],
        [item.prescribed_current_field.response for item in operators],
        [item.prescribed_current_field.current for item in operators],
        [item.declared_axis_flux for item in operators],
        [item.declared_boundary_flux for item in operators],
    ):
        assert len(_array_bytes(blocks)) == WIDTH
    for leaves in (
        member_data.seed_state,
        member_data.target_current,
        member_data.current,
        member_data.prescribed_current,
    ):
        assert len(_array_bytes(leaves)) == WIDTH


def _one_forward_iteration(operator, data):
    mapped = operator.flux_map(
        current=data.current,
        prescribed_current=data.prescribed_current,
    )(data.seed_state)
    return mapped + jnp.asarray(data.target_current) * jnp.asarray(1.0e-30)


def _operator_with_source(
    template: ForwardFluxOperator, source: ForwardSource
) -> ForwardFluxOperator:
    """Keep one member's host geometry while replacing only its source profile."""
    arguments = {
        "grid": template.grid,
        "wall": template.wall,
        "source": source,
        "external_current": template.external_current,
        "area": template.area,
        "cell_average_stencil": template.cell_average_stencil,
        "cell_average_weight": template.cell_average_weight,
        "polarity": template.polarity,
        "inside_material": template.inside_material,
        "moment_geometry": template.moment_geometry,
        "sample": template.sample,
        "use_linear_moments": template.use_linear_moments,
        "prescribed_current_field": template.prescribed_current_field,
    }
    if isinstance(template, _ReferenceAnchoredOperator):
        arguments.update(
            declared_axis_flux=template.declared_axis_flux,
            declared_boundary_flux=template.declared_boundary_flux,
            declared_support=template.declared_support,
        )
    return type(template)(**arguments)


def _operator_with_declared_support(
    template: ForwardFluxOperator, declared_support
) -> ForwardFluxOperator:
    """Keep one member unchanged except for its declared support mask."""
    arguments = {
        "grid": template.grid,
        "wall": template.wall,
        "source": template.source,
        "external_current": template.external_current,
        "area": template.area,
        "cell_average_stencil": template.cell_average_stencil,
        "cell_average_weight": template.cell_average_weight,
        "polarity": template.polarity,
        "inside_material": template.inside_material,
        "moment_geometry": template.moment_geometry,
        "sample": template.sample,
        "use_linear_moments": template.use_linear_moments,
        "prescribed_current_field": template.prescribed_current_field,
    }
    if isinstance(template, _ReferenceAnchoredOperator):
        arguments.update(
            declared_axis_flux=template.declared_axis_flux,
            declared_boundary_flux=template.declared_boundary_flux,
            declared_support=declared_support,
        )
    return type(template)(**arguments)


def _static_profile(scale: float) -> ForwardSource:
    """Build an address-independent pair of static profile closures."""

    def pressure(normalized_flux):
        return jnp.asarray(scale) * jnp.asarray(normalized_flux)

    def diamagnetic(normalized_flux):
        return jnp.asarray(scale) * (1.0 - jnp.asarray(normalized_flux))

    return ForwardSource(
        core=DomainProfile(p_prime=pressure, ff_prime=diamagnetic),
        boundary_pressure=0.0,
        boundary_field_function=0.0,
    )


def _rebuilt_diiid_source(template: ForwardFluxOperator) -> ForwardSource:
    """Rebuild one strict-exit source from its sampled profile tables."""

    def rebuild(profile: SampledFluxFunction) -> SampledFluxFunction:
        return SampledFluxFunction(profile.coordinate, profile.values)

    source = template.source
    return ForwardSource(
        core=DomainProfile(
            p_prime=rebuild(source.core.p_prime),
            ff_prime=rebuild(source.core.ff_prime),
        ),
        boundary_pressure=source.boundary_pressure,
        boundary_field_function=source.boundary_field_function,
        common_sol=source.common_sol,
        private_flux=source.private_flux,
        normalisation=source.normalisation,
    )


def test_persisted_mast_bank_has_six_profile_identities_for_twelve_arms():
    """Keep the real-bank grouping measurement outside the synthetic fixture."""
    _response, _targets, rows, _arrays = _read_carriers()
    identities = {str(row["identity"]) for row in rows}

    assert len(rows) == WIDTH
    assert len(identities) == WIDTH // 2
    assert all(
        sum(str(row["identity"]) == identity for row in rows) == 2
        for identity in identities
    )


def test_operator_pytree_contains_member_arrays_but_no_geometry_arrays():
    configure_dtypes()
    operators, _member_data = _synthetic_profile_batch()
    operator = operators[0]
    leaves = jax.tree_util.tree_leaves(operator)

    assert all(isinstance(value, jax.Array) for value in leaves)
    assert isinstance(operator.area, np.ndarray)
    assert isinstance(operator.inside_material, np.ndarray)
    assert isinstance(operator.grid.coordinate, np.ndarray)
    assert isinstance(operator.grid.null.stencil, np.ndarray)
    assert isinstance(operator.wall.coordinate, np.ndarray)
    assert not any(value is operator.area for value in leaves)
    assert not any(value is operator.inside_material for value in leaves)
    scalar_values = [float(value) for value in leaves if value.shape == ()]
    assert float(operator.source.boundary_pressure) in scalar_values
    assert float(operator.source.boundary_field_function) in scalar_values


@pytest.mark.slow
def test_equivalent_diiid_bank_operators_stack_into_one_program():
    """The strict-exit DIII-D construction admits its equivalent pair to vmap."""
    from benchmarks.strict_exit_incidence import _build_diiid_members

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    members, _inputs = _build_diiid_members(DIIID_MACHINE_ARTIFACT_CACHE)
    original = members[0].profile.operator
    rebuilt = _operator_with_source(original, _rebuilt_diiid_source(original))
    assert original.geometry_identity == rebuilt.geometry_identity
    assert original.geometry_identity.startswith("1a91f6e0")
    checked = stack_forward_operators((original, rebuilt))

    assert checked.geometry_identical
    assert checked.stacked is not None

    def profile_value(operator):
        return operator.source.core.p_prime(jnp.asarray(0.5))

    program = jax.jit(jax.vmap(profile_value)).lower(checked.stacked).compile()
    result = jax.block_until_ready(program(checked.stacked))
    assert result.shape == (2,)


def test_different_static_profile_callables_take_sequential_fallback():
    configure_dtypes()
    operators, _member_data = _synthetic_profile_batch()
    first = operators[0]
    left = _operator_with_source(first, _static_profile(1.0))
    right = _operator_with_source(first, _static_profile(2.0))
    checked = stack_forward_operators((left, right))

    assert left.geometry_identity == right.geometry_identity
    assert not checked.geometry_identical
    assert checked.stacked is None

    def profile_value(operator, normalized_flux):
        return operator.source.core.p_prime(normalized_flux)

    result = checked.map(profile_value, jnp.asarray((0.25, 0.25)))
    assert isinstance(result, tuple)
    np.testing.assert_array_equal(
        np.asarray(jnp.stack(result)), np.asarray((0.25, 0.5))
    )


def test_declared_support_is_member_data_for_batch_admission():
    configure_dtypes()
    operators, _member_data = _synthetic_profile_batch()
    first = operators[0]
    changed_support = np.asarray(first.declared_support).copy()
    changed_support[0] = ~changed_support[0]
    second = _operator_with_declared_support(first, changed_support)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        np.asarray(first.declared_support),
        np.asarray(second.declared_support),
    )
    assert first.geometry_identity == second.geometry_identity
    checked = stack_forward_operators((first, second))

    assert checked.geometry_identical
    assert checked.geometry_groups == ((0, 1),)
    assert checked.stacked is not None

    def declared_support(operator):
        return operator.declared_support

    program = jax.jit(jax.vmap(declared_support)).lower(checked.stacked).compile()
    result = jax.block_until_ready(program(checked.stacked))
    np.testing.assert_array_equal(
        np.asarray(result),
        np.stack((np.asarray(first.declared_support), changed_support)),
    )


def test_geometry_mismatch_selects_sequential_compiled_fallback():
    configure_dtypes()
    operators, member_data = _synthetic_profile_batch()
    first = operators[0]
    moved_wall = np.asarray(first.wall.coordinate).copy()
    moved_wall[0, 0] += 1.0e-6
    different = _ReferenceAnchoredOperator(
        grid=first.grid,
        wall=FluxTarget(
            first.wall.source_target,
            first.wall.plasma_target,
            Null1D(moved_wall),
        ),
        source=first.source,
        external_current=first.external_current,
        area=first.area,
        polarity=first.polarity,
        inside_material=first.inside_material,
        use_linear_moments=False,
        prescribed_current_field=first.prescribed_current_field,
        declared_axis_flux=first.declared_axis_flux,
        declared_boundary_flux=first.declared_boundary_flux,
        declared_support=first.declared_support,
    )
    checked = stack_forward_operators((first, different))

    assert not checked.geometry_identical
    assert checked.stacked is None
    assert checked.geometry_groups == ((0,), (1,))
    data = jax.tree_util.tree_map(lambda value: value[:2], member_data)
    result = checked.map(_one_forward_iteration, data)
    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.slow
def test_width_twelve_mast_batch_has_no_member_constants_and_matches_width_one():
    configure_dtypes()
    operators, member_data = _synthetic_profile_batch()
    _assert_member_data_varies(operators, member_data)
    checked = stack_forward_operators(operators)

    assert checked.width == WIDTH
    assert checked.geometry_identical
    assert checked.geometry_groups == (tuple(range(WIDTH)),)
    assert checked.stacked is not None

    def batched(operator, data):
        return jax.vmap(_one_forward_iteration)(operator, data)

    lowered = jax.jit(batched).lower(checked.stacked, member_data)
    closed = jax.make_jaxpr(batched)(checked.stacked, member_data)
    dynamic_leaves = jax.tree_util.tree_leaves((checked.stacked, member_data))
    assert all(value.shape[0] == WIDTH for value in dynamic_leaves)
    for constant in closed.consts:
        if not hasattr(constant, "shape"):
            continue
        assert not any(
            constant.shape == dynamic.shape
            and np.array_equal(np.asarray(constant), np.asarray(dynamic))
            for dynamic in dynamic_leaves
        )
    stable_hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert "tensor<12xf64>" in stable_hlo

    batched_result = jax.block_until_ready(
        lowered.compile()(checked.stacked, member_data)
    )
    width_one = jax.jit(_one_forward_iteration)
    single_results = []
    for index, operator in enumerate(operators):
        data = jax.tree_util.tree_map(lambda value: value[index], member_data)
        single_results.append(jax.block_until_ready(width_one(operator, data)))
    expected = jnp.stack(single_results)
    np.testing.assert_allclose(
        np.asarray(batched_result),
        np.asarray(expected),
        rtol=1e-15,
        atol=0.0,
    )
