"""Contracts of the fine-grid display flux map."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.greens import hybrid_greens
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.display_flux import (
    BRANCH_KEYS,
    DEFAULT_SPACING,
    DisplayCoupling,
    build_display_flux_map,
)
from nova.equilibrium.separatrix_branches import assemble_separatrix_branches
from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.linalg.interpolant import Linear
from nova.linalg.tensor_spline import fit_tensor_spline


def _polish_map_saddle(display, seed):
    """Polish ``display``'s own flux surface at ``seed``, independently.

    The build reports where its polished saddle landed, so a check that read the
    distance to that coordinate would agree with any point the build chose.
    This re-falsifies the claim from the published field alone.
    """
    surface = fit_tensor_spline(
        jnp.asarray(display.radius),
        jnp.asarray(display.height),
        jnp.asarray(display.flux.T),
    )
    polished = polish_stationary_points(
        surface,
        jnp.asarray(seed, dtype=jnp.float64)[None, :],
        jnp.ones((1,), dtype=bool),
    )
    return np.asarray(polished["position_rz"], dtype=np.float64).reshape(2), bool(
        np.asarray(polished["converged"]).reshape(())
    )


#: Radial and vertical extent [m] of the manufactured lattice.
RADIUS = (0.6, 1.4)
HEIGHT = (-0.45, 0.45)
#: Conductor sections [m] and the current vector the operator carries.
CONDUCTOR_RADIUS = np.asarray([0.75, 1.25])
CONDUCTOR_HEIGHT = np.asarray([0.0, 0.0])
CONDUCTOR_WIDTH = 0.05
CONDUCTOR_CURRENT = np.asarray([1.0e5, -1.0e5])
#: The current the display build carries.  A one-percent edit to this scale is
#: the declared negative control: the reproduction cannot hold under it.
DISPLAY_CURRENT_SCALE = 1.0


def _lattice(nodes: int = 9) -> FluxLattice:
    """Return the manufactured uniform lattice."""
    return FluxLattice(
        np.linspace(RADIUS[0], RADIUS[1], nodes),
        np.linspace(HEIGHT[0], HEIGHT[1], nodes),
    )


def _source() -> ForwardSource:
    """Return the absolute source the manufactured operator carries."""
    nodes = jnp.linspace(0.0, 1.0, 9)
    return ForwardSource(
        core=DomainProfile(
            p_prime=Linear(nodes, jnp.full(9, -1.0)),
            ff_prime=Linear(nodes, jnp.full(9, 2.0)),
        ),
        boundary_pressure=0.0,
        boundary_field_function=1.0,
    )


def _plasma_columns(target: np.ndarray, source: np.ndarray, dr: float, dz: float):
    """Return the uniform, radial and vertical plasma moment columns."""
    columns = [
        hybrid_greens(
            target[:, 0],
            target[:, 1],
            float(radius),
            float(height),
            float(dr),
            float(dz),
        )
        for radius, height in source
    ]
    uniform, radial, vertical = zip(*columns, strict=True)
    return (
        np.stack(uniform, axis=1),
        np.stack(radial, axis=1),
        np.stack(vertical, axis=1),
    )


def _profile(nodes: int = 9) -> tuple[ForwardProfile, FluxLattice]:
    """Return the manufactured forward operator and its lattice."""
    lattice = _lattice(nodes)
    coordinate = lattice.coordinate
    source_columns = [
        hybrid_greens(
            coordinate[:, 0],
            coordinate[:, 1],
            float(radius),
            float(height),
            CONDUCTOR_WIDTH,
            CONDUCTOR_WIDTH,
        )[0]
        for radius, height in zip(CONDUCTOR_RADIUS, CONDUCTOR_HEIGHT, strict=True)
    ]
    plasma_columns = _plasma_columns(
        coordinate, coordinate, lattice.radial_step, lattice.vertical_step
    )
    angle = 2.0 * np.pi * np.arange(12) / 12
    wall = np.c_[1.0 + 0.15 * np.cos(angle), 0.15 * np.sin(angle)]
    node = lattice.node_count
    profile = ForwardProfile.from_lattice(
        lattice,
        _source(),
        external_current=jnp.asarray(CONDUCTOR_CURRENT.copy()),
        source_to_grid=np.stack(source_columns, axis=1),
        plasma_to_grid=plasma_columns[0],
        plasma_to_grid_r=plasma_columns[1],
        plasma_to_grid_z=plasma_columns[2],
        source_to_wall=np.zeros((wall.shape[0], len(CONDUCTOR_CURRENT))),
        plasma_to_wall=np.zeros((wall.shape[0], node)),
        wall_coordinate=wall,
        cubic_cell_average=False,
    )
    return profile, lattice


def _coupling(profile: ForwardProfile) -> DisplayCoupling:
    """Return the target-point coupling the manufactured operator is built from."""
    lattice = profile.lattice
    cells = lattice.coordinate
    dr = lattice.radial_step
    dz = lattice.vertical_step
    drive = CONDUCTOR_CURRENT.copy()
    drive[0] *= DISPLAY_CURRENT_SCALE

    def plasma(points: np.ndarray):
        """Return the three plasma moment columns at ``points``."""
        return _plasma_columns(points, cells, dr, dz)

    def external(points: np.ndarray) -> np.ndarray:
        """Return the conductor-only flux at ``points`` [Wb]."""
        flux = np.zeros(points.shape[0], dtype=np.float64)
        for radius, height, current in zip(
            CONDUCTOR_RADIUS, CONDUCTOR_HEIGHT, drive, strict=True
        ):
            flux += (
                current
                * hybrid_greens(
                    points[:, 0],
                    points[:, 1],
                    float(radius),
                    float(height),
                    CONDUCTOR_WIDTH,
                    CONDUCTOR_WIDTH,
                )[0]
            )
        return flux

    return DisplayCoupling(plasma=plasma, external=external)


def _moments(lattice: FluxLattice) -> CellCurrentMoments:
    """Return a manufactured cell-current moment triple on ``lattice``.

    The radial and vertical moments are non-zero and differ from each other, so
    a build that drops one column, or contracts them in the wrong order, cannot
    reproduce the operator's own raster.
    """
    radius = lattice.coordinate[:, 0]
    height = lattice.coordinate[:, 1]
    core = np.hypot(radius - 1.0, height) < 0.25
    cell_current = np.where(core, 1.0e5, 0.0)
    radial = np.where(core, 2.5e4 * (radius - 1.0), 0.0)
    vertical = np.where(core, -4.0e4 * height, 0.0)
    return CellCurrentMoments(
        jnp.asarray(cell_current),
        jnp.asarray(radial),
        jnp.asarray(vertical),
    )


def _state(profile: ForwardProfile, moments: CellCurrentMoments) -> np.ndarray:
    """Return the state whose currents are ``moments``."""
    raster = np.asarray(profile.operator.raster_image(moments), dtype=np.float64)
    assert raster.size == profile.lattice.node_count
    return raster


def test_fine_map_reproduces_the_operator_raster_image():
    """Non-zero moment columns contract to the operator's own raster image."""
    profile, lattice = _profile()
    moments = _moments(lattice)
    display = build_display_flux_map(
        profile,
        _state(profile, moments),
        coupling=_coupling(profile),
        moments=moments,
        chunk=128,
    )
    assert display.reproduction_relative_error < 1.0e-10
    assert display.max_node_mismatch_wb < 1.0e-6


def test_the_moment_columns_carry_the_map():
    """The linear moments move the flux, so a dropped column cannot hide."""
    profile, lattice = _profile()
    moments = _moments(lattice)
    full = build_display_flux_map(
        profile,
        _state(profile, moments),
        coupling=_coupling(profile),
        moments=moments,
        chunk=128,
    )
    uniform_only = CellCurrentMoments(
        moments.cell_current,
        jnp.zeros_like(moments.radial_moment),
        jnp.zeros_like(moments.vertical_moment),
    )
    bare = build_display_flux_map(
        profile,
        _state(profile, moments),
        coupling=_coupling(profile),
        moments=uniform_only,
        chunk=128,
    )
    scale = max(float(np.max(np.abs(full.flux))), 1.0e-30)
    assert float(np.max(np.abs(full.flux - bare.flux))) / scale > 1.0e-2


def _moment_law(state: np.ndarray, node: int) -> CellCurrentMoments:
    """Return the manufactured source law the operator is iterated under.

    The law is linear in the state, so the write-then-read map contracts and
    has one fixed point; it is a source, not the identity, so the fixed point
    is not a state the operator produces in a single evaluation.
    """
    cell = 1.0e5 + 2.0e3 * state[:node]
    radial = 2.5e4 * (state[:node] - np.mean(state[:node]))
    vertical = -4.0e4 * (state[:node] - np.mean(state[:node]))
    return CellCurrentMoments(
        jnp.asarray(cell), jnp.asarray(radial), jnp.asarray(vertical)
    )


def _solve(profile, node: int, steps: int = 400, tolerance: float = 1.0e-12):
    """Return the converged state of the manufactured map and its residual."""
    state = np.zeros(node)
    residual = np.inf
    for _step in range(steps):
        next_state = np.asarray(
            profile.operator.raster_image(_moment_law(state, node)),
            dtype=np.float64,
        )
        residual = float(np.max(np.abs(next_state - state)))
        state = next_state
        if residual < tolerance:
            break
    image = np.asarray(
        profile.operator.raster_image(_moment_law(state, node)), dtype=np.float64
    )
    return np.asarray(state, dtype=np.float64), float(np.max(np.abs(image - state)))


#: Amplitude [Wb] of the compensating conductor flux the constrained route
#: carries.  It is far above the fixed-point residual, so a build that drops it
#: cannot hold the solved-node mismatch.
COMPENSATION_WB = 0.02


def _compensation(points: np.ndarray) -> np.ndarray:
    """Return the compensating conductor flux [Wb] at ``points``.

    A constrained route folds a compensating conductor current into its
    prescribed vector before imaging, and that current carries flux of its own.
    The manufactured term varies with radius, so a build that drops it, or that
    adds it in the wrong place, moves the map at the lattice nodes.
    """
    return COMPENSATION_WB * (points[:, 0] - 1.0)


def _solve_with_compensation(profile, node: int, steps: int = 400):
    """Return the fixed point of the operator's map including compensation."""
    coordinate = np.asarray(profile.lattice.coordinate)
    state = np.zeros(node)
    residual = np.inf
    for _step in range(steps):
        next_state = np.asarray(
            profile.operator.raster_image(_moment_law(state, node)),
            dtype=np.float64,
        ) + _compensation(coordinate)
        residual = float(np.max(np.max(np.abs(next_state - state))))
        state = next_state
        if residual < 1.0e-12:
            break
    image = np.asarray(
        profile.operator.raster_image(_moment_law(state, node)), dtype=np.float64
    ) + _compensation(coordinate)
    return state, float(np.max(np.abs(image - state)))


def test_promoted_nodes_are_excluded_from_the_solved_mismatch():
    """The mask the solve returns is what the receipt's solved figure excludes.

    The state is the fixed point of the map that carries the compensating
    conductor flux, so every node the solve claims matches it.  A trial state
    is pasted onto three promoted nodes, exactly as the closing trip keeps the
    trial state there, which makes the unmasked mismatch large while the figure
    over the claimed nodes stays at the fixed-point residual.
    """
    profile, lattice = _profile()
    node = lattice.node_count
    state, residual = _solve_with_compensation(profile, node)
    assert residual < 1.0e-9
    promoted = np.zeros(node, dtype=bool)
    promoted[[10, 11, 20]] = True
    trial = state.copy()
    trial[promoted] += 0.05
    xpoint = np.asarray([1.0, 0.0])
    display = build_display_flux_map(
        profile,
        trial,
        coupling=_coupling(profile),
        moments=_moment_law(state, node),
        promoted=promoted,
        compensation=_compensation,
        xpoint_rz=xpoint,
        chunk=128,
    )
    assert display.promoted_node_count == 3
    assert display.max_solved_node_mismatch_wb <= 10.0 * residual
    assert display.max_node_mismatch_wb >= 0.049
    assert display.max_solved_node_mismatch_wb == pytest.approx(
        residual, rel=1.0e-6, abs=1.0e-15
    )
    assert np.isnan(display.nearest_promoted_to_xpoint_m) == (
        not display.saddle_converged
    )
    if display.saddle_converged:
        polished, converged = _polish_map_saddle(display, xpoint)
        assert converged
        offset = np.asarray(lattice.coordinate)[promoted] - polished
        assert display.nearest_promoted_to_xpoint_m == pytest.approx(
            float(np.min(np.linalg.norm(offset, axis=1))), rel=1.0e-12
        )
    assert display.reproduction_relative_error < 1.0e-10


def test_the_promoted_mask_tail_must_be_the_non_lattice_rows():
    """A state-wide mask is cut only when its tail is the operator's wall rows.

    The solve keeps its wall rows after the lattice nodes, so a mask over the
    whole state is longer than the lattice by that wall.  A mask whose tail is
    any other length is not the state's layout, and truncating it would drop a
    node the map cannot account for from the promoted count.
    """
    profile, lattice = _profile()
    node = lattice.node_count
    state, _residual = _solve_with_compensation(profile, node)
    wall_rows = int(np.asarray(profile.operator.wall.null.coordinate).shape[0])
    assert wall_rows > 0
    accepted = np.zeros(node + wall_rows, dtype=bool)
    accepted[[10, 11, 20, node + 1]] = True
    display = build_display_flux_map(
        profile,
        np.concatenate([state, np.zeros(wall_rows)]),
        coupling=_coupling(profile),
        moments=_moment_law(state, node),
        promoted=accepted,
        xpoint_rz=np.asarray([1.0, 0.0]),
        chunk=128,
    )
    assert display.promoted_node_count == 3
    unexplained = np.zeros(node + wall_rows + 3, dtype=bool)
    with pytest.raises(ValueError, match="discards .* state rows past the"):
        build_display_flux_map(
            profile,
            np.concatenate([state, np.zeros(wall_rows + 3)]),
            coupling=_coupling(profile),
            moments=_moment_law(state, node),
            promoted=unexplained,
            xpoint_rz=np.asarray([1.0, 0.0]),
            chunk=128,
        )


def test_dropping_the_compensation_moves_the_solved_mismatch():
    """The compensating conductor flux is part of the field the solve imaged.

    The same state and the same mask are built with the compensation withheld.
    The figure over the claimed nodes then leaves the residual by the
    compensation's own amplitude, which is the failure the receipt exists to
    expose.
    """
    profile, lattice = _profile()
    node = lattice.node_count
    state, residual = _solve_with_compensation(profile, node)
    promoted = np.zeros(node, dtype=bool)
    promoted[[10, 11, 20]] = True
    trial = state.copy()
    trial[promoted] += 0.05
    withheld = build_display_flux_map(
        profile,
        trial,
        coupling=_coupling(profile),
        moments=_moment_law(state, node),
        promoted=promoted,
        chunk=128,
    )
    assert withheld.max_solved_node_mismatch_wb > 10.0 * residual


def test_a_converged_solve_reports_a_mismatch_within_ten_residuals():
    """A state converged under the operator's own map misses it by its residual."""
    profile, lattice = _profile()
    state, residual = _solve(profile, lattice.node_count)
    assert residual < 1.0e-9
    display = build_display_flux_map(
        profile,
        state,
        coupling=_coupling(profile),
        moments=_moment_law(state, lattice.node_count),
        chunk=128,
    )
    assert display.max_node_mismatch_wb <= 10.0 * residual
    assert display.max_node_mismatch_wb == pytest.approx(
        residual, rel=1.0e-6, abs=1.0e-15
    )
    assert display.reproduction_relative_error < 1.0e-10


#: Half extent [m] of the manufactured diverted field and the factor its
#: coordinates are scaled by, so the field is the repo's diverted null
#: configuration drawn inside a 330 mm box at the 10 mm display floor.
DIVERTED_EXTENT = 0.165
DIVERTED_SCALE = DIVERTED_EXTENT / 2.0
#: Radial centre [m] of the diverted box: the lattice radius stays positive.
DIVERTED_CENTRE = 0.330


def _diverted_coefficients() -> np.ndarray:
    """Return the monomial coefficients of the scaled diverted field."""
    scale = DIVERTED_SCALE
    cubic = 1.0 / (2.0 * np.sqrt(2.0) * scale**3)
    return np.asarray([-2.0 / scale**2, cubic, 3.0 * cubic, 3.0 * cubic, cubic])


def _diverted_basis(points: np.ndarray) -> np.ndarray:
    """Return the monomial basis columns of the diverted field."""
    radius = points[:, 0] - DIVERTED_CENTRE
    height = points[:, 1]
    return np.stack(
        (
            radius * height,
            radius**3,
            radius**2 * height,
            radius * height**2,
            height**3,
        ),
        axis=1,
    )


def _diverted_field(points: np.ndarray) -> np.ndarray:
    """Return the diverted flux [Wb] at ``points``."""
    return _diverted_basis(points) @ _diverted_coefficients()


def _diverted_profile() -> tuple[ForwardProfile, FluxLattice]:
    """Return an operator whose conductor coupling is the diverted field
    itself, so the display map carries a genuine saddle and separatrix."""
    nodes = 9
    lattice = FluxLattice(
        np.linspace(
            DIVERTED_CENTRE - DIVERTED_EXTENT, DIVERTED_EXTENT + DIVERTED_CENTRE, nodes
        ),
        np.linspace(-DIVERTED_EXTENT, DIVERTED_EXTENT, nodes),
    )
    coordinate = lattice.coordinate
    angle = 2.0 * np.pi * np.arange(12) / 12
    wall = np.c_[
        DIVERTED_CENTRE + DIVERTED_EXTENT * np.cos(angle),
        DIVERTED_EXTENT * np.sin(angle),
    ]
    node = lattice.node_count
    profile = ForwardProfile.from_lattice(
        lattice,
        _source(),
        external_current=jnp.asarray(_diverted_coefficients()),
        source_to_grid=_diverted_basis(coordinate),
        plasma_to_grid=np.zeros((node, node)),
        source_to_wall=np.zeros((wall.shape[0], _diverted_coefficients().size)),
        plasma_to_wall=np.zeros((wall.shape[0], node)),
        wall_coordinate=wall,
        cubic_cell_average=False,
    )
    return profile, lattice


def _diverted_coupling(node: int) -> DisplayCoupling:
    """Return the target-point coupling of the diverted field."""

    def plasma(points: np.ndarray):
        """Return zero plasma moment columns at ``points``."""
        return (
            np.zeros((points.shape[0], node)),
            np.zeros((points.shape[0], node)),
            np.zeros((points.shape[0], node)),
        )

    def external(points: np.ndarray) -> np.ndarray:
        """Return the diverted flux at ``points`` [Wb]."""
        return _diverted_field(points)

    return DisplayCoupling(plasma=plasma, external=external)


def test_the_map_assembles_its_own_separatrix_branches():
    """The branches come from the fine map at the map's own saddle level."""
    profile, lattice = _diverted_profile()
    zero = CellCurrentMoments(
        jnp.zeros(lattice.node_count),
        jnp.zeros(lattice.node_count),
        jnp.zeros(lattice.node_count),
    )
    axis = np.asarray([DIVERTED_CENTRE + 0.47 * DIVERTED_SCALE, 0.47 * DIVERTED_SCALE])
    display = build_display_flux_map(
        profile,
        _diverted_field(np.asarray(lattice.coordinate)),
        coupling=_diverted_coupling(lattice.node_count),
        moments=zero,
        extent=(
            DIVERTED_CENTRE - DIVERTED_EXTENT,
            DIVERTED_CENTRE + DIVERTED_EXTENT,
            -DIVERTED_EXTENT,
            DIVERTED_EXTENT,
        ),
        xpoint_rz=np.asarray([DIVERTED_CENTRE, 0.0]),
        axis_rz=axis,
        chunk=512,
    )
    assert display.saddle_converged
    assert set(display.branches) == set(BRANCH_KEYS)
    assert display.well_formed
    assert int(np.asarray(display.branches["closed_segment_count"])) > 0
    assert int(np.asarray(display.branches["open_branch_count"])) == 2
    direct = assemble_separatrix_branches(
        jnp.asarray(display.flux.T),
        jnp.asarray(display.radius),
        jnp.asarray(display.height),
        jnp.asarray(display.saddle_flux),
        jnp.asarray(axis),
    )
    for key in BRANCH_KEYS:
        np.testing.assert_array_equal(
            np.asarray(display.branches[key]), np.asarray(direct[key])
        )


def test_display_map_spacing_and_saddle_are_reported():
    """The map is drawn at its spacing floor and carries its own saddles."""
    profile, lattice = _diverted_profile()
    zero = CellCurrentMoments(
        jnp.zeros(lattice.node_count),
        jnp.zeros(lattice.node_count),
        jnp.zeros(lattice.node_count),
    )
    display = build_display_flux_map(
        profile,
        _diverted_field(np.asarray(lattice.coordinate)),
        coupling=_diverted_coupling(lattice.node_count),
        moments=zero,
        extent=(
            DIVERTED_CENTRE - DIVERTED_EXTENT,
            DIVERTED_CENTRE + DIVERTED_EXTENT,
            -DIVERTED_EXTENT,
            DIVERTED_EXTENT,
        ),
        xpoint_rz=np.asarray([DIVERTED_CENTRE, 0.0]),
        chunk=512,
    )
    assert display.saddle_converged
    assert np.linalg.norm(display.saddle_rz - np.asarray([DIVERTED_CENTRE, 0.0])) < 0.02
    assert display.spacing_m <= DEFAULT_SPACING
    np.testing.assert_allclose(np.diff(display.radius), display.spacing_m)
    assert display.radius.size > lattice.radius.size
    assert np.all(np.isfinite(display.saddle_rz))
    assert np.all(np.isfinite(display.lattice_saddle_rz))
    assert np.isfinite(display.saddle_offset_mm)
    assert np.all(np.isfinite(display.flux))
    assert display.max_node_mismatch_wb < 1.0e-6


def test_a_seed_on_an_extremum_reports_no_saddle():
    """A seed that lands on an O-point is refused rather than read as a saddle."""
    profile, lattice = _diverted_profile()
    zero = CellCurrentMoments(
        jnp.zeros(lattice.node_count),
        jnp.zeros(lattice.node_count),
        jnp.zeros(lattice.node_count),
    )
    o_point = np.asarray(
        [DIVERTED_CENTRE + 0.47 * DIVERTED_SCALE, 0.47 * DIVERTED_SCALE]
    )
    display = build_display_flux_map(
        profile,
        _diverted_field(np.asarray(lattice.coordinate)),
        coupling=_diverted_coupling(lattice.node_count),
        moments=zero,
        extent=(
            DIVERTED_CENTRE - DIVERTED_EXTENT,
            DIVERTED_CENTRE + DIVERTED_EXTENT,
            -DIVERTED_EXTENT,
            DIVERTED_EXTENT,
        ),
        xpoint_rz=o_point,
        axis_rz=o_point,
        chunk=512,
    )
    assert not display.saddle_converged
    assert np.all(np.isnan(display.saddle_rz))
    assert np.isnan(display.saddle_flux)
    assert display.branches == {}
    assert not display.well_formed


def test_a_coarser_than_declared_spacing_is_refused():
    """A map coarser than the declared floor is refused rather than drawn."""
    profile, lattice = _profile()
    moments = _moments(lattice)
    with pytest.raises(ValueError, match="coarser"):
        build_display_flux_map(
            profile,
            _state(profile, moments),
            coupling=_coupling(profile),
            moments=moments,
            spacing=2.0 * DEFAULT_SPACING,
        )
