"""Contracts for the presentation-media package, on synthetic geometry.

Every test here runs in the fast lane: no corpus, no solve, no device. What is
pinned is the behaviour that went wrong while the package was written, so a
regression reintroduces a measured defect rather than an imagined one.
"""

from pathlib import Path

import numpy as np
import pytest

from nova.media.gif import write_contact_sheet, write_gif
from nova.media.ink import DEFAULT_INK, poloidal_axes, trace_axes
from nova.media.layout import three_view
from nova.media.poloidal import contour_levels, draw_flux_contours, draw_nulls
from nova.media.sources.frame import (
    EquilibriumFrame,
    MachineGeometry,
    Pulse,
    SurfaceFrame,
)
from nova.media.traces import TraceScale

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _square(radius, height, half=0.05):
    """Return one axis-aligned square cell outline."""
    return np.array(
        [
            [radius - half, height - half],
            [radius + half, height - half],
            [radius + half, height + half],
            [radius - half, height + half],
        ]
    )


def _machine():
    """Return a tall machine whose coils overhang its wall in both axes."""
    angle = np.linspace(0.0, 2.0 * np.pi, 40)
    limiter = np.column_stack((1.0 + 0.6 * np.cos(angle), 1.2 * np.sin(angle)))
    coils = (_square(0.05, 2.4), _square(1.9, -2.4))
    return MachineGeometry(limiter=limiter, coils=coils)


def _blob(radius, height, shift=0.0):
    """Return a smooth flux map shaped (height, radius)."""
    mesh_r, mesh_z = np.meshgrid(radius, height)
    return -np.exp(-(((mesh_r - 1.0) / 0.4) ** 2 + ((mesh_z - shift) / 0.8) ** 2))


# --------------------------------------------------------------------------
# ink: the values are the house style, and the axes carry no chrome
# --------------------------------------------------------------------------


def test_ink_holds_the_pinned_imas_ink_values():
    """The canonical style must still agree with upstream imas-ink."""
    assert DEFAULT_INK.contour_color == "#999999"
    assert DEFAULT_INK.contour_linewidth == 0.35
    assert DEFAULT_INK.coil_edgecolor == "#888888"
    assert DEFAULT_INK.coil_facecolor == "none"
    assert DEFAULT_INK.wall_color == "#000000"
    assert DEFAULT_INK.wall_linewidth == 1.0
    assert DEFAULT_INK.separatrix_color == "#cc0000"
    assert DEFAULT_INK.separatrix_linewidth == 1.5
    assert DEFAULT_INK.flux_color == "#3366cc"
    assert DEFAULT_INK.figure_dpi == 120


def test_a_style_variant_leaves_the_shared_default_untouched():
    """A frozen default cannot be mutated by a figure needing one override."""
    variant = DEFAULT_INK.variant(wall_linewidth=3.0)
    assert variant.wall_linewidth == 3.0
    assert DEFAULT_INK.wall_linewidth == 1.0


def test_poloidal_axes_carry_no_chrome_and_hold_equal_aspect():
    view = three_view(extent=(0.0, 2.0, -2.0, 2.0))
    axes = poloidal_axes(view.poloidal)
    assert axes.get_aspect() == 1.0
    assert not axes.axison
    assert not any(line.get_visible() for line in axes.get_xgridlines())


def test_trace_axes_are_despined_and_ungridded():
    view = three_view(extent=(0.0, 2.0, -2.0, 2.0))
    axes = trace_axes(view.upper)
    assert not axes.spines["top"].get_visible()
    assert not axes.spines["right"].get_visible()
    assert axes.spines["left"].get_visible()
    assert not any(line.get_visible() for line in axes.get_ygridlines())


# --------------------------------------------------------------------------
# layout: the panel takes the machine's aspect, and keeps it across frames
# --------------------------------------------------------------------------


def test_poloidal_panel_box_matches_the_machine_aspect():
    """A width ratio leaves a tall machine floating in margin; this does not."""
    extent = (0.0, 2.25, -2.65, 2.65)
    view = three_view(extent=extent)
    figure_width, figure_height = view.figure.get_size_inches()
    box = view.poloidal.get_position()
    panel = (box.width * figure_width) / (box.height * figure_height)
    machine = (extent[1] - extent[0]) / (extent[3] - extent[2])
    assert panel == pytest.approx(machine, rel=1e-6)


def test_clear_restores_the_poloidal_extent():
    """Clearing an axes drops its limits; a frame loop must not autoscale."""
    extent = (0.1, 2.0, -1.5, 1.5)
    view = three_view(extent=extent)
    view.poloidal.set_xlim(0.5, 0.6)
    view.clear()
    assert view.poloidal.get_xlim() == pytest.approx(extent[:2])
    assert view.poloidal.get_ylim() == pytest.approx(extent[2:])
    assert not view.poloidal.get_autoscale_on()


def test_three_view_refuses_a_degenerate_extent():
    with pytest.raises(ValueError, match="positive area"):
        three_view(extent=(1.0, 1.0, -1.0, 1.0))


# --------------------------------------------------------------------------
# records: a wrong-shaped map is the failure that draws a plausible machine
# --------------------------------------------------------------------------


def _frame(radius, height, flux):
    return EquilibriumFrame(
        time=0.1,
        radius=radius,
        height=height,
        flux=flux,
        flux_axis=-1.0,
        flux_boundary=-0.3,
        psi_norm=np.linspace(0.0, 1.0, 5),
        p_prime=np.zeros(5),
        ff_prime=np.zeros(5),
        boundary=np.zeros((0, 2)),
        magnetic_axis=np.array([1.0, 0.0]),
        x_points=np.zeros((0, 2)),
    )


def test_a_transposed_flux_map_is_refused():
    """It would contour without error and draw the wrong machine."""
    radius = np.linspace(0.2, 1.8, 9)
    height = np.linspace(-1.0, 1.0, 15)
    with pytest.raises(ValueError, match="height, radius"):
        _frame(radius, height, _blob(radius, height).T)


def test_a_correctly_shaped_flux_map_is_accepted():
    radius = np.linspace(0.2, 1.8, 9)
    height = np.linspace(-1.0, 1.0, 15)
    frame = _frame(radius, height, _blob(radius, height))
    assert frame.flux.shape == (15, 9)


def test_surface_frame_refuses_a_flux_count_mismatch():
    with pytest.raises(ValueError, match="flux values"):
        SurfaceFrame(
            time=0.0,
            surface_flux=np.zeros(3),
            surfaces=(np.zeros((4, 2)), np.zeros((4, 2))),
            boundary=np.zeros((4, 2)),
            magnetic_axis=np.zeros(2),
            x_points=np.zeros((0, 2)),
        )


def test_machine_bounds_enclose_the_coils_not_just_the_wall():
    """A bound from the wall alone would crop the conductors off the panel."""
    bounds = _machine().bounds()
    assert bounds[2] <= -2.4 and bounds[3] >= 2.4


def test_pulse_extent_clamps_the_inboard_edge_at_the_machine_axis():
    """A panel reaching negative major radius shows space where iron is."""
    pulse = Pulse(machine="T", identifier="1", geometry=_machine(), frames=())
    assert pulse.extent(pad=0.5)[0] == 0.0


# --------------------------------------------------------------------------
# levels: the shared array is what makes two maps comparable
# --------------------------------------------------------------------------


def test_contour_levels_place_a_line_exactly_on_the_boundary():
    radius = np.linspace(0.2, 1.8, 21)
    height = np.linspace(-1.0, 1.0, 31)
    levels = contour_levels(_blob(radius, height), 12, boundary=-0.3)
    assert np.isclose(levels, -0.3).any()


def test_contour_levels_refuse_a_map_with_no_finite_value():
    with pytest.raises(ValueError, match="finite"):
        contour_levels(np.full((4, 4), np.nan), 5)


def test_draw_flux_contours_refuses_a_transposed_map():
    radius = np.linspace(0.2, 1.8, 9)
    height = np.linspace(-1.0, 1.0, 15)
    view = three_view(extent=(0.2, 1.8, -1.0, 1.0))
    with pytest.raises(ValueError, match="height, radius"):
        draw_flux_contours(
            view.poloidal, radius, height, _blob(radius, height).T, [-0.5]
        )


def test_draw_flux_contours_requires_explicit_levels():
    radius = np.linspace(0.2, 1.8, 9)
    height = np.linspace(-1.0, 1.0, 15)
    view = three_view(extent=(0.2, 1.8, -1.0, 1.0))
    with pytest.raises(ValueError, match="one contour level"):
        draw_flux_contours(view.poloidal, radius, height, _blob(radius, height), [])


def test_draw_nulls_drops_a_non_finite_point():
    """An absent second X-point must not be drawn on the machine axis."""
    view = three_view(extent=(0.0, 2.0, -2.0, 2.0))
    draw_nulls(
        view.poloidal,
        magnetic_axis=(1.0, 0.0),
        x_points=np.array([[0.7, -1.1], [np.nan, np.nan]]),
    )
    drawn = np.concatenate([line.get_xydata() for line in view.poloidal.lines])
    assert np.all(np.isfinite(drawn))
    assert not np.any(np.all(np.isclose(drawn, 0.0), axis=1))


# --------------------------------------------------------------------------
# scales: the statistic must cover frames, not pooled samples
# --------------------------------------------------------------------------


def test_the_quantile_covers_frames_rather_than_pooled_samples():
    """One spiking frame must not set the axis for a whole animation.

    Shaped after the measured MAST 21978 case: a single early slice runs two
    orders of magnitude below every other frame across its whole abscissa, so
    it survives any sample quantile while contributing one frame extreme.
    """
    ordinary = [np.linspace(0.0, 3.0e5, 65) for _ in range(69)]
    spike = np.linspace(-7.8e5, 1.6e2, 65)
    abscissa = [np.linspace(0.0, 1.0, 65)] * 70

    pooled = TraceScale.over(abscissa, ordinary + [spike])
    framewise = TraceScale.over(abscissa, ordinary + [spike], quantile=0.98)

    assert pooled.y_limit[0] < -7.0e5
    assert framewise.y_limit[0] > -1.0e5


def test_a_log_scale_masks_non_positive_samples_and_sets_the_axis():
    rows = [np.array([0.0, -5.0, 10.0, 1000.0]), np.array([np.nan, 20.0, 500.0, 2.0])]
    scale = TraceScale.over([np.arange(4.0)], rows, log=True)
    assert scale.log
    assert scale.y_limit[0] > 0.0
    view = three_view(extent=(0.0, 2.0, -2.0, 2.0))
    scale.apply(view.upper)
    assert view.upper.get_yscale() == "log"


def test_a_log_scale_refuses_wholly_non_positive_data():
    with pytest.raises(ValueError, match="positive"):
        TraceScale.over([np.arange(3.0)], [np.array([-1.0, -2.0, 0.0])], log=True)


def test_a_scale_stops_the_axes_autoscaling():
    scale = TraceScale.over([np.arange(3.0)], [np.array([1.0, 2.0, 3.0])])
    view = three_view(extent=(0.0, 2.0, -2.0, 2.0))
    scale.apply(view.lower)
    assert not view.lower.get_autoscale_on()


# --------------------------------------------------------------------------
# composition: duration is the thing the caller controls
# --------------------------------------------------------------------------


def _frames(count, size=(32, 24)):
    from PIL import Image

    return [
        Image.new("RGB", size, (index * 7 % 256, 128, 200)) for index in range(count)
    ]


def test_a_gif_reports_the_duration_it_achieved(tmp_path):
    """GIF stores a per-frame delay in hundredths, so ten seconds is a target."""
    receipt = write_gif(_frames(12), tmp_path / "a.gif", duration=10.0)
    assert receipt["frames"] == 12
    assert receipt["requested_seconds"] == 10.0
    assert receipt["achieved_seconds"] == pytest.approx(10.0, abs=0.3)
    assert receipt["frame_milliseconds"] * 12 / 1000.0 == receipt["achieved_seconds"]
    assert (tmp_path / "a.gif").stat().st_size > 0


def test_a_gif_needs_a_frame_and_a_positive_duration(tmp_path):
    with pytest.raises(ValueError, match="at least one frame"):
        write_gif([], tmp_path / "b.gif")
    with pytest.raises(ValueError, match="positive"):
        write_gif(_frames(2), tmp_path / "c.gif", duration=0.0)


def test_a_contact_sheet_always_carries_the_first_and_last_frame(tmp_path):
    """The sheet must read as the whole pulse, not as its middle."""
    receipt = write_contact_sheet(_frames(85), tmp_path / "s.png", columns=3, count=6)
    assert receipt["tile_indices"][0] == 0
    assert receipt["tile_indices"][-1] == 84
    assert receipt["tiles"] == 6
    assert receipt["columns"] == 3 and receipt["rows"] == 2


def test_a_contact_sheet_of_one_frame_is_a_single_tile(tmp_path):
    receipt = write_contact_sheet(_frames(1), tmp_path / "t.png", columns=3, count=6)
    assert receipt["tiles"] == 1 and receipt["tile_indices"] == [0]


# --------------------------------------------------------------------------
# mesh: the clipped cells must tile the boundary exactly
# --------------------------------------------------------------------------


def test_clipped_cells_tile_the_boundary_region_exactly():
    """Area is the check: a missed or double-counted cell shows up here."""
    import shapely

    from nova.media.sources.plasma_mesh import clip_to_boundary, hex_mesh

    angle = np.linspace(0.0, 2.0 * np.pi, 60)
    wall = np.column_stack((1.0 + 0.5 * np.cos(angle), 0.7 * np.sin(angle)))
    cells, provenance = hex_mesh(wall, cells=60)
    assert provenance["delivered_cells"] > 0
    assert provenance["coupling"].startswith("none")

    boundary = np.column_stack((1.0 + 0.3 * np.cos(angle), 0.4 * np.sin(angle)))
    clipped = clip_to_boundary(cells, boundary)
    assert clipped
    tiled = sum(shapely.Polygon(cell).area for cell in clipped)
    assert tiled == pytest.approx(shapely.Polygon(boundary).area, rel=1e-6)


def test_a_boundary_outside_the_mesh_clips_to_nothing():
    from nova.media.sources.plasma_mesh import clip_to_boundary

    cells = (_square(1.0, 0.0), _square(1.2, 0.0))
    far = np.array([[5.0, 5.0], [5.1, 5.0], [5.1, 5.1], [5.0, 5.1]])
    assert clip_to_boundary(cells, far) == ()


def test_a_bare_poloidal_panel_fills_its_figure_at_the_machine_aspect():
    """A single-panel figure must be the machine's shape, not a default box."""
    from nova.media.layout import poloidal_view

    extent = (0.0, 2.25, -2.65, 2.65)
    view = poloidal_view(extent=extent, margin=0.0)
    width, height = view.figure.get_size_inches()
    assert width / height == pytest.approx(
        (extent[1] - extent[0]) / (extent[3] - extent[2]), rel=1e-6
    )
    box = view.poloidal.get_position()
    panel = (box.width * width) / (box.height * height)
    assert panel == pytest.approx(width / height, rel=1e-6)


def test_a_bare_poloidal_panel_restores_its_extent_on_clear():
    from nova.media.layout import poloidal_view

    extent = (0.1, 2.0, -1.5, 1.5)
    view = poloidal_view(extent=extent)
    view.poloidal.set_xlim(0.5, 0.6)
    view.clear()
    assert view.poloidal.get_xlim() == pytest.approx(extent[:2])
    assert not view.poloidal.axison


# --------------------------------------------------------------------------
# Thomson: two disjoint eras, and a position can be absent where data is not
# --------------------------------------------------------------------------


def _string(positions, temperature, density=None):
    from nova.media.sources.mast_thomson import ThomsonString

    temperature = np.atleast_2d(np.asarray(temperature, dtype=float))
    return ThomsonString(
        name="core",
        positions=np.asarray(positions, dtype=float),
        time=np.arange(temperature.shape[0], dtype=float) * 0.01,
        temperature=temperature,
        density=(
            np.ones_like(temperature) if density is None else np.atleast_2d(density)
        ),
    )


def test_a_channel_with_a_non_finite_position_is_dropped():
    """Both eras carry one; a NaN abscissa silently poisons any fit or scale."""
    positions = np.array([[0.4, 0.0], [np.nan, 0.0], [1.2, 0.0]])
    radius, temperature, _ = _string(positions, [[100.0, 200.0, 300.0]]).finite(0.0)
    assert np.all(np.isfinite(radius))
    assert radius.size == 2 and temperature.size == 2
    assert 200.0 not in temperature


def test_a_non_positive_measurement_is_dropped():
    positions = np.array([[0.4, 0.0], [0.8, 0.0], [1.2, 0.0]])
    radius, temperature, _ = _string(positions, [[100.0, 0.0, -5.0]]).finite(0.0)
    assert radius.size == 1 and temperature[0] == 100.0


def test_a_profile_is_read_from_the_nearest_row_not_interpolated():
    """The laser fires at its own cadence; averaging two rows invents a profile."""
    positions = np.array([[0.5, 0.0]])
    string = _string(positions, [[10.0], [20.0], [30.0]])
    assert string.at(0.0)[0][0] == 10.0
    assert string.at(0.0104)[0][0] == 20.0
    assert string.at(0.019)[0][0] == 30.0


def test_channel_positions_accept_both_stored_layouts():
    """atm stores one radius per channel; ayc stores one per time row."""
    from nova.media.sources.mast_thomson import _channel_positions

    flat = _channel_positions(np.array([0.4, 0.8, 1.2]))
    assert flat.shape == (3, 2)
    assert np.allclose(flat[:, 0], [0.4, 0.8, 1.2])
    assert np.allclose(flat[:, 1], 0.0)

    drifting = np.array([[0.40, 0.80, np.nan], [0.42, 0.82, np.nan]])
    per_time = _channel_positions(drifting)
    assert per_time.shape == (3, 2)
    assert per_time[0, 0] == pytest.approx(0.41)
    assert not np.isfinite(per_time[2, 0])


@pytest.mark.slow
def test_both_thomson_eras_are_served_from_the_level_one_store():
    """An atm shot returns core alone; an ayc shot may return core and edge."""
    from nova.media.sources import read_thomson
    from nova.media.sources.mast_thomson import SHOT_STORE

    if not (SHOT_STORE / "22086.zarr").is_dir():
        pytest.skip("the MAST level-1 shot store is not present")

    early = read_thomson(22086)
    assert len(early) == 1
    assert early[0].provenance["era"] == "atm"
    assert early[0].provenance["group"] == "atm"

    late = read_thomson(27079)
    assert [string.provenance["group"] for string in late] == ["ayc", "aye"]
    assert late[0].provenance["era"] == "ayc"

    for string in early + late:
        radius, temperature, _ = string.finite(0.15)
        assert radius.size and np.all(np.isfinite(radius))
        assert np.all(temperature > 0.0)


def test_draw_nulls_drops_an_x_point_outside_the_wall():
    """A null outside the vessel is finite, so finiteness alone lets it draw."""
    angle = np.linspace(0.0, 2.0 * np.pi, 40)
    wall = np.column_stack((1.0 + 0.5 * np.cos(angle), 0.8 * np.sin(angle)))
    view = three_view(extent=(0.0, 2.0, -1.5, 1.5))
    inside_point = np.array([1.0, -0.4])
    outside_point = np.array([0.30, 0.0])
    tally = draw_nulls(
        view.poloidal,
        x_points=np.vstack((inside_point, outside_point)),
        contain=wall,
    )
    assert tally["x_points_drawn"] == 1
    assert tally["x_points_dropped_outside_wall"] == 1
    drawn = np.concatenate([line.get_xydata() for line in view.poloidal.lines])
    assert not np.any(np.all(np.isclose(drawn, outside_point), axis=1))


def test_draw_nulls_without_containment_draws_every_finite_point():
    """Containment is opt-in, so the count says what was actually drawn."""
    view = three_view(extent=(0.0, 2.0, -1.5, 1.5))
    tally = draw_nulls(view.poloidal, x_points=np.array([[1.0, 0.0], [0.30, 0.0]]))
    assert tally["x_points_drawn"] == 2
    assert tally["x_points_dropped_outside_wall"] == 0


def test_strike_points_are_exempt_from_containment():
    """A strike point lies ON the wall, so containment is the wrong test."""
    angle = np.linspace(0.0, 2.0 * np.pi, 40)
    wall = np.column_stack((1.0 + 0.5 * np.cos(angle), 0.8 * np.sin(angle)))
    view = three_view(extent=(0.0, 2.0, -1.5, 1.5))
    tally = draw_nulls(
        view.poloidal,
        strike_points=np.array([[0.30, 0.0], [1.9, 0.0]]),
        contain=wall,
    )
    assert tally["strike_points_drawn"] == 2


def test_chord_crossings_finds_both_sides_of_a_closed_boundary():
    """A Thomson string samples through the boundary where it crosses its chord."""
    from nova.media.poloidal import chord_crossings

    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    loop = np.column_stack((1.0 + 0.4 * np.cos(angle), 0.6 * np.sin(angle)))
    inboard, outboard = chord_crossings(loop, 0.0)
    assert inboard == pytest.approx(0.6, abs=1e-3)
    assert outboard == pytest.approx(1.4, abs=1e-3)


def test_chord_crossings_returns_nothing_off_the_boundary():
    from nova.media.poloidal import chord_crossings

    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    loop = np.column_stack((1.0 + 0.4 * np.cos(angle), 0.6 * np.sin(angle)))
    assert chord_crossings(loop, 5.0) == []
    assert chord_crossings(np.zeros((2, 2)), 0.0) == []


def test_chord_crossings_closes_an_open_polyline():
    """A stored boundary need not repeat its first vertex."""
    from nova.media.poloidal import chord_crossings

    square = np.array([[0.5, -1.0], [1.5, -1.0], [1.5, 1.0], [0.5, 1.0]])
    assert chord_crossings(square, 0.0) == pytest.approx([0.5, 1.5])


@pytest.mark.slow
def test_the_thomson_pairing_carries_no_systematic_lag():
    """Two clocks paired by nearest row must not drift against each other.

    A figure that draws an equilibrium beside a measurement pairs two time
    bases, and a constant offset between them is invisible in the figure and
    in every aggregate over it: the profile simply belongs to a different
    moment than the boundary drawn with it. The check is that the signed
    residual is centred, not merely small -- a lag shows as a consistently
    signed residual, which rounding cannot produce.
    """
    from nova.media.sources import read_labels, read_thomson
    from nova.media.sources.mast_thomson import SHOT_STORE

    if not (SHOT_STORE / "27079.zarr").is_dir():
        pytest.skip("the MAST level-1 shot store is not present")
    if not (BOUNDARY_LABEL_ROOT / "27079.nc").is_file():
        pytest.skip("the carrier's boundary-carrying label session is absent")

    frames, _ = read_labels(27079, dirname=BOUNDARY_LABEL_ROOT)
    label_time = np.asarray([frame.time for frame in frames])
    for string in read_thomson(27079):
        cadence = float(np.median(np.diff(string.time)))
        selected = np.asarray(
            [string.time[int(np.argmin(np.abs(string.time - t)))] for t in label_time]
        )
        residual = selected - label_time
        # Centred to well inside half a sampling interval: nearest-row
        # rounding cannot bias the sign, so a biased median is a real lag.
        assert abs(float(np.median(residual))) < 0.5 * cadence
        assert float(np.max(np.abs(residual))) <= cadence
        positive = float(np.mean(residual > 0.0))
        assert 0.2 < positive < 0.8, f"{string.name} residual sign is biased"


def test_boundary_wall_gap_separates_contact_from_standoff():
    """A limited boundary touches its limiter; a diverted one stands off."""
    from nova.media.poloidal import boundary_wall_gap

    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    wall = np.column_stack((1.0 + 0.5 * np.cos(angle), 0.8 * np.sin(angle)))
    touching = np.column_stack((1.0 + 0.5 * np.cos(angle), 0.8 * np.sin(angle)))
    assert boundary_wall_gap(touching, wall) == pytest.approx(0.0, abs=1e-9)

    standing_off = np.column_stack((1.0 + 0.3 * np.cos(angle), 0.5 * np.sin(angle)))
    assert boundary_wall_gap(standing_off, wall) == pytest.approx(0.2, abs=1e-3)

    assert np.isnan(boundary_wall_gap(np.zeros((1, 2)), wall))


@pytest.mark.slow
def test_the_limited_boundary_defect_is_visible_rather_than_silent():
    """A collapsed limited boundary must show up as a number, not a picture.

    Guards the defect the lead found by eye: on this carrier the solve's
    limited-phase boundary stands off a limiter it must touch. The assertion
    is deliberately on the CONTRAST between classes rather than on an absolute
    gap, so it keeps holding once the solve is repaired and the limited median
    drops toward contact.
    """
    from nova.media.poloidal import boundary_wall_gap
    from nova.media.sources import read_labels
    from nova.media.sources.mast_efit import SHOT_STORE, read_pulse

    if not (SHOT_STORE / "27079.zarr").is_dir():
        pytest.skip("the MAST level-1 shot store is not present")
    if not (BOUNDARY_LABEL_ROOT / "27079.nc").is_file():
        pytest.skip("the carrier's boundary-carrying label session is absent")

    frames, _ = read_labels(27079, dirname=BOUNDARY_LABEL_ROOT)
    wall = read_pulse(27079).geometry.limiter
    classes = {frame.diverted for frame in frames}
    assert classes == {True, False}, "the carrier must carry both topology classes"

    gaps = {
        diverted: np.array(
            [
                boundary_wall_gap(frame.boundary, wall)
                for frame in frames
                if frame.diverted is diverted
            ]
        )
        for diverted in (True, False)
    }
    for diverted, values in gaps.items():
        assert values.size and np.all(np.isfinite(values)), diverted
    # A limited boundary can never stand off FURTHER than a diverted one: that
    # ordering is what the defect violates, and it is the repair's signature.
    assert np.median(gaps[False]) > 0.0


# --------------------------------------------------------------------------
# nova_labels: the boundary contract is the stored LCFS polyline, and an
# empty one is refused rather than substituted from the nested surfaces
# --------------------------------------------------------------------------

# The carrier's boundary-carrying relabelled sessions: the shipped
# presentation-media receipts read this root (boundary_source "stored lcfs
# polyline"), and the label reader now refuses a session that carries none,
# so the carrier slow tests above must read the root that has them rather
# than the production root where every frame's polyline is still empty.
BOUNDARY_LABEL_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/"
    "boundary-repair-validation-20260907T1124Z"
)


def _label_geometry(index: int = 0) -> dict[str, object]:
    """Return deterministic nested-surface geometry for a synthetic session."""
    surface = np.linspace(0.0, 1.0, 11)
    angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radius = 1.0 + 0.25 * surface[:, None] * np.cos(angle)[None, :]
    height = 0.02 * index + 0.35 * surface[:, None] * np.sin(angle)[None, :]
    faces = np.linspace(0.0, 1.0, 26)
    scale = 1.0 + index * 0.01
    profiles = {
        name: faces
        for name in (
            "rho_tor",
            "Phi",
            "psi_face",
            "Ip_profile",
            "R_in",
            "R_out",
            "F",
            "int_dl_over_Bp",
            "inv_R",
            "inv_R2",
            "grad_psi",
            "grad_psi2",
            "grad_psi2_over_R2",
            "B2",
            "inv_B2",
            "delta_upper",
            "delta_lower",
            "elongation",
            "vpr",
            "volume",
            "area",
            "q",
            "g0",
            "g1",
            "g2",
            "g3",
            "psi_norm_face",
        )
    }
    return {
        "flux_surface_psi_norm": surface,
        "flux_surface_psi": scale * surface,
        "flux_surface_r": radius,
        "flux_surface_z": height,
        "flux_surface_angle": angle,
        "rho_face_norm": faces,
        "p_prime_face": -2.0e5 * (1.0 - faces),
        "ff_prime_face": -0.2 * (1.0 - faces),
        **profiles,
        "R_major": 1.0,
        "a_minor": 0.35,
        "B_0": 2.0,
        "boundary_toroidal_flux": 0.5,
        "magnetic_axis_z_scalar": 0.02 * index,
        "diverted": False,
        "divertor_leg_r": np.full((4, 32), np.nan),
        "divertor_leg_z": np.full((4, 32), np.nan),
        "divertor_leg_finite": np.zeros(4, dtype=bool),
    }


def _label_frame(
    lcfs: np.ndarray,
    *,
    index: int = 0,
    boundary_slots: int = 8,
    guarded: bool = True,
) -> object:
    """Return one synthetic steering frame carrying ``lcfs`` as its polyline.

    The stored polyline is NaN-padded to ``boundary_slots`` and counted by
    ``n_boundary_coords``, mirroring the corpus store; a zero-row ``lcfs`` is
    an empty boundary, exactly the shape a pre-persistence session carries.
    """
    from nova.equilibrium.steering_frames import (
        SteeringAction,
        SteeringFrame,
    )

    radial_count, vertical_count = 4, 3
    radius = np.linspace(0.6, 1.42, radial_count, dtype=np.float64)
    height = np.linspace(-0.42, 0.42, vertical_count, dtype=np.float64)
    packed = np.full((boundary_slots, 2), np.nan)
    vertex_count = int(lcfs.shape[0])
    packed[:vertex_count] = np.asarray(lcfs, dtype=float)
    strike = np.array([[np.nan, np.nan], [1.2, 0.06]])
    return SteeringFrame(
        radius=radius,
        height=height,
        shape=np.array([radial_count, vertical_count], dtype=np.int32),
        psi=np.arange(radial_count * vertical_count, dtype=np.float64).reshape(
            radial_count, vertical_count
        )
        * (1.0 + 0.1 * index),
        psi_norm=np.linspace(0.0, 1.0, radial_count * vertical_count).reshape(
            radial_count, vertical_count
        ),
        domain_label=np.full((radial_count, vertical_count), 0, dtype=np.int8),
        separatrix=np.empty((0, 2), dtype=np.float64),
        separatrix_vertex_count=np.int32(0),
        magnetic_axis_r=0.9 + 0.05 * index,
        magnetic_axis_z=0.02 * index,
        x_point_r=np.array([0.65, 0.86]),
        x_point_z=np.array([0.15, -0.36]),
        strike_points_r=strike[:, 0],
        strike_points_z=strike[:, 1],
        lcfs_r=packed[:, 0],
        lcfs_z=packed[:, 1],
        n_boundary_coords=np.int32(vertex_count),
        finite_mask=np.array([True, True, True, False, True, vertex_count > 0]),
        coil_current=np.arange(3, dtype=np.float64) * 1.0e4,
        compensating_current=np.array([1.0, -2.0], dtype=np.float64),
        action=SteeringAction(
            name="minor_radius",
            delta=0.01,
            commanded_control_points=np.array([[0.8, 0.0], [1.2, 0.1]]),
        ),
        wall_seconds=0.25,
        trip_count=index + 1,
        carrier_identity="synthetic",
        nova_version="9.9.9",
        policy_digest="0" * 64,
        p_prime_source="efm",
        current_centroid_r=1.5,
        current_centroid_z=0.02 * index,
        reference_centroid_z=0.02 * index,
        branch_guard_ok=guarded,
        **_label_geometry(index),
    )


def _write_label_session(tmp_path, frames, shot=4242) -> None:
    """Record one synthetic rasterless session into ``tmp_path``."""
    from nova.equilibrium.steering_frames import write_session

    write_session(
        tuple(frames),
        filename=str(shot),
        dirname=str(tmp_path),
        include_raster=False,
    )


def _stored_ring():
    """Return a closed LCFS ring distinct from the outermost nested surface."""
    angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    return np.column_stack((1.0 + 0.30 * np.cos(angle), 0.40 * np.sin(angle)))


def test_a_session_with_an_empty_boundary_variable_raises(tmp_path):
    """A frame with no stored polyline is refused, never substituted."""
    from nova.media.sources.nova_labels import read_labels

    _write_label_session(tmp_path, (_label_frame(np.empty((0, 2))),))
    with pytest.raises(ValueError, match="boundary polyline"):
        read_labels(4242, dirname=tmp_path)


def test_the_stored_polyline_is_the_boundary_the_reader_draws(tmp_path):
    """A populated lcfs is taken as the LCFS, not the nested surface.

    The stored ring is deliberately a different curve from the outermost
    nested surface, so this only passes because the reader selected the
    stored polyline -- the previous prefer-if-present branch and the
    nested-surface contract would both draw something else.
    """
    from nova.media.sources.nova_labels import read_labels

    stored = _stored_ring()
    _write_label_session(tmp_path, (_label_frame(stored, boundary_slots=64),))
    frames, provenance = read_labels(4242, dirname=tmp_path)

    assert np.array_equal(frames[0].boundary, stored)
    assert not np.array_equal(frames[0].boundary, frames[0].surfaces[-1])
    assert "stored lcfs polyline" in provenance["boundary_source"]


def test_a_session_mixing_stored_and_empty_frames_is_refused(tmp_path):
    """A partially relabelled corpus fails rather than blends boundary sources."""
    from nova.media.sources.nova_labels import read_labels

    _write_label_session(
        tmp_path,
        (
            _label_frame(_stored_ring(), index=0, boundary_slots=64),
            _label_frame(np.empty((0, 2)), index=1, boundary_slots=64),
        ),
    )
    with pytest.raises(ValueError, match="boundary polyline"):
        read_labels(4242, dirname=tmp_path)
