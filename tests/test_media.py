"""Contracts for the presentation-media package, on synthetic geometry.

Every test here runs in the fast lane: no corpus, no solve, no device. What is
pinned is the behaviour that went wrong while the package was written, so a
regression reintroduces a measured defect rather than an imagined one.
"""

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
