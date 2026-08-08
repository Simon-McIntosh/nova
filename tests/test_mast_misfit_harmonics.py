"""Constraint sets and harmonic misfit maps.

Every fixture here is synthetic: a known filament stands in for the unmodelled
conductor, two more stand in for described coils whose currents the fit is
allowed to move, and the sensor poses reproduce the shape of the real array --
a line on the centre column facing outward and an arc on the outboard wall
carrying both field components, plus toroidal loops that read flux.  That makes
the truth known, so the tests can assert what the fits RECOVER rather than
merely that they run.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from nova.biot import toroidalharmonic as th
from nova.biot.greens import greens_bz_br, greens_psi
from nova.imas import mast_misfit_harmonics as mh

SOURCE = (1.45, -1.20)
"""Filament standing in for the unmodelled conductor [m]."""

COILS = ((0.20, 0.00), (1.50, 1.10), (1.70, 0.50), (0.70, 1.50))
"""Filaments standing in for described coils the joint fit may move.

Placed clear of the source, so that projecting their span out leaves 96% of the
source's own pattern standing.  That separation is what makes the recovery tests
below a statement about the FIT: with a coil set that nearly spans the source
the split between the harmonic block and the coil block is ill-posed for any
solver, which is a property of the machine rather than of this code and is
asserted separately.
"""

CROWDING_COILS = ((0.20, 0.00), (1.70, -0.50), (0.70, -1.50), (1.50, 1.10))
"""A coil set closing in on the source, leaving a quarter of its pattern."""

COIL_SHARE = (0.040, -0.025, 0.018, -0.031)
"""Ampere-turn share of each described coil the synthetic coupling carries.

Sized so the coil columns alone reach 0.30 of the whitened power, which is where
the real drive groups sit (0.34 on the strongest).  The share matters to more
than realism: let the coils dominate and both radial families fit the small
remainder equally well, so the contrast between them -- the thing that
localises -- goes to zero and measures nothing.
"""


def probe_poses():
    """Return the centre-column and outboard probe poses of the test array."""
    column_z = np.linspace(-1.5, 1.4, 20)
    column = np.column_stack(
        [np.full(column_z.size, 0.18), column_z, np.ones(column_z.size), np.zeros(20)]
    )
    angle = np.linspace(-1.25 * np.pi, 0.25 * np.pi, 18)
    arc_r = 1.55 + 0.28 * np.cos(angle)
    arc_z = 0.75 * np.sin(angle)
    radial = np.column_stack([arc_r, arc_z, np.cos(angle), np.sin(angle)])
    axial = np.column_stack([arc_r, arc_z, -np.sin(angle), np.cos(angle)])
    return np.vstack([column, radial, axial])


def loop_poses():
    """Return the toroidal flux-loop positions of the test array."""
    column = np.column_stack([np.full(6, 0.18), np.linspace(-1.2, 1.2, 6)])
    outboard = np.column_stack(
        [np.array([1.60, 1.40, 1.40, 1.60]), np.array([-1.05, -1.15, 1.15, 1.05])]
    )
    return np.vstack([column, outboard])


def projected_field(poses, source, current=1.0):
    """Return a filament's field along each pose's sensitive axis [T]."""
    axial, radial = greens_bz_br(poses[:, 0], poses[:, 1], *source)
    return current * (poses[:, 2] * radial + poses[:, 3] * axial)


def build_classes(
    *,
    shots: int = 9,
    current: float = 3.0e-3,
    coil_share=COIL_SHARE,
    scatter: float = 0.02,
    seed: int = 7,
    loops: bool = True,
    coils=COILS,
):
    """Return the two sensor classes of a synthetic drive group.

    The couplings carry the source filament, a share of each described coil, and
    a per-shot multiplicative wobble, so a fit that recovers the source has had
    to separate it from the coil columns rather than absorb them.
    """
    generator = np.random.default_rng(seed)
    probes, rings = probe_poses(), loop_poses()
    probe_described = np.column_stack([projected_field(probes, coil) for coil in coils])
    loop_described = np.column_stack(
        [greens_psi(rings[:, 0], rings[:, 1], *coil) for coil in coils]
    )
    share = np.asarray(coil_share)
    probe_truth = projected_field(probes, SOURCE, current) + probe_described @ share
    loop_truth = (
        current * greens_psi(rings[:, 0], rings[:, 1], *SOURCE) + loop_described @ share
    )

    def sample(truth):
        """Return per-shot draws about a truth vector."""
        return truth[None, :] * (
            1.0 + generator.normal(0.0, scatter, (shots, truth.size))
        )

    classes = [
        mh.SensorClass(
            channel=tuple(f"probe{index:02d}" for index in range(probes.shape[0])),
            r=probes[:, 0],
            z=probes[:, 1],
            radial_cosine=probes[:, 2],
            axial_sine=probes[:, 3],
            reads_flux=False,
            coupling=sample(probe_truth),
            described=probe_described,
            floor=np.zeros(probes.shape[0]),
        )
    ]
    if loops:
        classes.append(
            mh.SensorClass(
                channel=tuple(f"loop{index:02d}" for index in range(rings.shape[0])),
                r=rings[:, 0],
                z=rings[:, 1],
                radial_cosine=np.ones(rings.shape[0]),
                axial_sine=np.zeros(rings.shape[0]),
                reads_flux=True,
                coupling=sample(loop_truth),
                described=loop_described,
                floor=np.zeros(rings.shape[0]),
            )
        )
    return classes


@pytest.fixture
def constraint():
    """Return the synthetic drive group's assembled constraint set."""
    return mh.assemble("group", build_classes(), shots=tuple(range(9)))


@pytest.fixture
def focus():
    """Return a focal circle displaced from the synthetic source."""
    return mh.place_focus(*SOURCE)


# --- the sensor classes and the pooling -------------------------------------


def test_sensor_class_rejects_a_pose_of_the_wrong_length():
    """A class whose arrays disagree about its channel count is refused."""
    with pytest.raises(mh.MisfitMapError, match="channels but"):
        mh.SensorClass(
            channel=("a", "b"),
            r=np.zeros(2),
            z=np.zeros(3),
            radial_cosine=np.zeros(2),
            axial_sine=np.zeros(2),
            reads_flux=False,
            coupling=np.zeros((4, 2)),
            described=np.zeros((2, 1)),
            floor=np.zeros(2),
        )


def test_sensor_class_rejects_a_coupling_of_the_wrong_width():
    """A coupling block that is not one column per channel is refused."""
    with pytest.raises(mh.MisfitMapError, match="against 2 channels"):
        mh.SensorClass(
            channel=("a", "b"),
            r=np.zeros(2),
            z=np.zeros(2),
            radial_cosine=np.zeros(2),
            axial_sine=np.zeros(2),
            reads_flux=False,
            coupling=np.zeros((4, 3)),
            described=np.zeros((2, 1)),
            floor=np.zeros(2),
        )


def test_pooling_takes_the_median_and_its_robust_standard_error():
    """The pooled value ignores an outlying shot and the error falls with count."""
    sample = np.tile(np.array([2.0, 5.0]), (9, 1))
    sample += np.tile(np.array([-0.1, 0.0, 0.1]), 3)[:, None]
    sample[0, 0] = 40.0
    value, error, count = mh.pooled_noise(sample, np.zeros(2))
    assert value[0] == pytest.approx(2.0, abs=0.05)
    assert count.tolist() == [9, 9]
    assert error[0] < 0.1


def test_pooling_floors_the_error_at_the_instrument_contribution():
    """A channel whose shots happen to agree still carries the sensor floor."""
    sample = np.full((16, 1), 3.0)
    _, error, _ = mh.pooled_noise(sample, np.array([0.8]))
    assert error[0] == pytest.approx(0.8 / 4.0)


def test_pooling_ignores_a_shot_that_did_not_serve_a_channel():
    """A NaN entry lowers the count rather than poisoning the median."""
    sample = np.full((6, 2), 4.0)
    sample[:4, 1] = np.nan
    value, _, count = mh.pooled_noise(sample, np.zeros(2))
    assert count.tolist() == [6, 2]
    assert value[1] == pytest.approx(4.0)


# --- assembling a constraint set --------------------------------------------


def test_assemble_carries_both_sensor_classes_in_their_own_units(constraint):
    """Field rows and flux rows arrive together, tagged by what they read."""
    assert constraint.rows == 56 + 10
    assert int(constraint.reads_flux.sum()) == 10
    assert constraint.sample.shape == (9, constraint.rows)
    assert constraint.shots == tuple(range(9))


def test_assemble_drops_an_excluded_channel(constraint):
    """A named faulty channel never reaches the solve."""
    trimmed = mh.assemble("group", build_classes(), excluded={"probe00", "loop00"})
    assert trimmed.rows == constraint.rows - 2
    assert "probe00" not in trimmed.channel


def test_assemble_refuses_a_channel_with_too_few_shots():
    """A channel one shot served has no measurable scatter and is not carried."""
    classes = build_classes(shots=4)
    classes[0].coupling[1:, 0] = np.nan
    kept = mh.assemble("group", classes, minimum_shots=2)
    assert "probe00" not in kept.channel


def test_assemble_refuses_a_group_no_channel_survives():
    """A set whose every channel fails the pooling requirement raises."""
    classes = build_classes(shots=3)
    classes[0].coupling[...] = np.nan
    with pytest.raises(mh.MisfitMapError, match="clears the pooling"):
        mh.assemble("group", classes[:1], minimum_shots=2)


def test_the_bands_split_the_array_at_the_centre_column(constraint):
    """Every row falls in exactly one band, split on the geometric gap."""
    assert (constraint.centre_column ^ constraint.outboard).all()
    assert constraint.r[constraint.centre_column].max() < mh.CENTRE_COLUMN_RADIUS
    assert constraint.r[constraint.outboard].min() > mh.CENTRE_COLUMN_RADIUS


def test_selecting_rows_keeps_the_per_shot_sample_aligned(constraint):
    """A row selection carries the resampling block with it."""
    chosen = constraint.select(constraint.reads_flux)
    assert chosen.rows == 10
    assert chosen.sample.shape == (9, 10)
    assert chosen.described.shape[0] == 10


def test_selecting_no_row_raises(constraint):
    """An empty selection is a caller error, not an empty solve."""
    with pytest.raises(mh.MisfitMapError, match="no row survives"):
        constraint.select(np.zeros(constraint.rows, dtype=bool))


# --- the design -------------------------------------------------------------


def test_each_row_reads_the_functional_its_class_measures(constraint, focus):
    """Flux rows take the column itself and field rows take its curl."""
    basis = th.ToroidalHarmonics(focus, order=3, families=(th.INNER,))
    design = mh.harmonic_design(basis, constraint)
    flux = constraint.reads_flux
    assert design.shape == (constraint.rows, len(basis.labels))
    np.testing.assert_allclose(
        design[flux], basis.flux(constraint.r[flux], constraint.z[flux])
    )
    np.testing.assert_allclose(
        design[~flux],
        basis.project(
            constraint.r[~flux],
            constraint.z[~flux],
            constraint.radial_cosine[~flux],
            constraint.axial_sine[~flux],
        ),
    )


# --- the fits ---------------------------------------------------------------


def test_the_joint_fit_recovers_the_source_the_coil_columns_hide(constraint, focus):
    """Fitting harmonics and coil currents together returns the true filament.

    The synthetic coupling carries a large described-coil share on top of the
    source, and the fit is given no hint which is which.
    """
    basis = th.ToroidalHarmonics(focus, order=8, families=(th.INNER,))
    fit = mh.fit_jointly(basis, constraint)
    estimate = th.locate_source(basis, fit.coefficients)
    assert np.hypot(estimate.r - SOURCE[0], estimate.z - SOURCE[1]) < 0.05
    assert fit.explained > 0.99


def test_the_joint_fit_recovers_the_coil_currents_it_was_not_told(constraint, focus):
    """The described-coil unknowns land on the share the coupling was built with."""
    basis = th.ToroidalHarmonics(focus, order=8, families=(th.INNER,))
    fit = mh.fit_jointly(basis, constraint)
    np.testing.assert_allclose(fit.currents, COIL_SHARE, atol=0.002)


def test_the_projected_fit_is_not_the_joint_fit_by_another_route(constraint, focus):
    """The shortcut answers a different question and cannot report coil currents.

    How far its source read is displaced depends on how nearly the coil span
    contains the source, which is a property of the machine; what is always true
    is that the reduction is not equivalent -- the projector deletes the data's
    coil-span component instead of explaining it, so no coil current is
    recoverable and the harmonic coefficients are not the joint fit's.
    """
    basis = th.ToroidalHarmonics(focus, order=8, families=(th.INNER,))
    joint = mh.fit_jointly(basis, constraint)
    shortcut = mh.fit_projected(basis, constraint)
    assert shortcut.currents.size == 0
    assert joint.currents.size == len(COIL_SHARE)

    def apart(rows):
        """Return how far the two reductions' coefficients stand, relatively."""
        first = mh.fit_jointly(basis, rows).coefficients
        second = mh.fit_projected(basis, rows).coefficients
        return float(np.linalg.norm(first - second) / np.linalg.norm(first))

    # The gap widens with how nearly the coil span contains the source, which is
    # the whole mechanism: the projector deletes what the span can imitate.
    crowded = mh.assemble("group", build_classes(coils=CROWDING_COILS))
    assert apart(constraint) > 1.0e-3
    assert apart(crowded) > 5.0 * apart(constraint)


def test_the_projected_fit_leaves_no_coil_span_in_its_residual(constraint, focus):
    """Its residual is orthogonal to every whitened described column, by build."""
    basis = th.ToroidalHarmonics(focus, order=6, families=(th.INNER,))
    fit = mh.fit_projected(basis, constraint)
    design = mh.harmonic_design(basis, constraint)
    weight = constraint.weight
    described = constraint.described * weight[:, None]
    orthogonal = np.eye(constraint.rows) - described @ np.linalg.pinv(described)
    whitened = orthogonal * weight[None, :]
    residual = whitened @ (design @ fit.coefficients - constraint.value)
    leak = np.linalg.norm(described.T @ residual) / (
        np.linalg.norm(described) * np.linalg.norm(residual)
    )
    assert leak < 1.0e-10


def test_a_fit_reports_the_coil_columns_own_reach(constraint, focus):
    """The coil block alone explains far less than the expansion beside it."""
    basis = th.ToroidalHarmonics(focus, order=8, families=(th.INNER,))
    fit = mh.fit_jointly(basis, constraint)
    assert fit.coil_explained < 0.9
    assert fit.explained > fit.coil_explained


def test_the_family_slice_addresses_one_side_of_a_two_family_fit(constraint, focus):
    """A two-family fit's coefficients split evenly, first family first."""
    basis = th.ToroidalHarmonics(focus, order=4, families=(th.INNER, th.OUTER))
    fit = mh.fit_jointly(basis, constraint)
    assert fit.family_slice(th.INNER) == slice(0, 9)
    assert fit.family_slice(th.OUTER) == slice(9, 18)
    with pytest.raises(mh.MisfitMapError, match="fit carries"):
        fit.family_slice("elsewhere")


# --- degree selection -------------------------------------------------------


def test_the_degree_is_chosen_on_channels_the_fold_did_not_fit(constraint, focus):
    """Selection returns an offered degree and scores every one of them."""
    order, scores = mh.select_degree(focus, constraint, (2, 4, 6, 8))
    assert order in (2, 4, 6, 8)
    assert set(scores) == {2, 4, 6, 8}
    assert all(np.isfinite(value) for value in scores.values())


def test_a_smooth_misfit_does_not_buy_the_highest_degree_offered(focus):
    """A source far from the focal circle is served by a low degree.

    Held-out prediction is what makes this measurable: the in-sample residual
    falls with degree whether or not the added degrees carry anything.
    """
    classes = build_classes(scatter=0.05, seed=11)
    constraint = mh.assemble("group", classes)
    order, _ = mh.select_degree(mh.place_focus(0.9, 0.0), constraint, (2, 4, 8, 12))
    assert order <= 8


# --- maps -------------------------------------------------------------------


def test_the_flux_map_reproduces_the_filament_it_was_fitted_to(constraint, focus):
    """Away from the source the mapped flux matches the filament's own."""
    basis = th.ToroidalHarmonics(focus, order=10, families=(th.INNER,))
    fit = mh.fit_jointly(basis, constraint)
    grid_r = np.linspace(0.6, 1.1, 9)
    grid_z = np.linspace(-0.4, 0.4, 9)
    mapped = mh.flux_map(basis, fit.coefficients, grid_r, grid_z)
    mesh_r, mesh_z = np.meshgrid(grid_r, grid_z)
    exact = 3.0e-3 * greens_psi(mesh_r.ravel(), mesh_z.ravel(), *SOURCE)
    exact = exact.reshape(mesh_r.shape)
    assert np.max(np.abs(mapped - exact)) < 0.05 * np.max(np.abs(exact))


def test_the_supported_mask_is_the_range_the_sensors_bracket(constraint, focus):
    """Points nearer the focal circle than every sensor are outside the mask."""
    grid_r = np.linspace(0.1, 2.0, 21)
    grid_z = np.linspace(-1.8, 1.8, 21)
    mask = mh.supported_mask(focus, constraint, grid_r, grid_z)
    assert mask.shape == (21, 21)
    assert mask.any() and not mask.all()
    span = th.focal_frame(constraint.r, constraint.z, focus).distance
    mesh_r, mesh_z = np.meshgrid(grid_r, grid_z)
    distance = th.focal_frame(mesh_r, mesh_z, focus).distance
    assert (distance[mask] >= span.min()).all()
    assert (distance[mask] <= span.max()).all()


def test_resampling_bands_come_from_the_shots(constraint, focus):
    """One map per draw, and a band that is positive where the data is thin."""
    basis = th.ToroidalHarmonics(focus, order=4, families=(th.INNER,))
    grid_r = np.linspace(0.5, 1.2, 6)
    grid_z = np.linspace(-0.5, 0.5, 6)
    draws = mh.resample_maps(basis, constraint, grid_r, grid_z, draws=12, seed=3)
    assert draws.shape == (12, 6, 6)
    assert np.std(draws, axis=0).max() > 0.0


def test_resampling_refuses_a_group_of_one_shot(constraint, focus):
    """A single shot carries no resampling distribution and says so."""
    single = dataclasses.replace(constraint, sample=constraint.sample[:1])
    basis = th.ToroidalHarmonics(focus, order=2, families=(th.INNER,))
    with pytest.raises(mh.MisfitMapError, match="cannot resample"):
        mh.resample_maps(basis, single, [0.8, 0.9], [0.0, 0.1], draws=4)


# --- localisation -----------------------------------------------------------


def test_the_family_contrast_favours_a_circle_placed_on_the_source(constraint):
    """The enclosed family wins near the source and not on the far side."""
    near = mh.family_contrast(constraint, mh.place_focus(*SOURCE), order=2)
    across = mh.family_contrast(constraint, th.FocalCircle(1.45, 1.20), order=2)
    assert near > 0.2
    assert near > across


def test_the_focus_scan_peaks_at_the_source(constraint):
    """Scanning the focal circle over the plane localises the filament."""
    grid_r = np.linspace(0.6, 2.0, 8)
    grid_z = np.linspace(-1.8, 1.4, 9)
    scan = mh.scan_focus(constraint, grid_r, grid_z, order=2)
    assert scan.shape == (9, 8)
    peak = np.unravel_index(int(np.argmax(scan)), scan.shape)
    assert abs(grid_r[peak[1]] - SOURCE[0]) < 0.35
    assert abs(grid_z[peak[0]] - SOURCE[1]) < 0.35


def test_the_iterated_focus_settles_on_the_source(constraint):
    """Re-placing the circle at each read converges and reports its support."""
    read = mh.iterate_focus(constraint, th.FocalCircle(1.2, -0.9), order=8)
    assert read.estimate is not None
    assert np.hypot(read.estimate.r - SOURCE[0], read.estimate.z - SOURCE[1]) < 0.06
    assert read.rows > 0
    assert len(read.track) >= 1


def test_a_read_outside_the_array_stops_the_iteration(constraint):
    """A position the sensors do not bracket ends the walk where it stood."""
    assert mh.rows_converge(
        th.ToroidalHarmonics(mh.place_focus(*SOURCE), order=4, families=(th.INNER,)),
        constraint.select(constraint.centre_column),
        3.0,
    )
    read = mh.iterate_focus(constraint, th.FocalCircle(1.2, -0.9), order=8, rounds=1)
    assert read.focus.radius > 0.0


# --- what one part of the array says about another --------------------------


def test_an_expansion_is_scored_against_silence_on_rows_it_never_saw(constraint, focus):
    """A fit that saw equivalent rows predicts the withheld ones better than zero."""
    basis = th.ToroidalHarmonics(focus, order=6, families=(th.INNER,))
    probes = constraint.select(~constraint.reads_flux)
    loops = constraint.select(constraint.reads_flux)
    score = mh.cross_prediction(basis, mh.fit_jointly(basis, probes), loops)
    assert score["baseline"] > 0.0
    assert score["ratio"] == pytest.approx(score["error"] / score["baseline"])


def test_both_bands_are_fitted_and_scored_across(constraint, focus):
    """Each band's expansion is measured on the band it did not see."""
    basis = th.ToroidalHarmonics(focus, order=4, families=(th.INNER,))
    fits, scores = mh.band_agreement(basis, constraint)
    assert set(fits) == {"centre_column", "outboard"}
    assert set(scores) == {"centre_column", "outboard"}
    assert all(value["error"] >= 0.0 for value in scores.values())


# --- reading a bank ---------------------------------------------------------


def test_a_coupling_bank_is_read_into_one_group(tmp_path):
    """The reader takes one drive group's rows and scales the floor by the spans."""
    archive = {
        "groups": np.array(["a", "b", "a"]),
        "shots": np.array([11, 12, 13]),
        "channel": np.array(["one", "two"]),
        "r": np.array([1.0, 1.2]),
        "z": np.array([0.0, 0.1]),
        "cos": np.array([1.0, 0.0]),
        "sin": np.array([0.0, 1.0]),
        "spans": np.array([100.0, 50.0, 100.0]),
        "coupling": np.arange(6, dtype=float).reshape(3, 2),
        "response": np.ones((2, 3)),
    }
    sensor = mh.sensor_class(archive, "a", reads_flux=False, floor=np.array([2.0, 4.0]))
    assert sensor.coupling.shape == (2, 2)
    assert sensor.floor == pytest.approx([0.02, 0.04])
    assert mh.bank_shots(archive, "a") == (11, 13)
    with pytest.raises(mh.MisfitMapError, match="no rows for drive group"):
        mh.sensor_class(archive, "c", reads_flux=False)


def test_a_flux_class_needs_no_sensitive_axis(tmp_path):
    """A loop bank has no pose columns and the reader does not ask for them."""
    archive = {
        "groups": np.array(["a"]),
        "shots": np.array([11]),
        "channel": np.array(["loop"]),
        "r": np.array([1.0]),
        "z": np.array([0.0]),
        "spans": np.array([10.0]),
        "coupling": np.zeros((1, 1)),
        "response": np.ones((1, 3)),
    }
    sensor = mh.sensor_class(archive, "a", reads_flux=True)
    assert sensor.reads_flux
    assert sensor.floor == pytest.approx([0.0])
