"""Pin the state scoreboard's comparator, checks and published table.

The scoreboard scores any reconstructed state supplied as an IMAS equilibrium
against a reference state with the registered bands.  These tests build a small
IMAS corpus, score it, and measure the four claims the record makes: a state
compared with itself passes every band; a state carrying a declared
perturbation is ranked worse on every metric and fails the bands it carries
defects in; the reference-free checks fire on a state whose axis leaves its own
boundary; and a column group added to the manifest changes the table without any
change to the scoring code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.estimator_scoreboard import (
    DEFAULT_PERTURBATION,
    emit_perturbed_states,
    main,
    map_boundary,
    perturb_state,
    state_from_record,
)
from nova.imas.mast_efit_referee import (
    EfitRefereeError,
    compare_states,
    q95_from_flux_map,
    read_equilibrium_state,
    read_equilibrium_wall,
    reference_free_checks,
)
from nova.imas.parity_tolerances import (
    STATE_SCOREBOARD_FIELDS,
    STATE_SCOREBOARD_TOLERANCES,
)

DD_VERSION = "4.1.0"


def write_state(
    path: Path,
    *,
    axis: tuple[float, float] = (0.8, 0.0),
    half_width: float = 0.5,
    half_height: float = 0.5,
    plasma_current: float = 1.0e6,
    flux_axis: float = 1.0,
    flux_boundary: float = 0.0,
    q95_scale: float = 1.0,
    points: int = 64,
    outline: np.ndarray | None = None,
) -> Path:
    """Write one single-slice IMAS equilibrium at a stated data dictionary."""

    import imas
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    equilibrium = imas.IDSFactory().equilibrium()
    equilibrium.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    equilibrium.time = np.array([1.0])
    equilibrium.time_slice.resize(1)
    frame = equilibrium.time_slice[0]
    if outline is None:
        angle = np.linspace(0.0, 2.0 * np.pi, points)
        outline = np.column_stack(
            [
                axis[0] + half_width * np.cos(angle),
                axis[1] + half_height * np.sin(angle),
            ]
        )
    frame.boundary.outline.r = outline[:, 0]
    frame.boundary.outline.z = outline[:, 1]
    frame.boundary.type = 1
    quantities = frame.global_quantities
    quantities.magnetic_axis.r = axis[0]
    quantities.magnetic_axis.z = axis[1]
    quantities.psi_axis = flux_axis
    quantities.psi_boundary = flux_boundary
    quantities.ip = plasma_current
    norm = np.linspace(0.0, 1.0, 65)
    frame.profiles_1d.psi = flux_axis + (flux_boundary - flux_axis) * norm
    frame.profiles_1d.psi_norm = norm
    frame.profiles_1d.q = q95_scale * (1.0 + 3.0 * norm**2)
    path.mkdir(parents=True, exist_ok=True)
    entry = imas.DBEntry(f"imas:hdf5?path={path}", "w", dd_version=DD_VERSION)
    entry.put(equilibrium)
    entry.close()
    return path


def write_wall(
    path: Path, *, half_width: float = 1.4, half_height: float = 1.4
) -> Path:
    """Write one machine description carrying a single limiter ring."""

    import imas
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    wall = imas.IDSFactory().wall()
    wall.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    wall.description_2d.resize(1)
    unit = wall.description_2d[0].limiter.unit
    unit.resize(1)
    ring = np.array(
        [
            (0.2, -half_height),
            (0.2 + 2.0 * half_width, -half_height),
            (0.2 + 2.0 * half_width, half_height),
            (0.2, half_height),
            (0.2, -half_height),
        ]
    )
    unit[0].outline.r = ring[:, 0]
    unit[0].outline.z = ring[:, 1]
    path.mkdir(parents=True, exist_ok=True)
    entry = imas.DBEntry(f"imas:hdf5?path={path}", "w", dd_version=DD_VERSION)
    entry.put(wall)
    entry.close()
    return path


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("estimator-scoreboard")
    reference = write_state(root / "reference")
    wall = write_wall(root / "machine")
    return {"root": root, "reference": reference, "wall": wall}


def test_state_compared_with_itself_passes_every_band(corpus):
    reference = read_equilibrium_state(corpus["reference"], wall_source=corpus["wall"])
    comparison = compare_states(reference, reference)

    assert comparison.magnetic_axis_distance_m == 0.0
    assert comparison.plasma_current_fraction == 0.0
    assert comparison.axis_flux_fraction == 0.0
    assert comparison.boundary_flux_fraction == 0.0
    assert comparison.q95_difference == 0.0
    assert set(comparison.verdicts()) == set(STATE_SCOREBOARD_FIELDS)
    assert all(comparison.verdicts().values())

    checks = reference_free_checks(reference)
    assert checks.axis_inside_wall is True
    assert checks.axis_inside_boundary is True
    assert 0.5 < checks.boundary_area_fraction < 1.0
    assert all(checks.verdicts().values())


def test_declared_perturbation_is_ranked_worse_on_every_metric(corpus):
    reference = read_equilibrium_state(corpus["reference"], wall_source=corpus["wall"])
    perturbed_path = perturb_state(
        corpus["reference"], corpus["root"] / "perturbed", **DEFAULT_PERTURBATION
    )
    perturbed = read_equilibrium_state(perturbed_path, wall_source=corpus["wall"])
    comparison = compare_states(reference, perturbed)

    for field, value in comparison.metrics.items():
        assert np.isfinite(value), f"{field} carries no measure on the perturbed state"
        assert value > 0.0, f"{field} does not move under the declared perturbation"

    verdicts = comparison.verdicts()
    tolerance = STATE_SCOREBOARD_TOLERANCES
    assert not verdicts["magnetic_axis_distance_m"]
    assert not verdicts["plasma_current_fraction"]
    assert not verdicts["axis_flux_fraction"]
    assert not verdicts["boundary_flux_fraction"]
    assert (
        comparison.magnetic_axis_distance_m
        >= tolerance["magnetic_axis_distance_m"].bound
    )
    assert comparison.q95_difference >= tolerance["q95_difference"].bound
    assert not verdicts["q95_difference"]
    # The state still passes the reference-free checks: the defect is a comparison
    # failure, not a refusal to be a state at all.
    assert all(reference_free_checks(perturbed).verdicts().values())


def test_reference_free_checks_fires_on_an_axis_outside_its_boundary(corpus):
    stranded = write_state(
        corpus["root"] / "axis-outside",
        axis=(0.0, 0.0),
        outline=np.array(
            [(2.8, -0.2), (3.2, -0.2), (3.2, 0.2), (2.8, 0.2), (2.8, -0.2)]
        ),
    )
    state = read_equilibrium_state(stranded, wall_source=corpus["wall"])
    checks = reference_free_checks(state)
    assert checks.axis_inside_boundary is False
    assert checks.verdicts()["axis_inside_boundary"] is False
    # The axis is outside the wall as well, so the wall check fires too.
    assert checks.axis_inside_wall is False


def test_degenerate_boundary_fails_the_area_check(corpus):
    sliver = write_state(
        corpus["root"] / "sliver",
        outline=np.array(
            [(0.5, 0.0), (4.5, 4.0), (4.5, 4.02), (0.5, 0.02), (0.5, 0.0)]
        ),
    )
    state = read_equilibrium_state(sliver, wall_source=corpus["wall"])
    checks = reference_free_checks(state)
    assert checks.boundary_area_fraction < 0.05
    assert checks.verdicts()["boundary_area_fraction"] is False


def test_state_digest_is_stable_and_moves_with_content(corpus):
    first = read_equilibrium_state(corpus["reference"], wall_source=corpus["wall"])
    second = read_equilibrium_state(corpus["reference"], wall_source=corpus["wall"])
    moved = write_state(corpus["root"] / "moved", axis=(0.81, 0.0))
    other = read_equilibrium_state(moved, wall_source=corpus["wall"])

    assert first.digest() == second.digest()
    assert first.digest() != other.digest()
    assert first.digest().startswith("sha256:")
    assert first.dd_version == DD_VERSION


def write_mapped_state(
    path: Path, *, amplitude: float = 5.0, flux_map: Path | None = None
) -> Path:
    """Write one state carrying its own flux map, gradient and reference radius."""

    import imas
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    equilibrium = imas.IDSFactory().equilibrium()
    equilibrium.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    equilibrium.time = np.array([1.0])
    if flux_map is None:
        radius = np.linspace(0.0, 2.0, 41)
        height = np.linspace(-1.0, 1.0, 41)
        rr, zz = np.meshgrid(radius, height, indexing="ij")
        flux = -((rr - 1.0) ** 2 + zz**2)
    else:
        stored = np.load(Path(flux_map), allow_pickle=False)
        radius = np.asarray(stored["radius"], dtype=float)
        height = np.asarray(stored["height"], dtype=float)
        flux = np.asarray(stored["flux"], dtype=float)
    equilibrium.time_slice.resize(1)
    frame = equilibrium.time_slice[0]
    angle = np.linspace(0.0, 2.0 * np.pi, 64)
    frame.boundary.outline.r = 1.0 + 0.45 * np.cos(angle)
    frame.boundary.outline.z = 0.45 * np.sin(angle)
    frame.boundary.type = 1
    quantities = frame.global_quantities
    quantities.magnetic_axis.r = 1.0
    quantities.magnetic_axis.z = 0.0
    quantities.psi_axis = -1.0
    quantities.psi_boundary = -0.2
    quantities.ip = 1.0e6
    norm = np.linspace(0.0, 1.0, 65)
    frame.profiles_1d.psi = -1.0 + 0.8 * norm
    frame.profiles_1d.f_df_dpsi = amplitude * (1.0 - norm)
    frame.profiles_1d.q = 1.0 + 2.0 * norm**2
    frame.profiles_2d.resize(1)
    profile = frame.profiles_2d[0]
    profile.grid.dim1 = radius
    profile.grid.dim2 = height
    profile.grid_type.index = 1
    profile.psi = flux
    equilibrium.vacuum_toroidal_field.b0 = np.array([-5.3])
    equilibrium.vacuum_toroidal_field.r0 = 1.0
    path.mkdir(parents=True, exist_ok=True)
    entry = imas.DBEntry(f"imas:hdf5?path={path}", "w", dd_version=DD_VERSION)
    entry.put(equilibrium)
    entry.close()
    return path


def test_a_state_without_a_boundary_is_refused(corpus):
    import imas
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    empty = corpus["root"] / "empty"
    equilibrium = imas.IDSFactory().equilibrium()
    equilibrium.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    equilibrium.time = np.array([1.0])
    equilibrium.time_slice.resize(1)
    empty.mkdir(parents=True, exist_ok=True)
    entry = imas.DBEntry(f"imas:hdf5?path={empty}", "w", dd_version=DD_VERSION)
    entry.put(equilibrium)
    entry.close()

    with pytest.raises(EfitRefereeError, match="boundary outline"):
        read_equilibrium_state(empty, wall_source=corpus["wall"])


def test_a_flux_span_that_is_absent_is_reported_not_zero(corpus):
    flat = write_state(corpus["root"] / "flat-flux", flux_axis=0.0, flux_boundary=0.0)
    state = read_equilibrium_state(flat, wall_source=corpus["wall"])
    comparison = compare_states(state, state)
    assert np.isnan(comparison.axis_flux_fraction)
    assert np.isnan(comparison.boundary_flux_fraction)
    assert comparison.verdicts()["axis_flux_fraction"] is False
    assert comparison.verdicts()["magnetic_axis_distance_m"] is True


def test_a_column_group_is_added_without_a_scoring_change(corpus):
    reference = read_equilibrium_state(corpus["reference"], wall_source=corpus["wall"])
    perturbed_path = perturb_state(
        corpus["reference"],
        corpus["root"] / "table-reference-perturbed",
        **DEFAULT_PERTURBATION,
    )
    manifest = {
        "cases": [
            {
                "id": "synthetic",
                "machine": "synthetic",
                "vacuum_b0": 5.3,
                "wall": str(corpus["wall"]),
                "reference": {"source": str(corpus["reference"])},
                "estimators": [
                    {"name": "stored", "source": str(corpus["reference"])},
                    {
                        "name": "perturbed",
                        "perturbed_of": "reference",
                        "out": "perturbed",
                    },
                    {"name": "absent", "source": str(corpus["root"] / "missing")},
                ],
            }
        ]
    }
    manifest_path = corpus["root"] / "cases.json"
    manifest_path.write_text(json.dumps(manifest))
    state_dir = corpus["root"] / "states"
    written = emit_perturbed_states(manifest, state_dir)
    assert written == [state_dir / "perturbed"]
    assert Path(written[0]).joinpath("master.h5").exists()
    assert perturbed_path.joinpath("master.h5").exists()

    html = corpus["root"] / "table.html"
    receipt = corpus["root"] / "receipt.json"
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--html",
                str(html),
                "--receipt",
                str(receipt),
                "--state-dir",
                str(state_dir),
            ]
        )
        == 0
    )
    record = json.loads(receipt.read_text())
    entries = {entry["name"]: entry for entry in record["cases"][0]["estimators"]}
    assert entries["stored"]["state"]["digest"] == reference.digest()
    assert (
        entries["perturbed"]["state"]["digest"]
        == read_equilibrium_state(written[0], wall_source=corpus["wall"]).digest()
    )
    assert entries["absent"]["status"] == "absent"
    # the corpus states carry no flux map, so the safety factor is not supplied
    # by the one calculator and that is the only band the stored state fails
    for name in ("stored", "perturbed"):
        assert np.isnan(entries[name]["q95_flux_map"])
    assert entries["stored"]["verdicts"]["q95_difference"] is False
    assert all(
        verdict
        for field, verdict in entries["stored"]["verdicts"].items()
        if field != "q95_difference"
    )
    assert not all(entries["perturbed"]["verdicts"].values())
    assert all(entries["stored"]["reference_free_verdicts"].values())

    published = html.read_text()
    assert published.count("<th colspan=") == 3
    assert "not supplied" in published
    assert reference.digest()[7:19] in published


def test_the_safety_factor_is_read_from_a_states_own_flux_map(tmp_path):
    """One calculator reads q95 off the map and the gradient the state stores."""

    mapped = write_mapped_state(tmp_path / "mapped", amplitude=5.0)
    state = read_equilibrium_state(mapped, wall_source=corpus_wall(tmp_path))
    assert state.flux_map is not None
    value = q95_from_flux_map(state, vacuum_b0=5.3)

    assert np.isfinite(value)
    # the map and gradient are what the value is read from: a steeper diamagnetic
    # gradient carries more toroidal flux per unit poloidal flux and raises q
    stronger = read_equilibrium_state(
        write_mapped_state(tmp_path / "steeper", amplitude=20.0),
        wall_source=corpus_wall(tmp_path),
    )
    assert q95_from_flux_map(stronger, vacuum_b0=5.3) > value > 0.0


def test_a_state_without_a_flux_map_reports_q95_not_supplied(tmp_path):
    """A scalar safety factor never stands in for the map the calculator reads."""

    plain = write_state(tmp_path / "plain", q95_scale=2.0)
    wall = write_wall(tmp_path / "machine")
    state = read_equilibrium_state(plain, wall_source=wall)
    assert state.flux_map is None
    assert np.isfinite(state.q95)
    assert np.isnan(q95_from_flux_map(state, vacuum_b0=5.3))

    manifest = {
        "cases": [
            {
                "id": "unmapped",
                "machine": "synthetic",
                "vacuum_b0": 5.3,
                "wall": str(wall),
                "reference": {"source": str(plain)},
                "estimators": [{"name": "plain", "source": str(plain)}],
            }
        ]
    }
    manifest_path = tmp_path / "cases.json"
    manifest_path.write_text(json.dumps(manifest))
    html = tmp_path / "table.html"
    receipt = tmp_path / "receipt.json"
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--html",
                str(html),
                "--receipt",
                str(receipt),
            ]
        )
        == 0
    )
    record = json.loads(receipt.read_text())["cases"][0]
    entry = record["estimators"][0]
    assert np.isnan(entry["q95_flux_map"])
    assert np.isnan(entry["metrics"]["q95_difference"])
    assert entry["verdicts"]["q95_difference"] is False
    assert entry["q95_stored"] == pytest.approx(state.q95)
    # the stored value is reported beside the absence, never in place of it.
    assert "not supplied" in html.read_text()
    assert "no flux map" in html.read_text()


def test_the_declared_perturbation_moves_the_map_the_calculator_reads(tmp_path):
    """The copy's q95 moves because its stored map moved, not its scalar q."""

    mapped = write_mapped_state(tmp_path / "source", amplitude=5.0)
    wall = corpus_wall(tmp_path)
    reference = read_equilibrium_state(mapped, wall_source=wall)
    copy = read_equilibrium_state(
        perturb_state(
            mapped,
            tmp_path / "copy",
            radial_shift_m=0.0,
            flux_fraction=0.0,
            flux_map_scale=DEFAULT_PERTURBATION["flux_map_scale"],
        ),
        wall_source=wall,
    )
    reference_q95 = q95_from_flux_map(reference, vacuum_b0=5.3)
    copy_q95 = q95_from_flux_map(copy, vacuum_b0=5.3)
    assert np.isfinite(reference_q95) and np.isfinite(copy_q95)
    difference = abs(abs(copy_q95) - abs(reference_q95))
    assert difference > STATE_SCOREBOARD_TOLERANCES["q95_difference"].bound
    comparison = compare_states(reference, copy, q95_values=(reference_q95, copy_q95))
    assert comparison.verdicts()["q95_difference"] is False


def corpus_wall(root: Path) -> Path:
    """Return the corpus wall written under one scratch root."""

    return write_wall(root / "machine")


def write_flux_map(path: Path) -> Path:
    """Write a paraboloid flux map whose boundary level is a circle."""

    radius = np.linspace(0.0, 2.0, 81)
    height = np.linspace(-1.0, 1.0, 81)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    flux = (rr - 1.0) ** 2 + zz**2
    np.savez(
        path,
        radius=radius,
        height=height,
        flux=-flux,
        wall=np.array([(0.2, -1.2), (2.2, -1.2), (2.2, 1.2), (0.2, 1.2), (0.2, -1.2)]),
    )
    return path


def test_a_state_is_written_from_a_topology_record_and_flux_map(tmp_path):
    flux_map = write_flux_map(tmp_path / "map.npz")
    record = tmp_path / "record.json"
    record.write_text(
        json.dumps(
            {
                "time": 7.5,
                "state": {
                    "axis": [1.0, 0.0],
                    "flux_axis": -1.0,
                    "flux_boundary": 0.0,
                    "plasma_current": 2.0e6,
                },
            }
        )
    )
    written = state_from_record(
        record, flux_map, tmp_path / "state", section="state", edge_psi_norm=0.99
    )
    state = read_equilibrium_state(written, wall_source=flux_map, q95=-1.5)

    assert state.time_s == 7.5
    assert state.boundary_m.shape[1] == 2
    assert state.boundary_m.shape[0] >= 3
    assert np.allclose(state.magnetic_axis_m, [1.0, 0.0])
    assert state.plasma_current_a == 2.0e6
    assert state.q95 == -1.5
    assert len(state.wall_loops) == 1
    checks = reference_free_checks(state)
    assert checks.values["axis_inside_wall"] is True
    assert checks.values["axis_inside_boundary"] is True
    assert checks.values["boundary_area_fraction"] > 0.5


def test_the_outline_is_traced_inside_the_separatrix(tmp_path):
    flux_map = write_flux_map(tmp_path / "map.npz")
    ring = map_boundary(flux_map, -0.05)
    radius = ring[:, 0]

    assert np.allclose(ring[0], ring[-1])
    assert radius.max() - radius.min() == pytest.approx(2.0 * np.sqrt(0.05), rel=0.05)


def test_a_stored_ring_is_read_as_the_wall(tmp_path):
    flux_map = write_flux_map(tmp_path / "map.npz")
    rings = read_equilibrium_wall(flux_map)

    assert len(rings) == 1
    assert rings[0].shape == (5, 2)
