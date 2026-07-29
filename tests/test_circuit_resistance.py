"""Passive-resistance calibration and conductor-topology discovery.

The load-bearing contracts:

* group labels honour the case identity and the normalised region rule, and
  applying a calibration fails loud on an unknown group;
* on a synthetic system with a known resistance multiplier the pooled objective
  is minimised at the truth, from held-back circuit-current supervision alone;
* a held-back channel is supervision ONLY -- corrupting the measured target moves
  the held-back loss and nothing else;
* the series reduction reproduces the classical inductance algebra;
* the adjacency stamp couples a pair and preserves positive definiteness, and a
  zero coupling reproduces the diagonal spectrum exactly;
* the galvanic wiring edit lands only on the wired row, and the structured loss
  is minimised at the true wiring gain;
* an empty structure reduces to the plain diagonal eigen-reduction.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.circuit.passive import PassiveCircuitSystem, reduce_passive_system
from nova.circuit.propagate import zoh_mode_response
from nova.circuit.resistance import (
    ModeMaps,
    PassiveStructure,
    ResistanceCalibration,
    VacuumShotData,
    build_structure_hypothesis,
    campaign_mode_maps,
    case_parent_channels,
    load_calibration,
    load_structure,
    neighbour_edges,
    pooled_loss,
    resistance_group_labels,
    save_calibration,
    save_structure,
    series_reduction,
    structure_hypothesis_parts,
    structured_mode_maps,
    structured_reduced_basis,
    structured_shot_loss,
    updown_pair_channels,
)

RNG = np.random.default_rng(7)


def _toy_system(n_measured: int = 1) -> PassiveCircuitSystem:
    """Two passive circuits, three drive channels, one held-back measurement."""
    return PassiveCircuitSystem(
        circuits=np.array([101, 14]),
        centroid_r=np.array([0.3, 1.5]),
        centroid_z=np.array([0.0, 1.1]),
        lmat=np.array([[2.0, 0.6], [0.6, 1.5]]) * 1e-6,
        r_diag=np.array([4.0e-5, 8.0e-5]),
        a_circuit=RNG.normal(size=(5, 2)) * 1e-6,
        g_grid=RNG.normal(size=(12, 2)) * 1e-6,
        m_channel=RNG.normal(size=(2, 3)) * 1e-5,
        channels=["a_current", "b_current", "sol_current"],
        measured_channel_row={"p2u_case_current": 1} if n_measured else {},
        resistivity=7.2e-7,
        section_scale=np.array([0.05, 0.05]),
    )


def _toy_system_three() -> PassiveCircuitSystem:
    """Three passive circuits (row two is instrumented), three drive channels."""
    rng = np.random.default_rng(11)
    return PassiveCircuitSystem(
        circuits=np.array([201, 202, 14]),
        centroid_r=np.array([1.0, 1.0, 1.4]),
        centroid_z=np.array([0.0, 0.08, 1.0]),
        lmat=np.array([[2.0, 0.5, 0.2], [0.5, 1.8, 0.3], [0.2, 0.3, 1.2]]) * 1e-6,
        r_diag=np.array([4.0e-5, 6.0e-5, 9.0e-5]),
        a_circuit=rng.normal(size=(5, 3)) * 1e-6,
        g_grid=rng.normal(size=(12, 3)) * 1e-6,
        m_channel=rng.normal(size=(3, 3)) * 1e-5,
        channels=["p4l_coil_current", "p4u_coil_current", "sol_current"],
        measured_channel_row={"p4u_case_current": 2},
        resistivity=7.2e-7,
        section_scale=np.array([0.1, 0.1, 0.05]),
    )


#: the drive linkage of the three-circuit toy, as the wiring hypothesis reads it
_TOY_LAM = np.array([[5.0, 1.0, 0.3], [1.0, 4.0, 0.2], [0.3, 0.2, 8.0]]) * 1e-5


def test_group_labels_follow_case_identity_and_the_normalised_regions():
    circuits = np.array([14, 18, 200, 201, 202, 203])
    case_of = {14: "p2u", 18: "p4u"}
    radius = np.array([1.5, 1.5, 0.2, 1.9, 1.0, 1.0])
    height = np.array([1.1, 1.1, 0.0, 0.1, 2.0, 0.2])
    labels = resistance_group_labels(
        circuits, radius, height, "regions-percase", case_of=case_of
    )
    assert labels[0] == "case:p2u" and labels[1] == "case:p4u"
    assert labels[2] == "vessel:inboard"  # normalised radius zero
    assert labels[3] == "vessel:outboard"  # normalised radius one
    assert labels[4] == "vessel:ends"  # mid radius, largest |z|
    assert labels[5] == "vessel:mid"
    paired = resistance_group_labels(
        circuits, radius, height, "regions-casepairs", case_of=case_of
    )
    assert paired[0] == "case:p2" and paired[1] == "case:p4"
    coarse = resistance_group_labels(
        circuits, radius, height, "vessel-case", case_of=case_of
    )
    assert coarse[:2] == ["case", "case"]
    assert set(coarse[2:]) == {"vessel"}
    assert set(
        resistance_group_labels(circuits, radius, height, "global", case_of=case_of)
    ) == {"all"}
    with pytest.raises(ValueError, match="unknown ladder level"):
        resistance_group_labels(circuits, radius, height, "per-filament")


def test_calibration_round_trips_and_fails_loud_on_a_missing_group(tmp_path):
    calibration = ResistanceCalibration(
        level="vessel-case",
        group_multipliers={"vessel": 3.7, "case": 1.4},
        provenance={"pool": "unit-test"},
    )
    path = tmp_path / "calibration.json"
    save_calibration(path, calibration)
    restored = load_calibration(path)
    assert restored.level == calibration.level
    assert restored.group_multipliers == calibration.group_multipliers
    np.testing.assert_allclose(
        restored.per_circuit(
            np.array([14, 200]),
            np.array([1.5, 0.2]),
            np.array([1.1, 0.0]),
            case_of={14: "p2u"},
        ),
        [1.4, 3.7],
    )
    incomplete = ResistanceCalibration("vessel-case", {"vessel": 3.7}, {})
    with pytest.raises(KeyError, match="case"):
        incomplete.per_circuit(
            np.array([14]), np.array([1.5]), np.array([1.1]), case_of={14: "p2u"}
        )
    (tmp_path / "junk.json").write_text('{"kind": "other"}')
    with pytest.raises(ValueError, match="not a resistance calibration"):
        load_calibration(tmp_path / "junk.json")


def test_uniform_multiplier_scales_decay_times_and_keeps_the_vectors():
    system = _toy_system()
    nominal = campaign_mode_maps(system, np.ones(2))
    scaled = campaign_mode_maps(system, np.full(2, 4.0))
    np.testing.assert_allclose(scaled.tau, nominal.tau / 4.0, rtol=1e-12)
    # L-orthonormal eigenvectors are sign-fixed up to column order; a uniform
    # resistance scale preserves both
    np.testing.assert_allclose(np.abs(scaled.v), np.abs(nominal.v), rtol=1e-10)


def test_measured_rows_follow_the_sorted_channel_order():
    system = _toy_system()
    system.measured_channel_row = {"p2u_case_current": 1, "a2l_case_current": 0}
    maps = campaign_mode_maps(system, np.ones(2))
    assert isinstance(maps, ModeMaps)
    np.testing.assert_array_equal(maps.measured_v[0], maps.v[0])
    np.testing.assert_array_equal(maps.measured_v[1], maps.v[1])


def _simulate_shot(system, multipliers, seed=0, n_t=1200, interval=1e-3):
    """A synthetic coil-only shot: the held-back current carries the truth."""
    rng = np.random.default_rng(seed)
    i_drive = np.cumsum(rng.normal(0, 30.0, size=(n_t, 3)), axis=0)
    psi_circuit = i_drive @ system.m_channel.T
    maps = campaign_mode_maps(system, multipliers)
    state = zoh_mode_response(maps.tau, interval, psi_circuit @ maps.v)
    return VacuumShotData(
        shot=1000 + seed,
        campaign="toy",
        stratum="dedicated_vacuum",
        interval=interval,
        psi_circuit=psi_circuit,
        residual=state @ maps.a_sensor_modes.T + rng.normal(0, 1e-5, size=(n_t, 5)),
        sigma=np.full(5, 1e-5),
        measured=state @ maps.measured_v.T + rng.normal(0, 0.05, size=(n_t, 1)),
    )


def test_pooled_loss_is_minimised_at_the_true_multiplier():
    system = _toy_system()
    truth = np.array([4.0, 4.0])
    shots = [_simulate_shot(system, truth, seed=seed) for seed in range(3)]
    sigma_sensor = {"toy": np.full(5, 1e-5)}
    sigma_measured = {"toy": np.array([np.nanstd(shots[0].measured)])}
    grid = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    losses = [
        pooled_loss(
            np.array([scale]),
            {"toy": np.zeros(2, dtype=np.int64)},
            {"toy": system},
            shots,
            sigma_sensor,
            sigma_measured,
        )["combined"]
        for scale in grid
    ]
    assert grid[int(np.argmin(losses))] == 4.0


def test_held_back_supervision_never_leaks_into_the_drive():
    """Corrupting the measured target moves the held-back loss and ONLY that: the
    drive flux is assembled from the drive channels alone."""
    system = _toy_system()
    theta = np.array([4.0, 4.0])
    shot = _simulate_shot(system, theta, seed=1)
    args = (
        {"toy": np.zeros(2, dtype=np.int64)},
        {"toy": system},
        [shot],
        {"toy": np.full(5, 1e-5)},
        {"toy": np.array([1.0])},
    )
    base = pooled_loss(theta, *args)
    corrupted = VacuumShotData(
        **{
            **shot.__dict__,
            "measured": shot.measured
            + 100.0 * np.sin(np.arange(shot.n_samples)[:, None] / 50.0),
        }
    )
    hit = pooled_loss(theta, *args[:2], [corrupted], *args[3:])
    assert hit["measured"] > base["measured"] * 2
    np.testing.assert_allclose(hit["sensor"], base["sensor"], rtol=1e-12)


# --- structure discovery ---------------------------------------------------
def test_series_reduction_reproduces_the_classical_inductance_algebra():
    lmat = np.array([[2.0, 0.6], [0.6, 1.5]])
    series = series_reduction(2, [(0, 1, +1)])
    anti = series_reduction(2, [(0, 1, -1)])
    np.testing.assert_allclose(series.T @ lmat @ series, [[2.0 + 1.5 + 1.2]])
    np.testing.assert_allclose(anti.T @ lmat @ anti, [[2.0 + 1.5 - 1.2]])
    resistance = np.diag([4.0, 8.0])
    np.testing.assert_allclose(series.T @ resistance @ series, [[12.0]])
    np.testing.assert_allclose(anti.T @ resistance @ anti, [[12.0]])
    drive = np.array([1.0, 0.25])
    np.testing.assert_allclose(series.T @ drive, [1.25])
    np.testing.assert_allclose(anti.T @ drive, [0.75])
    with pytest.raises(ValueError, match="disjoint"):
        series_reduction(3, [(0, 1, 1), (1, 2, 1)])


def test_neighbour_edges_use_the_size_normalised_rule():
    radius = np.array([1.0, 1.0, 1.0, 2.0])
    height = np.array([0.0, 0.1, 0.5, 0.0])
    scale = np.array([0.1, 0.1, 0.1, 0.1])
    edges = neighbour_edges(radius, height, scale, factor=1.5)
    assert (0, 1) in edges  # 0.1 apart, threshold 0.15
    assert (0, 2) not in edges and (1, 2) not in edges
    assert all(3 not in edge for edge in edges)
    assert neighbour_edges(radius, height, scale, factor=1.5, exclude_rows={1}) == []


def test_channel_label_rules_find_parents_and_updown_pairs():
    channels = [
        "p2il_coil_current",
        "p2iu_coil_current",
        "p2ol_coil_current",
        "p2ou_coil_current",
        "p4u_coil_current",
        "p4l_coil_current",
        "p6u_current",
        "p6l_current",
        "sol_current",
    ]
    assert case_parent_channels("p2l_case_current", channels) == [
        "p2il_coil_current",
        "p2ol_coil_current",
    ]
    assert case_parent_channels("p4u_case_current", channels) == ["p4u_coil_current"]
    pairs = updown_pair_channels(channels)
    assert ("p2iu_coil_current", "p2il_coil_current") in pairs
    assert ("p4u_coil_current", "p4l_coil_current") in pairs
    assert ("p6u_current", "p6l_current") in pairs
    assert all("sol" not in channel for pair in pairs for channel in pair)


def test_an_empty_structure_reduces_to_the_diagonal_model():
    system = _toy_system_three()
    multipliers = np.array([1.5, 2.0, 0.8])
    diagonal = campaign_mode_maps(system, multipliers)
    structured = structured_mode_maps(
        build_structure_hypothesis(system, np.arange(3)), multipliers
    )
    np.testing.assert_allclose(
        np.sort(structured.tau), np.sort(diagonal.tau), rtol=1e-12
    )
    # identical sensor response to the same drive history
    rng = np.random.default_rng(0)
    i_drive = np.cumsum(rng.normal(size=(400, 3)), axis=0) * 10.0
    diagonal_state = zoh_mode_response(
        diagonal.tau, 1e-3, (i_drive @ system.m_channel.T) @ diagonal.v
    )
    structured_state = zoh_mode_response(
        structured.tau, 1e-3, i_drive @ structured.drive_flux.T
    )
    np.testing.assert_allclose(
        structured_state @ structured.a_sensor_modes.T,
        diagonal_state @ diagonal.a_sensor_modes.T,
        rtol=1e-8,
        atol=1e-24,
    )


def test_adjacency_stamp_couples_a_pair_and_stays_positive_definite():
    system = _toy_system_three()
    hypothesis = build_structure_hypothesis(system, np.arange(3), edges=[(0, 1)])
    diagonal = campaign_mode_maps(system, np.ones(3))
    coupled = structured_mode_maps(hypothesis, np.ones(3), edge_r=np.array([1.0]))
    # a strong shared branch makes the pair's differential mode decay at once
    assert np.min(coupled.tau) < np.min(diagonal.tau) * 1e-3
    assert np.all(coupled.tau > 0)
    uncoupled = structured_mode_maps(hypothesis, np.ones(3), edge_r=np.array([0.0]))
    np.testing.assert_allclose(
        np.sort(uncoupled.tau), np.sort(diagonal.tau), rtol=1e-12
    )


def test_galvanic_wiring_edits_only_the_wired_row():
    system = _toy_system_three()
    hypothesis = build_structure_hypothesis(
        system,
        np.arange(3),
        wiring_cases=["p4u_case_current"],
        drive_linkage=(list(system.channels), _TOY_LAM),
    )
    # the parent of the p4u case is the p4u winding: drive column one
    np.testing.assert_allclose(hypothesis.wiring_select, [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(hypothesis.wiring_lam, [_TOY_LAM[1]])
    maps = structured_mode_maps(
        hypothesis, np.ones(3), g_v=np.array([2.0]), r_w=np.array([3e-3])
    )
    expected = system.m_channel.copy()
    expected[2] -= 2.0 * _TOY_LAM[1]
    np.testing.assert_allclose(
        maps.drive_flux, maps.v_physical.T @ expected, rtol=1e-12
    )
    volt_columns = np.zeros((3, 3))
    volt_columns[2, 1] = 3e-3
    np.testing.assert_allclose(
        maps.drive_volt, maps.v_physical.T @ volt_columns, rtol=1e-12
    )


def test_structured_loss_recovers_the_true_wiring_gain():
    """Synthetic truth with a galvanic case wiring: the structured loss is
    minimised at the true gain, from held-back supervision alone."""
    system = _toy_system_three()
    hypothesis = build_structure_hypothesis(
        system,
        np.arange(3),
        wiring_cases=["p4u_case_current"],
        drive_linkage=(list(system.channels), _TOY_LAM),
    )
    true_gain = 3.0
    rng = np.random.default_rng(5)
    i_drive = np.cumsum(rng.normal(0, 30.0, size=(1500, 3)), axis=0)
    truth = structured_mode_maps(hypothesis, np.ones(3), g_v=np.array([true_gain]))
    state = zoh_mode_response(truth.tau, 1e-3, i_drive @ truth.drive_flux.T)
    measured = state @ truth.measured_map.T + rng.normal(0, 0.02, size=(1500, 1))
    data = VacuumShotData(
        shot=1,
        campaign="toy",
        stratum="dedicated_vacuum",
        interval=1e-3,
        psi_circuit=i_drive @ system.m_channel.T,
        residual=state @ truth.a_sensor_modes.T + rng.normal(0, 1e-5, size=(1500, 5)),
        sigma=np.full(5, 1e-5),
        measured=measured,
        i_drive=i_drive,
    )
    sigma_measured = np.array([max(float(np.nanstd(measured)), 1.0)])
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
    losses = []
    for gain in grid:
        maps = structured_mode_maps(hypothesis, np.ones(3), g_v=np.array([gain]))
        terms = structured_shot_loss(data, maps, np.full(5, 1e-5), sigma_measured)
        losses.append(terms[0] / max(terms[1], 1) + terms[2] / max(terms[3], 1))
    assert grid[int(np.argmin(losses))] == true_gain


def test_structured_loss_requires_the_raw_drives():
    system = _toy_system_three()
    maps = structured_mode_maps(
        build_structure_hypothesis(system, np.arange(3)), np.ones(3)
    )
    shot = _simulate_shot(_toy_system(), np.array([4.0, 4.0]), seed=1)
    with pytest.raises(ValueError, match="i_drive"):
        structured_shot_loss(shot, maps, np.full(5, 1e-5), np.array([1.0]))


def test_a_series_wired_pair_predicts_both_channels_equal():
    system = _toy_system_three()
    system.measured_channel_row = {
        "p4u_case_current": 2,
        "p4l_case_current": 1,
    }
    hypothesis = build_structure_hypothesis(
        system,
        np.arange(3),
        case_series=[("p4l_case_current", "p4u_case_current", +1)],
    )
    assert hypothesis.reduction.shape == (3, 2)
    maps = structured_mode_maps(hypothesis, np.ones(3))
    rng = np.random.default_rng(9)
    i_drive = np.cumsum(rng.normal(size=(300, 3)), axis=0) * 15.0
    state = zoh_mode_response(maps.tau, 1e-3, i_drive @ maps.drive_flux.T)
    predicted = state @ maps.measured_map.T  # sorted channels: p4l then p4u
    np.testing.assert_allclose(predicted[:, 0], predicted[:, 1], rtol=1e-12)


def test_structured_reduced_basis_matches_the_diagonal_reduction():
    system = _toy_system_three()
    case_of = {14: "p4u"}
    multipliers = {
        "case:p4u": 1.3,
        "vessel:inboard": 2.0,
        "vessel:outboard": 4.0,
        "vessel:mid": 8.0,
        "vessel:ends": 16.0,
    }
    empty = PassiveStructure(
        case_series_pairs=[],
        case_wiring={},
        pair_drive_gains=[],
        adjacency={},
        neighbour_rule={},
        r_level="regions-percase",
        r_group_multipliers=multipliers,
        provenance={},
    )
    scale = np.full(5, 1e-5)
    cell_index = np.array([3, 7, 11])
    _hypothesis, parts = structure_hypothesis_parts(system, empty, case_of=case_of)
    reference = reduce_passive_system(
        system,
        sensor_scale=scale,
        n_modes=2,
        cell_index=cell_index,
        r_multipliers=parts["multipliers"],
    )
    structured = structured_reduced_basis(
        system,
        empty,
        sensor_scale=scale,
        n_modes=2,
        cell_index=cell_index,
        case_of=case_of,
    )
    np.testing.assert_allclose(structured.tau, reference.tau, rtol=1e-12)
    np.testing.assert_allclose(np.abs(structured.v), np.abs(reference.v), rtol=1e-9)
    np.testing.assert_allclose(
        np.abs(structured.m_channel), np.abs(reference.m_channel), rtol=1e-9
    )
    np.testing.assert_allclose(
        np.abs(structured.m_cell), np.abs(reference.m_cell), rtol=1e-9
    )
    assert structured.volt_channel is None

    wired = PassiveStructure(
        case_series_pairs=[],
        case_wiring={
            "p4u_case_current": {
                "parents": ["p4u_coil_current"],
                "g_v": 5.0,
                "r_w": 2e-3,
            }
        },
        pair_drive_gains=[],
        adjacency={},
        neighbour_rule={},
        r_level=empty.r_level,
        r_group_multipliers=multipliers,
        provenance={},
    )
    with_wiring = structured_reduced_basis(
        system,
        wired,
        sensor_scale=scale,
        n_modes=3,
        cell_index=cell_index,
        drive_linkage=(list(system.channels), _TOY_LAM),
        case_of=case_of,
    )
    assert with_wiring.volt_channel is not None
    assert with_wiring.volt_channel.shape == (3, 3)
    with pytest.raises(ValueError, match="drive_linkage"):
        structured_reduced_basis(
            system,
            wired,
            sensor_scale=scale,
            n_modes=2,
            cell_index=cell_index,
            case_of=case_of,
        )


def test_structure_artifact_round_trips(tmp_path):
    structure = PassiveStructure(
        case_series_pairs=[
            {"channels": ["p3l_case_current", "p3u_case_current"], "sign": 1}
        ],
        case_wiring={
            "p2l_case_current": {
                "parents": ["p2il_coil_current", "p2ol_coil_current"],
                "g_v": 11.2,
                "r_w": 2.4e-3,
            }
        },
        pair_drive_gains=[
            {
                "channels": ["p4u_coil_current", "p4l_coil_current"],
                "common": 0.02,
                "differential": -0.01,
            }
        ],
        adjacency={"first": [{"i": 201, "j": 202, "r_couple": 3.0e-4}]},
        neighbour_rule={"factor": 1.5, "metric": "pair-mean section scale"},
        r_level="regions-percase",
        r_group_multipliers={"vessel:mid": 12.7},
        provenance={"pool": "unit-test"},
    )
    path = tmp_path / "structure.json"
    save_structure(path, structure)
    restored = load_structure(path)
    assert restored.case_wiring["p2l_case_current"]["g_v"] == 11.2
    assert restored.adjacency["first"][0]["r_couple"] == 3.0e-4
    assert restored.case_series_pairs == structure.case_series_pairs
    (tmp_path / "junk.json").write_text('{"kind": "other"}')
    with pytest.raises(ValueError, match="not a passive-structure"):
        load_structure(tmp_path / "junk.json")
