from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from nova.equilibrium.conductor_current import (
    InferenceLikelihood,
    LikelihoodValue,
    UnknownCurrentPrior,
    solve_conductor_currents,
)
from nova.imas import diiid_current, diiid_description
from nova.imas.diiid_description import (
    CIRCUIT_DRIVEN_CONDUCTORS,
    PF_ACTIVE_CIRCUIT,
    POLOIDAL_CONDUCTORS,
)


def _circuit_bypass_priors() -> dict[str, UnknownCurrentPrior]:
    return {
        name: UnknownCurrentPrior(
            mean_a=10_000.0 + index * 1000.0,
            standard_deviation_a=3000.0,
            lower_a=-20_000.0,
            upper_a=40_000.0,
            provenance="declared demonstration prior",
        )
        for index, name in enumerate(diiid_current.CIRCUIT_BYPASS_PRIOR_CONDUCTORS)
    }


def _shipped() -> dict[str, float]:
    result = {
        name: float(index + 1) * 1000.0
        for index, name in enumerate(POLOIDAL_CONDUCTORS)
    }
    result["ECOILA"] = 10_000.0
    return result


def test_diiid_tiers_carry_fit_once_values_and_relation_uncertainty() -> None:
    resolution = diiid_current.resolve_diiid_currents(POLOIDAL_CONDUCTORS, _shipped())
    by_name = {name: index for index, name in enumerate(resolution.names)}

    assert len(resolution.names) == 24
    assert resolution.names[:19] == POLOIDAL_CONDUCTORS
    assert resolution.unknown_names == ()
    expected = {
        "ECOILB": 20_000.0,
        "E567UP": 10_000.0,
        "E567DN": 10_000.0,
        "E89UP": 10_456.947569496173,
        "E89DN": 10_456.240764323717,
    }
    for drive in PF_ACTIVE_CIRCUIT.drives:
        index = by_name[drive.conductor]
        assert resolution.template_a[index] == pytest.approx(expected[drive.conductor])
        assert resolution.prescribed_standard_deviation_a[index] == pytest.approx(
            abs(expected[drive.conductor]) * drive.uncertainty.residual_rms_fraction
        )
        declaration = resolution.declarations[index]
        assert declaration.relation is diiid_current.CIRCUIT_RELATIONS[drive.conductor]
        assert "leave-one-shot-out R-squared" in declaration.relation.provenance
        assert "label flux" in declaration.relation.transfer_caveat


def test_circuit_map_drives_all_registered_conductors_from_ecoila() -> None:
    shipped = _shipped()

    reconstructed = diiid_current.circuit_current_map(shipped)
    resolution = diiid_current.resolve_diiid_currents(POLOIDAL_CONDUCTORS, shipped)

    assert tuple(reconstructed) == CIRCUIT_DRIVEN_CONDUCTORS
    np.testing.assert_allclose(
        list(reconstructed.values()),
        [20_000.0, 10_000.0, 10_000.0, 10_456.947569496173, 10_456.240764323717],
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        resolution.current(())[: len(POLOIDAL_CONDUCTORS)],
        list(shipped.values()),
    )
    np.testing.assert_allclose(
        resolution.current(())[len(POLOIDAL_CONDUCTORS) :],
        list(reconstructed.values()),
    )


def test_banked_circuit_receipt_matches_runtime_calibration() -> None:
    root = Path(__file__).parents[2]
    receipt = json.loads(
        (
            root / "docs/figures/coil-circuit-discovery/pf_active_circuit_receipt.json"
        ).read_text()
    )
    ensemble = json.loads(
        (
            root
            / "docs/figures/coil-circuit-discovery"
            / "ensemble_relation_assessment_receipt.json"
        ).read_text()
    )
    frame_receipt = json.loads(
        (root / ensemble["inputs"]["fitted_current"]["path"]).read_text()
    )
    reproduced = diiid_description.adjudicate_circuit_wiring(
        frame_receipt["ensemble_ready_fitted_current_table"], ensemble
    )
    banked = receipt["adjudication"]["drives"]

    assert receipt["reconstruction_fixture"]["complete_response_current_count"] == 24
    assert receipt["reconstruction_fixture"]["free_current_count_with_circuit"] == 0
    assert reproduced == tuple(drive.wiring for drive in PF_ACTIVE_CIRCUIT.drives)
    assert [row["effective_gain"] for row in banked] == [
        drive.gain for drive in PF_ACTIVE_CIRCUIT.drives
    ]
    for row, drive in zip(banked, PF_ACTIVE_CIRCUIT.drives, strict=True):
        assert row["wiring"] == drive.wiring.as_record()
        assert row["uncertainty"] == drive.uncertainty.as_record()
    assert "nearly degenerate" in " ".join(receipt["caveats"])
    assert "1 of 60" in " ".join(receipt["caveats"])


def test_circuit_bypass_exposes_three_prior_driven_diagnostic_slots() -> None:
    resolution = diiid_current.resolve_diiid_currents(
        POLOIDAL_CONDUCTORS,
        _shipped(),
        _circuit_bypass_priors(),
        use_circuit=False,
    )
    by_name = {name: index for index, name in enumerate(resolution.names)}

    assert resolution.unknown_names == ("ECOILB", "E567DN", "E89UP")
    assert resolution.template_a[by_name["E567UP"]] == pytest.approx(9929.0)
    assert resolution.template_a[by_name["E89DN"]] == pytest.approx(10_165.0)
    assert all(
        np.isnan(resolution.template_a[by_name[name]])
        for name in diiid_current.CIRCUIT_BYPASS_PRIOR_CONDUCTORS
    )


class LabelRejectingRow(dict):
    def __getitem__(self, key):
        if "efit" in str(key).lower() or "recovered" in str(key).lower():
            raise AssertionError(f"label artifact read: {key}")
        return super().__getitem__(key)


@dataclass(frozen=True)
class Turns:
    applied_multiplier: float = 1.0


@dataclass(frozen=True)
class Conductor:
    name: str
    input_column: str
    turns: Turns = Turns()


@dataclass(frozen=True)
class Description:
    conductors: tuple[Conductor, ...]


def test_shipped_interpolation_never_reads_label_artifacts() -> None:
    row = LabelRejectingRow(
        magnetics_time=np.asarray([0.0, 1.0]),
        **{f"magnetics_{name}": np.asarray([1.0, 3.0]) for name in POLOIDAL_CONDUCTORS},
    )
    description = Description(
        tuple(Conductor(name, f"magnetics_{name}") for name in POLOIDAL_CONDUCTORS)
    )

    result = diiid_current.shipped_current_at(
        row, description, POLOIDAL_CONDUCTORS, 0.5
    )

    assert tuple(result) == POLOIDAL_CONDUCTORS
    assert set(result.values()) == {2000.0}


@dataclass(frozen=True)
class Target:
    source_target: np.ndarray
    coordinate: np.ndarray


@dataclass(frozen=True)
class Operator:
    """The operator surface the adapter reads.

    The sampling target is present on every operator and is ``None`` until the
    booked current route attaches one, so a stand-in for the operator carries
    the field even where the case under test has no sampling nodes.
    """

    grid: Target
    wall: Target
    external_current: np.ndarray
    sample: Target | None = None


@dataclass(frozen=True)
class Profile:
    operator: Operator


def _shipped_vector() -> np.ndarray:
    """The build seam's vector: the shipped currents in response-column order."""

    shipped = _shipped()
    return np.asarray([shipped[name] for name in POLOIDAL_CONDUCTORS])


@pytest.mark.parametrize("changed", [False, True])
def test_shipped_comparison_is_exact_and_matches_conductor_names(changed) -> None:
    shipped = _shipped()
    names = tuple(reversed(POLOIDAL_CONDUCTORS)) + CIRCUIT_DRIVEN_CONDUCTORS
    installed = np.asarray([shipped.get(name, -1.0) for name in names])
    built = _shipped_vector()
    if changed:
        built[0] = np.nextafter(built[0], np.inf)
        with pytest.raises(ValueError, match="disagrees.*F1A"):
            diiid_current._refuse_silent_shipped_current_replacement(
                built, installed, names, POLOIDAL_CONDUCTORS
            )
    else:
        diiid_current._refuse_silent_shipped_current_replacement(
            built, installed, names, POLOIDAL_CONDUCTORS
        )


def test_complete_adapter_preserves_response_order(monkeypatch) -> None:
    shipped_count = len(POLOIDAL_CONDUCTORS)
    profile = Profile(
        Operator(
            grid=Target(
                np.arange(2 * shipped_count).reshape(2, shipped_count),
                np.asarray([[1.0, 0.0], [2.0, 0.0]]),
            ),
            wall=Target(
                np.arange(3 * shipped_count).reshape(3, shipped_count),
                np.asarray([[1.0, -1.0], [2.0, 0.0], [1.0, 1.0]]),
            ),
            external_current=_shipped_vector(),
        )
    )

    def response(entry, dd_version, names, target_r, target_z):
        del entry, dd_version, target_z
        values = np.asarray(
            [np.full(len(target_r), 100.0 + index) for index in range(len(names))]
        )
        return tuple(names), values, {"target_points": len(target_r)}

    monkeypatch.setattr(diiid_current, "active_coil_response_from_imas", response)
    adapter = diiid_current.complete_profile_current_adapter(
        profile,
        shipped_names=POLOIDAL_CONDUCTORS,
        shipped_current_a=_shipped(),
        active_coil_entry="unused.nc",
    )

    assert adapter.resolution.names == (
        *POLOIDAL_CONDUCTORS,
        *CIRCUIT_DRIVEN_CONDUCTORS,
    )
    np.testing.assert_array_equal(
        np.asarray(adapter.profile.operator.grid.source_target)[:, :shipped_count],
        profile.operator.grid.source_target,
    )
    np.testing.assert_allclose(
        np.asarray(adapter.profile.operator.grid.source_target)[:, shipped_count:],
        [[100.0, 101.0, 102.0, 103.0, 104.0]] * 2,
    )
    assert adapter.response_receipt["complete_count"] == 24
    assert adapter.response_receipt["current_authority"] == "pf_active circuit"
    assert adapter.resolution.unknown_names == ()
    assert len(adapter.response_receipt["pf_active"]["circuit"]["drives"]) == 5


FRAME = 137
BUILD_SEAM_CONDUCTOR = "F1A"
BUILD_SEAM_SHIFT_A_TURN = 54_058.0


def _corpus_137d():
    """Return the benchmark module and frame 137D's corpus row."""

    from benchmarks import diiid_forward_gs_match as bench

    paths = sorted(bench.DEFAULT_DATA.glob("*.parquet"))
    if not paths:
        pytest.skip(f"no DIII-D corpus rows under {bench.DEFAULT_DATA}")
    selected = bench.select_frames(
        paths, bench.EXECUTION_FRAME_COUNT, bench.polarity_population()
    )
    match = [item for item in selected if item.frame == FRAME]
    if len(match) != 1:
        pytest.skip(f"frame {FRAME} did not select exactly one corpus row")
    columns = (
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
    )
    return bench, bench._read(match[0].path, columns)


def _bench_shipped(bench, row):
    description = bench.dataset_machine_description(
        row, source_row=str(row.get("_source_path", "corpus row"))
    ).physical
    time_ms = float(row["efit_times"][FRAME])
    return (
        bench.shipped_current_at(row, description, POLOIDAL_CONDUCTORS, time_ms),
        time_ms,
    )


@pytest.mark.timeout(1800)
def test_complete_adapter_refuses_a_build_time_current_it_would_replace(
    monkeypatch,
) -> None:
    """Build frame 137D, patch its build-time conductor current, require a refusal.

    The benchmark stores the conductor current on the operator at the build seam
    and the adapter re-reads the corpus columns before the solve, so a current
    patched at the seam would be discarded without a word unless the adapter
    compares the two. The unpatched build is the control: it must still pass.
    """

    bench, row = _corpus_137d()
    shipped, _time_ms = _bench_shipped(bench, row)
    expansion = 0.02

    profile, _seed, _label, _wall, _reliable, _statement = bench.build_profile(
        row, FRAME, expansion
    )
    adapter = diiid_current.complete_profile_current_adapter(
        profile,
        shipped_names=POLOIDAL_CONDUCTORS,
        shipped_current_a=shipped,
    )
    np.testing.assert_array_equal(
        np.asarray(profile.operator.external_current, dtype=float),
        np.asarray([shipped[name] for name in POLOIDAL_CONDUCTORS], dtype=float),
        err_msg="the unpatched build did not carry the shipped conductor current",
    )
    installed = np.asarray(adapter.resolution.current(()), dtype=float)
    np.testing.assert_array_equal(
        installed[: len(POLOIDAL_CONDUCTORS)],
        np.asarray([shipped[name] for name in POLOIDAL_CONDUCTORS], dtype=float),
    )
    assert len(installed) == len(POLOIDAL_CONDUCTORS) + len(CIRCUIT_DRIVEN_CONDUCTORS)

    names = tuple(POLOIDAL_CONDUCTORS)
    index = names.index(BUILD_SEAM_CONDUCTOR)
    original = bench._source_current

    def patched(source):
        def seam(seam_row, description, seam_names, time_ms):
            vector = np.asarray(
                source(seam_row, description, seam_names, time_ms), dtype=float
            )
            if tuple(seam_names) == names:
                vector = vector.copy()
                vector[index] += BUILD_SEAM_SHIFT_A_TURN
            return vector

        return seam

    monkeypatch.setattr(bench, "_source_current", patched(original))
    patched_profile, *_rest = bench.build_profile(row, FRAME, expansion)
    assert (
        float(np.asarray(patched_profile.operator.external_current)[index])
        == shipped[BUILD_SEAM_CONDUCTOR] + BUILD_SEAM_SHIFT_A_TURN
    )

    with pytest.raises(ValueError, match="disagrees.*" + BUILD_SEAM_CONDUCTOR) as error:
        diiid_current.complete_profile_current_adapter(
            patched_profile,
            shipped_names=POLOIDAL_CONDUCTORS,
            shipped_current_a=shipped,
        )
    assert (
        f"build {shipped[BUILD_SEAM_CONDUCTOR] + BUILD_SEAM_SHIFT_A_TURN:.6f} A"
        in str(error.value)
    )
    assert f"resolved {shipped[BUILD_SEAM_CONDUCTOR]:.6f} A" in str(error.value)


def test_complete_adapter_refuses_a_build_current_of_the_wrong_width(
    monkeypatch,
) -> None:
    """A build vector that is not the shipped population cannot be compared."""

    shipped_count = len(POLOIDAL_CONDUCTORS)
    profile = Profile(
        Operator(
            grid=Target(
                np.zeros((2, shipped_count)), np.asarray([[0.0, 0.0], [1.0, 1.0]])
            ),
            wall=Target(
                np.zeros((2, shipped_count)), np.asarray([[0.0, 0.0], [1.0, -1.0]])
            ),
            external_current=np.zeros(shipped_count + 5),
        )
    )
    monkeypatch.setattr(
        diiid_current,
        "active_coil_response_from_imas",
        lambda entry, version, names, radius, height: (
            tuple(names),
            np.zeros((len(names), len(radius))),
            {"stub": True},
        ),
    )

    with pytest.raises(ValueError, match="build-time conductor current carries"):
        diiid_current.complete_profile_current_adapter(
            profile,
            shipped_names=POLOIDAL_CONDUCTORS,
            shipped_current_a=_shipped(),
        )


class RecordingProfile:
    def __init__(self):
        self.currents = []

    def solve(self, initial_flux, *, current, **options):
        del initial_flux, options
        self.currents.append(np.asarray(current, dtype=float))
        return self.currents[-1]


def test_outer_loop_updates_only_three_circuit_bypass_prior_slots() -> None:
    resolution = diiid_current.resolve_diiid_currents(
        POLOIDAL_CONDUCTORS,
        _shipped(),
        _circuit_bypass_priors(),
        use_circuit=False,
    )
    profile = RecordingProfile()
    target = np.asarray([11_500.0, 13_500.0, 15_500.0])
    likelihood = InferenceLikelihood(
        name="inference-available three-channel demonstration",
        provenance="banked demonstration observation",
        evaluator=lambda equilibrium, current: LikelihoodValue(
            residual=current[resolution.unknown_indices] - target,
            covariance=np.eye(3) * 100.0**2,
        ),
    )

    result = solve_conductor_currents(
        profile, np.zeros(1), resolution, likelihood=likelihood
    )

    assert result.receipt["likelihood_rank"] == 3
    assert result.receipt["posterior_status"] == "recovered"
    assert len(result.receipt["conductors"]) == 24
    assert {row["name"] for row in result.receipt["conductors"]} == set(
        resolution.names
    )
    assert {row["disposition"] for row in result.receipt["conductors"]} == {
        "prescribed",
        "predicted",
        "solved",
    }
    assert len(profile.currents) > 1
    fixed_indices = np.setdiff1d(
        np.arange(len(resolution.names)), resolution.unknown_indices
    )
    for current in profile.currents:
        np.testing.assert_allclose(
            current[fixed_indices],
            resolution.current(resolution.prior_mean_a)[fixed_indices],
        )
    assert not np.allclose(
        result.current_a[resolution.unknown_indices],
        resolution.prior_mean_a,
    )
