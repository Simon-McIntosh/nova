"""Guards on the reduction, the control and the arm-selection rule.

The gate this study ships re-reads the summary the production reduction wrote,
so a summary that has drifted from the rows it claims to reduce would pass a
gate that reads only the summary.  These tests exercise the functions that
produce those numbers -- the lag-one estimate, the two error measures, the
noise generator and the arm-selection rule -- so a change to any of them fails
here rather than silently changing a table.

They also pin the row-set rule: both codes reconstruct from one row set, so an
arm selection that would compare two information sets is a failure, not a
choice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyse  # noqa: E402
import check  # noqa: E402
import study  # noqa: E402


def rows_of(axis, flux_boundary=3.0, q95=3.0, converged=True):
    return [
        dict(
            slice=index,
            axis=[float(value[0]), float(value[1])],
            flux_boundary=float(flux_boundary),
            q95=float(q95),
            plasma_current=-1.0e7,
            converged=converged,
        )
        for index, value in enumerate(axis)
    ]


def test_lag_one_reads_an_independent_sequence_as_zero():
    rng = np.random.default_rng(11)
    value = rng.standard_normal(4000)
    assert abs(analyse.lag_one(value)) < 0.1


def test_lag_one_recovers_a_first_order_process():
    rng = np.random.default_rng(12)
    value = np.zeros(4000)
    for index in range(1, value.size):
        value[index] = 0.9 * value[index - 1] + rng.standard_normal()
    assert abs(analyse.lag_one(value) - 0.9) < 0.05


def test_lag_one_refuses_a_constant_stream():
    assert np.isnan(analyse.lag_one(np.ones(100)))


def test_series_holds_an_absent_quantity_as_nan():
    rows = [
        dict(axis=[1.0, 2.0], flux_boundary=None, q95=1e31, plasma_current=0.0),
        dict(axis=[1.0, 2.0], flux_boundary=3.0, q95=2.0, plasma_current=0.0),
    ]
    values = analyse.series(rows)
    assert np.isnan(values["flux"][0])
    assert np.isnan(values["q95"][0])
    assert np.isfinite(values["q95"][1])


def test_rms_and_block_error_average_the_errors_they_name():
    truth = np.array([0.0, 0.0])
    value = np.array([[0.003, 0.004], [0.003, 0.004], [0.003, 0.004], [0.003, 0.004]])
    assert analyse.rms_error(value, truth) == pytest.approx(0.005)
    assert analyse.block_error(value, truth, block=2) == pytest.approx(0.005)


def test_noise_sequences_carry_the_control_and_the_drift():
    sigma = np.full(72, 1.0)
    white = study.noise_columns("white", 200, sigma, 7, 1.0)
    drift = study.noise_columns("drift", 200, sigma, 7, 1.0)
    assert abs(analyse.lag_one(white)) < 0.15
    assert analyse.lag_one(drift) > 0.5
    # The colouring redistributes power; it must not change the noise size.
    assert np.std(white) == pytest.approx(np.std(drift), rel=0.25)


def test_current_draw_carries_the_sequence_colour_and_keeps_its_size():
    sigma_ip = 1.5e5
    white = study.current_draw("white", 200, sigma_ip, 7)
    drift = study.current_draw("drift", 200, sigma_ip, 7)
    # The current is a measurement row, so it must carry the sequence's own
    # memory: a white draw on the drift sequence is the unmodelled per-slice
    # jump that decorrelates the axis position the control measures.
    assert abs(analyse.lag_one(white)) < 0.15
    assert analyse.lag_one(drift) > analyse.DRIFT_LAG_ONE_FLOOR
    assert np.std(drift) == pytest.approx(sigma_ip, rel=0.25)
    assert np.std(white) == pytest.approx(sigma_ip, rel=0.25)


def fake_case(sensors=72):
    """A case small enough to iterate: only the keys run_nova's sequence loop reads."""
    from types import SimpleNamespace

    return dict(
        reference=dict(
            plasma_current=-1.4e7,
            q95=3.0,
            b0=2.5,
            axis=[6.0, 0.5],
            flux_boundary=3.0,
            flux_span=77.0,
            flux_axis=-70.0,
        ),
        generated=SimpleNamespace(
            terminal_state=SimpleNamespace(
                topology=SimpleNamespace(
                    axis=[6.0, 0.5],
                    axis_flux=-70.0,
                    boundary_flux=3.0,
                    flux_span=77.0,
                )
            )
        ),
        sigma_loop=1.0e-4,
        sigma_probe=1.0e-4,
        total=np.zeros(sensors),
        profile=None,
        initial=None,
        sensors=None,
        magnetic_source=None,
        basis=None,
        request=None,
        vacuum=None,
    )


def nova_arguments(**over):
    from types import SimpleNamespace

    values = dict(
        cells=500,
        reuse_rows=True,
        shard=0,
        shards=1,
        factor=0.01,
        sequences="white,drift",
        slices=200,
        seed=7,
        rows="all",
        jacobian=False,
        skip_sweep=True,
        sweep_factors="",
        sweep_part="",
        sweep_slices=0,
        index_sweep="",
        corner_sweep="",
    )
    values.update(over)
    return SimpleNamespace(**values)


def run_nova_over_a_fake_card(monkeypatch, tmp_path, seed=7):
    """Drive run_nova's sequence loop with the draw left real; return its file.

    Everything the loop reaches before the fit is left real -- the noise
    columns, the
    plasma-current draw and the record it writes -- so the current the file
    carries is the one the production call site produced.  The case, the fit,
    the computed map and the metrics are replaced because this guard is about
    which current reaches the record, not about the reconstruction.
    """
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace

    class FakeFit:
        def __init__(self, *args, **kwargs):
            pass

    fake_fit = FakeFit()
    fake_fit.fit = lambda *args, **kwargs: SimpleNamespace(
        forward=SimpleNamespace(terminal_state=None)
    )

    module = ModuleType("nova.equilibrium.magnetics_fit")
    module.MagneticsFit = lambda *args, **kwargs: fake_fit
    for name in ("nova", "nova.equilibrium", "nova.equilibrium.magnetics_fit"):
        monkeypatch.setitem(
            sys.modules, name, module if name.endswith("fit") else ModuleType(name)
        )

    folder = tmp_path / ("seed%d" % seed)
    folder.mkdir(exist_ok=True)
    monkeypatch.setattr(study, "STUDY", folder)
    monkeypatch.setattr(study, "serial_pools", lambda: None)
    monkeypatch.setattr(study, "build_case", lambda *args, **kwargs: fake_case())
    monkeypatch.setattr(study, "write_prepare", lambda case: None)
    monkeypatch.setattr(study, "q95_of", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        study,
        "nova_flux_map",
        lambda case, state: SimpleNamespace(
            flux=np.zeros((2, 2)),
            topology=SimpleNamespace(axis=np.array([6.0, 0.5])),
        ),
    )
    monkeypatch.setattr(study, "nova_metrics", lambda case, fitted: dict(slice=0))
    monkeypatch.setattr(study, "guard", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(study, "land", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        study,
        "fitted_measurement",
        lambda case, measured, sigma, current, mask=None: None,
    )
    study.run_nova(nova_arguments(seed=seed))
    return folder / "nova-drift-200-0.0100.jsonl"


def recorded_currents(path):
    """The plasma current each written slice carries, in slice order."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return np.array([row["plasma_current_measured"] for row in rows])


SEEDS = (7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def mean_recorded_lag(monkeypatch, tmp_path):
    """The mean lag-one of the recorded current over several seeds.

    One realisation of a coloured draw moves about 0.1 around its own mean, so
    a single seed can sit either side of a floor by chance; the mean over ten
    seeds is the statistic the guard states.
    """
    lags = [
        analyse.lag_one(
            recorded_currents(run_nova_over_a_fake_card(monkeypatch, tmp_path, seed))
        )
        for seed in SEEDS
    ]
    return float(np.mean(lags))


def test_run_nova_records_a_current_draw_that_holds_the_sequence_memory(
    monkeypatch, tmp_path
):
    """The call site, not just the generator, must colour the current row.

    ``current_draw`` can be correct while ``run_nova`` asks for a different
    sequence or drops the draw; this drives the loop that writes the record and
    reduces the current read back out of the file it produced.
    """
    path = run_nova_over_a_fake_card(monkeypatch, tmp_path)
    assert recorded_currents(path).size == 200
    assert mean_recorded_lag(monkeypatch, tmp_path) > 0.53


def test_run_nova_guard_catches_a_white_current_draw(monkeypatch, tmp_path):
    """The negative control: reinstating a white current draw fails the guard.

    A guard that cannot fail proves nothing, so this run replaces the draw with
    the white Gaussian the defect used and requires the same reduction to fall
    inside the white control instead of holding the drift memory.
    """
    monkeypatch.setattr(
        study,
        "current_draw",
        lambda sequence, slices, sigma, seed, **kwargs: (
            sigma * np.random.default_rng(seed).standard_normal(slices)
        ),
    )
    lag = mean_recorded_lag(monkeypatch, tmp_path)
    assert abs(lag) < analyse.LAG_ONE_LIMIT
    assert lag <= analyse.DRIFT_LAG_ONE_FLOOR


def test_thresholds_refuse_a_sequence_made_from_white():
    rng = np.random.default_rng(3)
    white = rng.standard_normal((600, 2)) * 1e-3
    lag = analyse.lag_one(white)
    white_arm = dict(axis_lag1=lag, flux_lag1=lag, q95_lag1=lag)
    # A drift column that is white in all but its name is exactly the case the
    # floor exists to catch: it passes the white clause and fails the drift one.
    verdict = analyse.thresholds((white_arm, white_arm))
    assert verdict["white_lag1_pass"] is True
    assert verdict["drift_lag1_pass"] is False


def test_thresholds_accept_a_held_drift_sequence():
    rng = np.random.default_rng(4)
    white = rng.standard_normal((600, 2)) * 1e-3
    drift = np.zeros((600, 2))
    for index in range(1, 600):
        drift[index] = 0.85 * drift[index - 1] + rng.standard_normal(2)
    verdict = analyse.thresholds(
        (
            dict(
                axis_lag1=analyse.lag_one(white),
                flux_lag1=analyse.lag_one(white),
                q95_lag1=analyse.lag_one(white),
            ),
            dict(
                axis_lag1=analyse.lag_one(drift),
                flux_lag1=analyse.lag_one(drift),
                q95_lag1=analyse.lag_one(drift),
            ),
        )
    )
    assert verdict["white_lag1_pass"] is True
    assert verdict["drift_lag1_pass"] is True


def test_reduce_arm_reports_the_distance_from_the_generating_axis():
    truth = dict(axis=[6.0, 0.5], flux_boundary=3.0, flux_span=77.0, q95=3.0)
    rows = rows_of([[6.003, 0.504]] * 8)
    arm = analyse.reduce_arm(rows, truth)
    assert arm["slices"] == 8
    assert arm["axis_rms"] == pytest.approx(0.005)
    assert arm["slices_with_boundary"] == 8


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))


def test_arm_selection_takes_the_row_set_the_code_declares(tmp_path, monkeypatch):
    monkeypatch.setattr(analyse, "STUDY", tmp_path)
    fuller = rows_of([[6.0, 0.5]] * 5)
    loops = rows_of([[6.0, 0.5]] * 3)
    for code in ("nova", "efit"):
        write_rows(tmp_path / ("%s-white-200-0.0100.jsonl" % code), fuller)
        write_rows(tmp_path / ("%s-white-200-0.0100-loops-shard0.jsonl" % code), loops)
    assert analyse.DECLARED_ROWS == {"nova": "all", "efit": "loops"}
    assert len(analyse.selected_rows("nova", "white", 200, 0.01)) == 5
    assert len(analyse.selected_rows("efit", "white", 200, 0.01)) == 3


def test_row_matched_arm_is_available_by_name(tmp_path, monkeypatch):
    import figures

    monkeypatch.setattr(analyse, "STUDY", tmp_path)
    monkeypatch.setattr(figures, "STUDY", tmp_path)
    write_rows(tmp_path / "nova-drift-200-0.0100.jsonl", rows_of([[6.0, 0.5]] * 4))
    write_rows(
        tmp_path / "nova-drift-200-0.0100-loops.jsonl", rows_of([[6.0, 0.5]] * 3)
    )
    assert len(figures.arm_rows("nova", "drift")) == 4
    assert len(figures.arm_rows("nova", "drift", "loops")) == 3


def test_grouped_refuses_a_slice_one_arm_carries_twice(tmp_path):
    """A record left at the arm's own name must not be merged into the arm.

    An independent fit of a slice already carried by a shard of the same stem
    double-weights that slice in every statistic the arm reports, so the
    reduction refuses the arm and names both files rather than reducing it.
    """
    rows = rows_of([[6.0, 0.5]] * 3)
    write_rows(tmp_path / "nova-white-200-0.0100-shard0.jsonl", rows)
    write_rows(tmp_path / "nova-white-200-0.0100.jsonl", rows[:1])
    with pytest.raises(ValueError, match="repeats slice 0"):
        analyse.grouped(sorted(tmp_path.glob("nova-white-200-0.0100*.jsonl")))


def test_grouped_merges_shards_that_do_not_repeat_a_slice(tmp_path):
    rows = rows_of([[6.0, 0.5]] * 4)
    write_rows(tmp_path / "nova-white-200-0.0100-shard0.jsonl", rows[:2])
    write_rows(tmp_path / "nova-white-200-0.0100-shard1.jsonl", rows[2:])
    runs = analyse.grouped(sorted(tmp_path.glob("nova-white-200-0.0100*.jsonl")))
    assert [row["slice"] for row in runs["nova-white-200-0.0100"]] == [0, 1, 2, 3]


def test_arm_selection_names_each_code_row_set(tmp_path, monkeypatch):
    assert analyse.arm_row_set("nova-white-200-0.0100") == "all"
    assert analyse.arm_row_set("nova-white-200-0.0100-loops") == "loops"
    assert analyse.declared_stem("nova", "white") == "nova-white-200-0.0100"
    assert analyse.declared_stem("efit", "white") == "efit-white-200-0.0100-loops"


def test_arm_selection_reports_missing_rather_than_comparing_row_sets(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(analyse, "STUDY", tmp_path)
    write_rows(tmp_path / "efit-white-200-0.0100.jsonl", rows_of([[6.0, 0.5]] * 4))
    assert analyse.selected_rows("efit", "white", 200, 0.01) == []


def complete_payload(white_lag=0.05, drift_lag=0.8, slices=200, row_set=None):
    def arm(code, lag):
        return dict(
            slices=slices,
            slices_with_boundary=slices,
            axis_rms=0.001,
            flux_rms=0.001,
            q95_rms=0.01,
            axis_lag1=lag,
            flux_lag1=lag,
            q95_lag1=lag,
            row_set=analyse.DECLARED_ROWS[code] if row_set is None else row_set,
        )

    control = {}
    for code in ("nova", "efit"):
        control[code] = dict(
            white=arm(code, white_lag),
            drift=arm(code, drift_lag),
            white_lag1_pass=abs(white_lag) <= analyse.LAG_ONE_LIMIT,
            drift_lag1_pass=drift_lag > analyse.DRIFT_LAG_ONE_FLOOR,
        )
    return dict(control=control)


def test_gate_counts_only_its_assertions():
    failures, performed, lines = check.check(complete_payload())
    assert failures == []
    assert performed == 26
    assert len(lines) == 4


def test_gate_fails_an_arm_that_holds_another_row_set():
    failures, _, _ = check.check(complete_payload(row_set="loops"))
    assert any("not the declared" in line for line in failures)


def test_gate_fails_a_drift_arm_that_does_not_hold_its_memory():
    payload = complete_payload(drift_lag=0.46)
    failures, performed, _ = check.check(payload)
    assert failures
    assert performed == 26


def test_gate_fails_a_short_arm():
    failures, _, _ = check.check(complete_payload(slices=120))
    assert any("short of 200" in message for message in failures)
