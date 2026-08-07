"""Whether an injected coupling comes back, and whether the guards refuse what they claim to.

The manufactured truth is a machine that is described slightly wrongly.  A sensor is
given a described response to every drive, an undescribed source is added that couples
to one drive only, and the fit has to return that coupling and nothing on the drives
it was not added to.  Zero is as much a result as the injected number: a fit that
reports a coupling to a circuit nothing was injected on has spread the fault across
the design.

The joint fit gets the case that says why it is joint.  Two correlated drives, a
coupling injected on one of them, and a fit taken one drive at a time returns a
coupling on both -- each absorbing what the column it omitted would have taken.  The
joint fit returns the injected number on one and zero on the other, from the same data.

Each guard is tested by the fit it exists to refuse: a circuit that barely moved, a
pair whose waveforms are one waveform, and a design whose columns cannot be told
apart.  The last two look alike and are not: a merged pair is reported honestly as a
pair, while a design still ill-conditioned after merging is refused outright.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.calibrate.coupling import (
    CONDITION_LIMIT,
    CouplingError,
    active_columns,
    baseline_free_residual,
    build_drive_block,
    joint_drive_fit,
    pool_couplings,
    scaled_condition,
)

SAMPLES = 512
"""Samples per manufactured pulse."""

DRIVES = ("inner_lower", "inner_upper", "outer_lower", "outer_upper")
"""Four circuits, named the way an adapter would name them."""


def waveforms(*, correlated: bool = False, quiet: int = 40) -> np.ndarray:
    """Return one pulse of ampere-turn histories, held at zero over a quiet window."""

    time = np.linspace(0.0, 1.0, SAMPLES)
    columns = []
    for index in range(len(DRIVES)):
        shape = np.sin((1.3 + 0.7 * index) * np.pi * time) * (8.0e3 + 2.0e3 * index)
        if correlated and index == 1:
            shape = columns[0] * 0.97 + 40.0 * np.sin(31.0 * time)
        columns.append(shape)
    drive = np.column_stack(columns)
    drive[:quiet, :] = 0.0
    return drive


def test_a_residual_is_what_the_described_response_does_not_predict():
    drive = waveforms()
    response = np.asarray([2.0e-6, -1.5e-6, 8.0e-7, 3.0e-7])
    baseline = np.zeros(SAMPLES, dtype=bool)
    baseline[:40] = True
    extra, offset = 5.0e-8, 0.031
    signal = drive @ response + extra * drive[:, 2] + offset
    residual = baseline_free_residual(
        signal, drive, response, baseline_mask=baseline
    )
    assert np.allclose(residual, extra * drive[:, 2], atol=1e-12)


def test_a_response_of_the_wrong_width_is_refused():
    with pytest.raises(CouplingError, match="cannot be contracted"):
        baseline_free_residual(np.zeros(SAMPLES), waveforms(), np.zeros(3))


def test_only_the_circuits_that_swept_are_admitted():
    drive = waveforms()
    drive[:, 3] *= 1.0e-4
    admitted = active_columns(drive, floor=2000.0, share=0.05)
    assert admitted == (0, 1, 2)


def test_a_circuit_below_the_absolute_floor_is_dropped_however_alone_it_is():
    drive = np.zeros((SAMPLES, 1))
    drive[:, 0] = np.linspace(0.0, 100.0, SAMPLES)
    assert active_columns(drive, floor=2000.0, share=0.0) == ()


def test_a_pair_driven_together_is_merged_and_named_as_a_pair():
    drive = waveforms(correlated=True)
    block = build_drive_block(
        drive,
        DRIVES,
        merge_groups={("inner_lower", "inner_upper"): "inner_pair"},
    )
    assert "inner_pair" in block.names
    assert "inner_lower" not in block.names and "inner_upper" not in block.names
    assert block.merged == (("inner_lower", "inner_upper"),)
    assert block.conditioned


def test_a_pair_driven_independently_stays_two_columns():
    block = build_drive_block(
        waveforms(),
        DRIVES,
        merge_groups={("inner_lower", "inner_upper"): "inner_pair"},
    )
    assert "inner_pair" not in block.names
    assert {"inner_lower", "inner_upper"} <= set(block.names)


def test_merging_is_what_saves_a_pair_pulse_from_the_conditioning_guard():
    drive = waveforms(correlated=True)
    unmerged = build_drive_block(drive, DRIVES)
    merged = build_drive_block(
        drive, DRIVES, merge_groups={("inner_lower", "inner_upper"): "inner_pair"}
    )
    assert not unmerged.conditioned and unmerged.condition > CONDITION_LIMIT
    assert merged.conditioned


def test_the_condition_number_is_taken_after_scaling_each_column():
    """A kiloampere circuit beside a ten-ampere one is not ill-conditioned."""

    time = np.linspace(0.0, 1.0, SAMPLES)
    block = np.column_stack([1.0e4 * np.sin(3.0 * time), 10.0 * np.cos(5.0 * time)])
    assert scaled_condition(block) < 5.0
    assert np.linalg.cond(block) > 100.0


def test_an_injected_coupling_comes_back_and_the_others_come_back_at_zero():
    drive = waveforms()
    block = build_drive_block(drive, DRIVES)
    injected = 3.7e-8
    residual = injected * drive[:, block.names.index("outer_lower")]
    fit = joint_drive_fit(residual, block, channel="p01")
    assert fit.coupling("outer_lower") == pytest.approx(injected, rel=1e-9)
    for name in set(block.names) - {"outer_lower"}:
        assert fit.coupling(name) == pytest.approx(0.0, abs=1e-16)
    assert fit.variance_explained == pytest.approx(1.0, abs=1e-12)


def test_fitting_one_drive_at_a_time_spreads_the_coupling_and_the_joint_fit_does_not():
    """Two drives that partly track each other, with the coupling on only one.

    A fit taken against the second column alone returns most of the injected number,
    because nothing in that fit knows the first column exists and the second is
    correlated enough to stand in for it.  The joint fit, on the same samples, puts the
    coupling on the column it was injected on and zero on the other.
    """

    time = np.linspace(0.0, 1.0, SAMPLES)
    first = 9.0e3 * np.sin(2.0 * np.pi * time)
    second = 0.75 * first + 6.0e3 * np.cos(5.0 * np.pi * time)
    drive = np.column_stack([first, second])
    names = ("inner_lower", "inner_upper")
    injected = 4.0e-8
    residual = injected * first

    marginal = float(second @ residual / (second @ second))
    block = build_drive_block(drive, names)
    assert block.conditioned
    assert 0.4 * injected < marginal < 0.9 * injected

    fit = joint_drive_fit(residual, block)
    assert fit.coupling("inner_lower") == pytest.approx(injected, rel=1e-9)
    assert fit.coupling("inner_upper") == pytest.approx(0.0, abs=1e-17)


def test_an_ill_conditioned_block_is_refused_rather_than_reported():
    block = build_drive_block(waveforms(correlated=True), DRIVES)
    with pytest.raises(CouplingError, match="too nearly collinear"):
        joint_drive_fit(np.zeros(SAMPLES), block)


def test_a_channel_with_too_few_finite_samples_says_nothing():
    drive = waveforms()
    block = build_drive_block(drive, DRIVES)
    residual = np.full(SAMPLES, np.nan)
    residual[:100] = drive[:100, 0] * 1.0e-8
    assert joint_drive_fit(residual, block) is None


def test_a_channel_with_gaps_is_fitted_on_the_samples_it_has():
    drive = waveforms()
    block = build_drive_block(drive, DRIVES)
    injected = 2.2e-8
    residual = injected * drive[:, 2]
    residual[::7] = np.nan
    fit = joint_drive_fit(residual, block)
    assert fit.sample_count < SAMPLES
    assert fit.coupling("outer_lower") == pytest.approx(injected, rel=1e-8)


def test_pooling_is_a_median_so_one_wrong_pulse_costs_one_vote():
    rows = [
        np.asarray([1.0, 2.0]),
        np.asarray([1.1, 2.1]),
        np.asarray([1.05, 2.05]),
        np.asarray([50.0, 2.02]),
    ]
    pooled, counts = pool_couplings(rows)
    assert pooled[0] == pytest.approx(1.075)
    assert pooled[1] == pytest.approx(2.035)
    assert list(counts) == [4, 4]


def test_a_channel_backed_by_one_pulse_pools_to_nothing():
    rows = [np.asarray([1.0, np.nan]), np.asarray([1.2, 5.0])]
    pooled, counts = pool_couplings(rows, minimum_fits=2)
    assert pooled[0] == pytest.approx(1.1)
    assert np.isnan(pooled[1])
    assert list(counts) == [2, 1]
