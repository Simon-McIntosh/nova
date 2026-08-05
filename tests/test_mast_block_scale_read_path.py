"""Whether the correction reaches the consumers, and reports itself when it does.

The block table is only worth having if a fit that never heard of it reads corrected
signals, so these cases go through the readers rather than through the table.  Two
readers matter: the one the vacuum fits use, and the one that serves the solve-input
map.  Each is checked on a store written here, so the case says what the reader does
rather than what one archive happens to contain.

What is being defended is the pair of opposite mistakes.  A reader that forgets the
table leaves a channel reading twice the field it saw, and every fit pooling that
channel across the switch inherits a phantom campaign dependence.  A reader that
applies it silently is worse, because a later reader cannot tell a corrected channel
from an uncorrected one and the second correction is invisible.  So every read carries
its corrections back, and the raw path stays reachable by naming an empty table.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.imas.mast_block_scale import (
    MEASURED,
    UNMEASURED,
    BlockScale,
    BlockScaleTable,
)
from nova.imas.mast_error_field_screen import read_probe_signals
from nova.imas.mast_vacuum_cohort import read_shot_waveforms

SHOT = 14061
"""A shot number inside the doubled block the store below is written for."""

SAMPLES = 64
"""Enough samples that the reader's window tests have something to run on."""


@pytest.fixture
def store(tmp_path):
    """Write one shot carrying two probe channels and one driven coil."""

    import zarr

    root = tmp_path / "shots"
    group = zarr.open_group(f"{root}/{SHOT}.zarr", mode="w")
    time = np.linspace(0.0, 0.1, SAMPLES)
    fields = group.create_group("amb")
    fields["time"] = time
    fields["obr02"] = np.full(SAMPLES, 4.0)
    fields["obv06"] = np.full(SAMPLES, 4.0)
    currents = group.create_group("amc")
    currents["time"] = time
    currents["p4u_case_current"] = np.zeros(SAMPLES)
    currents["plasma_current"] = np.zeros(SAMPLES)
    return root


@pytest.fixture
def doubled():
    """A table saying obr02 was recorded at twice its ordinary range on this shot."""

    return BlockScaleTable.create(
        [
            BlockScale("obr02", 1.985, (SHOT,), rung=2.0),
        ],
        route="far-field response ratio on plasma-free shots",
    )


def test_the_vacuum_reader_divides_the_doubled_channel_and_leaves_the_other(
    store, doubled
):
    """One shot's channels are corrected independently, as the defect is."""

    waveforms = read_shot_waveforms(SHOT, store=store, block_scale=doubled)
    assert waveforms.probes["obr02"].tolist() == [2.0] * SAMPLES
    assert waveforms.probes["obv06"].tolist() == [4.0] * SAMPLES


def test_the_vacuum_reader_reports_what_it_divided_and_what_it_did_not(store, doubled):
    """A silent correction is worse than none, because a second one is invisible."""

    waveforms = read_shot_waveforms(SHOT, store=store, block_scale=doubled)
    assert waveforms.scaled_channels == ("obr02",)
    assert waveforms.unscaled_channels == ("obv06",)
    dispositions = {row.channel: row.disposition for row in waveforms.scale_corrections}
    assert dispositions == {"obr02": MEASURED, "obv06": UNMEASURED}


def test_an_empty_table_keeps_the_vacuum_reader_on_the_raw_archive(store):
    """The route the sweep that measures the settings has to read on."""

    waveforms = read_shot_waveforms(SHOT, store=store, block_scale=BlockScaleTable())
    assert waveforms.probes["obr02"].tolist() == [4.0] * SAMPLES
    assert waveforms.scaled_channels == ()
    assert waveforms.unscaled_channels == ("obr02", "obv06")


def test_the_correction_survives_the_window_tests_the_reader_applies(store, doubled):
    """Masking samples must not put the raw channel back."""

    waveforms = read_shot_waveforms(
        SHOT, store=store, block_scale=doubled, quiescent_ramp_fraction=0.1
    )
    kept = waveforms.probes["obr02"][waveforms.sample_mask]
    assert kept.size > 0
    assert np.allclose(kept, 2.0)


def test_the_screen_reader_serves_field_rather_than_range(store, doubled):
    """A coupling slope measured on a doubled channel would be twice too steep."""

    signals = read_probe_signals(SHOT, store=store, block_scale=doubled)
    assert signals["obr02"].tolist() == [2.0] * SAMPLES
    assert signals["obv06"].tolist() == [4.0] * SAMPLES


def test_the_screen_reader_stays_raw_on_an_empty_table(store):
    signals = read_probe_signals(SHOT, store=store, block_scale=BlockScaleTable())
    assert signals["obr02"].tolist() == [4.0] * SAMPLES
