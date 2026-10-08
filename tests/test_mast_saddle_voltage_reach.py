"""Two contracts the saddle-loop reach benchmark rests on.

The benchmark admits a whole-store cohort through
:func:`nova.imas.mast_vacuum_cohort.light_census` rather than re-declaring the
cohort's thresholds, so the census must judge a shot exactly as the full survey
does.  And it resolves each loop's raw key from the store's own key set, because
the ``m`` family spells its tenth to twelfth loops with a three-digit index while
every other loop is two-digit; a single width reports those three loops absent
when the store carries them.

Both contracts are stated on data written here, so each case says what the code
does rather than what one archive happens to contain.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.mast_saddle_voltage_reach import _resolve_raw, loop_identities
from nova.imas.mast_vacuum_cohort import (
    MINIMUM_PROBES,
    PLASMA_FREE_CURRENT,
    light_census,
    select_vacuum_cohort,
    survey_shot,
)

SAMPLES = 16
"""Enough samples that the reader's window tests have something to run on."""

DRIVE_CHANNEL = "p3u_feed_current"
"""The excitation channel of the ``p3_upper`` coil, driven on every shot below."""

PROBES = tuple(f"ccbv{number:02d}" for number in range(1, MINIMUM_PROBES + 1))
"""A probe set just large enough to over-determine the fit."""


def write_shot(
    root,
    shot: int,
    *,
    plasma_peak: float = 0.0,
    drive_peak: float = 20.0e3,
    probes: tuple[str, ...] = PROBES,
) -> None:
    """Write one synthetic shot carrying a current group and a field group.

    Currents are written in kiloamperes, as the store records them, and the
    relevant peak is passed in amperes so the case reads against the thresholds
    the cohort module holds.
    """

    import zarr

    group = zarr.open_group(f"{root}/{shot}.zarr", mode="w")
    time = np.linspace(0.0, 0.1, SAMPLES)
    currents = group.create_group("amc")
    currents["time"] = time
    currents["plasma_current"] = np.full(SAMPLES, plasma_peak / 1.0e3)
    currents[DRIVE_CHANNEL] = np.full(SAMPLES, drive_peak / 1.0e3)
    fields = group.create_group("amb")
    fields["time"] = time
    for name in probes:
        fields[name] = np.full(SAMPLES, 0.01)


@pytest.fixture
def store(tmp_path):
    """Five synthetic shots spanning every admission outcome.

    One admitted; one carrying plasma; one with too few probes; one with no
    deliberate excitation; and one more admitted shot so the split has something
    to hold out.
    """

    root = tmp_path / "shots"
    write_shot(root, 9001, plasma_peak=0.0, drive_peak=2.0e4)
    write_shot(root, 9002, plasma_peak=PLASMA_FREE_CURRENT * 2.0, drive_peak=2.0e4)
    write_shot(root, 9003, plasma_peak=0.0, drive_peak=2.0e4, probes=PROBES[:10])
    write_shot(root, 9004, plasma_peak=0.0, drive_peak=0.0)
    write_shot(root, 9005, plasma_peak=0.0, drive_peak=2.0e4)
    return root


# --- raw-key resolution --------------------------------------------------


def test_each_family_resolves_at_its_own_key_width():
    """The ``m`` family resolves at three digits where every other family does not."""

    keys = {
        "sad_out_m010",
        "sad_out_m011",
        "sad_out_m012",
        "sad_out_l12",
        "sad_out_u01",
    }
    loops = {loop["loop"]: loop for loop in loop_identities()}
    assert _resolve_raw(keys, loops["saddle_m_10"]) == "sad_out_m010"
    assert _resolve_raw(keys, loops["saddle_m_11"]) == "sad_out_m011"
    assert _resolve_raw(keys, loops["saddle_m_12"]) == "sad_out_m012"
    assert _resolve_raw(keys, loops["saddle_l_12"]) == "sad_out_l12"
    assert _resolve_raw(keys, loops["saddle_u_1"]) == "sad_out_u01"


def test_a_loop_absent_from_the_store_resolves_to_none():
    """A key the store does not carry is reported absent, not guessed at."""

    loops = {loop["loop"]: loop for loop in loop_identities()}
    assert _resolve_raw({"sad_out_m010"}, loops["saddle_l_12"]) is None


# --- light admission agreement -------------------------------------------


def test_light_census_reads_the_same_admission_inputs_as_the_full_survey(store):
    """The cheap census reproduces the survey's thresholds and probe count."""

    for shot in (9001, 9002, 9003, 9004):
        full = survey_shot(shot, store=store)
        light = light_census(shot, store=store)
        assert light is not None
        assert light.plasma_current_peak == pytest.approx(full.plasma_current_peak)
        assert light.excited_families == full.excited_families
        assert light.probe_count == len(full.field_channels)


def test_light_admission_agrees_with_the_full_predicate(store):
    """Every shot is admitted or refused identically by the two paths."""

    full = select_vacuum_cohort(
        [survey_shot(shot, store=store) for shot in (9001, 9002, 9003, 9004, 9005)]
    )
    admitted = set(full.training) | set(full.held_out)
    assert admitted == {9001, 9005}
    for shot in (9001, 9002, 9003, 9004, 9005):
        light = light_census(shot, store=store)
        assert light is not None
        assert light.admitted() is (shot in admitted)


def test_a_shot_with_no_readable_current_group_is_refused(store):
    """A shot the census cannot read is refused rather than admitted on silence."""

    assert light_census(4242, store=store) is None