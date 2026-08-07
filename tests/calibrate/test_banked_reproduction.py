"""Whether the numbers the published record states still come out of the package.

The kernels here were migrated from one-shot scripts that had already produced
results, and those results are written down.  A migration that changes an answer while
keeping every unit test green is the failure mode worth guarding against, so a small
slice of the banked arrays is committed beside these tests and the published numbers
are asserted against what the package computes from it now.

``data/banked_couplings.npz`` holds the sensor geometry and axes, the described
response of thirteen conductors, each sensor's measured noise floor, and the per-pulse
signed couplings for two drive groups.  Nothing in it needs a data mount, and none of
these tests reads one.

Tolerances are tight on purpose.  The spectrum is asserted to a relative part in
ten billion because it is a pure linear-algebra function of a committed matrix and
nothing about it should move at all; the fitted results are asserted to the precision
the record itself quotes them at.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nova.calibrate.correction_model import CorrectionKind
from nova.calibrate.correction_set import read_correction_set
from nova.calibrate.corrections import apply_corrections, build_chain
from nova.calibrate.coupling import pool_couplings
from nova.calibrate.gain import pool_scalar_gains
from nova.calibrate.inversion import identifiability, whiten
from nova.calibrate.localize import (
    filament_scan,
    span_projector,
    surviving_fraction,
)

BANKED = Path(__file__).parent / "data" / "banked_couplings.npz"

SINGULAR_VALUES = (
    0.0360013605365339,
    0.03216931213054753,
    0.025943083209571813,
    0.023716974015707864,
    0.01346715072155109,
    0.012668535796502706,
    0.009064150752718783,
    0.007361934119861648,
    0.004648841096438636,
    0.004060403625103204,
    0.0003475166875313702,
    0.00012463659752907235,
    9.711752739981491e-05,
)
"""The whitened spectrum of the described response over the thirteen conductors."""

CONDITION_NUMBER = 370.6988995747693
"""What the same spectrum's extremes divide to."""

EXCLUDED_CHANNEL = "obr17"
"""The one sensor held out of every pooled fit, its own gain being under adjudication."""

SCAN_RADIUS = np.linspace(0.15, 2.4, 91)
SCAN_HEIGHT = np.linspace(-2.4, 2.4, 129)
"""The grid the published localization was scanned on."""

POOLED = {
    "p4_lower": {
        "sensors": 76,
        "surviving_rms": 3.698e-08,
        "surviving_share": 0.786,
        "largest": ("ccbv22", -243.7),
        "peak_radius": 0.20,
        "peak_height": -0.15,
        "peak_score": 0.601,
        "peak_current": -21.08e-3,
        "convention_ratio": -0.498,
    },
    "p5_upper": {
        "sensors": 76,
        "surviving_rms": 2.653e-08,
        "surviving_share": 0.841,
        "largest": ("ccbv10", -165.7),
        "peak_radius": 0.15,
        "peak_height": 0.75,
        "peak_score": 0.558,
        "peak_current": 43.28e-3,
        "convention_ratio": -0.511,
    },
}
"""What the record states for the two drive groups the banked slice carries.

``surviving_rms`` is in tesla per ampere turn, ``largest`` names the sensor carrying
the most surviving misfit and its value in nanotesla per ampere turn, and
``peak_current`` is the amperes per ampere turn a single filament at the peak would
carry.  ``convention_ratio`` is the excluded sensor's own pooled coupling against its
described response, which reads its gain off data no gain fit contributed to.
"""

FAMILY_GAINS = (
    0.5131900663217215,
    0.48883010026519635,
    0.5026701501895766,
    0.4996370417109336,
    0.49976805756277465,
    0.4892471999752461,
)
"""The excluded sensor's independently fitted scale, one per driving conductor."""

POOLED_FAMILY_GAIN = 0.49889043600424143
"""What those six pool to."""

PROMOTED_GAIN = 0.5011
"""The scale the correction document applies to that sensor on the read path."""


@pytest.fixture(scope="module")
def banked():
    """Return the committed slice of the banked calibration arrays."""

    with np.load(BANKED) as data:
        return {key: data[key] for key in data.files}


def sensor_names(banked):
    """Return the sensor channel names in the order the arrays carry them."""

    return [str(name) for name in banked["channel"]]


def pooled_group(banked, group, *, minimum_fits=2):
    """Return one drive group's pooled coupling over the pulses that measured it."""

    groups = [str(name) for name in banked["group"]]
    rows = [banked["coupling"][index] for index, g in enumerate(groups) if g == group]
    return pool_couplings(rows, minimum_fits=minimum_fits)[0]


def surviving(banked, group):
    """Return the pooled coupling, the sensors kept, and what the span cannot absorb."""

    pooled = pooled_group(banked, group)
    names = sensor_names(banked)
    keep = np.isfinite(pooled) & np.asarray(
        [name != EXCLUDED_CHANNEL for name in names]
    )
    projector = span_projector(banked["response"][keep])
    return pooled, keep, projector, projector.residual(pooled[keep])


def test_the_identifiability_spectrum_still_has_the_values_the_record_reports(banked):
    whitened = whiten(banked["response"], banked["floor"])
    spectrum = identifiability(whitened, [str(name) for name in banked["drives"]])
    assert np.allclose(
        spectrum.singular_values, SINGULAR_VALUES, rtol=1e-10, atol=0.0
    )
    assert spectrum.condition_number == pytest.approx(CONDITION_NUMBER, rel=1e-10)


def test_the_weakest_directions_are_the_solenoid_against_the_inner_conductors(banked):
    """The published finding the spectrum was computed to state.

    A condition number alone says the design is nearly degenerate; which conductors
    make the degenerate direction is what a consumer needs, and it is the solenoid
    mixed with the inner pairs rather than anything the outer conductors do.
    """

    whitened = whiten(banked["response"], banked["floor"])
    spectrum = identifiability(whitened, [str(name) for name in banked["drives"]])
    for mode in spectrum.modes[-3:]:
        assert "sol" in {name for name, _ in mode.dominant}
    assert spectrum.unresolved(1.0e-2) == spectrum.modes[-3:]


@pytest.mark.parametrize("group", sorted(POOLED))
def test_the_surviving_misfit_is_the_size_the_record_states(banked, group):
    expected = POOLED[group]
    pooled, keep, _, residual = surviving(banked, group)
    assert int(keep.sum()) == expected["sensors"]
    assert float(np.sqrt(np.mean(residual**2))) == pytest.approx(
        expected["surviving_rms"], rel=1e-3
    )
    assert surviving_fraction(pooled[keep], residual) == pytest.approx(
        expected["surviving_share"], abs=5e-4
    )


@pytest.mark.parametrize("group", sorted(POOLED))
def test_the_largest_surviving_sensor_is_the_one_the_record_names(banked, group):
    expected = POOLED[group]
    _, keep, _, residual = surviving(banked, group)
    names = [name for name, kept in zip(sensor_names(banked), keep) if kept]
    largest = int(np.argmax(np.abs(residual)))
    assert names[largest] == expected["largest"][0]
    assert float(residual[largest]) * 1e9 == pytest.approx(
        expected["largest"][1], abs=0.05
    )


@pytest.mark.parametrize("group", sorted(POOLED))
def test_the_filament_scan_peaks_where_the_record_puts_it(banked, group):
    expected = POOLED[group]
    _, keep, projector, residual = surviving(banked, group)
    peak = filament_scan(
        residual,
        banked["r"][keep],
        banked["z"][keep],
        banked["cos"][keep],
        banked["sin"][keep],
        projector=projector,
        radius=SCAN_RADIUS,
        height=SCAN_HEIGHT,
    ).peak
    assert peak.radius == pytest.approx(expected["peak_radius"], abs=0.005)
    assert peak.height == pytest.approx(expected["peak_height"], abs=0.005)
    assert peak.score == pytest.approx(expected["peak_score"], abs=5e-4)
    assert peak.current == pytest.approx(expected["peak_current"], rel=1e-3)


@pytest.mark.parametrize("group", sorted(POOLED))
def test_the_excluded_sensor_reads_half_of_its_described_response(banked, group):
    """The gain that made the sensor suspect, measured without fitting a gain.

    Its pooled coupling to a drive is what it reads beyond the description, and it
    comes to almost exactly minus half the described response -- which is what a sensor
    reading half of what it should, with the sign reversed, produces.  Nothing in this
    route is a gain fit, so it corroborates the promoted scale rather than restating it.
    """

    pooled = pooled_group(banked, group)
    names = sensor_names(banked)
    drives = [str(name) for name in banked["drives"]]
    row = names.index(EXCLUDED_CHANNEL)
    described = float(banked["response"][row, drives.index(group)])
    assert pooled[row] / described == pytest.approx(
        POOLED[group]["convention_ratio"], abs=5e-4
    )


def test_the_independently_fitted_scales_pool_to_the_recorded_number():
    pooled = pool_scalar_gains(FAMILY_GAINS, channel=EXCLUDED_CHANNEL)
    assert pooled.slope == pytest.approx(POOLED_FAMILY_GAIN, rel=1e-15)
    assert pooled.identified


def test_the_promoted_scale_reaches_a_read_through_the_correction_document():
    """The number in the record has to arrive at the samples, not merely be stored."""

    document = read_correction_set("mast", "magnetics")
    chain = build_chain(document, EXCLUDED_CHANNEL, pulse=20000)
    gains = [step for step in chain.steps if step.kind is CorrectionKind.gain]
    assert [step.value for step in gains] == [PROMOTED_GAIN]
    corrected, _ = apply_corrections(
        document, EXCLUDED_CHANNEL, np.full(8, PROMOTED_GAIN), pulse=20000
    )
    assert np.allclose(corrected, 1.0, atol=1e-12)


def test_a_read_between_the_measured_pulses_says_so():
    """The document's own record of which pulses were measured survives the read."""

    document = read_correction_set("mast", "magnetics")
    measured = build_chain(document, EXCLUDED_CHANNEL, pulse=14080)
    between = build_chain(document, EXCLUDED_CHANNEL, pulse=14082)
    assert not measured.extrapolated
    assert between.extrapolated
    assert measured.multiplier == between.multiplier
