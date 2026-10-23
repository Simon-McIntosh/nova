"""The COCOS 17 relation between the toroidal current and the flux functions.

Two independent routes establish the sign that multiplies
``R p' + FF'/(mu0 R)`` under COCOS 17, and a third contributes partly:

* the general digit form, evaluated at COCOS 11 and COCOS 17;
* the DD4-native regression, which is COCOS 17 by definition;
* imas-ambix's convention owner, which confirms the Ampere side and the
  ``2 pi`` but takes the bracket's sign as an input, so it adds nothing on
  the sign that is in question.

The DD4 regression measures ``+2*pi`` when ``p'`` is the derivative against
the Data Dictionary's own signed poloidal flux.  Nova's owner states
``-2*pi`` beside ``p' = -dp/dPhi``, so it is self-consistent with the flux
functions :func:`nova.equilibrium.map_extraction.extract_flux_functions`
publishes from a map, and anti-parallel to the map current when it is handed
the Data Dictionary's own derivatives.  Changing either site alone moves the
disagreement rather than closing it, which is why the strict xfail below
asserts both directions.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import mu_0

from nova_cocos import DODPSI_LIKE, PSI_LIKE, convention_transform

MEASURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "flux_function_sign"
    / "measure.py"
)

COCOS_11_DIGITS = {"sigma_bp": +1, "e_bp": 1}
COCOS_17_DIGITS = {"sigma_bp": -1, "e_bp": 1}


def _measure():
    """Return the measurement module by path, without importing a package."""

    spec = importlib.util.spec_from_file_location(
        "flux_function_sign_measure", MEASURE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit as exit_:
        pytest.skip(str(exit_.code))
    return module


def _dd4_slices(measure):
    """Return the stored fields of the pinned DD4 layer's sampled slices."""

    layer, _, _ = measure.pinned_dd4_layer()
    if not layer.is_file():
        pytest.skip(f"pinned DD4 layer not cached at {layer}")
    import imas

    try:
        directory, dd_version = measure.single_dd4_layer()
    except SystemExit as exit_:
        pytest.skip(str(exit_.code))
    entry = imas.DBEntry(f"imas:hdf5?path={directory}", "r", dd_version=dd_version)
    equilibrium = entry.get("equilibrium")
    count = len(equilibrium.time_slice)
    sample = sorted(set(np.linspace(0, count - 1, min(count, 3)).astype(int)))
    fields = [measure.read_slice(entry, int(index)) for index in sample]
    entry.close()
    return fields


def _relation(digits, radius, p_prime, ff_prime):
    """Return the digit form's current density for one convention."""

    sigma_bp = digits["sigma_bp"]
    e_bp = digits["e_bp"]
    flux_functions = radius * p_prime + ff_prime / (mu_0 * radius)
    return -sigma_bp * (2.0 * math.pi) ** e_bp * flux_functions


def test_the_digit_form_gives_the_same_current_in_cocos_11_and_17():
    """A selection is invariant under the 11 to 17 mapping.

    COCOS 11 to 17 negates psi and therefore the derivative taken against it,
    and negates the digit ``sigma_Bp`` beside it.  The two minus signs cancel,
    so the same physical current leaves the written equation unchanged.  This
    is why the DD4 regression and the raw DD3-source regressions carry the
    same printed slope and must not be read as two conventions disagreeing.
    """

    radius = np.array([3.0, 5.0, 17.0])
    p_prime_11 = np.array([-4.0e4, -2.0e4, -1.0e3])
    ff_prime_11 = np.array([-2.0, -1.0, -0.1])
    transform = convention_transform(source=11, target=17)
    assert transform.factor(PSI_LIKE) == -1.0
    assert transform.factor(DODPSI_LIKE) == -1.0
    p_prime_17 = transform.factor(DODPSI_LIKE) * p_prime_11
    ff_prime_17 = transform.factor(DODPSI_LIKE) * ff_prime_11
    assert np.allclose(
        _relation(COCOS_11_DIGITS, radius, p_prime_11, ff_prime_11),
        _relation(COCOS_17_DIGITS, radius, p_prime_17, ff_prime_17),
    )


def test_the_cocos_17_relation_carries_plus_two_pi():
    """The relation, measured on the DD4 layer by the measurement module.

    The Data Dictionary supplies ``dpressure_dpsi`` and ``f_df_dpsi`` against
    its own COCOS 17 flux, so the relation is applied to them directly and the
    only factor in play is the ``2*pi`` its total-flux definition carries.
    The same call with the flux map negated is the negative control: every
    slope reverses and no magnitude moves.
    """

    measure = _measure()
    slices = _dd4_slices(measure)
    direct = [measure.measure_slice(fields)["plus_2pi"] for fields in slices]
    negated = [
        measure.measure_slice(fields, negate_flux=True)["plus_2pi"] for fields in slices
    ]
    slopes = [row["slope"] for row in direct]
    correlations = [row["correlation"] for row in direct]
    control = [row["slope"] for row in negated]
    assert all(slope > 0.0 for slope in slopes), slopes
    assert all(correlation > 0.0 for correlation in correlations), correlations
    assert all(slope == pytest.approx(1.0, abs=0.05) for slope in slopes), slopes
    assert all(value < 0.0 for value in control), control
    assert all(
        value == pytest.approx(-slope, rel=0.02)
        for value, slope in zip(control, slopes)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "nova/equilibrium/convention.py documents p' as -dp/dPhi and computes "
        "-2*pi*(R p' + FF'/(mu0 R)). Fed the Data Dictionary's own dp/dpsi, "
        "which is what a reader reaches for and what the DD4 layer stores, that "
        "returns a current anti-parallel to the flux map's. The COCOS 17 "
        "relation is +2*pi against the DD's derivative. A bare flip of the "
        "leading sign would satisfy the first assertion and break the second: "
        "the flux functions map_extraction.py publishes are in the same "
        "negated sense, so the derivative convention and the sign have to move "
        "together, at map_extraction.py:292 and its paired sites."
    ),
)
def test_nova_owner_matches_the_cocos_17_relation():
    """The owner must produce the physical current from the state it is given.

    The first assertion drives the owner with the Data Dictionary's own
    derivatives and requires the current the map carries.  The second drives
    it with the derivatives the map extractor publishes from the same state
    and requires the same current, so the two sites stay in one sense: either
    change alone breaks one of the two.
    """

    from nova.equilibrium.convention import toroidal_current_density

    measure = _measure()
    for fields in _dd4_slices(measure):
        receipt = measure.extraction_roundtrip(fields)
        shells = receipt["shells"]
        assert shells, "no reliable shells in the DD4 state"
        for row in shells:
            assert row["stored"] == pytest.approx(row["physical"], rel=1.0e-9)
        assert receipt["slope_extracted"] == pytest.approx(1.0, abs=0.1)
    radius = np.array([3.0, 30.0, 60.0])
    p_prime = np.array([-4.0e4, -2.0e4, -1.0e3])
    ff_prime = np.array([-2.0, -1.0, -0.1])
    expected = 2.0 * math.pi * (radius * p_prime + ff_prime / (mu_0 * radius))
    assert np.allclose(toroidal_current_density(radius, p_prime, ff_prime), expected)
