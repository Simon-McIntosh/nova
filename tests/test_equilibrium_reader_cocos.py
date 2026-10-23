"""The equilibrium reader serves its signed quantities in COCOS 17.

The reader renormalises its profiles to ``psi_norm``, and a sign reversal of
psi leaves that normalisation unchanged, so a test of normalised quantities
cannot see the translation at all. These tests assert the signed quantities
against the written convention of the pulse they are read from, and they assert
that the reader refuses the two DD3 pulses whose own digits contradict COCOS 11.

The pulse paths and their written Data Dictionary versions are restated from
``tests/test_dd_translator.py``'s ``DD3_PULSES`` and ``NON_CONFORMING`` tables
(lines 46 to 60), which are the authoritative statement of what each imasdb
pulse was written in.
"""

from __future__ import annotations

from pathlib import Path

import imas
import numpy as np
import pytest

from nova.imas.equilibrium import EquilibriumData
from nova.io.sourcemap import NOVA_CONVENTION
from nova_cocos import DODPSI_LIKE, PSI_LIKE, convention, convention_transform

IMASDB = Path("/work/imas/shared/imasdb/ITER/3")

#: conforming DD3 pulses, as ``pulse/run -> written DD version``.
CASES = {
    "130506/6": "3.41.0",
    "105052/1": "3.38.1",
    "105069/3": "3.39.0",
}

#: pulses whose digits contradict their DD version, as ``pulse/run`` ->
#: ``(version, failing digit)``.
REFUSED = {
    "105014/1": ("3.38.1", "sigma_Bp=-1"),
    "105013/1": ("3.38.1", "sigma_rho_theta_phi=-1"),
}


def open_ids(case: str, dd_version: str):
    """Open one pulse's equilibrium IDS at the version it was written in."""
    path = IMASDB / case
    if not path.is_dir():
        pytest.skip(f"imasdb pulse {case} not present")
    entry = imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version=dd_version)
    with entry:
        return entry.get("equilibrium", lazy=False, autoconvert=False)


def open_equilibrium(case: str, dd_version: str) -> EquilibriumData:
    """Return the equilibrium reader on one pulse at its written DD version.

    The reader is constructed from ``(pulse, run, dd_version)`` rather than
    from an already-opened IDS: a reader built around a passed IDS carries no
    URI and no Data Dictionary major of its own, and its machine-description
    read then resolves at the latest published major instead of the pulse's.
    """
    pulse, run = (int(part) for part in case.split("/"))
    return EquilibriumData(pulse, run, dd_version=dd_version)


@pytest.mark.parametrize("case", sorted(CASES))
def test_signed_quantities_carry_the_internal_convention(case):
    """psi and the flux-function gradients move by the 11 -> 17 factor."""
    equilibrium = open_equilibrium(case, CASES[case])
    equilibrium.itime = 0
    assert equilibrium.data.attrs["cocos"] == NOVA_CONVENTION

    raw = open_ids(case, CASES[case]).time_slice[0]
    factor = convention_transform(source=11, target=NOVA_CONVENTION)
    psi_factor = factor.factor(PSI_LIKE)
    dpsi_factor = factor.factor(DODPSI_LIKE)
    assert psi_factor == convention(NOVA_CONVENTION).sigma_bp
    assert psi_factor == -1.0

    # The reader translates and the raw IDS does not, so each signed quantity
    # differs from its raw counterpart by the two conventions' factor.
    for attr, read, raw_value in [
        (
            "psi_axis",
            float(equilibrium["psi_axis"]),
            float(raw.global_quantities.psi_axis),
        ),
        (
            "psi_boundary",
            float(equilibrium["psi_boundary"]),
            float(raw.global_quantities.psi_boundary),
        ),
    ]:
        assert np.isclose(read, psi_factor * raw_value), attr
        assert read != raw_value or raw_value == 0.0, attr

    # The gradients are taken with respect to the negated total flux, so they
    # carry the same sign change as psi.
    reader_p_prime = np.asarray(equilibrium["dpressure_dpsi"], float)
    raw_p_prime = np.asarray(raw.profiles_1d.dpressure_dpsi, float)
    assert np.isclose(reader_p_prime[0], dpsi_factor * raw_p_prime[0])

    reader_ff_prime = np.asarray(equilibrium["f_df_dpsi"], float)
    raw_ff_prime = np.asarray(raw.profiles_1d.f_df_dpsi, float)
    assert np.sign(reader_ff_prime[0]) == dpsi_factor * np.sign(raw_ff_prime[0])

    # Ip and q are unchanged by the 11 -> 17 factor: a control that must still
    # read its raw counterpart while the rows above have moved.
    assert np.isclose(float(equilibrium["ip"]), float(raw.global_quantities.ip))
    assert np.isclose(float(equilibrium["q_95"]), float(raw.global_quantities.q_95))


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_reader_refuses_non_conforming_data_by_name(case):
    """A DD3 pulse whose digits contradict COCOS 11 is refused with its digit."""
    dd_version, digit = REFUSED[case]
    with pytest.raises(ValueError) as refused:
        open_equilibrium(case, dd_version)
    message = str(refused.value)
    assert "does not conform" in message
    assert digit in message
