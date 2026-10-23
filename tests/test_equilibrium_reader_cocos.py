"""The equilibrium reader serves its signed quantities in COCOS 17.

The reader renormalises its profiles to ``psi_norm``, and a sign reversal of
psi leaves that normalisation unchanged, so a test of normalised quantities
cannot see the translation at all. These tests assert the signed quantities
against the raw read of the same slice, and they assert the refusal of the two
DD3 pulses whose measured digits contradict the convention their Data
Dictionary version carries.
"""

from pathlib import Path

import imas
import numpy as np
import pytest

from nova.imas.equilibrium import EquilibriumData
from nova.io.sourcemap import NOVA_CONVENTION
from nova_cocos import convention, convention_transform, DODPSI_LIKE, PSI_LIKE

IMASDB = Path("/work/imas/shared/imasdb/ITER/3")

CASES = {
    "130506/6": "3.41.0",
    "105069/3": "3.39.0",
}
NON_CONFORMING = {"105014/1": "sigma_Bp=-1", "105013/1": "sigma_rho_theta_phi=-1"}


def open_ids(case: str, dd_version: str):
    path = IMASDB / case
    if not path.is_dir():
        pytest.skip(f"imasdb pulse {case} not present")
    entry = imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version=dd_version)
    with entry:
        return entry.get("equilibrium", lazy=False, autoconvert=False)


def open_equilibrium(case: str, dd_version: str) -> EquilibriumData:
    """Return the equilibrium reader on one pulse at its written DD version."""
    return EquilibriumData(ids=open_ids(case, dd_version), filename=case)


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

    # The reader translates; the raw IDS still carries its own DD version's
    # convention, so each signed quantity differs by the two conventions'
    # factor and not by a factor of one.
    for attr, read, raw_value in [
        ("psi_axis", float(equilibrium["psi_axis"]), float(raw.global_quantities.psi_axis)),
        (
            "psi_boundary",
            float(equilibrium["psi_boundary"]),
            float(raw.global_quantities.psi_boundary),
        ),
    ]:
        assert np.isclose(read, psi_factor * raw_value), attr
        assert read != raw_value or raw_value == 0.0, attr

    psi_span = float(equilibrium["psi_boundary"]) - float(equilibrium["psi_axis"])
    assert np.sign(psi_span) == psi_factor * np.sign(float(equilibrium["ip"]))

    # The gradients are taken with respect to the negated total flux, so they
    # carry the same sign change as psi.
    reader_p_prime = np.asarray(equilibrium["dpressure_dpsi"], float)
    raw_p_prime = np.asarray(raw.profiles_1d.dpressure_dpsi, float)
    assert np.isclose(reader_p_prime[0], dpsi_factor * raw_p_prime[0])
    assert np.sign(reader_p_prime[0]) == dpsi_factor * np.sign(raw_p_prime[0])

    reader_ff_prime = np.asarray(equilibrium["f_df_dpsi"], float)
    raw_ff_prime = np.asarray(raw.profiles_1d.f_df_dpsi, float)
    assert np.sign(reader_ff_prime[0]) == dpsi_factor * np.sign(raw_ff_prime[0])

    # Ip and q are unchanged by the 11 -> 17 factor: a control that must read
    # the raw value while the rows above have moved.
    assert np.isclose(float(equilibrium["ip"]), float(raw.global_quantities.ip))
    assert np.isclose(
        float(equilibrium["q_95"]), float(raw.global_quantities.q_95)
    )


@pytest.mark.parametrize("case", sorted(NON_CONFORMING))
def test_reader_refuses_non_conforming_data_by_name(case):
    """A DD3 pulse whose digits contradict COCOS 11 is refused with its digit."""
    with pytest.raises(ValueError) as refused:
        open_equilibrium(case, "3.38.1")
    message = str(refused.value)
    assert case in message
    assert NON_CONFORMING[case] in message