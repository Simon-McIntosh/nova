"""Pin the identified MAST source coordinate convention and its DDv4 transform.

The MAST equilibrium reconstruction served through FAIR-MAST was measured to be
COCOS 3 and the target Data Dictionary is COCOS 17.  Both facts, and every
factor of the transform between them, are asserted here so that a change to
either side of the map turns red instead of silently re-signing a flux or a
safety factor.

The banked sign combinations below are the measured corpus statistics, not
illustrative numbers -- see ``nova/scripts/identify_source_cocos.py``, which
re-derives them from the level-1 store.  The corpus-backed tests at the end
re-run that derivation when the mirror is reachable and skip when it is not, so
the convention stays pinned in an environment with no MAST data at all.
"""

from pathlib import Path

import numpy as np
import pytest

from nova.scripts.identify_source_cocos import (
    LEVEL1,
    cocos_from_digits,
    determine_digits,
    fingerprint_shot,
    flux_loop_two_pi_ratio,
)

# --- the measured verdict ---------------------------------------------------

#: identified source convention of the FAIR-MAST ``efm`` reconstruction.
MAST_SOURCE_COCOS = 3

#: the Data Dictionary v4 convention every authored IDS must carry.
TARGET_COCOS = 17

#: digits of the source convention, each pinned by its own measurement:
#: sigma_Bp from sign(psi_edge - psi_axis) = sigma_Ip * sigma_Bp (unanimous over
#: 11154 shots) and independently from sign(dp/dpsi); e_Bp from the flux-loop
#: ratio test (median 6.3007 against 2*pi = 6.2832 over 2662 loop-slices);
#: sigma_rho_theta_phi from sign(q) = sigma_Ip * sigma_B0 * sigma_rho_theta_phi;
#: sigma_RphiZ declared as the standard right-handed (R, phi, Z) frame.
SOURCE_DIGITS = {
    "sigma_bp": -1,
    "e_bp": 0,
    "sigma_r_phi_z": +1,
    "sigma_rho_theta_phi": -1,
}

#: the two sign cohorts the corpus actually contains.  The forward cohort is
#: 10775 of 11154 shots; the reversed cohort is 379 shots (3.40%) in two blocks,
#: 13469..13696 and 22376..22626, where BOTH the plasma current and the toroidal
#: field are reversed.  The convention is identical for both -- which is the
#: point of banking them together.
FORWARD_COHORT = {
    "psi_axis": 0.0815,
    "psi_boundary": -0.0285,
    "ip": 8.113e5,
    "b0": -0.406,
    "q_95": 4.513,
    "dp_dpsi": 3.990e4,
}
REVERSED_COHORT = {
    "psi_axis": -0.0815,
    "psi_boundary": 0.0285,
    "ip": -8.113e5,
    "b0": +0.406,
    "q_95": 4.513,
    "dp_dpsi": -3.990e4,
}


def _sigma_bp(sample: dict) -> int:
    """sigma_Bp from Sauter Eq. 23: sign(psi_edge - psi_axis) = sigma_Ip sigma_Bp."""
    return int(
        np.sign(sample["psi_boundary"] - sample["psi_axis"]) * np.sign(sample["ip"])
    )


def _sigma_rho_theta_phi(sample: dict) -> int:
    """sigma_rho_theta_phi from sign(q) = sigma_Ip sigma_B0 sigma_rho_theta_phi."""
    return int(np.sign(sample["q_95"]) * np.sign(sample["ip"]) * np.sign(sample["b0"]))


# --- Table I round trip -----------------------------------------------------


def test_source_digits_give_cocos_three():
    assert cocos_from_digits(**SOURCE_DIGITS) == MAST_SOURCE_COCOS


def test_target_digits_give_cocos_seventeen():
    assert (
        cocos_from_digits(sigma_bp=-1, e_bp=1, sigma_r_phi_z=+1, sigma_rho_theta_phi=+1)
        == TARGET_COCOS
    )


def test_table_is_a_bijection():
    """Every valid COCOS maps to a distinct digit quadruple and back."""
    valid = [*range(1, 9), *range(11, 19)]
    seen = set()
    for value in valid:
        digits = next(
            d
            for d in [
                {
                    "sigma_bp": bp,
                    "e_bp": e,
                    "sigma_r_phi_z": rz,
                    "sigma_rho_theta_phi": rtp,
                }
                for bp in (-1, 1)
                for e in (0, 1)
                for rz in (-1, 1)
                for rtp in (-1, 1)
            ]
            if cocos_from_digits(**d) == value
        )
        key = tuple(sorted(digits.items()))
        assert key not in seen
        seen.add(key)
    assert len(seen) == 16


# --- the source convention, from the banked sign cohorts --------------------


@pytest.mark.parametrize(
    ("name", "cohort"),
    [("forward", FORWARD_COHORT), ("reversed", REVERSED_COHORT)],
)
def test_both_sign_cohorts_are_cocos_three(name, cohort):
    """Reversing Ip and B_phi together leaves the convention unchanged."""
    digits = {
        "sigma_bp": _sigma_bp(cohort),
        "e_bp": SOURCE_DIGITS["e_bp"],
        "sigma_r_phi_z": SOURCE_DIGITS["sigma_r_phi_z"],
        "sigma_rho_theta_phi": _sigma_rho_theta_phi(cohort),
    }
    assert digits == SOURCE_DIGITS, name
    assert cocos_from_digits(**digits) == MAST_SOURCE_COCOS


def test_axis_flux_extremum_is_not_a_convention_invariant():
    """psi_axis > psi_boundary holds only for the forward-current cohort.

    Guards the tempting shortcut of testing the convention by asserting the axis
    is a flux maximum: 379 shots of the corpus have it the other way round while
    being the same COCOS.  sigma_Bp is the invariant; the extremum is not.
    """
    assert FORWARD_COHORT["psi_axis"] > FORWARD_COHORT["psi_boundary"]
    assert REVERSED_COHORT["psi_axis"] < REVERSED_COHORT["psi_boundary"]
    assert _sigma_bp(FORWARD_COHORT) == _sigma_bp(REVERSED_COHORT) == -1


@pytest.mark.parametrize(
    ("name", "cohort"),
    [("forward", FORWARD_COHORT), ("reversed", REVERSED_COHORT)],
)
def test_pressure_gradient_cross_checks_sigma_bp(name, cohort):
    """sign(dp/dpsi) = -sigma_Ip sigma_Bp -- an independent read on the same digit."""
    expected = -np.sign(cohort["ip"]) * _sigma_bp(cohort)
    assert np.sign(cohort["dp_dpsi"]) == expected, name


# --- the source-to-target transform ----------------------------------------


def test_transform_flips_only_the_q_sign_and_scales_psi():
    """COCOS 3 -> 17 composes to +1 on sigma_Bp and sigma_RphiZ, -1 on q, 2*pi on psi.

    Derived from the digit algebra rather than asserted as a literal: the
    effective factors are the products of the two conventions' digits, so an
    error in either digit table shows up here.
    """
    source = SOURCE_DIGITS
    target = {
        "sigma_bp": -1,
        "e_bp": 1,
        "sigma_r_phi_z": +1,
        "sigma_rho_theta_phi": +1,
    }
    sigma_bp_eff = source["sigma_bp"] * target["sigma_bp"]
    sigma_rz_eff = source["sigma_r_phi_z"] * target["sigma_r_phi_z"]
    sigma_rtp_eff = source["sigma_rho_theta_phi"] * target["sigma_rho_theta_phi"]
    e_bp_eff = target["e_bp"] - source["e_bp"]

    assert sigma_bp_eff == +1  # psi keeps its sign
    assert sigma_rz_eff == +1  # Ip, B_phi, F keep their signs
    assert sigma_rtp_eff == -1  # q changes sign
    assert e_bp_eff == +1  # psi picks up one factor of 2*pi

    psi_factor = sigma_bp_eff * sigma_rz_eff * (2 * np.pi) ** e_bp_eff
    assert np.isclose(psi_factor, 2 * np.pi)
    # dp/dpsi and f df/dpsi carry the reciprocal scale
    assert np.isclose(
        sigma_bp_eff * sigma_rz_eff * (2 * np.pi) ** (-e_bp_eff), 1 / (2 * np.pi)
    )
    assert sigma_rtp_eff * sigma_rz_eff == -1  # the q factor


@pytest.mark.parametrize(
    ("name", "cohort"),
    [("forward", FORWARD_COHORT), ("reversed", REVERSED_COHORT)],
)
def test_transformed_quantities_read_back_as_cocos_seventeen(name, cohort):
    """Apply the transform, re-derive the digits, and land on 17."""
    transformed = {
        "psi_axis": 2 * np.pi * cohort["psi_axis"],
        "psi_boundary": 2 * np.pi * cohort["psi_boundary"],
        "ip": cohort["ip"],
        "b0": cohort["b0"],
        "q_95": -cohort["q_95"],
        "dp_dpsi": cohort["dp_dpsi"] / (2 * np.pi),
    }
    digits = {
        "sigma_bp": _sigma_bp(transformed),
        "e_bp": 1,
        "sigma_r_phi_z": +1,
        "sigma_rho_theta_phi": _sigma_rho_theta_phi(transformed),
    }
    assert cocos_from_digits(**digits) == TARGET_COCOS, name
    # the pressure-gradient identity must survive the transform too
    assert (
        np.sign(transformed["dp_dpsi"])
        == -np.sign(transformed["ip"]) * digits["sigma_bp"]
    )


def test_target_q_is_negative_for_the_forward_cohort():
    """A MAST forward shot authored in DDv4 carries q < 0.

    Not a bug and not a knob: COCOS 17 has sigma_rho_theta_phi = +1, so
    sign(q) = sigma_Ip sigma_B0 = -1 whenever the current is positive and the
    toroidal field negative, which is the MAST forward configuration.  Anything
    downstream that assumes a positive q on DDv4 MAST data is wrong.
    """
    assert -FORWARD_COHORT["q_95"] < 0
    assert np.sign(FORWARD_COHORT["ip"]) * np.sign(FORWARD_COHORT["b0"]) == -1


# --- corpus-backed re-derivation (skipped without the mirror) --------------

_MIRROR = Path(LEVEL1)
_needs_mirror = pytest.mark.skipif(
    not _MIRROR.is_dir(), reason=f"MAST level-1 mirror not present at {_MIRROR}"
)

#: one representative shot per identified machine description, each chosen
#: because its reconstructed flux map and fitted loop fluxes are both populated
#: at its peak-current slice -- so the same three shots exercise the fingerprint
#: check and the 2*pi measurement.
REPRESENTATIVE_SHOTS = (11794, 12417, 18502)

#: the three machine-description fingerprints the whole corpus resolves to.
KNOWN_FINGERPRINTS = {
    "mp78-fl46-fc1004-lim37-9425ae4a8bf3bc15",
    "mp78-fl46-fc1004-lim37-edd753d282903679",
    "mp78-fl46-fc938-lim37-1cb6f2ee742c4ee4",
}


@_needs_mirror
@pytest.mark.parametrize("shot", REPRESENTATIVE_SHOTS)
def test_shot_resolves_to_a_known_fingerprint(shot):
    row = fingerprint_shot(shot)
    if row is None:
        pytest.skip(f"shot {shot} unreadable in this mirror")
    assert row["fingerprint"] in KNOWN_FINGERPRINTS
    assert row["n_probe"] == 78
    assert row["n_loop"] == 46
    assert row["n_limiter"] == 37


@_needs_mirror
def test_corpus_sample_re_derives_cocos_three():
    rows = [fingerprint_shot(s) for s in REPRESENTATIVE_SHOTS]
    rows = [r for r in rows if r is not None and r.get("has_signs")]
    if not rows:
        pytest.skip("no readable shot with reconstructed signs")
    digits = determine_digits(rows)
    assert digits["sigma_bp"] == SOURCE_DIGITS["sigma_bp"]
    assert digits["sigma_rho_theta_phi"] == SOURCE_DIGITS["sigma_rho_theta_phi"]
    assert (
        cocos_from_digits(
            sigma_bp=digits["sigma_bp"],
            e_bp=SOURCE_DIGITS["e_bp"],
            sigma_r_phi_z=SOURCE_DIGITS["sigma_r_phi_z"],
            sigma_rho_theta_phi=digits["sigma_rho_theta_phi"],
        )
        == MAST_SOURCE_COCOS
    )


@_needs_mirror
@pytest.mark.parametrize("shot", REPRESENTATIVE_SHOTS)
def test_flux_loop_ratio_measures_the_two_pi(shot):
    """The e_Bp digit, measured: loop flux [Wb] over psi at the loop [Wb/rad].

    Asserted once per machine description, because the flux map is stored on a
    different grid layout in different campaigns and a wrong grid produces a
    plausible-looking but meaningless ratio.
    """
    result = flux_loop_two_pi_ratio(shot)
    if result is None:
        pytest.skip(f"shot {shot} has no usable reconstructed flux map")
    assert result["n_loops"] >= 20
    # within 2% of 2*pi, and nowhere near 1 (which would mean total flux)
    assert result["ratio_median"] == pytest.approx(2 * np.pi, rel=0.02)
    assert abs(result["ratio_median"] - 1.0) > 1.0
    # the loops must agree with each other, or the grid is wrong
    assert result["ratio_iqr_fraction"] < 0.05
