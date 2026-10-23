"""The Data Dictionary equilibrium translator: DD3 in, COCOS 17 out, by name.

nova reads DD3 data as COCOS 11 and DD4 data as COCOS 17, works in COCOS 17,
and treats data that does not conform to its own DD version's convention as
invalid.  The translator is :func:`nova.io.sourcemap.translate_equilibrium`,
which declares each equilibrium quantity as a source-map row whose
``convention_factor`` comes from the two conventions, and refuses a
non-conforming read through the map's blocked path.

The factors asserted here are transcribed from the two conventions' digits in
this file rather than imported from the algebra, so a change to either side of
the map has to disagree with the arithmetic below instead of with itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.io.sourcemap import (
    DD3_CONVENTION,
    DD4_CONVENTION,
    NOVA_CONVENTION,
    SourceMapError,
    dd_equilibrium_convention,
    translate_equilibrium,
)
from nova_cocos import B0_LIKE, DODPSI_LIKE, IP_LIKE, ONE_LIKE, PSI_LIKE, Q_LIKE

pytestmark = pytest.mark.slow

IMASDB = Path("/work/imas/shared/imasdb/ITER/3")
DD3_CORSICA = IMASDB / "130506/6"
DD3_CORSICA_ALIGNED = IMASDB / "130506/403"

#: (run, written DD version) of the DD3 pulses whose measured digits name COCOS 11.
#: Each carries a reversed plasma current against a 5.3 T toroidal field stored
#: negative, so sign(q) = sign(ip) * sign(b0) = +1 and both sign-bearing digits
#: come out at their COCOS 11 values on every slice above 1 MA.
DD3_PULSES = {
    "CORSICA 130506/6": (DD3_CORSICA, "3.41.0"),
    "DINA 105052/1": (IMASDB / "105052/1", "3.38.1"),
    "DINA 105069/3": (IMASDB / "105069/3", "3.39.0"),
}

#: The two 2.65 T DINA pulses store their toroidal field mirrored against their
#: stored flux ordering, so neither names COCOS 11.  105014/1 reads exactly
#: COCOS 17's digits -- what a writer that produced COCOS 17 data under a DD3
#: label looks like -- and 105013/1 is its q-mirror.  Each is refused with the
#: digit that fails it named in the reason.
NON_CONFORMING = {
    "DINA 105014/1": (IMASDB / "105014/1", "3.38.1", "sigma_Bp=-1"),
    "DINA 105013/1": (IMASDB / "105013/1", "3.38.1", "sigma_rho_theta_phi=-1"),
}

MANIFEST = Path(__file__).resolve().parent / "data-manifest.json"
CASE_KEY = "iter_corsica_130506"

#: the two conventions as their digits, transcribed from Sauter & Medvedev
#: Table I: (sigma_Bp, e_Bp, sigma_RphiZ, sigma_rho_theta_phi).
COCOS_11 = (+1, 1, +1, +1)
COCOS_17 = (-1, 1, +1, +1)

#: a slice carries no convention information below this |Ip|.
MIN_IP_A = 1.0e6


def _analytic_factor(transformation: str, source: tuple, target: tuple) -> float:
    """The factor two conventions impose, composed from their digits alone."""

    sigma_bp = source[0] * target[0]
    sigma_r_phi_z = source[2] * target[2]
    sigma_rho_theta_phi = source[3] * target[3]
    flux_exponent = target[1] - source[1]
    flux = sigma_bp * sigma_r_phi_z
    if transformation == ONE_LIKE:
        return 1.0
    if transformation in (IP_LIKE, B0_LIKE):
        return float(sigma_r_phi_z)
    if transformation == Q_LIKE:
        return float(sigma_rho_theta_phi * sigma_r_phi_z)
    if transformation == PSI_LIKE:
        return float(flux) * (2.0 * math.pi) ** flux_exponent
    return float(flux) * (2.0 * math.pi) ** (-flux_exponent)


def _open(path: Path, dd_version: str):
    if not path.is_dir():
        pytest.skip(f"imasdb pulse not present at {path}")
    import imas

    return imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version=dd_version)


def _vacuum_field(equilibrium, index: int) -> float:
    values = np.asarray(equilibrium.vacuum_toroidal_field.b0).ravel()
    if values.size == 0:
        return float("nan")
    return float(values[0] if values.size == 1 else values[index])


def _observables(entry) -> list[dict[str, float]]:
    """One row per time slice carrying the observables the digits are read from."""

    equilibrium = entry.get("equilibrium")
    rows = []
    for index in range(len(equilibrium.time_slice)):
        quantities = equilibrium.time_slice[index].global_quantities
        rows.append(
            {
                "ip": float(quantities.ip),
                "psi_axis": float(quantities.psi_axis),
                "psi_boundary": float(quantities.psi_boundary),
                "q": float(quantities.q_95),
                "b0": _vacuum_field(equilibrium, index),
            }
        )
    return rows


def _above_one_mega_ampere(row: dict[str, float]) -> bool:
    return bool(np.isfinite(row["ip"]) and abs(row["ip"]) > MIN_IP_A)


def _translate(name: str, dd_version: str, row: dict[str, float]):
    return translate_equilibrium(source_name=name, dd_version=dd_version, **row)


def _fixture_digest() -> str:
    record = json.loads(MANIFEST.read_text())["efitpp_validator_layers"]
    case = record["cases"][CASE_KEY]
    reference = next(layer for layer in case["layers"] if layer["role"] == "reference")
    return reference["digest"].removeprefix("sha256:")


# --- the two DD conventions -------------------------------------------------


def test_a_dd_version_names_its_convention():
    assert dd_equilibrium_convention("3.41.0") == DD3_CONVENTION
    assert dd_equilibrium_convention("3.38.1") == DD3_CONVENTION
    assert dd_equilibrium_convention("4.1.0") == DD4_CONVENTION
    assert NOVA_CONVENTION == DD4_CONVENTION


def test_a_version_with_no_declared_convention_is_refused():
    with pytest.raises(SourceMapError, match="no COCOS is declared"):
        dd_equilibrium_convention("2.7.0")


# --- DD3 in, COCOS 17 out ---------------------------------------------------


def test_a_dd3_slice_translates_by_the_analytic_mapping():
    """Every quantity of the conforming DD3 pulse takes the 11 to 17 factor.

    Checked quantity by quantity: the map's ``convention_factor`` must equal the
    factor the two conventions' digits compose to, and applying the row to the
    raw value must give the analytically mapped value.
    """

    entry = _open(DD3_CORSICA, "3.41.0")
    rows = [row for row in _observables(entry) if _above_one_mega_ampere(row)]
    assert rows, "no slice of CORSICA 130506/6 above 1 MA"
    checked = 0
    for row in rows:
        source_map = _translate("CORSICA 130506/6", "3.41.0", row)
        assert not source_map.blocked, source_map.blocked
        assert source_map.signals
        for signal in source_map.signals:
            assert signal.source_convention == DD3_CONVENTION
            assert signal.target_convention == NOVA_CONVENTION
            assert signal.convention_factor == pytest.approx(
                _analytic_factor(signal.transformation, COCOS_11, COCOS_17)
            )
            raw = np.array([1.0, -2.5, 0.0])
            mapped = signal.apply(raw)
            assert np.allclose(
                mapped,
                raw * _analytic_factor(signal.transformation, COCOS_11, COCOS_17),
            )
            checked += 1
    assert checked > 0


def test_the_dd3_transform_flips_the_flux_and_leaves_the_current_alone():
    """COCOS 11 to 17 composes to -1 on psi and its derivative, +1 on Ip and q."""

    assert _analytic_factor(PSI_LIKE, COCOS_11, COCOS_17) == pytest.approx(-1.0)
    assert _analytic_factor(DODPSI_LIKE, COCOS_11, COCOS_17) == pytest.approx(-1.0)
    assert _analytic_factor(IP_LIKE, COCOS_11, COCOS_17) == pytest.approx(1.0)
    assert _analytic_factor(B0_LIKE, COCOS_11, COCOS_17) == pytest.approx(1.0)
    assert _analytic_factor(Q_LIKE, COCOS_11, COCOS_17) == pytest.approx(1.0)


def test_the_conforming_dd3_pulses_are_accepted_on_every_slice_above_one_ma():
    for name, (path, dd_version) in DD3_PULSES.items():
        entry = _open(path, dd_version)
        rows = [row for row in _observables(entry) if _above_one_mega_ampere(row)]
        assert rows, f"{name} has no slice above 1 MA"
        for row in rows:
            source_map = _translate(name, dd_version, row)
            assert not source_map.blocked, (name, source_map.blocked)
            assert source_map.signals


def test_the_non_conforming_dd3_pulses_are_refused_by_name_on_every_slice():
    """Both mirrored 2.65 T pulses are refused, each naming the digit that fails.

    The two pulses fail on different digits, so a check that happened to test
    only one of them would still look green against the other: 105014/1 fails
    the DD the plan's acceptance list named and 105013/1 fails on the poloidal
    orientation, which the census read as underivable from its sampled slices.
    """

    for name, (path, dd_version, failed_digit) in NON_CONFORMING.items():
        entry = _open(path, dd_version)
        rows = [row for row in _observables(entry) if _above_one_mega_ampere(row)]
        assert rows, f"{name} has no slice above 1 MA"
        for row in rows:
            source_map = _translate(name, dd_version, row)
            assert not source_map.signals
            assert len(source_map.blocked) == 1
            blocked = source_map.blocked[0]
            assert blocked.source_channel == name
            assert name in blocked.reason
            assert failed_digit in blocked.reason
            assert dd_version in blocked.reason
            assert f"COCOS {DD3_CONVENTION}" in blocked.reason


# --- DD4 through, bit-identically -------------------------------------------


def test_the_dd4_fixture_passes_through_bit_identically(tmp_path):
    """The repository's DD 4.1.0 conversion of CORSICA 130506 is DD4 data.

    Read through the translator every quantity's factor must be exactly one and
    the converted samples must be the raw samples, bit for bit -- a DD4 read is
    not rescaled, re-signed or re-ordered on the way in.
    """

    record = json.loads(MANIFEST.read_text())["efitpp_validator_layers"]
    case = record["cases"][CASE_KEY]
    reference = next(layer for layer in case["layers"] if layer["role"] == "reference")
    layer = Path.home() / ".cache" / "imas-efit-publish-169" / reference["name"]
    if not layer.is_file():
        pytest.skip(f"pinned DD4 layer not cached at {layer}")
    assert hashlib.sha256(layer.read_bytes()).hexdigest() == _fixture_digest()

    from nova.imas.validator_case import netcdf_to_hdf5

    converted = netcdf_to_hdf5(
        layer, tmp_path / "dd4", case["dd_version"], ["equilibrium"]
    )
    import imas

    entry = imas.DBEntry(
        f"imas:hdf5?path={converted}", "r", dd_version=case["dd_version"]
    )
    rows = _observables(entry)
    assert rows
    for row in rows:
        source_map = _translate(
            f"CORSICA 130506 (DD4 {case['dd_version']})", case["dd_version"], row
        )
        assert not source_map.blocked, source_map.blocked
        for signal in source_map.signals:
            assert signal.source_convention == DD4_CONVENTION
            assert signal.target_convention == NOVA_CONVENTION
            # The analytic composition carries the measure: DD4 is COCOS 17
            # already, so every transformation's factor is the identity's. The
            # sample comparison beside it only shows ``apply`` carries whatever
            # factor the row holds, since it compares the converted samples with
            # the samples the test handed in.
            assert signal.convention_factor == pytest.approx(
                _analytic_factor(signal.transformation, COCOS_17, COCOS_17)
            )
            raw = np.array([0.0, -1.5, np.nan, 7.0])
            assert np.array_equal(signal.apply(raw), raw, equal_nan=True)


def test_the_dd3_read_reproduces_the_dd4_fixture_of_the_same_case():
    """The DD3 read of shot 130506 and its DD4 conversion agree once translated.

    The DD4 reference layer carries no psi_axis, so the two are compared on the
    quantities both publish: the safety factor, which no convention between 11
    and 17 rescales, and the sign of the boundary flux, which the mapping flips.
    The DD3 run that covers the fixture's times is 130506/403.
    """

    record = json.loads(MANIFEST.read_text())["efitpp_validator_layers"]
    case = record["cases"][CASE_KEY]
    reference = next(layer for layer in case["layers"] if layer["role"] == "reference")
    layer = Path.home() / ".cache" / "imas-efit-publish-169" / reference["name"]
    if not layer.is_file():
        pytest.skip(f"pinned DD4 layer not cached at {layer}")
    if not DD3_CORSICA_ALIGNED.is_dir():
        pytest.skip(f"imasdb pulse not present at {DD3_CORSICA_ALIGNED}")

    import imas

    from nova.imas.validator_case import netcdf_to_hdf5

    with tempfile.TemporaryDirectory() as tmp:
        converted = netcdf_to_hdf5(
            layer, Path(tmp) / "dd4", case["dd_version"], ["equilibrium"]
        )
        dd4 = imas.DBEntry(
            f"imas:hdf5?path={converted}", "r", dd_version=case["dd_version"]
        )
        dd4_rows = _observables(dd4)
        dd4_times = np.asarray(dd4.get("equilibrium").time).ravel()
    dd3 = _open(DD3_CORSICA_ALIGNED, "3.38.1")
    dd3_times = np.asarray(dd3.get("equilibrium").time).ravel()
    dd3_rows = _observables(dd3)

    compared = 0
    for fixture_row, fixture_time in zip(dd4_rows, dd4_times, strict=True):
        matches = np.flatnonzero(dd3_times == fixture_time)
        if not matches.size:
            continue
        row = dd3_rows[int(matches[0])]
        source_map = _translate("CORSICA 130506/403", "3.38.1", row)
        assert not source_map.blocked, source_map.blocked
        flux = next(
            row_ for row_ in source_map.signals if row_.transformation == PSI_LIKE
        )
        safety = next(
            row_ for row_ in source_map.signals if row_.transformation == Q_LIKE
        )
        translated_boundary = flux.apply([row["psi_boundary"]])[0]
        assert np.sign(translated_boundary) == np.sign(fixture_row["psi_boundary"])
        # the 11 to 17 mapping leaves q alone, so the factor is exactly one and
        # the translated value is the stored value bit for bit
        assert safety.convention_factor == 1.0
        assert safety.apply([row["q"]])[0] == row["q"]
        # the two are separate reconstructions of the case rather than one
        # converted into the other, so they are compared on the physical
        # agreement between them, not to identity
        assert safety.apply([row["q"]])[0] == pytest.approx(fixture_row["q"], rel=2e-2)
        compared += 1
    assert compared, "no shared time between the DD3 run and the DD4 fixture"


# --- the literal negative control -------------------------------------------


def test_negating_psi_on_a_conforming_dd3_slice_is_refused_by_name():
    """Flip the flux on a conforming DD3 slice and the read must be refused.

    A guard that never refuses anything is not a conformance check, so this is
    the mutation the check has to fail against: negating psi_axis and
    psi_boundary flips sigma_Bp to -1, which is COCOS 17's digit, and the slice
    is then no longer data at DD 3.41.0.
    """

    entry = _open(DD3_CORSICA, "3.41.0")
    rows = [row for row in _observables(entry) if _above_one_mega_ampere(row)]
    assert rows
    row = rows[0]
    assert not _translate("CORSICA 130506/6", "3.41.0", row).blocked

    negated = dict(row, psi_axis=-row["psi_axis"], psi_boundary=-row["psi_boundary"])
    source_map = _translate("CORSICA 130506/6", "3.41.0", negated)
    assert not source_map.signals
    assert len(source_map.blocked) == 1
    blocked = source_map.blocked[0]
    assert "CORSICA 130506/6" in blocked.reason
    assert "3.41.0" in blocked.reason
    assert "sigma_Bp=-1" in blocked.reason
    assert f"COCOS {DD4_CONVENTION}" in blocked.reason
