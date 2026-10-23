"""The equilibrium reader serves its signed quantities in COCOS 17.

The reader renormalises its profiles to ``psi_norm``, and a sign reversal of
psi leaves that normalisation unchanged, so a test of normalised quantities
cannot see the translation at all. These tests assert the signed quantities the
reader's convention build produces against the convention of the pulse they
were read from, assert that the read seam applies the same map, and assert that
the reader refuses the two DD3 pulses whose own digits contradict COCOS 11.

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
from nova.imas.getslice import GetSlice
from nova.io.sourcemap import NOVA_CONVENTION, translate_equilibrium
from nova_cocos import DODPSI_LIKE, IP_LIKE, PSI_LIKE, convention, convention_transform

IMASDB = Path("/work/imas/shared/imasdb/ITER/3")

#: conforming DD3 pulses, as ``pulse/run -> written DD version``.
CASES = {
    "130506/6": "3.41.0",
    "105069/3": "3.39.0",
}

#: pulses whose digits contradict their DD version, as ``pulse/run`` ->
#: ``(version, failing digit)``.
REFUSED = {
    "105014/1": ("3.38.1", "sigma_Bp=-1"),
    "105013/1": ("3.38.1", "sigma_rho_theta_phi=-1"),
}

#: a conforming DD3 pulse none of these tests builds for any other reason, so
#: the one test that builds the whole reader meets it uncached.
UNCACHED = ("105052/1", "3.38.1")


class NoMachine:
    """A stand-in for the machine geometry the reader reads beside its pulse.

    The wall inserts nothing, and the divertor strike the contour build reads
    from the same machine carries two points, so the build completes without
    the machine description. Neither the wall outline nor the strike point
    takes part in the convention the convention arm asserts.
    """

    points = np.array([[0.0, 0.0], [1.0, 0.0]])

    def insert(self, data):
        return None

    def update(self, surfaces):
        return None


@pytest.fixture
def no_machine(monkeypatch):
    """Serve the convention build without the machine the reader reads with it.

    The wall read is a defect of its own, outside this reader's convention
    seam, and it is what the last test in this file exercises. Substituting it
    lets the convention arm assert the reader's own output for a pulse whose
    machine description cannot be built.
    """
    machine = NoMachine()
    monkeypatch.setattr(EquilibriumData, "wall", property(lambda self: machine))
    monkeypatch.setattr(EquilibriumData, "strike", property(lambda self: machine))


def open_ids(case: str, dd_version: str):
    """Open one pulse's equilibrium IDS at the version it was written in."""
    path = IMASDB / case
    if not path.is_dir():
        pytest.skip(f"imasdb pulse {case} not found")
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


def source_map(case: str, dd_version: str):
    """Return the signal map one pulse's own observables pin."""
    ids = open_ids(case, dd_version)
    raw = ids.time_slice[0]
    field = np.asarray(ids.vacuum_toroidal_field.b0).ravel()
    if field.size == 0:
        field = np.asarray(raw.vacuum_toroidal_field.b0).ravel()
    return translate_equilibrium(
        source_name=case,
        dd_version=dd_version,
        ip=float(raw.global_quantities.ip),
        psi_axis=float(raw.global_quantities.psi_axis),
        psi_boundary=float(raw.global_quantities.psi_boundary),
        q=float(raw.global_quantities.q_95),
        b0=float(field[0]) if field.size else float("nan"),
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_signed_quantities_carry_the_internal_convention(case, no_machine):
    """Every convention-bearing field the reader holds is in COCOS 17."""
    equilibrium = open_equilibrium(case, CASES[case])
    equilibrium.itime = 0
    assert equilibrium.data.attrs["cocos"] == NOVA_CONVENTION

    raw = open_ids(case, CASES[case]).time_slice[0]
    factor = convention_transform(source=11, target=NOVA_CONVENTION)
    psi_factor = factor.factor(PSI_LIKE)
    dpsi_factor = factor.factor(DODPSI_LIKE)
    assert psi_factor == convention(NOVA_CONVENTION).sigma_bp
    assert psi_factor == -1.0, "the psi rows must move, or this asserts nothing"

    # psi: the profile, the axis, and the separatrix flux the boundary field
    # carries. The raw read still holds its own DD version's convention, so
    # each signed quantity differs from its raw counterpart by that factor.
    assert np.isclose(
        float(equilibrium["psi1d"][0]), psi_factor * float(raw.profiles_1d.psi[0])
    )
    assert np.isclose(
        float(equilibrium["psi_axis"]),
        psi_factor * float(raw.global_quantities.psi_axis),
    )
    assert np.isclose(
        float(equilibrium["psi_boundary"]),
        psi_factor * float(raw.boundary_separatrix.psi),
    )
    # Where the two stored nodes differ, the boundary field must track the
    # separatrix one, which is the assertion a declared path naming the global
    # quantity would fail. On a pulse whose two nodes agree it says nothing,
    # and it is skipped rather than spelled as an inequality that both satisfy.
    boundary_psi = float(raw.boundary_separatrix.psi)
    global_psi = float(raw.global_quantities.psi_boundary)
    if not np.isclose(boundary_psi, global_psi):
        assert np.isclose(
            float(equilibrium["psi_boundary"]), psi_factor * boundary_psi
        )
        assert not np.isclose(
            float(equilibrium["psi_boundary"]), psi_factor * global_psi
        ), "the boundary field carries the separatrix flux, not the global quantity"

    # p' and FF' are taken with respect to the negated total flux.
    assert np.isclose(
        float(np.asarray(equilibrium["dpressure_dpsi"], float)[0]),
        dpsi_factor * float(raw.profiles_1d.dpressure_dpsi[0]),
    )
    assert np.sign(
        float(np.asarray(equilibrium["f_df_dpsi"], float)[0])
    ) == dpsi_factor * np.sign(float(raw.profiles_1d.f_df_dpsi[0]))

    # Ip and q do not move under the 11 -> 17 factor: a control that must
    # still read its raw counterpart while the rows above have moved.
    assert factor.factor(IP_LIKE) == 1.0
    assert np.isclose(float(equilibrium["ip"]), float(raw.global_quantities.ip))
    assert np.isclose(float(equilibrium["q_95"]), float(raw.global_quantities.q_95))


@pytest.mark.parametrize("case", sorted(CASES))
def test_read_seam_applies_the_map(case):
    """GetSlice.translated moves exactly the fields the map marks."""
    signal_map = source_map(case, CASES[case])
    assert not signal_map.blocked

    values = np.array([1.0, -2.0, 3.5])
    factor = convention_transform(source=11, target=NOVA_CONVENTION)
    for path, expected in [
        ("equilibrium/profiles_1d/psi", factor.factor(PSI_LIKE)),
        ("equilibrium/global_quantities/psi_boundary", factor.factor(PSI_LIKE)),
        ("equilibrium/boundary_separatrix/psi", factor.factor(PSI_LIKE)),
        ("equilibrium/profiles_1d/dpressure_dpsi", factor.factor(DODPSI_LIKE)),
        ("equilibrium/global_quantities/ip", 1.0),
        ("equilibrium/global_quantities/q_95", 1.0),
    ]:
        assert np.allclose(
            GetSlice.translated(signal_map, path, values), expected * values
        ), path


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_reader_refuses_non_conforming_data_by_name(case):
    """A DD3 pulse whose digits contradict COCOS 11 is refused with its digit."""
    dd_version, digit = REFUSED[case]
    with pytest.raises(ValueError) as refused:
        open_equilibrium(case, dd_version)
    message = str(refused.value)
    assert "does not conform" in message
    assert digit in message


@pytest.mark.xfail(
    strict=True,
    reason="the ITER wall build raises KeyError 'plasma' in the frame's "
    "columnar store, so the reader's full build cannot complete",
)
def test_full_build_completes_on_a_dd3_pulse():
    """The whole reader build, wall geometry included, on a DD3 pulse."""
    equilibrium = open_equilibrium(*UNCACHED)
    assert equilibrium.data.attrs["cocos"] == NOVA_CONVENTION
    assert equilibrium.wall is not None