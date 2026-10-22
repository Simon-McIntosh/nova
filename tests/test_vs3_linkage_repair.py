"""The repaired sectioned set stores a positive-definite joint linkage, re-keyed.

The stored ``linkage.npy`` is the joint coil-plus-passive matrix the build
actually uses, so every eigenvalue of it must be positive: a non-positive one
is an inductance matrix that stores negative energy for some current vector.  A
pre-repair entry stores the raw indefinite matrix and is refused by name.  This
gate measures the repaired keys, the repaired entry's spectra, and the loader's
refusal.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ellipe, ellipk

from nova.circuit.passive import (
    JointLinkageNotPositiveDefinite,
    load_sectioned_machine,
)
from nova.circuit.screening import plasma_self_inductance

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit")
#: the sectioned-set root the arms read.  A fenced worker cannot write the shared
#: cache, so a build under test names its own root here and the arms measure the
#: entries that build produced rather than skipping on an uninstalled one.
ENTRY_ROOT = Path(
    os.environ.get("NOVA_SECTIONED_SET_ROOT", CACHE / "sectioned-conductor-set")
)

#: the sectioned-set keys the repair retired, named by the nodes that pinned
#: them.  They may still exist as directories under the shared cache, which is
#: where a scratch copy of a pre-repair entry is taken from.
RETIRED_KEYS = {
    "12mm": "68df15665ab2ea45f427",
    "6mm": "06ff06d3eb7a4448f0fe",
}

#: the keys the repaired build derives, and the keys every in-tree pin moves to.
REPAIRED_KEYS = {
    "12mm": "66c10162f3e37adb2679",
    "6mm": "f3a24fdd12eae6348f3a",
}

#: why the 6 mm arms are expected to fail.  The measured state is the one the
#: loader refuses: 40 negative eigenvalues with a smallest value of
#: -4.009411e-08 H in the passive block, whose smallest eigenvalue is
#: -9.192799e-04 of the block's mean self term against +5.692051e-05 at 12 mm.
#: The mechanism behind that near-singular collective mode is still open, so the
#: arms are strict xfails rather than relaxed: an XPASS is a refusal, which is
#: what tells the reader the entry has changed.
SIX_MM_REFUSED = (
    "the 6 mm entry's passive block stores 40 negative eigenvalues, smallest "
    "-4.009411e-08 H, and the block is collectively near-singular rather than "
    "over-coupled; the mechanism is under investigation"
)

#: the two meshes, with the 6 mm arms expected to fail on that entry's defect.
LABELS = [
    pytest.param("12mm"),
    pytest.param("6mm", marks=pytest.mark.xfail(strict=True, reason=SIX_MM_REFUSED)),
]

PINNED_PLANT_ENTRY = "28148563521d92a2198b"
PINNED_PLANT = CACHE / "coupled-plant" / PINNED_PLANT_ENTRY / "plant.npz"
PINNED_PLANT_SHA256 = "7800a4d8db5ea452fca7e0a18179f8939f85e08e8503b61e4e1bbee8d3c06b6c"

#: the annulus each VS3 turn is wound from, in the machine description [m].
TURN_ANNULUS = (0.01695, 0.02300)

#: the four turn annuli of each vertical coil, in series, at the (R, Z) centres
#: ``pf_active`` gives them [m].  A vertical coil is four turns, not one ring, so
#: its bare inductance is the sum of the four self terms and the six distinct
#: mutual pairs between them -- which is why a single-ring closed form at R = 6 m
#: (4.370e-05 H) is not a reference for either coil.
VS3_TURNS = {
    "VSU": (
        (5.80665907, 4.93980766),
        (5.86330474, 4.91128576),
        (5.78032370, 4.88598875),
        (5.83643430, 4.85653910),
    ),
    "VSL": (
        (7.55055806, -2.50001788),
        (7.49633281, -2.54109841),
        (7.51139746, -2.44931726),
        (7.45665771, -2.49006112),
    ),
}

#: tolerance on each coil's bare inductance and on the coil-to-coil mutual,
#: against their own turn references.  Stated from the measurement: the stored
#: series map stands 0.25 per cent above its reference for VSU (5.630347e-04
#: against 5.644469e-04) and 0.28 per cent for VSL (7.560341e-04 against
#: 7.581931e-04), while the VSU-VSL mutual agrees to 0.0006 per cent
#: (4.241052e-05 against 4.241026e-05).  One per cent is about four times the
#: largest measured disagreement, and the disagreement that remains sits in the
#: self terms, where the sector discretisation lives.
TURN_REFERENCE_TOLERANCE = 0.01

#: permeability of free space [H/m]
MU0 = 4e-7 * np.pi


def _coaxial_mutual(r1: float, r2: float, dz: float) -> float:
    """Mutual inductance of two coaxial circular loops [H]."""
    k2 = 4.0 * r1 * r2 / ((r1 + r2) ** 2 + dz**2)
    k = np.sqrt(k2)
    return float(
        MU0 * np.sqrt(r1 * r2) * ((2.0 / k - k) * ellipk(k2) - (2.0 / k) * ellipe(k2))
    )


def _turn_self(r: float) -> float:
    """One turn's self inductance, from the repository's closed form [H]."""
    return float(plasma_self_inductance(r, np.sqrt(np.prod(TURN_ANNULUS))))


def _turn_reference(turns) -> float:
    """A coil's bare inductance as its four turn annuli in series [H].

    The four self terms plus twice each of the six distinct mutual pairs, which
    is the energy of one ampere through all four turns in series.  Every term is
    an independent closed form: the repository's ring self inductance and the
    coaxial ring mutual.
    """
    total = 0.0
    for index, (r, z) in enumerate(turns):
        total += _turn_self(r)
        for r2, z2 in turns[index + 1 :]:
            total += 2.0 * _coaxial_mutual(r, r2, abs(z - z2))
    return float(total)


def _coupling_reference(first, second) -> float:
    """The mutual inductance between two vertical coils, turn by turn [H]."""
    return float(
        sum(
            _coaxial_mutual(r1, r2, abs(z1 - z2))
            for r1, z1 in first
            for r2, z2 in second
        )
    )


#: files the retired-digest search must skip, and why.  This module is the
#: record of what was retired, so it is the one file that has to name the
#: retired keys; every other file in the tree now names the derived ones.
SELF = Path(__file__).name
OUT_OF_SCOPE_DIGEST_PINS = (SELF,)


def _entry(label: str) -> Path:
    return ENTRY_ROOT / REPAIRED_KEYS[label]


def _skip_without(entry: Path, *names: str) -> None:
    if not all((entry / name).exists() for name in names):
        pytest.skip(f"the repaired entry {entry.name} is not present on this host")


@pytest.mark.parametrize("label", LABELS)
def test_stored_joint_linkage_is_positive_definite(label):
    """The stored joint linkage is positive definite, with no floor applied."""
    entry = _entry(label)
    _skip_without(entry, "linkage.npy", "system.npz", "geometry.npz")
    matrix = np.load(entry / "linkage.npy")
    values = np.linalg.eigvalsh(matrix)
    assert int((values < 0).sum()) == 0, float(values.min())
    assert float(values.min()) > 0.0
    machine = load_sectioned_machine(entry)
    assert matrix.shape[0] > machine.passive_circuits.size, (
        "the stored linkage carries no finite-source coil block: it was written "
        "before the block was spliced in"
    )


@pytest.mark.parametrize("label", LABELS)
def test_every_coil_bare_inductance_is_positive(label):
    """Each of the 14 coil channels stores positive inductive energy."""
    entry = _entry(label)
    _skip_without(entry, "linkage.npy", "system.npz", "geometry.npz")
    machine = load_sectioned_machine(entry)
    coil_matrix = machine.series.T @ machine.section_inductance @ machine.series
    bare = np.diag(coil_matrix)
    assert np.all(bare > 0.0), dict(zip(machine.coil_names, bare, strict=True))


@pytest.mark.parametrize("label", LABELS)
def test_vertical_coil_bare_inductance_matches_its_turn_reference(label):
    """VSU's and VSL's bare inductances match their own four-turn references.

    Each vertical coil is four turns in series at the four turn annuli
    ``pf_active`` gives, so the reference is built per coil from the four self
    terms and the six mutual pairs between them -- never from a generic single
    ring.  The reduced series map stands 0.25 per cent above the reference for
    VSU and 0.28 per cent for VSL, so the tolerance is one per cent, about four
    times the measured disagreement.  The units are henries, so it is relative.
    """
    entry = _entry(label)
    _skip_without(entry, "linkage.npy", "system.npz", "geometry.npz")
    machine = load_sectioned_machine(entry)
    coil_matrix = machine.series.T @ machine.section_inductance @ machine.series
    names = list(machine.coil_names)
    for name, turns in VS3_TURNS.items():
        measured = float(np.diag(coil_matrix)[names.index(name)])
        reference = _turn_reference(turns)
        assert measured == pytest.approx(reference, rel=TURN_REFERENCE_TOLERANCE), (
            name,
            measured,
            reference,
        )


@pytest.mark.parametrize("label", LABELS)
def test_vertical_coil_coupling_matches_its_turn_reference(label):
    """The VSU-VSL mutual agrees with the same per-coil reference."""
    entry = _entry(label)
    _skip_without(entry, "linkage.npy", "system.npz", "geometry.npz")
    machine = load_sectioned_machine(entry)
    coil_matrix = machine.series.T @ machine.section_inductance @ machine.series
    names = list(machine.coil_names)
    measured = float(coil_matrix[names.index("VSU"), names.index("VSL")])
    reference = _coupling_reference(VS3_TURNS["VSU"], VS3_TURNS["VSL"])
    assert measured == pytest.approx(reference, rel=TURN_REFERENCE_TOLERANCE), (
        measured,
        reference,
    )


def test_turn_reference_is_built_from_the_repository_closed_form():
    """The per-coil reference is four turns deep, not one ring.

    One turn's self term is ``plasma_self_inductance`` at the annulus's
    geometric-mean wire radius, and the coil reference adds the six mutual pairs
    between its turns.  A single ring at R = 6 m (4.370e-05 H) is one turn at a
    radius no VS3 turn sits at, which is why neither coil could match the
    single-ring constant: the coil reference is about 12.9 times it.
    """
    one_turn = _turn_self(6.0)
    reference = _turn_reference(VS3_TURNS["VSU"])
    assert one_turn == pytest.approx(4.370137793331819e-05, rel=0.01)
    assert reference == pytest.approx(5.64446939481016e-04, rel=1e-6)
    assert reference > 12.0 * one_turn


def _pre_repair_copy(scratch: Path) -> None:
    """Copy a pre-repair entry into ``scratch``, reproducing the defect if absent."""
    retired = ENTRY_ROOT / RETIRED_KEYS["12mm"]
    if (retired / "linkage.npy").exists():
        for name in ("linkage.npy", "system.npz", "geometry.npz"):
            source = retired / name
            if source.exists():
                shutil.copy2(source, scratch / name)
        return
    repaired = _entry("12mm")
    if not (repaired / "linkage.npy").exists():
        pytest.skip("neither a retired nor a repaired sectioned-set entry is present")
    for name in ("linkage.npy", "system.npz", "geometry.npz"):
        shutil.copy2(repaired / name, scratch / name)
    matrix = np.load(scratch / "linkage.npy")
    values, vectors = np.linalg.eigh(matrix)
    values[0] = -abs(values[0])
    np.save(scratch / "linkage.npy", (vectors * values) @ vectors.T)


def test_loader_refuses_a_pre_repair_entry_by_name(tmp_path):
    """An entry whose stored joint linkage is not positive definite is refused.

    Removed repairably: with the refusal in ``load_sectioned_machine`` deleted,
    the ``pytest.raises`` below fails on its own assertion because the loader
    returns a machine built from a matrix that stores negative energy.
    """
    scratch = tmp_path / "sectioned-conductor-set" / "pre-repair"
    scratch.mkdir(parents=True)
    _pre_repair_copy(scratch)
    matrix = np.load(scratch / "linkage.npy")
    assert float(np.linalg.eigvalsh(matrix).min()) < 0.0, (
        "the scratch copy is not a pre-repair entry: it stores no negative "
        "eigenvalue, so it cannot exercise the refusal"
    )
    with pytest.raises(JointLinkageNotPositiveDefinite, match="non-positive"):
        load_sectioned_machine(scratch)


def test_repaired_entry_is_not_refused(tmp_path):
    """A repaired copy of the same entry loads, so the refusal separates them."""
    entry = _entry("12mm")
    _skip_without(entry, "linkage.npy", "system.npz", "geometry.npz")
    scratch = tmp_path / "sectioned-conductor-set" / "repaired"
    scratch.mkdir(parents=True)
    for name in ("linkage.npy", "system.npz", "geometry.npz"):
        shutil.copy2(entry / name, scratch / name)
    machine = load_sectioned_machine(scratch)
    assert machine.passive_circuits.size > 0


def test_retired_digests_are_gone_from_the_tree():
    """No in-tree source or test source still names a retired sectioned-set key."""
    searched = sorted((ROOT / "nova").rglob("*.py"))
    searched += sorted((ROOT / "scripts").rglob("*.py"))
    searched += [
        path
        for path in sorted((ROOT / "tests").rglob("*.py"))
        if path.name not in OUT_OF_SCOPE_DIGEST_PINS
    ]
    assert len(searched) > 50, "the digest search found too few files to be a search"
    found = {
        str(path.relative_to(ROOT)): [
            digest for digest in RETIRED_KEYS.values() if digest in path.read_text()
        ]
        for path in searched
        if any(digest in path.read_text() for digest in RETIRED_KEYS.values())
    }
    assert found == {}, found


def test_repaired_keys_are_pinned_where_the_retired_ones_were():
    """Every in-tree pin moved to the repaired 12 mm key."""
    for relative in (
        "scripts/iter_vessel_evolution/evolve.py",
        "tests/test_coupled_plant.py",
        "tests/test_sectioned_set_geometry_cache.py",
    ):
        text = (ROOT / relative).read_text()
        assert REPAIRED_KEYS["12mm"] in text, relative


def test_pinned_plant_entry_keeps_its_digest():
    """The pinned 105069/3 plant entry is neither re-keyed nor rewritten."""
    if not PINNED_PLANT.exists():
        pytest.skip("the pinned coupled-plant entry is not present on this host")
    digest = hashlib.sha256(PINNED_PLANT.read_bytes()).hexdigest()
    assert digest == PINNED_PLANT_SHA256
