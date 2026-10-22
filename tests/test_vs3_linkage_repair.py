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
import shutil
from pathlib import Path

import numpy as np
import pytest

from nova.circuit.passive import (
    JointLinkageNotPositiveDefinite,
    load_sectioned_machine,
)
from nova.circuit.screening import plasma_self_inductance

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit")
ENTRY_ROOT = CACHE / "sectioned-conductor-set"

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

PINNED_PLANT_ENTRY = "28148563521d92a2198b"
PINNED_PLANT = CACHE / "coupled-plant" / PINNED_PLANT_ENTRY / "plant.npz"
PINNED_PLANT_SHA256 = "7800a4d8db5ea452fca7e0a18179f8939f85e08e8503b61e4e1bbee8d3c06b6c"

#: the single-conductor-in-jacket dense coaxial reference the vertical coils'
#: bare inductances are stated against [H], and the tolerance on them.
COAXIAL_REFERENCE_H = 4.1581e-05
COAXIAL_TOLERANCE = 0.03

#: machine-description cross-section of one VS3 conductor annulus [m]
CONDUCTOR_ANNULUS = (0.01695, 0.02300)

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


@pytest.mark.parametrize("label", ["12mm", "6mm"])
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


@pytest.mark.parametrize("label", ["12mm", "6mm"])
def test_every_coil_bare_inductance_is_positive(label):
    """Each of the 14 coil channels stores positive inductive energy."""
    entry = _entry(label)
    _skip_without(entry, "linkage.npy", "system.npz", "geometry.npz")
    machine = load_sectioned_machine(entry)
    coil_matrix = machine.series.T @ machine.section_inductance @ machine.series
    bare = np.diag(coil_matrix)
    assert np.all(bare > 0.0), dict(zip(machine.coil_names, bare, strict=True))


@pytest.mark.parametrize("label", ["12mm", "6mm"])
def test_vertical_coil_bare_inductance_matches_the_coaxial_reference(label):
    """VSU's and VSL's bare inductances match the dense coaxial reference.

    Recorded pre-repair evidence, kept as the numbers the repair had to move:
    the reduced annulus arm gave 4.1926e-5 H at 8 sectors and 4.1627e-5 H at 16
    against the reference's 4.1581e-5 H, with a coupling of 0.93 against the
    reference's 0.9759.  The tolerance is stated against the reference, not
    against the arm's own earlier output.  The units are henries throughout, so
    the tolerance is a relative one.
    """
    entry = _entry(label)
    _skip_without(entry, "linkage.npy", "system.npz", "geometry.npz")
    machine = load_sectioned_machine(entry)
    coil_matrix = machine.series.T @ machine.section_inductance @ machine.series
    names = list(machine.coil_names)
    for name in ("VSU", "VSL"):
        measured = float(np.diag(coil_matrix)[names.index(name)])
        assert measured == pytest.approx(COAXIAL_REFERENCE_H, rel=COAXIAL_TOLERANCE), (
            name,
            measured,
        )


def test_coaxial_reference_is_the_repository_closed_form():
    """The reference the tolerance is stated against agrees with the closed form.

    Recorded pre-repair evidence: the closed form gives 4.2054e-5 H at the
    annulus's geometric-mean wire radius, 1.1 per cent from the dense coaxial
    sum of 4.1581e-5 H that ``COAXIAL_REFERENCE_H`` records.  The two are the
    same physical quantity at the same radius, so this is what ties the stated
    tolerance to a form the repository already owns.
    """
    radius = 6.0
    closed_form = float(
        plasma_self_inductance(radius, np.sqrt(np.prod(CONDUCTOR_ANNULUS)))
    )
    assert closed_form == pytest.approx(4.2054e-05, rel=0.01)


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
