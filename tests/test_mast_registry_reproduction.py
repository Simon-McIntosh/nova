"""Hold the packaged MAST registry reproducible from the catalogs behind it.

The registry is generated output that doubles as an identity: consumers select a
machine by its physical digest and pin that digest by value.  So a reader that
drifts from the packaged file does not announce itself as a bug.  The next census
simply hashes a different payload, finds no configuration matching it, and reports
a hardware reconfiguration that never happened -- from archived sources that
cannot change.

Reading the sources correctly is therefore only half of the guarantee.  The other
half is that the reader and the file stay the same statement about the machine,
which is what these tests hold: the payload the reader builds, and the bytes the
packaged file carries, both from the one shot the registry names as its
representative.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova.catalog.mast_geometry import (
    DEFAULT_LEVEL1_ROOT,
    DEFAULT_LEVEL2_ROOT,
    DEFAULT_REGISTRY_PATH,
    physical_digest,
    physical_snapshot,
    registry_payload,
)

REPRESENTATIVE_SHOT = 11766


@pytest.fixture(scope="module")
def packaged() -> dict:
    """Return the packaged registry exactly as it is stored."""

    return json.loads(DEFAULT_REGISTRY_PATH.read_text())


@pytest.fixture(scope="module")
def catalog_roots() -> tuple[Path, Path]:
    """Return the catalog roots, or skip where they are not mounted."""

    roots = (DEFAULT_LEVEL1_ROOT, DEFAULT_LEVEL2_ROOT)
    if not all((root / f"{REPRESENTATIVE_SHOT}.zarr").exists() for root in roots):
        pytest.skip("MAST L1/L2 catalogs are not mounted")
    return roots


@pytest.fixture(scope="module")
def rebuilt(catalog_roots) -> dict:
    """Return the physical payload read fresh from the catalogs."""

    return physical_snapshot(REPRESENTATIVE_SHOT, *catalog_roots)


def test_the_representative_shot_is_the_one_the_registry_names(packaged) -> None:
    assert packaged["provenance"]["representative_shot"] == REPRESENTATIVE_SHOT


def test_the_reader_rebuilds_the_packaged_geometry(packaged, rebuilt) -> None:
    """Every recovered component, pose and path matches what is published."""

    stored = next(iter(packaged["configurations"].values()))["geometry"]

    assert set(rebuilt) == set(stored)
    for family in sorted(rebuilt):
        assert rebuilt[family] == stored[family], family


def test_the_reader_rebuilds_the_packaged_identity(packaged, rebuilt) -> None:
    """The identity consumers pin follows from the sources, not from the file."""

    assert physical_digest(rebuilt) == next(iter(packaged["configurations"]))


def test_the_packaged_registry_is_byte_for_byte_reproducible(packaged, rebuilt) -> None:
    """Nothing in the published file is reachable only by hand-editing it.

    This covers the fields a payload comparison alone would let drift -- the
    registry digest, the shot ranges, the superseded identities and the
    serialisation itself.
    """

    payload = registry_payload(rebuilt, packaged["incomplete_evidence"])

    assert payload == packaged
    assert json.dumps(payload, indent=2, sort_keys=True) + "\n" == (
        DEFAULT_REGISTRY_PATH.read_text()
    )
