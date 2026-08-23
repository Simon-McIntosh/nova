"""Pins for the validity-repaired DIII-D machine artifact."""

from __future__ import annotations

from pathlib import Path

import imas
import numpy as np
import pytest
from shapely.geometry import Polygon

from benchmarks.diiid_ids_machine_description import (
    VERBATIM_ARTIFACT_CONTENT_SHA256,
    VERBATIM_ARTIFACT_OCI_TAG,
    VERBATIM_ARTIFACT_PHYSICAL_DIGEST,
    VERBATIM_ARTIFACT_SEMANTIC_IDENTITY,
    write_repaired_artifact_receipt,
)
from nova.imas.diiid_machine_ids import SOURCE_PATH, build_diiid_machine_ids
from nova.imas.machine_artifact import MachineArtifactManifest, publication_dd_version
from nova.scripts.diiid_machine_artifact import publish_diiid_machine_artifact


def _provenance_sources(node: object) -> list[str]:
    if hasattr(node, "sources"):
        return [str(value) for value in node.sources]
    return [str(reference.name) for reference in node.reference]


def test_source_chain_repairs_to_one_declared_simple_ring() -> None:
    if not SOURCE_PATH.is_file():
        pytest.skip("the authoritative DIII-D netCDF source is not mounted")

    bundle = build_diiid_machine_ids(SOURCE_PATH)
    repair = bundle.limiter_repair
    wall = bundle.ids["wall"]
    outline = wall.description_2d[0].limiter.unit[0].outline
    ring = np.column_stack((np.asarray(outline.r), np.asarray(outline.z)))
    polygon = Polygon(ring[:-1])

    assert bundle.dd_version == publication_dd_version()
    assert bundle.dd_version.startswith("4.")
    assert bundle.source_dd_version == "3.41.0"
    assert repair.source_chain_valid is False
    assert repair.source_chain_sha256 == (
        "32971da7a46af5e2b4f523aa745888070cccf3d780a5f4687098bfafe6619b15"
    )
    assert repair.source_vertex_count == 82
    assert repair.validity_component_count == 2
    assert repair.valid_material_relative_area_difference == pytest.approx(
        4.1593370133e-05,
        rel=1.0e-10,
    )
    assert repair.published_vertex_count == 84
    assert repair.excluded_component_area_m2 == pytest.approx(5.7936427519e-05)
    assert ring.shape == (84, 2)
    assert np.array_equal(ring[0], ring[-1])
    assert polygon.is_valid
    assert polygon.exterior.is_simple
    assert polygon.geom_type == "Polygon"

    sources = _provenance_sources(wall.ids_properties.provenance.node[0])
    repair_source = next(
        source for source in sources if "limiter validity repair" in source
    )
    assert f"source_chain_sha256={repair.source_chain_sha256}" in repair_source
    assert "authority=repaired-ring-only" in repair_source
    assert "valid_material_relative_area_difference=4.159337013" in repair_source


def test_repaired_artifact_round_trips_and_has_distinct_publication_identity(
    tmp_path: Path,
) -> None:
    if not SOURCE_PATH.is_file():
        pytest.skip("the authoritative DIII-D netCDF source is not mounted")
    ids_path = tmp_path / "diiid_machine_description.nc"
    manifest_path = tmp_path / "diiid_machine_description.manifest.json"
    recipe_path = tmp_path / "PUBLISH.md"
    artifact_receipt_path = tmp_path / "artifact.receipt.json"
    receipt = publish_diiid_machine_artifact(
        repository="ghcr.io/test-account/diii-d-machine-description",
        cache_directory=tmp_path / "cache",
        source_path=SOURCE_PATH,
        output=ids_path,
        receipt_path=artifact_receipt_path,
        manifest_path=manifest_path,
        recipe_path=recipe_path,
    )
    manifest = MachineArtifactManifest.from_bytes(manifest_path.read_bytes())
    identity_receipt_path = tmp_path / "repaired-ring.receipt.json"
    identity = write_repaired_artifact_receipt(
        source_path=SOURCE_PATH,
        ids_path=ids_path,
        manifest_path=manifest_path,
        artifact_receipt_path=artifact_receipt_path,
        recipe_path=recipe_path,
        receipt_path=identity_receipt_path,
    )

    assert receipt["round_trip"] == {
        "authored_leaf_count": 1550,
        "exact_equal": True,
        "maximum_absolute_difference": 0.0,
    }
    assert manifest.dd_version == publication_dd_version()
    assert manifest.files[0].sha256 != VERBATIM_ARTIFACT_CONTENT_SHA256
    assert manifest.physical_digest != VERBATIM_ARTIFACT_PHYSICAL_DIGEST
    assert manifest.semantic_identity() != VERBATIM_ARTIFACT_SEMANTIC_IDENTITY
    assert manifest.oci.tag != VERBATIM_ARTIFACT_OCI_TAG
    assert all(identity["identity_comparison"].values())
    assert identity["artifact"]["semantic_identity"] == manifest.semantic_identity()
    assert identity["publication"]["network_publication_attempted"] is False
    assert receipt["network_publication_attempted"] is False
    assert [gap.split(":", 1)[0] for gap in receipt["manifest"]["unresolved_gaps"]] == [
        "Thomson scattering line-of-sight endpoints",
        "pf_passive",
        "tf static conductor geometry",
    ]

    with imas.DBEntry(ids_path, "r", dd_version=manifest.dd_version) as database:
        wall = database.get("wall", 0, autoconvert=False)
        outline = wall.description_2d[0].limiter.unit[0].outline
        ring = np.column_stack((np.asarray(outline.r), np.asarray(outline.z)))
        assert Polygon(ring[:-1]).is_valid
        assert database.list_all_occurrences("pf_passive") == []
        assert database.list_all_occurrences("tf") == []
        assert database.list_all_occurrences("equilibrium") == []

    recipe = recipe_path.read_text()
    assert "oras push --image-spec v1.1" in recipe
    assert 'oras pull "$REFERENCE"' in recipe
    assert manifest.oci.tag in recipe
    assert manifest.files[0].sha256 in recipe
