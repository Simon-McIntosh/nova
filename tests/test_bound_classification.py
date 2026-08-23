from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks import efit_parity_criterion_provenance as provenance
from benchmarks.efit_topology_boundary_score import resolve_semantic_machine_artifact
from nova.imas.machine_artifact import (
    ArtifactShotRange,
    MachineArtifactError,
    create_machine_artifact_manifest,
    materialize_machine_artifact,
)

REGISTRY_DIGEST = "a" * 64
PHYSICAL_DIGEST = "b" * 16


def _materialized_artifact(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "machine.nc").write_bytes(b"semantic citation fixture")
    manifest = create_machine_artifact_manifest(
        source,
        machine="mast",
        dd_version="4.1.1",
        registry_digest=REGISTRY_DIGEST,
        physical_digest=PHYSICAL_DIGEST,
        shot_ranges=(
            ArtifactShotRange(
                first_shot=11766,
                last_shot=30471,
                physical_digest=PHYSICAL_DIGEST,
                evidence="observed",
            ),
        ),
        complete=False,
        unresolved_gaps=("diagnostic calibration is unresolved",),
    )
    stored = materialize_machine_artifact(source, tmp_path / "cache", manifest)
    return tmp_path / "cache", stored


def test_semantic_resolution_verifies_the_selected_materialisation(tmp_path: Path):
    cache, stored = _materialized_artifact(tmp_path)

    resolved = resolve_semantic_machine_artifact(
        cache,
        stored.manifest.semantic_identity(),
        expected_physical_digest=PHYSICAL_DIGEST,
        expected_registry_digest=REGISTRY_DIGEST,
    )

    assert resolved["semantic_identity"] == stored.manifest.semantic_identity()
    assert resolved["materialisation_digest"] == stored.digest
    assert resolved["fully_verified"] is True
    assert resolved["complete"] is False
    assert resolved["unresolved_gap_count"] == 1


def test_semantic_resolution_fails_closed_on_missing_identity(tmp_path: Path):
    cache, _ = _materialized_artifact(tmp_path)

    with pytest.raises(MachineArtifactError, match="is absent"):
        resolve_semantic_machine_artifact(
            cache,
            "sha256:" + "0" * 64,
            expected_physical_digest=PHYSICAL_DIGEST,
            expected_registry_digest=REGISTRY_DIGEST,
        )


def test_semantic_resolution_fails_closed_on_physical_mismatch(tmp_path: Path):
    cache, stored = _materialized_artifact(tmp_path)

    with pytest.raises(MachineArtifactError, match="physical digest mismatch"):
        resolve_semantic_machine_artifact(
            cache,
            stored.manifest.semantic_identity(),
            expected_physical_digest="c" * 16,
            expected_registry_digest=REGISTRY_DIGEST,
        )


def test_classification_table_covers_all_five_carried_bounds(tmp_path: Path):
    cache, stored = _materialized_artifact(tmp_path)
    resolution = resolve_semantic_machine_artifact(
        cache,
        stored.manifest.semantic_identity(),
        expected_physical_digest=PHYSICAL_DIGEST,
        expected_registry_digest=REGISTRY_DIGEST,
    )

    rows = provenance._bound_classification_table(resolution)

    assert len(rows) == 5
    assert {row["classification"] for row in rows} == {
        "supported",
        "merely contained",
        "reclassified",
    }
    by_field = {row["field"]: row for row in rows}
    assert by_field["magnetic_axis_distance_m"]["classification"] == (
        "merely contained"
    )
    assert by_field["lcfs_distance_m"]["classification"] == "merely contained"
    assert by_field["x_point_distance_m"]["classification"] == "supported"
    assert by_field["topology_class_agreement_fraction"]["classification"] == (
        "supported"
    )
    fixed_point = by_field["fixed_point_defect"]
    assert fixed_point["classification"] == "reclassified"
    assert "stopping policy" in fixed_point["replacement_reading"]


def test_axis_and_lcfs_citations_use_semantic_identity(tmp_path: Path):
    cache, stored = _materialized_artifact(tmp_path)
    resolution = resolve_semantic_machine_artifact(
        cache,
        stored.manifest.semantic_identity(),
        expected_physical_digest=PHYSICAL_DIGEST,
        expected_registry_digest=REGISTRY_DIGEST,
    )

    rows = provenance._bound_classification_table(resolution)
    citations = [row["semantic_citation"] for row in rows if row["semantic_citation"]]

    assert len(citations) == 2
    assert {citation["metric"] for citation in citations} == {
        "magnetic_axis_distance_m",
        "lcfs_distance_m",
    }
    assert all(
        citation["identity_kind"] == "MachineArtifactManifest.semantic_identity"
        for citation in citations
    )
    assert all(
        citation["semantic_identity"] == stored.manifest.semantic_identity()
        for citation in citations
    )
    assert all(citation["fully_verified"] for citation in citations)
