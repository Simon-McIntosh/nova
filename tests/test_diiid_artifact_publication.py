from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova.imas.diiid_machine_ids import SOURCE_PATH
from nova.imas.machine_artifact import MachineArtifactError, publication_dd_version
from nova.scripts.diiid_machine_artifact import (
    create_diiid_artifact_manifest,
    publish_diiid_machine_artifact,
    write_prior_major_probe,
)

REGISTRY_DIGEST = "a" * 64
PHYSICAL_DIGEST = "b" * 64
UNRESOLVED_GAPS = ("passive structure is not supplied",)


def test_builds_manifest_cache_receipt_and_publication_recipe(tmp_path: Path) -> None:
    if not SOURCE_PATH.is_file():
        pytest.skip("the authoritative DIII-D netCDF source is not mounted")
    output = tmp_path / "diiid_machine_description.nc"
    receipt_path = tmp_path / "artifact.receipt.json"
    manifest_path = tmp_path / "diiid_machine_description.manifest.json"
    recipe_path = tmp_path / "PUBLISH.md"
    receipt = publish_diiid_machine_artifact(
        repository="ghcr.io/test-account/diiid-machine-description",
        cache_directory=tmp_path / "cache",
        source_path=SOURCE_PATH,
        output=output,
        receipt_path=receipt_path,
        manifest_path=manifest_path,
        recipe_path=recipe_path,
    )

    publication = receipt["publication"]
    assert publication["data_dictionary"] == publication_dd_version()
    assert publication["data_dictionary"].startswith("4.")
    assert publication["data_dictionary_resolver"] == "publication_dd_version()"
    assert publication["manifest_schema"] == "nova-diii-d-machine-artifact"
    assert "diii-d" in publication["artifact_type"]
    assert "diii-d" in publication["file_media_type"]
    assert publication["oci_tag"].startswith(
        f"dd-{publication['data_dictionary']}-physical-"
    )
    assert publication["oci_reference"] == (
        f"ghcr.io/test-account/diiid-machine-description:{publication['oci_tag']}"
    )
    assert receipt["cache"]["digests_equal"] is True
    assert (
        receipt["cache"]["materialized_digest"] == receipt["cache"]["resolved_digest"]
    )
    assert receipt["network_publication_attempted"] is False
    assert receipt["data_dictionary_floor"]["refused"] is True
    assert receipt["data_dictionary_floor"]["exception"] == "MachineArtifactError"
    assert (
        "requires data dictionary major 4"
        in receipt["data_dictionary_floor"]["message"]
    )
    assert receipt["round_trip"]["exact_equal"] is True
    assert receipt["round_trip"]["maximum_absolute_difference"] == 0.0
    assert json.loads(receipt_path.read_text()) == receipt
    assert json.loads(manifest_path.read_text()) == receipt["manifest"]

    recipe = recipe_path.read_text()
    assert "GHCR_ACCOUNT='<registry-account>'" in recipe
    assert "GHCR_TOKEN='<registry-token>'" in recipe
    assert "oras push --image-spec v1.1" in recipe
    assert 'oras pull "$REFERENCE"' in recipe
    assert publication["artifact_type"] in recipe
    assert publication["file_media_type"] in recipe
    assert publication["oci_tag"] in recipe
    assert receipt["output"]["sha256"] in recipe
    assert f'"$PULL_DIRECTORY/{output.as_posix()}"' in recipe
    assert "no CI workflow" in recipe
    assert "local operator action" in recipe


def test_publication_path_refuses_ids_declaring_prior_major(tmp_path: Path) -> None:
    ids_path = tmp_path / "prior-major-machine-description.nc"
    declared = write_prior_major_probe(ids_path)
    assert declared.startswith("3.")
    with pytest.raises(
        MachineArtifactError,
        match="requires data dictionary major 4",
    ):
        create_diiid_artifact_manifest(
            ids_path,
            registry_digest=REGISTRY_DIGEST,
            physical_digest=PHYSICAL_DIGEST,
            source_shot=0,
            unresolved_gaps=UNRESOLVED_GAPS,
        )


def test_repository_is_cli_supplied_and_must_target_ghcr(tmp_path: Path) -> None:
    with pytest.raises(
        MachineArtifactError,
        match="requires ghcr.io/<account>/<repository>",
    ):
        publish_diiid_machine_artifact(
            repository="registry.example.invalid/account/machine-description",
            cache_directory=tmp_path / "cache",
        )
