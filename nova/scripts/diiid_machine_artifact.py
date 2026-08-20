"""Build and locally verify the publishable DIII-D machine artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import imas
from imas_data_dictionaries import dd_xml_versions, parse_dd_version

from nova.imas.diiid_machine_ids import (
    IDS_NAMES,
    SOURCE_PATH,
    DiiidMachineIds,
    build_diiid_machine_ids,
    machine_ids_snapshot,
    round_trip_leaf_receipt,
)
from nova.imas.machine_artifact import (
    ArtifactShotRange,
    MachineArtifactError,
    MachineArtifactManifest,
    create_machine_artifact_manifest,
    materialize_machine_artifact,
    oci_artifact_reference,
    oci_artifact_tag,
    publication_dd_version,
    resolve_machine_artifact,
)

MACHINE = "DIII-D"
DEFAULT_OUTPUT = Path(
    "docs/figures/diiid-forward-onboarding/ids-set/diiid_machine_description.nc"
)
DEFAULT_RECEIPT = DEFAULT_OUTPUT.with_name("diiid_machine_artifact.receipt.json")
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(".manifest.json")
DEFAULT_RECIPE = DEFAULT_OUTPUT.with_name("PUBLISH.md")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _declared_dd_version(path: Path) -> str:
    """Read an IDS file's own dictionary version without conversion."""

    with imas.DBEntry(path.resolve(), "r") as database:
        wall = database.get("wall", 0, lazy=True, autoconvert=False)
        return str(wall.ids_properties.version_put.data_dictionary)


def _write_and_verify_ids(
    bundle: DiiidMachineIds,
    output: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Write the native IDSs and require exact leaf equality after reopening."""

    output.parent.mkdir(parents=True, exist_ok=True)
    expected = machine_ids_snapshot(bundle.ids)
    database = imas.DBEntry(output.resolve(), "w", dd_version=bundle.dd_version)
    try:
        for name in IDS_NAMES:
            database.put(bundle.ids[name])
    finally:
        database.close()

    database = imas.DBEntry(output.resolve(), "r", dd_version=bundle.dd_version)
    try:
        reopened = {
            name: database.get(name, 0, autoconvert=False) for name in IDS_NAMES
        }
    finally:
        database.close()
    comparison = round_trip_leaf_receipt(
        expected,
        machine_ids_snapshot(reopened),
    )
    leaves = [result for values in comparison.values() for result in values.values()]
    if not all(result["exact_equal"] for result in leaves):
        raise ValueError("DIII-D machine IDS leaves changed during netCDF round trip")
    maximum_difference = max(
        (
            float(result["maximum_absolute_difference"])
            for result in leaves
            if "maximum_absolute_difference" in result
        ),
        default=0.0,
    )
    return expected, {
        "authored_leaf_count": len(leaves),
        "exact_equal": True,
        "maximum_absolute_difference": maximum_difference,
    }


def _artifact_digests(
    snapshot: Mapping[str, Mapping[str, Any]],
    bundle: DiiidMachineIds,
) -> tuple[str, str]:
    registry_record = {
        "declared_absent": [absence.as_dict() for absence in bundle.absent],
        "ids": snapshot,
        "source_data_dictionary": bundle.source_dd_version,
        "source_path": str(bundle.source_path),
        "target_data_dictionary": bundle.dd_version,
    }
    physical_record = {
        "declared_absent": [absence.as_dict() for absence in bundle.absent],
        "ids": {
            name: {
                path: value
                for path, value in leaves.items()
                if not path.startswith("ids_properties/")
            }
            for name, leaves in snapshot.items()
        },
    }
    return (
        _sha256_bytes(_canonical_bytes(registry_record)),
        _sha256_bytes(_canonical_bytes(physical_record)),
    )


def create_diiid_artifact_manifest(
    ids_path: Path,
    *,
    registry_digest: str,
    physical_digest: str,
    source_shot: int,
    unresolved_gaps: Sequence[str],
) -> MachineArtifactManifest:
    """Create a manifest using the dictionary declared by the IDS file itself."""

    declared_dd_version = _declared_dd_version(ids_path)
    return create_machine_artifact_manifest(
        ids_path.parent,
        machine=MACHINE,
        dd_version=declared_dd_version,
        registry_digest=registry_digest,
        physical_digest=physical_digest,
        shot_ranges=(
            ArtifactShotRange(
                first_shot=source_shot,
                last_shot=source_shot,
                physical_digest=physical_digest,
                evidence="observed",
            ),
        ),
        complete=False,
        unresolved_gaps=unresolved_gaps,
    )


def write_prior_major_probe(path: Path) -> str:
    """Write a real IDS netCDF file in the newest installed prior major."""

    versions = tuple(
        version for version in dd_xml_versions() if parse_dd_version(version).major == 3
    )
    if not versions:
        raise RuntimeError("no prior-major Data Dictionary is available for the probe")
    dd_version = str(max(versions, key=parse_dd_version))
    factory = imas.IDSFactory(version=dd_version)
    wall = factory.new("wall")
    wall.ids_properties.homogeneous_time = 0
    wall.description_2d.resize(1)
    wall.description_2d[0].type.index = 0
    wall.description_2d[0].limiter.unit.resize(1)
    outline = wall.description_2d[0].limiter.unit[0].outline
    outline.r = [1.0, 2.0, 1.0, 1.0]
    outline.z = [0.0, 0.0, 1.0, 0.0]
    path.parent.mkdir(parents=True, exist_ok=True)
    database = imas.DBEntry(path.resolve(), "w", dd_version=dd_version)
    try:
        database.put(wall)
    finally:
        database.close()
    return dd_version


def _publication_floor_receipt(
    directory: Path,
    *,
    registry_digest: str,
    physical_digest: str,
    unresolved_gaps: Sequence[str],
) -> dict[str, Any]:
    probe = directory / "prior-major-machine-description.nc"
    declared_dd_version = write_prior_major_probe(probe)
    try:
        create_diiid_artifact_manifest(
            probe,
            registry_digest=registry_digest,
            physical_digest=physical_digest,
            source_shot=0,
            unresolved_gaps=unresolved_gaps,
        )
    except MachineArtifactError as error:
        return {
            "declared_data_dictionary": declared_dd_version,
            "exception": type(error).__name__,
            "message": str(error),
            "refused": True,
        }
    raise RuntimeError("the machine-artifact publication floor accepted a prior major")


def _ghcr_repository_parts(repository: str) -> tuple[str, str]:
    parts = repository.split("/")
    if len(parts) != 3 or parts[0] != "ghcr.io":
        raise MachineArtifactError(
            "DIII-D publication requires ghcr.io/<account>/<repository>"
        )
    return parts[1], parts[2]


def _write_publication_recipe(
    path: Path,
    *,
    repository: str,
    output: Path,
    manifest_path: Path,
    manifest: MachineArtifactManifest,
    payload_sha256: str,
) -> None:
    _account, repository_name = _ghcr_repository_parts(repository)
    recipe_reference = f"ghcr.io/${{GHCR_ACCOUNT}}/{repository_name}:{manifest.oci.tag}"
    login_command = (
        "printf '%s' \"$GHCR_TOKEN\" | oras login ghcr.io "
        '--username "$GHCR_ACCOUNT" --password-stdin'
    )
    push_command = (
        'oras push --image-spec v1.1 --artifact-type "$ARTIFACT_TYPE" '
        '--config "$MANIFEST_PATH:$ARTIFACT_TYPE" "$REFERENCE" '
        '"$PAYLOAD_PATH:$FILE_MEDIA_TYPE"'
    )
    verify_command = (
        f"printf '%s  %s\\n' '{payload_sha256}' "
        f'"$PULL_DIRECTORY/{output.as_posix()}" | sha256sum --check -'
    )
    text = (
        "\n".join(
            [
                "# Publish the DIII-D machine description",
                "",
                (
                    "This repository has no CI workflow for machine-description "
                    "publication. Publication is a local operator action. These "
                    "commands are documentation only; the build command does not "
                    "contact a registry."
                ),
                "",
                "Set the registry account and a token with package write permission:",
                "",
                "```sh",
                "GHCR_ACCOUNT='<registry-account>'",
                "GHCR_TOKEN='<registry-token>'",
                login_command,
                "```",
                "",
                (
                    "Push the canonical machine manifest as the OCI config and the "
                    "netCDF IDS set as its machine-specific layer:"
                ),
                "",
                "```sh",
                f"REFERENCE='{recipe_reference}'",
                f"ARTIFACT_TYPE='{manifest.oci.artifact_type}'",
                f"FILE_MEDIA_TYPE='{manifest.oci.file_media_type}'",
                f"MANIFEST_PATH='{manifest_path.as_posix()}'",
                f"PAYLOAD_PATH='{output.as_posix()}'",
                push_command,
                "```",
                "",
                (
                    "Pull the layer and verify the exact payload bytes used to build "
                    "the manifest:"
                ),
                "",
                "```sh",
                f"REFERENCE='{recipe_reference}'",
                "PULL_DIRECTORY='<pull-directory>'",
                'mkdir -p "$PULL_DIRECTORY"',
                'oras pull "$REFERENCE" --output "$PULL_DIRECTORY"',
                verify_command,
                "```",
                "",
                (
                    f"The computed tag is `{manifest.oci.tag}`. The OCI manifest "
                    f"media type is `{manifest.oci.manifest_media_type}`; ORAS "
                    "selects it through `--image-spec v1.1`."
                ),
            ]
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def publish_diiid_machine_artifact(
    *,
    repository: str,
    cache_directory: Path,
    source_path: Path = SOURCE_PATH,
    output: Path = DEFAULT_OUTPUT,
    receipt_path: Path = DEFAULT_RECEIPT,
    manifest_path: Path = DEFAULT_MANIFEST,
    recipe_path: Path = DEFAULT_RECIPE,
) -> dict[str, Any]:
    """Author, manifest, cache, resolve, and document one local artifact."""

    _ghcr_repository_parts(repository)
    dd_version = publication_dd_version()
    bundle = build_diiid_machine_ids(source_path)
    if bundle.dd_version != dd_version:
        raise MachineArtifactError(
            f"native IDS builder selected {bundle.dd_version}, publication selected "
            f"{dd_version}"
        )
    snapshot, round_trip = _write_and_verify_ids(bundle, output)
    declared_dd_version = _declared_dd_version(output)
    if declared_dd_version != dd_version:
        raise MachineArtifactError(
            f"authored IDS declares {declared_dd_version}, expected {dd_version}"
        )
    registry_digest, physical_digest = _artifact_digests(snapshot, bundle)
    unresolved_gaps = tuple(
        f"{absence.quantity}: {absence.reason}" for absence in bundle.absent
    )
    try:
        source_shot = int(source_path.stem)
    except ValueError as error:
        raise MachineArtifactError(
            f"source IDS filename must identify its shot: {source_path.name}"
        ) from error

    cache_directory = cache_directory.resolve()
    cache_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".diiid-machine-artifact-",
        dir=cache_directory.parent,
    ) as temporary:
        bundle_directory = Path(temporary)
        staged_ids = bundle_directory / output.name
        shutil.copyfile(output, staged_ids)
        manifest = create_diiid_artifact_manifest(
            staged_ids,
            registry_digest=registry_digest,
            physical_digest=physical_digest,
            source_shot=source_shot,
            unresolved_gaps=unresolved_gaps,
        )
        materialized = materialize_machine_artifact(
            bundle_directory,
            cache_directory,
            manifest,
        )
        resolved = resolve_machine_artifact(
            cache_directory,
            materialized.digest,
            expected_dd_version=dd_version,
            expected_registry_digest=registry_digest,
            expected_physical_digest=physical_digest,
            allow_incomplete=True,
        )
        floor_receipt = _publication_floor_receipt(
            bundle_directory / "publication-floor",
            registry_digest=registry_digest,
            physical_digest=physical_digest,
            unresolved_gaps=unresolved_gaps,
        )

    if materialized.digest != resolved.digest:
        raise MachineArtifactError("resolved artifact digest differs from materialized")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest.canonical_bytes())
    payload_sha256 = _sha256_file(output)
    reference = oci_artifact_reference(repository, manifest)
    computed_tag = oci_artifact_tag(dd_version, physical_digest)
    if computed_tag != manifest.oci.tag:
        raise MachineArtifactError("computed OCI tag differs from the manifest")
    _write_publication_recipe(
        recipe_path,
        repository=repository,
        output=output,
        manifest_path=manifest_path,
        manifest=manifest,
        payload_sha256=payload_sha256,
    )
    receipt = {
        "cache": {
            "directory": str(cache_directory),
            "digests_equal": materialized.digest == resolved.digest,
            "materialized_digest": materialized.digest,
            "resolved_digest": resolved.digest,
            "resolved_directory": str(resolved.directory),
        },
        "data_dictionary_floor": floor_receipt,
        "digest_basis": {
            "physical": (
                "authored static IDS leaves excluding ids_properties, plus "
                "declared absences"
            ),
            "registry": (
                "all authored IDS leaves, source identity, dictionary versions, "
                "and declared absences"
            ),
        },
        "manifest": manifest.as_dict(),
        "network_publication_attempted": False,
        "output": {
            "path": str(output),
            "sha256": payload_sha256,
            "size_bytes": output.stat().st_size,
        },
        "publication": {
            "artifact_type": manifest.oci.artifact_type,
            "data_dictionary": dd_version,
            "data_dictionary_resolver": "publication_dd_version()",
            "file_media_type": manifest.oci.file_media_type,
            "manifest_digest": manifest.digest,
            "manifest_media_type": manifest.oci.manifest_media_type,
            "manifest_path": str(manifest_path),
            "manifest_schema": manifest.schema,
            "oci_reference": reference,
            "oci_tag": computed_tag,
            "physical_digest": physical_digest,
            "registry_digest": registry_digest,
            "repository": repository,
        },
        "publication_recipe": {
            "local_operator_action": True,
            "no_ci_workflow": True,
            "path": str(recipe_path),
        },
        "round_trip": round_trip,
        "source": {
            "data_dictionary": bundle.source_dd_version,
            "path": str(source_path),
            "shot": source_shot,
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and locally verify the DIII-D machine artifact."
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = publish_diiid_machine_artifact(
        repository=args.repository,
        cache_directory=args.cache,
        source_path=args.source,
        output=args.output,
        receipt_path=args.receipt,
        manifest_path=args.manifest,
        recipe_path=args.recipe,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
