from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from nova.imas.mast_artifact import (
    MANIFEST_FILENAME,
    OCI_ARTIFACT_TYPE,
    OCI_FILE_MEDIA_TYPE,
    OCI_MANIFEST_MEDIA_TYPE,
    ArtifactFile,
    ArtifactShotRange,
    IncompleteMachineArtifactError,
    MachineArtifactError,
    MachineArtifactManifest,
    create_machine_artifact_manifest,
    materialize_machine_artifact,
    oci_artifact_reference,
    oci_artifact_tag,
    pinned_dd_version,
    resolve_machine_artifact,
)

REGISTRY_DIGEST = "a" * 64
PHYSICAL_DIGEST = "b" * 16


def _write_bundle(path: Path) -> None:
    path.mkdir()
    (path / "master.h5").write_bytes(b"master")
    (path / "ids").mkdir()
    (path / "ids" / "pf_active.h5").write_bytes(b"active")


def _manifest(
    source: Path,
    *,
    complete: bool = False,
) -> MachineArtifactManifest:
    gaps = () if complete else ("toroidal probe orientation is unresolved",)
    return create_machine_artifact_manifest(
        source,
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
        complete=complete,
        unresolved_gaps=gaps,
    )


def _materialized(tmp_path: Path, *, complete: bool = False):
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source, complete=complete)
    return source, materialize_machine_artifact(source, tmp_path / "cache", manifest)


def _rewrite_manifest(
    artifact_directory: Path,
    payload: dict,
    *,
    cache_directory: Path,
) -> str:
    data = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    digest = hashlib.sha256(data).hexdigest()
    target = cache_directory / "sha256" / digest
    target.mkdir(parents=True)
    for path in artifact_directory.rglob("*"):
        if path.is_file() and path.name != MANIFEST_FILENAME:
            relative = path.relative_to(artifact_directory)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (target / MANIFEST_FILENAME).write_bytes(data)
    return f"sha256:{digest}"


def test_manifest_bytes_and_digest_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_bundle(first)
    second.mkdir()
    (second / "ids").mkdir()
    (second / "ids" / "pf_active.h5").write_bytes(b"active")
    (second / "master.h5").write_bytes(b"master")

    left = _manifest(first)
    right = _manifest(second)

    assert left.canonical_bytes() == right.canonical_bytes()
    assert left.digest == right.digest
    assert b"mtime" not in left.canonical_bytes()


def test_materialize_and_resolve_round_trip(tmp_path: Path) -> None:
    source, stored = _materialized(tmp_path)

    assert stored.directory != source
    assert (stored.directory / "master.h5").read_bytes() == b"master"
    assert (stored.directory / "ids" / "pf_active.h5").read_bytes() == b"active"
    assert pinned_dd_version(stored) == "4.1.1"
    assert (
        stored.manifest
        == resolve_machine_artifact(
            tmp_path / "cache",
            stored.digest,
            expected_dd_version="4.1.1",
            expected_registry_digest=REGISTRY_DIGEST,
            expected_physical_digest=PHYSICAL_DIGEST,
        ).manifest
    )


def test_repeated_materialization_is_idempotent(tmp_path: Path) -> None:
    source, stored = _materialized(tmp_path)

    repeated = materialize_machine_artifact(source, tmp_path / "cache", stored.manifest)

    assert repeated == stored
    assert len(list((tmp_path / "cache" / "sha256").glob("[0-9a-f]*"))) == 1


@pytest.mark.parametrize("change", ["tamper", "missing", "unexpected"])
def test_resolver_rejects_file_set_and_content_changes(
    tmp_path: Path,
    change: str,
) -> None:
    _, stored = _materialized(tmp_path)
    if change == "tamper":
        (stored.directory / "master.h5").write_bytes(b"altered")
    elif change == "missing":
        (stored.directory / "master.h5").unlink()
    else:
        (stored.directory / "unexpected.h5").write_bytes(b"surprise")

    expected = change.replace("tamper", "mismatch")
    with pytest.raises(MachineArtifactError, match=expected):
        resolve_machine_artifact(tmp_path / "cache", stored.digest)


@pytest.mark.parametrize("name", ["/absolute.h5", "../escape.h5", "ids/../escape.h5"])
def test_manifest_rejects_path_traversal(tmp_path: Path, name: str) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    altered_file = replace(manifest.files[0], name=name)

    with pytest.raises(MachineArtifactError, match="unsafe"):
        replace(manifest, files=(altered_file, *manifest.files[1:])).canonical_bytes()


def test_manifest_rejects_duplicate_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)

    with pytest.raises(MachineArtifactError, match="duplicate"):
        replace(
            manifest,
            files=(manifest.files[0], manifest.files[0]),
        ).canonical_bytes()


def test_resolver_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    _, stored = _materialized(tmp_path)

    with pytest.raises(MachineArtifactError, match="physical digest mismatch"):
        resolve_machine_artifact(
            tmp_path / "cache",
            stored.digest,
            expected_physical_digest="c" * 16,
        )

    manifest_path = stored.directory / MANIFEST_FILENAME
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(MachineArtifactError, match="cache address"):
        resolve_machine_artifact(tmp_path / "cache", stored.digest)


def test_resolver_rejects_identity_change_inside_manifest(tmp_path: Path) -> None:
    _, stored = _materialized(tmp_path)
    payload = stored.manifest.as_dict()
    payload["physical_digest"] = "c" * 16
    cache = tmp_path / "changed-cache"
    digest = _rewrite_manifest(stored.directory, payload, cache_directory=cache)

    with pytest.raises(MachineArtifactError, match="disagrees"):
        resolve_machine_artifact(cache, digest)


def test_resolver_rejects_noncanonical_manifest_bytes(tmp_path: Path) -> None:
    _, stored = _materialized(tmp_path)
    payload = stored.manifest.as_dict()
    data = json.dumps(payload, indent=2, sort_keys=True).encode()
    digest = hashlib.sha256(data).hexdigest()
    cache = tmp_path / "noncanonical-cache"
    target = cache / "sha256" / digest
    target.mkdir(parents=True)
    for artifact_file in stored.manifest.files:
        source = stored.directory / artifact_file.name
        destination = target / artifact_file.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (target / MANIFEST_FILENAME).write_bytes(data)

    with pytest.raises(MachineArtifactError, match="not canonical"):
        resolve_machine_artifact(cache, f"sha256:{digest}")


def test_incomplete_artifact_can_be_stored_but_not_operator_ready(
    tmp_path: Path,
) -> None:
    _, stored = _materialized(tmp_path)

    resolved = resolve_machine_artifact(tmp_path / "cache", stored.digest)
    assert not resolved.manifest.complete
    assert resolved.manifest.unresolved_gaps
    with pytest.raises(IncompleteMachineArtifactError, match="operator-ready"):
        resolve_machine_artifact(
            tmp_path / "cache",
            stored.digest,
            require_complete=True,
        )


def test_complete_artifact_is_operator_ready(tmp_path: Path) -> None:
    _, stored = _materialized(tmp_path, complete=True)

    resolved = resolve_machine_artifact(
        tmp_path / "cache",
        stored.digest,
        require_complete=True,
    )

    assert resolved.manifest.complete
    assert resolved.manifest.unresolved_gaps == ()


def test_completeness_is_consistent_with_gaps_and_missing_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    complete = _manifest(source, complete=True)
    missing_range = replace(complete.shot_ranges[0], evidence="missing")

    with pytest.raises(MachineArtifactError, match="complete artifact"):
        replace(
            complete,
            shot_ranges=(missing_range,),
            unresolved_gaps=("catalog store is missing",),
        ).canonical_bytes()
    with pytest.raises(MachineArtifactError, match="must state"):
        replace(complete, complete=False).canonical_bytes()


def test_oci_reference_and_media_types_are_explicit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)

    assert manifest.oci.artifact_type == OCI_ARTIFACT_TYPE
    assert manifest.oci.manifest_media_type == OCI_MANIFEST_MEDIA_TYPE
    assert manifest.oci.file_media_type == OCI_FILE_MEDIA_TYPE
    assert manifest.oci.tag == f"dd-4.1.1-physical-{PHYSICAL_DIGEST}"
    assert oci_artifact_tag("4.1.1", PHYSICAL_DIGEST) == manifest.oci.tag
    assert (
        oci_artifact_reference("ghcr.io/example/mast-md", manifest)
        == f"ghcr.io/example/mast-md:{manifest.oci.tag}"
    )
    with pytest.raises(MachineArtifactError, match="repository"):
        oci_artifact_reference("https://ghcr.io/example/mast-md", manifest)


def test_malformed_hashes_and_unsafe_source_entries_are_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)

    with pytest.raises(MachineArtifactError, match="registry digest"):
        replace(manifest, registry_digest="not-a-hash").canonical_bytes()
    with pytest.raises(MachineArtifactError, match="sha256"):
        replace(
            manifest,
            files=(replace(manifest.files[0], sha256="A" * 64), *manifest.files[1:]),
        ).canonical_bytes()
    (source / "linked.h5").symlink_to(source / "master.h5")
    with pytest.raises(MachineArtifactError, match="symlink"):
        create_machine_artifact_manifest(
            source,
            dd_version="4.1.1",
            registry_digest=REGISTRY_DIGEST,
            physical_digest=PHYSICAL_DIGEST,
            shot_ranges=manifest.shot_ranges,
            complete=False,
            unresolved_gaps=("source gap",),
        )


def test_file_size_mismatch_is_reported_before_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    first = manifest.files[0]
    wrong = ArtifactFile(first.name, first.sha256, first.size + 1)
    altered = replace(manifest, files=tuple(sorted((wrong, *manifest.files[1:]))))

    with pytest.raises(MachineArtifactError, match="size mismatch"):
        materialize_machine_artifact(source, tmp_path / "cache", altered)
