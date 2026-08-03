from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import nova.imas.mast_artifact as mast_artifact_module
from nova.imas.machine_evidence import (
    EvidenceError,
    EvidenceRecord,
    FieldEvidence,
    SourceReference,
)
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
SOURCE = SourceReference(
    title="A machine description",
    url="https://example.invalid/machine.pdf",
    locator="p. 7",
    machine="mast",
    text_verified=True,
)
MEASURED_FIELD = EvidenceRecord(
    path="wall/description_2d/limiter/unit/outline",
    evidence=FieldEvidence.MEASURED,
    first_shot=11766,
    last_shot=30471,
    statement="the limiter contour is the catalog wall cycle",
    source=SOURCE,
)
UNRESOLVED_FIELD = EvidenceRecord(
    path="tf/r0",
    evidence=FieldEvidence.UNRESOLVED,
    first_shot=11766,
    last_shot=30471,
    statement="the official reference radius is not sourced",
    assumptions=("no document states the machine constant",),
)


def _write_bundle(path: Path) -> None:
    path.mkdir()
    (path / "master.h5").write_bytes(b"master")
    (path / "ids").mkdir()
    (path / "ids" / "pf_active.h5").write_bytes(b"active")


def _manifest(
    source: Path,
    *,
    complete: bool = False,
    field_evidence=(MEASURED_FIELD,),
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
        field_evidence=field_evidence,
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
            allow_incomplete=True,
        ).manifest
    )


def test_repeated_materialization_is_idempotent(tmp_path: Path) -> None:
    source, stored = _materialized(tmp_path)

    repeated = materialize_machine_artifact(source, tmp_path / "cache", stored.manifest)

    assert repeated == stored
    assert len(list((tmp_path / "cache" / "sha256").glob("[0-9a-f]*"))) == 1


def test_stale_temporary_directory_does_not_block_materialization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    object_root = tmp_path / "cache" / "sha256"
    stale = object_root / f".{manifest.digest.removeprefix('sha256:')}.abandoned"
    stale.mkdir(parents=True)
    (stale / "partial").write_bytes(b"incomplete")

    stored = materialize_machine_artifact(source, tmp_path / "cache", manifest)

    assert stored.digest == manifest.digest
    assert stale.is_dir()


def test_concurrent_materializers_share_verified_winner(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    cache = tmp_path / "cache"

    def materialize():
        return materialize_machine_artifact(source, cache, manifest)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(lambda _: materialize(), range(8)))

    assert {result.digest for result in results} == {manifest.digest}
    assert {result.directory for result in results} == {results[0].directory}
    assert not tuple((cache / "sha256").glob(f".{manifest.digest[7:]}.*"))


def test_existing_empty_destination_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    destination = (
        tmp_path / "cache" / "sha256" / manifest.digest.removeprefix("sha256:")
    )
    destination.mkdir(parents=True)

    with pytest.raises(MachineArtifactError, match="manifest is missing"):
        materialize_machine_artifact(source, tmp_path / "cache", manifest)

    assert list(destination.iterdir()) == []


def test_symlinked_object_root_is_rejected_without_outside_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    cache.mkdir()
    outside.mkdir()
    (cache / "sha256").symlink_to(outside, target_is_directory=True)

    with pytest.raises(MachineArtifactError, match="object root.*symlink"):
        materialize_machine_artifact(source, cache, manifest)
    with pytest.raises(MachineArtifactError, match="object root.*symlink"):
        resolve_machine_artifact(cache, manifest.digest, allow_incomplete=True)

    assert list(outside.iterdir()) == []


def test_symlinked_digest_destination_is_rejected_without_outside_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    cache = tmp_path / "cache"
    object_root = cache / "sha256"
    outside = tmp_path / "outside"
    object_root.mkdir(parents=True)
    outside.mkdir()
    digest_hex = manifest.digest.removeprefix("sha256:")
    (object_root / digest_hex).symlink_to(outside, target_is_directory=True)

    with pytest.raises(MachineArtifactError, match="destination.*symlink"):
        materialize_machine_artifact(source, cache, manifest)
    with pytest.raises(MachineArtifactError, match="destination.*symlink"):
        resolve_machine_artifact(cache, manifest.digest, allow_incomplete=True)

    assert list(outside.iterdir()) == []


def test_object_root_path_swap_cannot_redirect_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    cache = tmp_path / "cache"
    visible_root = cache / "sha256"
    pinned_root = cache / "pinned-object-root"
    outside = tmp_path / "outside"
    outside.mkdir()
    create_private_directory = mast_artifact_module._create_private_directory

    def swap_visible_root(descriptor: int, digest_hex: str):
        visible_root.rename(pinned_root)
        visible_root.symlink_to(outside, target_is_directory=True)
        return create_private_directory(descriptor, digest_hex)

    monkeypatch.setattr(
        mast_artifact_module,
        "_create_private_directory",
        swap_visible_root,
    )

    with pytest.raises(MachineArtifactError, match="object root changed"):
        materialize_machine_artifact(source, cache, manifest)

    digest_hex = manifest.digest.removeprefix("sha256:")
    assert (pinned_root / digest_hex / MANIFEST_FILENAME).is_file()
    assert list(outside.iterdir()) == []
    with pytest.raises(MachineArtifactError, match="object root.*symlink"):
        resolve_machine_artifact(cache, manifest.digest, allow_incomplete=True)


def test_resolver_rejects_symlinked_manifest(tmp_path: Path) -> None:
    _, stored = _materialized(tmp_path)
    manifest_path = stored.directory / MANIFEST_FILENAME
    outside = tmp_path / "outside-manifest.json"
    outside.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(outside)

    with pytest.raises(MachineArtifactError, match="manifest.*symlink"):
        resolve_machine_artifact(
            tmp_path / "cache",
            stored.digest,
            allow_incomplete=True,
        )


def test_resolver_rejects_symlinked_payload(tmp_path: Path) -> None:
    _, stored = _materialized(tmp_path)
    payload = stored.directory / stored.manifest.files[0].name
    outside = tmp_path / "outside-payload.h5"
    outside.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(outside)

    with pytest.raises(MachineArtifactError, match="contains symlink"):
        resolve_machine_artifact(
            tmp_path / "cache",
            stored.digest,
            allow_incomplete=True,
        )


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


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "/absolute.h5",
        "../escape.h5",
        "ids/../escape.h5",
        "ids\\escape.h5",
        "C:/escape.h5",
        "\\\\server\\share\\escape.h5",
        "ids//escape.h5",
        "ids/./escape.h5",
        "ids/trailing.",
        "ids/trailing ",
        "MANIFEST.JSON",
        "ids/less<than.h5",
        "ids/greater>than.h5",
        'ids/quote"name.h5',
        "ids/pipe|name.h5",
        "ids/question?.h5",
        "ids/star*.h5",
        "ids/non_ascii_\u00e9.h5",
        "CON",
        "ids/aux.txt",
        "ids/COM1.h5",
        "ids/lpt9",
    ],
)
def test_manifest_rejects_nonportable_names(tmp_path: Path, name: str) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    altered_file = replace(manifest.files[0], name=name)

    with pytest.raises(MachineArtifactError, match="unsafe"):
        replace(manifest, files=(altered_file, *manifest.files[1:])).canonical_bytes()


def test_manifest_rejects_casefold_colliding_names(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    first = manifest.files[0]
    colliding = replace(first, name=first.name.upper())

    with pytest.raises(MachineArtifactError, match="case-insensitive"):
        replace(
            manifest,
            files=tuple(sorted((first, colliding, *manifest.files[1:]))),
        ).canonical_bytes()


def test_manifest_rejects_casefold_colliding_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    first = manifest.files[0]
    second = replace(first, name="IDS/other.h5")

    with pytest.raises(MachineArtifactError, match="case-insensitive"):
        replace(
            manifest,
            files=tuple(sorted((first, second, *manifest.files[1:]))),
        ).canonical_bytes()


def test_source_inventory_rejects_casefold_collisions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "coil.h5").write_bytes(b"lower")
    (source / "COIL.h5").write_bytes(b"upper")

    with pytest.raises(MachineArtifactError, match="case-insensitive"):
        create_machine_artifact_manifest(
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
            complete=False,
            unresolved_gaps=("source gap",),
        )


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

    with pytest.raises(IncompleteMachineArtifactError, match="operator-ready"):
        resolve_machine_artifact(tmp_path / "cache", stored.digest)

    resolved = resolve_machine_artifact(
        tmp_path / "cache",
        stored.digest,
        allow_incomplete=True,
    )
    assert not resolved.manifest.complete
    assert resolved.manifest.unresolved_gaps


def test_complete_artifact_is_operator_ready(tmp_path: Path) -> None:
    _, stored = _materialized(tmp_path, complete=True)

    resolved = resolve_machine_artifact(
        tmp_path / "cache",
        stored.digest,
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
    with pytest.raises(MachineArtifactError, match="at most 128"):
        oci_artifact_tag(f"{'1' * 120}.1.1", PHYSICAL_DIGEST)


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


def test_field_evidence_survives_the_manifest_round_trip(tmp_path: Path) -> None:
    _, stored = _materialized(tmp_path)

    restored = resolve_machine_artifact(
        tmp_path / "cache",
        stored.digest,
        allow_incomplete=True,
    ).manifest

    assert restored.field_evidence == (MEASURED_FIELD,)
    assert restored.evidence.state_counts()["measured"] == 1
    assert restored == stored.manifest


def test_complete_artifact_cannot_carry_an_unresolved_field(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)

    with pytest.raises(MachineArtifactError, match="unresolved fields: tf/r0"):
        _manifest(source, complete=True, field_evidence=(UNRESOLVED_FIELD,))

    incomplete = _manifest(source, field_evidence=(UNRESOLVED_FIELD, MEASURED_FIELD))
    assert incomplete.evidence.paths_with_state(FieldEvidence.UNRESOLVED) == ("tf/r0",)


def test_field_evidence_must_stay_inside_the_artifact_shot_extent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    outside = replace(MEASURED_FIELD, first_shot=11000)

    with pytest.raises(MachineArtifactError, match="outside the artifact extent"):
        _manifest(source, field_evidence=(outside,))


def test_manifest_rejects_conflicting_evidence_for_one_field(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    conflicting = replace(
        UNRESOLVED_FIELD,
        path=MEASURED_FIELD.path,
    )

    with pytest.raises(EvidenceError, match="two evidence states"):
        _manifest(source, field_evidence=(MEASURED_FIELD, conflicting))


def test_semantic_identity_is_stable_across_container_rewrites(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_bundle(first)
    _write_bundle(second)
    (second / "master.h5").write_bytes(b"master rewritten by the same authoring")

    left = _manifest(first)
    right = _manifest(second)

    assert left.digest != right.digest
    assert left.semantic_identity() == right.semantic_identity()
    assert left.oci.tag == right.oci.tag


def test_semantic_identity_changes_with_authored_semantics(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    plain = _manifest(source)
    seeded = _manifest(source, field_evidence=(MEASURED_FIELD, UNRESOLVED_FIELD))

    assert plain.semantic_identity() != seeded.semantic_identity()
    assert plain.oci.tag == seeded.oci.tag


def test_file_size_mismatch_is_reported_before_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = _manifest(source)
    first = manifest.files[0]
    wrong = ArtifactFile(first.name, first.sha256, first.size + 1)
    altered = replace(manifest, files=tuple(sorted((wrong, *manifest.files[1:]))))

    with pytest.raises(MachineArtifactError, match="size mismatch"):
        materialize_machine_artifact(source, tmp_path / "cache", altered)
