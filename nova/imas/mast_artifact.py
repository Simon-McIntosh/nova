"""Create and verify content-addressed MAST machine artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA = "nova-mast-machine-artifact"
OCI_ARTIFACT_TYPE = "application/vnd.iter.nova.mast-machine-description.v1"
OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
OCI_FILE_MEDIA_TYPE = "application/vnd.iter.nova.mast-machine-description.ids.v1"

_DD_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_HEX_PATTERN = re.compile(r"[0-9a-f]+")
_OCI_REPOSITORY_PATTERN = re.compile(
    r"[a-z0-9]+(?:[.-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
)
_EVIDENCE_STATES = frozenset({"observed", "inherited", "missing"})


class MachineArtifactError(ValueError):
    """Base exception for an invalid or altered machine artifact."""


class IncompleteMachineArtifactError(MachineArtifactError):
    """Raised when operator-ready semantics are requested from incomplete data."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _require_exact_keys(
    row: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(row)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise MachineArtifactError(
            f"{context} fields differ: missing={missing}, unexpected={unexpected}"
        )


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise MachineArtifactError(f"{context} must be a string")
    return value


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MachineArtifactError(f"{context} must be an integer")
    return value


def _validate_hex(value: str, lengths: tuple[int, ...], context: str) -> None:
    if len(value) not in lengths or _HEX_PATTERN.fullmatch(value) is None:
        allowed = " or ".join(str(length) for length in lengths)
        raise MachineArtifactError(
            f"{context} must be lowercase hexadecimal with length {allowed}"
        )


def _safe_relative_name(value: str) -> str:
    if not value or "\\" in value:
        raise MachineArtifactError(f"unsafe artifact file name {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MachineArtifactError(f"unsafe artifact file name {value!r}")
    normalized = path.as_posix()
    if normalized == MANIFEST_FILENAME:
        raise MachineArtifactError(f"{MANIFEST_FILENAME!r} is reserved")
    return normalized


def _decode_json(data: bytes) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in pairs:
            if key in row:
                raise MachineArtifactError(f"duplicate JSON field {key!r}")
            row[key] = value
        return row

    try:
        decoded = json.loads(data, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise MachineArtifactError("manifest is not valid UTF-8 JSON") from error
    if not isinstance(decoded, Mapping):
        raise MachineArtifactError("manifest root must be an object")
    return decoded


@dataclass(frozen=True, order=True)
class ArtifactFile:
    """Content identity for one file in the authored IDS bundle."""

    name: str
    sha256: str
    size: int

    def validate(self) -> None:
        """Reject unsafe paths and malformed file identities."""

        if _safe_relative_name(self.name) != self.name:
            raise MachineArtifactError(
                f"non-canonical artifact file name {self.name!r}"
            )
        _validate_hex(self.sha256, (64,), f"sha256 for {self.name!r}")
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise MachineArtifactError(f"size for {self.name!r} must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {"name": self.name, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ArtifactFile:
        """Build a validated file identity from decoded JSON."""

        _require_exact_keys(row, {"name", "sha256", "size"}, "file")
        result = cls(
            name=_require_string(row["name"], "file name"),
            sha256=_require_string(row["sha256"], "file sha256"),
            size=_require_int(row["size"], "file size"),
        )
        result.validate()
        return result


@dataclass(frozen=True, order=True)
class ArtifactShotRange:
    """Physical identity and evidence over a closed shot interval."""

    first_shot: int
    last_shot: int
    physical_digest: str
    evidence: str

    def validate(self) -> None:
        """Reject empty intervals, malformed identities, and unknown evidence."""

        if (
            isinstance(self.first_shot, bool)
            or not isinstance(self.first_shot, int)
            or self.first_shot < 0
        ):
            raise MachineArtifactError("first shot must be a non-negative integer")
        if (
            isinstance(self.last_shot, bool)
            or not isinstance(self.last_shot, int)
            or self.last_shot < self.first_shot
        ):
            raise MachineArtifactError("last shot must not precede first shot")
        _validate_hex(self.physical_digest, (16, 64), "range physical digest")
        if self.evidence not in _EVIDENCE_STATES:
            raise MachineArtifactError(f"unknown evidence state {self.evidence!r}")

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "evidence": self.evidence,
            "first_shot": self.first_shot,
            "last_shot": self.last_shot,
            "physical_digest": self.physical_digest,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ArtifactShotRange:
        """Build a validated shot range from decoded JSON."""

        _require_exact_keys(
            row,
            {"evidence", "first_shot", "last_shot", "physical_digest"},
            "shot range",
        )
        result = cls(
            first_shot=_require_int(row["first_shot"], "first shot"),
            last_shot=_require_int(row["last_shot"], "last shot"),
            physical_digest=_require_string(
                row["physical_digest"], "range physical digest"
            ),
            evidence=_require_string(row["evidence"], "evidence state"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class OciArtifactConvention:
    """OCI types and deterministic human-readable tag for this bundle."""

    artifact_type: str
    manifest_media_type: str
    file_media_type: str
    tag: str

    def validate(self, dd_version: str, physical_digest: str) -> None:
        """Reject convention drift or a tag that disagrees with the identity."""

        expected = oci_artifact_tag(dd_version, physical_digest)
        values = (
            (self.artifact_type, OCI_ARTIFACT_TYPE, "artifact type"),
            (
                self.manifest_media_type,
                OCI_MANIFEST_MEDIA_TYPE,
                "manifest media type",
            ),
            (self.file_media_type, OCI_FILE_MEDIA_TYPE, "file media type"),
            (self.tag, expected, "artifact tag"),
        )
        for actual, required, context in values:
            if actual != required:
                raise MachineArtifactError(
                    f"{context} {actual!r} does not match {required!r}"
                )

    def as_dict(self) -> dict[str, str]:
        """Return the canonical JSON representation."""

        return {
            "artifact_type": self.artifact_type,
            "file_media_type": self.file_media_type,
            "manifest_media_type": self.manifest_media_type,
            "tag": self.tag,
        }

    @classmethod
    def create(cls, dd_version: str, physical_digest: str) -> OciArtifactConvention:
        """Return the fixed conventions for a machine artifact identity."""

        return cls(
            artifact_type=OCI_ARTIFACT_TYPE,
            manifest_media_type=OCI_MANIFEST_MEDIA_TYPE,
            file_media_type=OCI_FILE_MEDIA_TYPE,
            tag=oci_artifact_tag(dd_version, physical_digest),
        )

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> OciArtifactConvention:
        """Build OCI conventions from decoded JSON."""

        expected = {
            "artifact_type",
            "file_media_type",
            "manifest_media_type",
            "tag",
        }
        _require_exact_keys(row, expected, "OCI convention")
        return cls(
            artifact_type=_require_string(row["artifact_type"], "artifact type"),
            manifest_media_type=_require_string(
                row["manifest_media_type"], "manifest media type"
            ),
            file_media_type=_require_string(row["file_media_type"], "file media type"),
            tag=_require_string(row["tag"], "artifact tag"),
        )


@dataclass(frozen=True)
class MachineArtifactManifest:
    """Canonical identity and completeness record for an authored IDS bundle."""

    schema: str
    dd_version: str
    registry_digest: str
    physical_digest: str
    shot_ranges: tuple[ArtifactShotRange, ...]
    complete: bool
    unresolved_gaps: tuple[str, ...]
    files: tuple[ArtifactFile, ...]
    oci: OciArtifactConvention

    def validate(self) -> None:
        """Reject ambiguous, incomplete, or non-canonical manifest state."""

        if self.schema != MANIFEST_SCHEMA:
            raise MachineArtifactError(f"unsupported manifest schema {self.schema!r}")
        if _DD_VERSION_PATTERN.fullmatch(self.dd_version) is None:
            raise MachineArtifactError(
                f"malformed data dictionary version {self.dd_version!r}"
            )
        _validate_hex(self.registry_digest, (64,), "registry digest")
        _validate_hex(self.physical_digest, (16, 64), "physical digest")
        if not isinstance(self.complete, bool):
            raise MachineArtifactError("complete must be a boolean")
        if not self.shot_ranges:
            raise MachineArtifactError("manifest must contain at least one shot range")
        if tuple(sorted(self.shot_ranges)) != self.shot_ranges:
            raise MachineArtifactError("shot ranges must be canonically ordered")
        previous_last: int | None = None
        for shot_range in self.shot_ranges:
            shot_range.validate()
            if shot_range.physical_digest != self.physical_digest:
                raise MachineArtifactError(
                    "shot range physical digest disagrees with manifest identity"
                )
            if previous_last is not None and shot_range.first_shot <= previous_last:
                raise MachineArtifactError("shot ranges overlap")
            previous_last = shot_range.last_shot
        if not self.files:
            raise MachineArtifactError("manifest must contain at least one IDS file")
        if tuple(sorted(self.files)) != self.files:
            raise MachineArtifactError("files must be canonically ordered")
        names: set[str] = set()
        for artifact_file in self.files:
            artifact_file.validate()
            if artifact_file.name in names:
                raise MachineArtifactError(
                    f"duplicate artifact file {artifact_file.name!r}"
                )
            names.add(artifact_file.name)
        if tuple(sorted(self.unresolved_gaps)) != self.unresolved_gaps:
            raise MachineArtifactError("unresolved gaps must be canonically ordered")
        if len(set(self.unresolved_gaps)) != len(self.unresolved_gaps):
            raise MachineArtifactError("unresolved gaps must be unique")
        if any(not gap or gap.strip() != gap for gap in self.unresolved_gaps):
            raise MachineArtifactError("unresolved gaps must be non-empty trimmed text")
        evidence_is_incomplete = any(
            shot_range.evidence == "missing" for shot_range in self.shot_ranges
        )
        if self.complete and (self.unresolved_gaps or evidence_is_incomplete):
            raise MachineArtifactError(
                "complete artifact cannot carry unresolved or missing evidence"
            )
        if not self.complete and not self.unresolved_gaps:
            raise MachineArtifactError(
                "incomplete artifact must state at least one unresolved gap"
            )
        self.oci.validate(self.dd_version, self.physical_digest)

    def as_dict(self) -> dict[str, Any]:
        """Return the complete canonical manifest payload."""

        return {
            "complete": self.complete,
            "dd_version": self.dd_version,
            "files": [artifact_file.as_dict() for artifact_file in self.files],
            "oci": self.oci.as_dict(),
            "physical_digest": self.physical_digest,
            "registry_digest": self.registry_digest,
            "schema": self.schema,
            "shot_ranges": [shot_range.as_dict() for shot_range in self.shot_ranges],
            "unresolved_gaps": list(self.unresolved_gaps),
        }

    def canonical_bytes(self) -> bytes:
        """Serialize to timestamp-free, byte-stable JSON."""

        self.validate()
        return _canonical_json(self.as_dict())

    @property
    def digest(self) -> str:
        """Return the content address of the canonical manifest."""

        return f"sha256:{_sha256_bytes(self.canonical_bytes())}"

    def require_complete(self) -> None:
        """Require operator-ready semantics without treating gaps as defaults."""

        self.validate()
        if not self.complete:
            gaps = "; ".join(self.unresolved_gaps)
            raise IncompleteMachineArtifactError(
                f"machine artifact is not operator-ready: {gaps}"
            )

    @classmethod
    def from_bytes(cls, data: bytes) -> MachineArtifactManifest:
        """Parse strict canonical JSON into a validated manifest."""

        row = _decode_json(data)
        expected = {
            "complete",
            "dd_version",
            "files",
            "oci",
            "physical_digest",
            "registry_digest",
            "schema",
            "shot_ranges",
            "unresolved_gaps",
        }
        _require_exact_keys(row, expected, "manifest")
        files = row["files"]
        shot_ranges = row["shot_ranges"]
        gaps = row["unresolved_gaps"]
        oci = row["oci"]
        if not isinstance(files, list):
            raise MachineArtifactError("files must be an array")
        if not isinstance(shot_ranges, list):
            raise MachineArtifactError("shot ranges must be an array")
        if not isinstance(gaps, list):
            raise MachineArtifactError("unresolved gaps must be an array")
        if not isinstance(oci, Mapping):
            raise MachineArtifactError("OCI convention must be an object")
        result = cls(
            schema=_require_string(row["schema"], "schema"),
            dd_version=_require_string(row["dd_version"], "DD version"),
            registry_digest=_require_string(row["registry_digest"], "registry digest"),
            physical_digest=_require_string(row["physical_digest"], "physical digest"),
            shot_ranges=tuple(
                ArtifactShotRange.from_dict(item)
                if isinstance(item, Mapping)
                else _raise_row_error("shot range")
                for item in shot_ranges
            ),
            complete=row["complete"],
            unresolved_gaps=tuple(
                _require_string(item, "unresolved gap") for item in gaps
            ),
            files=tuple(
                ArtifactFile.from_dict(item)
                if isinstance(item, Mapping)
                else _raise_row_error("file")
                for item in files
            ),
            oci=OciArtifactConvention.from_dict(oci),
        )
        result.validate()
        if result.canonical_bytes() != data:
            raise MachineArtifactError("manifest bytes are not canonical")
        return result


def _raise_row_error(context: str) -> Any:
    raise MachineArtifactError(f"{context} entry must be an object")


@dataclass(frozen=True)
class VerifiedMachineArtifact:
    """A local artifact directory whose manifest and files were verified."""

    directory: Path
    manifest: MachineArtifactManifest
    digest: str


def oci_artifact_tag(dd_version: str, physical_digest: str) -> str:
    """Format the deterministic OCI tag for one physical configuration."""

    if _DD_VERSION_PATTERN.fullmatch(dd_version) is None:
        raise MachineArtifactError(f"malformed data dictionary version {dd_version!r}")
    _validate_hex(physical_digest, (16, 64), "physical digest")
    return f"dd-{dd_version}-physical-{physical_digest}"


def oci_artifact_reference(
    repository: str,
    manifest: MachineArtifactManifest,
) -> str:
    """Format a tagged OCI reference without contacting a registry."""

    manifest.validate()
    if _OCI_REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise MachineArtifactError(f"malformed OCI repository {repository!r}")
    return f"{repository}:{manifest.oci.tag}"


def _inventory_files(directory: Path, *, allow_manifest: bool) -> dict[str, Path]:
    if not directory.is_dir():
        raise MachineArtifactError(f"artifact directory does not exist: {directory}")
    inventory: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise MachineArtifactError(f"artifact contains symlink {relative!r}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise MachineArtifactError(f"artifact contains non-file {relative!r}")
        if relative == MANIFEST_FILENAME and allow_manifest:
            continue
        safe = _safe_relative_name(relative)
        if safe in inventory:
            raise MachineArtifactError(f"duplicate artifact file {safe!r}")
        inventory[safe] = path
    return inventory


def create_machine_artifact_manifest(
    source_directory: Path | str,
    *,
    dd_version: str,
    registry_digest: str,
    physical_digest: str,
    shot_ranges: Iterable[ArtifactShotRange],
    complete: bool,
    unresolved_gaps: Iterable[str],
) -> MachineArtifactManifest:
    """Hash an authored IDS directory into a canonical manifest."""

    source = Path(source_directory)
    inventory = _inventory_files(source, allow_manifest=False)
    files = tuple(
        sorted(
            ArtifactFile(name=name, sha256=digest, size=size)
            for name, path in inventory.items()
            for digest, size in [_file_identity(path)]
        )
    )
    manifest = MachineArtifactManifest(
        schema=MANIFEST_SCHEMA,
        dd_version=dd_version,
        registry_digest=registry_digest,
        physical_digest=physical_digest,
        shot_ranges=tuple(sorted(shot_ranges)),
        complete=complete,
        unresolved_gaps=tuple(sorted(unresolved_gaps)),
        files=files,
        oci=OciArtifactConvention.create(dd_version, physical_digest),
    )
    manifest.validate()
    return manifest


def _verify_directory_files(
    directory: Path,
    manifest: MachineArtifactManifest,
    *,
    allow_manifest: bool,
) -> None:
    inventory = _inventory_files(directory, allow_manifest=allow_manifest)
    expected_names = {artifact_file.name for artifact_file in manifest.files}
    actual_names = set(inventory)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise MachineArtifactError(
            f"artifact files differ: missing={missing}, unexpected={unexpected}"
        )
    for artifact_file in manifest.files:
        digest, size = _file_identity(inventory[artifact_file.name])
        if size != artifact_file.size:
            raise MachineArtifactError(
                f"size mismatch for {artifact_file.name!r}: "
                f"expected {artifact_file.size}, got {size}"
            )
        if digest != artifact_file.sha256:
            raise MachineArtifactError(
                f"checksum mismatch for {artifact_file.name!r}: "
                f"expected {artifact_file.sha256}, got {digest}"
            )


def _digest_hex(digest: str) -> str:
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise MachineArtifactError("artifact digest must use the sha256 algorithm")
    value = digest.removeprefix("sha256:")
    _validate_hex(value, (64,), "artifact digest")
    return value


def materialize_machine_artifact(
    source_directory: Path | str,
    cache_directory: Path | str,
    manifest: MachineArtifactManifest,
) -> VerifiedMachineArtifact:
    """Atomically copy a verified bundle into the content-addressed cache."""

    manifest.validate()
    source = Path(source_directory)
    _verify_directory_files(source, manifest, allow_manifest=False)
    digest_hex = _digest_hex(manifest.digest)
    object_root = Path(cache_directory) / "sha256"
    object_root.mkdir(parents=True, exist_ok=True)
    destination = object_root / digest_hex
    if destination.exists():
        resolved = resolve_machine_artifact(cache_directory, manifest.digest)
        if resolved.manifest.canonical_bytes() != manifest.canonical_bytes():
            raise MachineArtifactError(
                f"cache object {manifest.digest} has a different manifest"
            )
        return resolved

    lock_path = object_root / f".{digest_hex}.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise MachineArtifactError(
            f"cache object {manifest.digest} is being materialized"
        ) from error
    os.close(descriptor)
    temporary = Path(tempfile.mkdtemp(prefix=f".{digest_hex}.", dir=object_root))
    try:
        for artifact_file in manifest.files:
            target = temporary / artifact_file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / artifact_file.name, target)
        (temporary / MANIFEST_FILENAME).write_bytes(manifest.canonical_bytes())
        _verify_directory_files(temporary, manifest, allow_manifest=True)
        if destination.exists():
            resolved = resolve_machine_artifact(cache_directory, manifest.digest)
            if resolved.manifest.canonical_bytes() != manifest.canonical_bytes():
                raise MachineArtifactError(
                    f"cache object {manifest.digest} has a different manifest"
                )
        else:
            temporary.rename(destination)
        return resolve_machine_artifact(cache_directory, manifest.digest)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        lock_path.unlink(missing_ok=True)


def resolve_machine_artifact(
    cache_directory: Path | str,
    digest: str,
    *,
    expected_dd_version: str | None = None,
    expected_registry_digest: str | None = None,
    expected_physical_digest: str | None = None,
    require_complete: bool = False,
) -> VerifiedMachineArtifact:
    """Resolve and fully verify one content-addressed local artifact."""

    digest_hex = _digest_hex(digest)
    directory = Path(cache_directory) / "sha256" / digest_hex
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise MachineArtifactError(f"artifact manifest is missing at {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != digest_hex:
        raise MachineArtifactError("manifest identity does not match cache address")
    manifest = MachineArtifactManifest.from_bytes(manifest_bytes)
    expected = (
        ("DD version", expected_dd_version, manifest.dd_version),
        ("registry digest", expected_registry_digest, manifest.registry_digest),
        ("physical digest", expected_physical_digest, manifest.physical_digest),
    )
    for context, requested, actual in expected:
        if requested is not None and requested != actual:
            raise MachineArtifactError(
                f"{context} mismatch: expected {requested!r}, got {actual!r}"
            )
    _verify_directory_files(directory, manifest, allow_manifest=True)
    if require_complete:
        manifest.require_complete()
    return VerifiedMachineArtifact(
        directory=directory,
        manifest=manifest,
        digest=digest,
    )


def pinned_dd_version(
    artifact: MachineArtifactManifest | VerifiedMachineArtifact,
) -> str:
    """Return the exact dictionary pin callers must use when opening IDS data."""

    manifest = (
        artifact.manifest if isinstance(artifact, VerifiedMachineArtifact) else artifact
    )
    manifest.validate()
    return manifest.dd_version
