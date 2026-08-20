"""Create and verify content-addressed machine-description artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
from ctypes import CDLL, c_char_p, c_int, get_errno
from dataclasses import dataclass
from errno import EEXIST, EINVAL, ENOSYS, ENOTEMPTY, EOPNOTSUPP, EPERM
from pathlib import Path, PurePosixPath
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import Any, Iterable, Mapping

from nova.imas.machine_drive import ChannelDrive, DriveMap
from nova.imas.machine_evidence import (
    EvidenceLedger,
    EvidenceRecord,
    FieldEvidence,
    MachineDescriptionError,
    canonical_json,
    require_bool,
    require_exact_keys,
    require_int,
    require_string,
)

MANIFEST_FILENAME = "manifest.json"
OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"

_DD_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_HEX_PATTERN = re.compile(r"[0-9a-f]+")
_MACHINE_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MANIFEST_SCHEMA_PATTERN = re.compile(
    r"nova-(?P<machine>[a-z0-9]+(?:-[a-z0-9]+)*)-machine-artifact"
)
_OCI_REPOSITORY_PATTERN = re.compile(
    r"[a-z0-9]+(?:[.-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
)
_OCI_TAG_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")
_PORTABLE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_SHOT_RANGE_EVIDENCE_STATES = frozenset({"observed", "inherited", "missing"})
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_HDF5_CONSISTENCY_FIELDS = {
    0: (20, 4),
    1: (20, 4),
    2: (11, 1),
    3: (11, 1),
}
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_RENAME_NO_REPLACE = 1
_RENAME_UNSUPPORTED_ERRORS = frozenset({EINVAL, ENOSYS, EOPNOTSUPP, EPERM})


class MachineArtifactError(MachineDescriptionError):
    """Base exception for an invalid or altered machine artifact."""


class IncompleteMachineArtifactError(MachineArtifactError):
    """Raised when operator-ready semantics are requested from incomplete data."""


def machine_name(machine: str) -> str:
    """Return the canonical registry-safe spelling of a machine name."""

    if not isinstance(machine, str):
        raise MachineArtifactError("machine name must be a string")
    canonical = machine.casefold()
    if _MACHINE_PATTERN.fullmatch(canonical) is None:
        raise MachineArtifactError(
            "machine name must contain lowercase letters, digits, and single hyphens"
        )
    return canonical


def manifest_schema(machine: str) -> str:
    """Return the manifest schema identifier for one machine."""

    return f"nova-{machine_name(machine)}-machine-artifact"


def oci_artifact_type(machine: str) -> str:
    """Return the OCI artifact media type for one machine description."""

    return f"application/vnd.iter.nova.{machine_name(machine)}-machine-description.v1"


def oci_file_media_type(machine: str) -> str:
    """Return the OCI payload media type for one machine IDS set."""

    return (
        f"application/vnd.iter.nova.{machine_name(machine)}-machine-description.ids.v1"
    )


def _machine_from_schema(schema: str) -> str:
    if not isinstance(schema, str):
        raise MachineArtifactError("manifest schema must be a string")
    match = _MANIFEST_SCHEMA_PATTERN.fullmatch(schema)
    if match is None:
        raise MachineArtifactError(f"unsupported manifest schema {schema!r}")
    return match.group("machine")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hdf5_consistency_field(
    descriptor: int,
    size: int,
    path: Path,
) -> tuple[int, int] | None:
    """Locate the transient file-consistency field in an HDF5 superblock.

    HDF5 permits a user block before the superblock, but only at byte zero or
    at powers of two starting at 512.  Superblock versions zero and one carry
    a four-byte consistency field; versions two and three carry a one-byte
    field.  No user-block byte or other superblock byte is excluded.
    """

    offset = 0
    while offset + len(_HDF5_SIGNATURE) <= size:
        signature = os.pread(descriptor, len(_HDF5_SIGNATURE), offset)
        if signature == _HDF5_SIGNATURE:
            version_bytes = os.pread(descriptor, 1, offset + len(_HDF5_SIGNATURE))
            if len(version_bytes) != 1:
                raise MachineArtifactError(f"truncated HDF5 superblock in {path}")
            version = version_bytes[0]
            try:
                relative_offset, width = _HDF5_CONSISTENCY_FIELDS[version]
            except KeyError as error:
                raise MachineArtifactError(
                    f"unsupported HDF5 superblock version {version} in {path}"
                ) from error
            field_offset = offset + relative_offset
            if field_offset + width > size:
                raise MachineArtifactError(f"truncated HDF5 superblock in {path}")
            return field_offset, width
        offset = 512 if offset == 0 else offset * 2
    return None


def _file_identity(path: Path) -> tuple[str, int]:
    """Return content identity with only HDF5 open-state flags canonicalized."""

    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MachineArtifactError(f"cannot open artifact file {path}") from error
    metadata = os.fstat(descriptor)
    if not S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise MachineArtifactError(f"artifact path is not a regular file: {path}")
    consistency_field = _hdf5_consistency_field(descriptor, metadata.st_size, path)
    with os.fdopen(descriptor, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if consistency_field is not None:
                field_offset, field_width = consistency_field
                block_stop = size + len(block)
                overlap_start = max(size, field_offset)
                overlap_stop = min(block_stop, field_offset + field_width)
                if overlap_start < overlap_stop:
                    canonical = bytearray(block)
                    canonical[overlap_start - size : overlap_stop - size] = b"\x00" * (
                        overlap_stop - overlap_start
                    )
                    block = canonical
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MachineArtifactError(f"cannot open artifact file {path}") from error
    metadata = os.fstat(descriptor)
    if not S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise MachineArtifactError(f"artifact path is not a regular file: {path}")
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def _require_exact_keys(
    row: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    require_exact_keys(row, expected, context, MachineArtifactError)


def _require_string(value: Any, context: str) -> str:
    return require_string(value, context, MachineArtifactError)


def _require_int(value: Any, context: str) -> int:
    return require_int(value, context, MachineArtifactError)


def _validate_hex(value: str, lengths: tuple[int, ...], context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or _HEX_PATTERN.fullmatch(value) is None
    ):
        allowed = " or ".join(str(length) for length in lengths)
        raise MachineArtifactError(
            f"{context} must be lowercase hexadecimal with length {allowed}"
        )


def _safe_relative_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value == "."
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise MachineArtifactError(f"unsafe artifact file name {value!r}")
    components = value.split("/")
    if any(
        not component
        or component in {".", ".."}
        or _PORTABLE_COMPONENT_PATTERN.fullmatch(component) is None
        or component.endswith((".", " "))
        or component.split(".", 1)[0].upper() in _WINDOWS_DEVICE_NAMES
        for component in components
    ):
        raise MachineArtifactError(f"unsafe artifact file name {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise MachineArtifactError(f"unsafe artifact file name {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise MachineArtifactError(f"non-canonical artifact file name {value!r}")
    if normalized.casefold() == MANIFEST_FILENAME.casefold():
        raise MachineArtifactError(
            f"unsafe artifact file name {value!r}: {MANIFEST_FILENAME!r} is reserved"
        )
    return normalized


def _validate_portable_name_set(names: Iterable[str]) -> None:
    seen: dict[str, str] = {}
    for name in names:
        safe = _safe_relative_name(name)
        parts = PurePosixPath(safe).parts
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            folded = prefix.casefold()
            previous = seen.get(folded)
            if previous is not None and previous != prefix:
                raise MachineArtifactError(
                    f"case-insensitive artifact path collision: "
                    f"{previous!r} and {prefix!r}"
                )
            seen[folded] = prefix


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
        if self.evidence not in _SHOT_RANGE_EVIDENCE_STATES:
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

    def validate(
        self,
        machine: str,
        dd_version: str,
        physical_digest: str,
    ) -> None:
        """Reject convention drift or a tag that disagrees with the identity."""

        expected = oci_artifact_tag(dd_version, physical_digest)
        values = (
            (self.artifact_type, oci_artifact_type(machine), "artifact type"),
            (
                self.manifest_media_type,
                OCI_MANIFEST_MEDIA_TYPE,
                "manifest media type",
            ),
            (self.file_media_type, oci_file_media_type(machine), "file media type"),
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
    def create(
        cls,
        machine: str,
        dd_version: str,
        physical_digest: str,
    ) -> OciArtifactConvention:
        """Return the fixed conventions for a machine artifact identity."""

        return cls(
            artifact_type=oci_artifact_type(machine),
            manifest_media_type=OCI_MANIFEST_MEDIA_TYPE,
            file_media_type=oci_file_media_type(machine),
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
    field_evidence: tuple[EvidenceRecord, ...] = ()
    channel_drive: tuple[ChannelDrive, ...] = ()

    @property
    def machine(self) -> str:
        """Return the machine identity encoded by the manifest schema."""

        return _machine_from_schema(self.schema)

    @property
    def evidence(self) -> EvidenceLedger:
        """Return the field-level provenance carried by this artifact."""

        return EvidenceLedger(records=self.field_evidence)

    @property
    def drive_map(self) -> DriveMap:
        """Return which measured channel drives which conductor, and how hard."""

        return DriveMap(drives=self.channel_drive)

    def driven_columns(self) -> tuple[tuple[str, str], ...]:
        """Return the conductors a campaign's channels can drive."""

        return self.drive_map.columns()

    def forward_model_blockers(self) -> tuple[str, ...]:
        """Return unresolved fields that stop an axisymmetric forward model."""

        return self.evidence.forward_model_blockers()

    def validate(self) -> None:
        """Reject ambiguous, incomplete, or non-canonical manifest state."""

        machine = self.machine
        if self.schema != manifest_schema(machine):
            raise MachineArtifactError(f"unsupported manifest schema {self.schema!r}")
        if (
            not isinstance(self.dd_version, str)
            or _DD_VERSION_PATTERN.fullmatch(self.dd_version) is None
        ):
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
        _validate_portable_name_set(names)
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
        self._validate_field_evidence()
        self._validate_channel_drive()
        self.oci.validate(machine, self.dd_version, self.physical_digest)

    def _validate_field_evidence(self) -> None:
        """Require field provenance consistent with the artifact's shot extent."""

        ledger = self.evidence
        ledger.validate()
        first_shot = min(shot_range.first_shot for shot_range in self.shot_ranges)
        last_shot = max(shot_range.last_shot for shot_range in self.shot_ranges)
        for record in ledger.records:
            if record.first_shot < first_shot or record.last_shot > last_shot:
                raise MachineArtifactError(
                    f"field {record.path!r} claims shots "
                    f"{record.first_shot}-{record.last_shot} outside the artifact "
                    f"extent {first_shot}-{last_shot}"
                )
        unresolved = ledger.paths_with_state(FieldEvidence.UNRESOLVED)
        if self.complete and unresolved:
            raise MachineArtifactError(
                f"complete artifact cannot carry unresolved fields: "
                f"{', '.join(unresolved)}"
            )

    def _validate_channel_drive(self) -> None:
        """Require every drive weight to point at provenance the artifact carries.

        A weight is a claim about the machine, so it is inadmissible on its own
        terms: the record it names has to be in the ledger, or the artifact would
        be publishing a number a consumer scales its whole vacuum field by with
        nothing behind it.
        """

        drive_map = self.drive_map
        drive_map.validate()
        paths = {record.path for record in self.field_evidence}
        missing = sorted(
            {drive.path for drive in drive_map.drives if drive.path not in paths}
        )
        if missing:
            raise MachineArtifactError(
                f"channel drives cite absent evidence records: {', '.join(missing)}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the complete canonical manifest payload."""

        return {
            "channel_drive": self.drive_map.as_list(),
            "complete": self.complete,
            "dd_version": self.dd_version,
            "field_evidence": self.evidence.as_list(),
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
        return canonical_json(self.as_dict())

    @property
    def digest(self) -> str:
        """Return the content address of the canonical manifest."""

        return f"sha256:{_sha256_bytes(self.canonical_bytes())}"

    def semantic_identity(self) -> str:
        """Return the address of the authored semantics alone.

        The stored files are dictionary containers, and their bytes carry
        library metadata that changes between writes, so two authoring runs over
        identical inputs publish different manifest digests.  This address covers
        the dictionary pin, the physical and registry identity, the shot extent
        and every field's provenance, and is therefore reproducible: it answers
        whether two revisions describe the same machine in the same way, which a
        file checksum cannot.
        """

        self.validate()
        payload = {
            key: value
            for key, value in self.as_dict().items()
            if key not in {"files", "oci"}
        }
        return f"sha256:{_sha256_bytes(canonical_json(payload))}"

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
            "channel_drive",
            "complete",
            "dd_version",
            "field_evidence",
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
            complete=require_bool(row["complete"], "complete", MachineArtifactError),
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
            field_evidence=EvidenceLedger.from_list(row["field_evidence"]).records,
            channel_drive=DriveMap.from_list(row["channel_drive"]).drives,
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

    if (
        not isinstance(dd_version, str)
        or _DD_VERSION_PATTERN.fullmatch(dd_version) is None
    ):
        raise MachineArtifactError(f"malformed data dictionary version {dd_version!r}")
    _validate_hex(physical_digest, (16, 64), "physical digest")
    tag = f"dd-{dd_version}-physical-{physical_digest}"
    if _OCI_TAG_PATTERN.fullmatch(tag) is None:
        raise MachineArtifactError(
            "OCI artifact tag must match the distribution grammar and be at most "
            "128 characters"
        )
    return tag


def oci_artifact_reference(
    repository: str,
    manifest: MachineArtifactManifest,
) -> str:
    """Format a tagged OCI reference without contacting a registry."""

    manifest.validate()
    if _OCI_REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise MachineArtifactError(f"malformed OCI repository {repository!r}")
    return f"{repository}:{manifest.oci.tag}"


def _entry_metadata(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise MachineArtifactError(f"cannot inspect artifact path {path}") from error


def _require_contained(path: Path, root: Path, context: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise MachineArtifactError(f"cannot resolve {context}: {path}") from error
    if not resolved.is_relative_to(root):
        raise MachineArtifactError(
            f"{context} escapes canonical cache root {root}: {resolved}"
        )
    return resolved


def _canonical_cache_root(cache_directory: Path | str, *, create: bool) -> Path:
    requested = Path(cache_directory)
    if create:
        try:
            requested.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MachineArtifactError(
                f"cannot create cache root {requested}"
            ) from error
    try:
        root = requested.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise MachineArtifactError(f"cannot resolve cache root {requested}") from error
    if not root.is_dir():
        raise MachineArtifactError(f"cache root is not a directory: {root}")
    return root


def _verified_object_root(cache_directory: Path | str, *, create: bool) -> Path:
    cache_root = _canonical_cache_root(cache_directory, create=create)
    object_root = cache_root / "sha256"
    metadata = _entry_metadata(object_root)
    if metadata is None and create:
        try:
            object_root.mkdir()
        except FileExistsError:
            pass
        except OSError as error:
            raise MachineArtifactError(
                f"cannot create cache object root {object_root}"
            ) from error
        metadata = _entry_metadata(object_root)
    if metadata is None:
        raise MachineArtifactError(f"cache object root is missing: {object_root}")
    if object_root.is_symlink():
        raise MachineArtifactError(
            f"cache object root must not be a symlink: {object_root}"
        )
    if not object_root.is_dir():
        raise MachineArtifactError(
            f"cache object root is not a directory: {object_root}"
        )
    resolved = _require_contained(object_root, cache_root, "cache object root")
    if resolved != object_root:
        raise MachineArtifactError(f"cache object root is not canonical: {object_root}")
    return object_root


def _verified_destination(object_root: Path, digest_hex: str) -> Path | None:
    destination = object_root / digest_hex
    metadata = _entry_metadata(destination)
    if metadata is None:
        return None
    if destination.is_symlink():
        raise MachineArtifactError(
            f"cache digest destination must not be a symlink: {destination}"
        )
    if not destination.is_dir():
        raise MachineArtifactError(
            f"cache digest destination is not a directory: {destination}"
        )
    resolved = _require_contained(destination, object_root.parent, "cache object")
    if resolved != destination:
        raise MachineArtifactError(
            f"cache digest destination is not canonical: {destination}"
        )
    return destination


def _inventory_files(
    directory: Path,
    *,
    allow_manifest: bool,
    containment_root: Path | None = None,
) -> dict[str, Path]:
    if not directory.is_dir():
        raise MachineArtifactError(f"artifact directory does not exist: {directory}")
    if directory.is_symlink():
        raise MachineArtifactError(
            f"artifact directory must not be a symlink: {directory}"
        )
    if containment_root is not None:
        resolved = _require_contained(directory, containment_root, "artifact directory")
        if resolved != directory:
            raise MachineArtifactError(
                f"artifact directory is not canonical: {directory}"
            )
    inventory: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise MachineArtifactError(f"artifact contains symlink {relative!r}")
        if containment_root is not None:
            _require_contained(path, containment_root, f"artifact path {relative!r}")
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
    _validate_portable_name_set(inventory)
    return inventory


def create_machine_artifact_manifest(
    source_directory: Path | str,
    *,
    machine: str,
    dd_version: str,
    registry_digest: str,
    physical_digest: str,
    shot_ranges: Iterable[ArtifactShotRange],
    complete: bool,
    unresolved_gaps: Iterable[str],
    field_evidence: Iterable[EvidenceRecord] = (),
    channel_drive: Iterable[ChannelDrive] = (),
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
        schema=manifest_schema(machine),
        dd_version=dd_version,
        registry_digest=registry_digest,
        physical_digest=physical_digest,
        shot_ranges=tuple(sorted(shot_ranges)),
        complete=complete,
        unresolved_gaps=tuple(sorted(unresolved_gaps)),
        files=files,
        oci=OciArtifactConvention.create(machine, dd_version, physical_digest),
        field_evidence=EvidenceLedger.create(field_evidence).records,
        channel_drive=DriveMap.create(channel_drive).drives,
    )
    manifest.validate()
    return manifest


def _verify_directory_files(
    directory: Path,
    manifest: MachineArtifactManifest,
    *,
    allow_manifest: bool,
    containment_root: Path | None = None,
) -> None:
    inventory = _inventory_files(
        directory,
        allow_manifest=allow_manifest,
        containment_root=containment_root,
    )
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


def _linux_rename_no_replace() -> Any:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise MachineArtifactError(
            "descriptor-relative artifact publication requires Linux open flags"
        )
    required_dir_fd = (os.mkdir, os.open, os.stat)
    if any(function not in os.supports_dir_fd for function in required_dir_fd):
        raise MachineArtifactError(
            "descriptor-relative artifact publication is unavailable"
        )
    if not Path("/proc/self/fd").is_dir():
        raise MachineArtifactError(
            "descriptor-relative artifact paths require the Linux proc filesystem"
        )
    library = CDLL(None, use_errno=True)
    try:
        rename_no_replace = library.renameat2
    except AttributeError as error:
        raise MachineArtifactError(
            "atomic no-clobber directory publication is unavailable"
        ) from error
    rename_no_replace.argtypes = (c_int, c_char_p, c_int, c_char_p, c_int)
    rename_no_replace.restype = c_int
    return rename_no_replace


def _open_pinned_object_root(object_root: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(object_root, flags)
    except OSError as error:
        raise MachineArtifactError(
            f"cannot pin cache object root {object_root}"
        ) from error
    opened = os.fstat(descriptor)
    visible = _entry_metadata(object_root)
    if (
        visible is None
        or S_ISLNK(visible.st_mode)
        or not S_ISDIR(visible.st_mode)
        or (visible.st_dev, visible.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise MachineArtifactError(
            f"cache object root changed while being pinned: {object_root}"
        )
    return descriptor


def _pinned_root_path(descriptor: int, cache_root: Path) -> Path:
    proc_path = Path("/proc/self/fd") / str(descriptor)
    resolved = _require_contained(proc_path, cache_root, "pinned cache object root")
    opened = os.fstat(descriptor)
    current = resolved.stat()
    if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
        raise MachineArtifactError("pinned cache object root identity changed")
    return resolved


def _visible_root_matches_descriptor(object_root: Path, descriptor: int) -> bool:
    visible = _entry_metadata(object_root)
    if visible is None or S_ISLNK(visible.st_mode) or not S_ISDIR(visible.st_mode):
        return False
    opened = os.fstat(descriptor)
    return (visible.st_dev, visible.st_ino) == (opened.st_dev, opened.st_ino)


def _destination_exists_at(descriptor: int, digest_hex: str) -> bool:
    try:
        metadata = os.stat(digest_hex, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise MachineArtifactError(
            f"cannot inspect cache object {digest_hex}"
        ) from error
    if S_ISLNK(metadata.st_mode):
        raise MachineArtifactError(
            f"cache digest destination must not be a symlink: {digest_hex}"
        )
    if not S_ISDIR(metadata.st_mode):
        raise MachineArtifactError(
            f"cache digest destination is not a directory: {digest_hex}"
        )
    return True


def _create_private_directory(descriptor: int, digest_hex: str) -> tuple[str, int]:
    for _ in range(32):
        name = f".{digest_hex}.{secrets.token_hex(12)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=descriptor)
        except FileExistsError:
            continue
        except OSError as error:
            raise MachineArtifactError(
                "cannot create private cache directory"
            ) from error
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            temporary_descriptor = os.open(name, flags, dir_fd=descriptor)
        except OSError as error:
            raise MachineArtifactError("cannot pin private cache directory") from error
        return name, temporary_descriptor
    raise MachineArtifactError("cannot allocate a unique private cache directory")


def _copy_file_at(source: Path, directory_descriptor: int, name: str) -> None:
    parts = PurePosixPath(name).parts
    current = os.dup(directory_descriptor)
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=current)
            except FileExistsError:
                pass
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
        source_flags = os.O_RDONLY | os.O_NOFOLLOW
        source_descriptor = os.open(source, source_flags)
        source_metadata = os.fstat(source_descriptor)
        if not S_ISREG(source_metadata.st_mode):
            os.close(source_descriptor)
            raise MachineArtifactError(
                f"artifact source is not a regular file: {source}"
            )
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            target_descriptor = os.open(
                parts[-1],
                target_flags,
                0o600,
                dir_fd=current,
            )
        except OSError:
            os.close(source_descriptor)
            raise
        with (
            os.fdopen(source_descriptor, "rb") as source_stream,
            os.fdopen(target_descriptor, "wb") as target_stream,
        ):
            shutil.copyfileobj(source_stream, target_stream)
    except OSError as error:
        raise MachineArtifactError(f"cannot copy artifact file {name!r}") from error
    finally:
        os.close(current)


def _write_bytes_at(directory_descriptor: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
    except OSError as error:
        raise MachineArtifactError(f"cannot write artifact file {name!r}") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _publish_directory_no_replace(
    rename_no_replace: Any,
    object_descriptor: int,
    source_name: str,
    destination_name: str,
) -> bool:
    """Atomically publish within a pinned directory and report a winner."""

    result = rename_no_replace(
        object_descriptor,
        os.fsencode(source_name),
        object_descriptor,
        os.fsencode(destination_name),
        _RENAME_NO_REPLACE,
    )
    if result == 0:
        return True
    error_number = get_errno()
    if error_number in {EEXIST, ENOTEMPTY}:
        return False
    error = OSError(error_number, os.strerror(error_number), destination_name)
    if error_number in _RENAME_UNSUPPORTED_ERRORS:
        raise MachineArtifactError(
            "the cache filesystem does not support atomic no-clobber directory "
            "rename, so an artifact cannot be published there without risking a "
            "half-visible object; several parallel filesystems reject the "
            "operation outright"
        ) from error
    raise MachineArtifactError(
        f"cannot publish cache object {destination_name}"
    ) from error


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
    rename_no_replace = _linux_rename_no_replace()
    object_root = _verified_object_root(cache_directory, create=True)
    object_descriptor = _open_pinned_object_root(object_root)
    temporary_name: str | None = None
    temporary_descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        if _destination_exists_at(object_descriptor, digest_hex):
            if not _visible_root_matches_descriptor(object_root, object_descriptor):
                raise MachineArtifactError(
                    "cache object root changed during materialization"
                )
            resolved = resolve_machine_artifact(
                cache_directory,
                manifest.digest,
                allow_incomplete=not manifest.complete,
            )
            if resolved.manifest.canonical_bytes() != manifest.canonical_bytes():
                raise MachineArtifactError(
                    f"cache object {manifest.digest} has a different manifest"
                )
            return resolved

        temporary_name, temporary_descriptor = _create_private_directory(
            object_descriptor,
            digest_hex,
        )
        temporary_path = Path("/proc/self/fd") / str(object_descriptor) / temporary_name
        for artifact_file in manifest.files:
            _copy_file_at(
                source / artifact_file.name,
                temporary_descriptor,
                artifact_file.name,
            )
        _write_bytes_at(
            temporary_descriptor,
            MANIFEST_FILENAME,
            manifest.canonical_bytes(),
        )
        _verify_directory_files(
            temporary_path,
            manifest,
            allow_manifest=True,
        )
        os.close(temporary_descriptor)
        temporary_descriptor = None
        _publish_directory_no_replace(
            rename_no_replace,
            object_descriptor,
            temporary_name,
            digest_hex,
        )
        _pinned_root_path(object_descriptor, object_root.parent)
        if not _visible_root_matches_descriptor(object_root, object_descriptor):
            raise MachineArtifactError(
                "cache object root changed during materialization"
            )
        resolved = resolve_machine_artifact(
            cache_directory,
            manifest.digest,
            allow_incomplete=not manifest.complete,
        )
        if resolved.manifest.canonical_bytes() != manifest.canonical_bytes():
            raise MachineArtifactError(
                f"cache object {manifest.digest} has a different manifest"
            )
        return resolved
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_path is not None:
            shutil.rmtree(temporary_path, ignore_errors=True)
        os.close(object_descriptor)


def resolve_machine_artifact(
    cache_directory: Path | str,
    digest: str,
    *,
    expected_dd_version: str | None = None,
    expected_registry_digest: str | None = None,
    expected_physical_digest: str | None = None,
    allow_incomplete: bool = False,
) -> VerifiedMachineArtifact:
    """Resolve and fully verify one content-addressed local artifact."""

    if not isinstance(allow_incomplete, bool):
        raise MachineArtifactError("allow_incomplete must be a boolean")
    digest_hex = _digest_hex(digest)
    object_root = _verified_object_root(cache_directory, create=False)
    directory = _verified_destination(object_root, digest_hex)
    if directory is None:
        raise MachineArtifactError(
            f"cache object {digest} is missing under {object_root}"
        )
    manifest_path = directory / MANIFEST_FILENAME
    metadata = _entry_metadata(manifest_path)
    if metadata is None:
        raise MachineArtifactError(f"artifact manifest is missing at {manifest_path}")
    if manifest_path.is_symlink():
        raise MachineArtifactError(
            f"artifact manifest must not be a symlink: {manifest_path}"
        )
    _require_contained(manifest_path, object_root.parent, "artifact manifest")
    manifest_bytes = _read_regular_bytes(manifest_path)
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
    _verify_directory_files(
        directory,
        manifest,
        allow_manifest=True,
        containment_root=object_root.parent,
    )
    if not allow_incomplete:
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
