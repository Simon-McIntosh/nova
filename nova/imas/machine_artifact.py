"""Create and verify content-addressed machine-description artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Iterable, Mapping

from imas_data_dictionaries import dd_xml_versions, parse_dd_version

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
from nova.database.content_store import (
    MANIFEST_FILENAME,
    ContentStoreError,
    _create_private_directory as _create_private_directory,
    digest_hex as _digest_hex,
    entry_metadata as _entry_metadata,
    file_content_identity as _file_identity,
    hdf5_consistency_field as _hdf5_consistency_field,
    inventory_files as _inventory_files,
    _linux_rename_no_replace as _linux_rename_no_replace,
    publish_files as _publish_files,
    read_regular_bytes as _read_regular_bytes,
    require_contained as _require_contained,
    safe_relative_name as _safe_relative_name,
    sha256_bytes as _sha256_bytes,
    validate_hex as _validate_hex,
    validate_portable_name_set as _validate_portable_name_set,
    verified_destination as _verified_destination,
    verified_object_root as _verified_object_root,
    verify_directory_files as _verify_directory_files,
)

OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"

_DD_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_MACHINE_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MANIFEST_SCHEMA_PATTERN = re.compile(
    r"nova-(?P<machine>[a-z0-9]+(?:-[a-z0-9]+)*)-machine-artifact"
)
_OCI_REPOSITORY_PATTERN = re.compile(
    r"[a-z0-9]+(?:[.-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
)
_OCI_TAG_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")
_SHOT_RANGE_EVIDENCE_STATES = frozenset({"observed", "inherited", "missing"})
_PUBLICATION_DD_MAJOR = 4


class MachineArtifactError(MachineDescriptionError):
    """Base exception for an invalid or altered machine artifact."""


class IncompleteMachineArtifactError(MachineArtifactError):
    """Raised when operator-ready semantics are requested from incomplete data."""


def _lifted(function: Any) -> Any:
    """Re-raise a content-store refusal as the artifact error callers catch.

    The store is machine-agnostic and raises its own exception base; artifact
    callers keep catching the artifact error, so the two vocabularies meet here
    rather than leaking a store type through the artifact API.
    """

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except ContentStoreError as error:
            raise MachineArtifactError(str(error)) from error

    return wrapper


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


def latest_publication_dd_version() -> str:
    """Return the newest available data dictionary in the publication major."""

    candidates = tuple(
        version
        for version in dd_xml_versions()
        if parse_dd_version(version).major == _PUBLICATION_DD_MAJOR
    )
    if not candidates:
        raise MachineArtifactError(
            "no data dictionary is available for machine-artifact publication"
        )
    return str(max(candidates, key=parse_dd_version))


def publication_dd_version(dd_version: str | None = None) -> str:
    """Resolve the runtime default and refuse publication from another major."""

    latest = latest_publication_dd_version()
    selected = latest if dd_version is None else dd_version
    if not isinstance(selected, str) or _DD_VERSION_PATTERN.fullmatch(selected) is None:
        raise MachineArtifactError(f"malformed data dictionary version {selected!r}")
    if parse_dd_version(selected).major != parse_dd_version(latest).major:
        raise MachineArtifactError(
            f"machine-artifact publication requires data dictionary major "
            f"{parse_dd_version(latest).major}, got {selected!r}"
        )
    return selected


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

    @_lifted
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

    @_lifted
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

    @_lifted
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


@_lifted
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


@_lifted
def create_machine_artifact_manifest(
    source_directory: Path | str,
    *,
    machine: str,
    dd_version: str | None = None,
    registry_digest: str,
    physical_digest: str,
    shot_ranges: Iterable[ArtifactShotRange],
    complete: bool,
    unresolved_gaps: Iterable[str],
    field_evidence: Iterable[EvidenceRecord] = (),
    channel_drive: Iterable[ChannelDrive] = (),
) -> MachineArtifactManifest:
    """Hash an authored IDS directory into a canonical manifest."""

    dd_version = publication_dd_version(dd_version)
    source = Path(source_directory)
    inventory = _inventory_files(source, allow_manifest=False)
    files = tuple(
        sorted(
            ArtifactFile(name=name, sha256=digest, size=size)
            for name, path in inventory.items()
            for digest, size in [
                _file_identity(path, consistency_field_locator=_hdf5_consistency_field)
            ]
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


@_lifted
def materialize_machine_artifact(
    source_directory: Path | str,
    cache_directory: Path | str,
    manifest: MachineArtifactManifest,
) -> VerifiedMachineArtifact:
    """Atomically copy a verified bundle into the content-addressed cache."""

    manifest.validate()
    source = Path(source_directory)
    _verify_directory_files(source, manifest.files, allow_manifest=False)
    _publish_files(
        cache_directory,
        manifest.digest,
        manifest.files,
        {item.name: source / item.name for item in manifest.files},
        manifest.canonical_bytes(),
        private_directory_factory=_create_private_directory,
        rename_factory=_linux_rename_no_replace,
    )
    return resolve_machine_artifact(
        cache_directory,
        manifest.digest,
        allow_incomplete=not manifest.complete,
    )


@_lifted
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
        manifest.files,
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
