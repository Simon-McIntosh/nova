"""Provenance, uncertainty and shot support for machine-description fields.

Every authored field carries one evidence state.  A state is a claim about how
a value was obtained, not about how good it is: a generated seed and a fitted
parameter are both usable, and both stay distinguishable from a measurement.
An unresolved field has no admissible value at all, so it is never written into
an IDS and never silently defaults to one.

Paths name the field a record governs.  A device-level field is written as
``<ids>/<node>``; one member of a named array of structures is written as
``<ids>/<array>(<name>)/<node>``, because a Data Dictionary index alone does
not identify a physical component.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping


class MachineDescriptionError(ValueError):
    """Base exception for invalid machine-description provenance or identity."""


class EvidenceError(MachineDescriptionError):
    """Raised when a field's evidence record is malformed or inadmissible."""


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Serialize to timestamp-free, byte-stable JSON."""

    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def require_exact_keys(
    row: Mapping[str, Any],
    expected: set[str],
    context: str,
    error: type[MachineDescriptionError],
) -> None:
    """Reject a decoded object whose field set differs from the schema."""

    actual = set(row)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise error(
            f"{context} fields differ: missing={missing}, unexpected={unexpected}"
        )


def require_string(
    value: Any,
    context: str,
    error: type[MachineDescriptionError],
) -> str:
    """Require a string field."""

    if not isinstance(value, str):
        raise error(f"{context} must be a string")
    return value


def require_int(
    value: Any,
    context: str,
    error: type[MachineDescriptionError],
) -> int:
    """Require an integer field, rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise error(f"{context} must be an integer")
    return value


def require_bool(
    value: Any,
    context: str,
    error: type[MachineDescriptionError],
) -> bool:
    """Require a boolean field."""

    if not isinstance(value, bool):
        raise error(f"{context} must be a boolean")
    return value


def require_float(
    value: Any,
    context: str,
    error: type[MachineDescriptionError],
) -> float:
    """Require a finite real field, rejecting booleans and non-finite values."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise error(f"{context} must be a real number")
    if not math.isfinite(value):
        raise error(f"{context} must be finite")
    return float(value)


def _trimmed(value: Any, context: str) -> str:
    text = require_string(value, context, EvidenceError)
    if not text or text.strip() != text:
        raise EvidenceError(f"{context} must be non-empty trimmed text")
    return text


def _trimmed_sequence(values: Any, context: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise EvidenceError(f"{context} must be a sequence of strings")
    result = tuple(_trimmed(value, f"{context} entry") for value in values)
    if len(set(result)) != len(result):
        raise EvidenceError(f"{context} must be unique")
    return result


class FieldEvidence(StrEnum):
    """How one machine-description field value was obtained."""

    MEASURED = "measured"
    PUBLISHED = "published"
    GENERATED = "generated"
    FITTED = "fitted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, order=True)
class SourceReference:
    """One document, the place inside it, and the machine it describes."""

    title: str
    url: str
    locator: str
    machine: str
    text_verified: bool

    def validate(self) -> None:
        """Reject a citation that cannot be followed back to a document."""

        for value, context in (
            (self.title, "source title"),
            (self.url, "source url"),
            (self.locator, "source locator"),
            (self.machine, "source machine"),
        ):
            _trimmed(value, context)
        if not self.url.startswith("https://"):
            raise EvidenceError(f"source url must be https: {self.url!r}")
        if self.machine != self.machine.lower():
            raise EvidenceError(f"source machine must be lowercase: {self.machine!r}")
        require_bool(self.text_verified, "source text_verified", EvidenceError)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "locator": self.locator,
            "machine": self.machine,
            "text_verified": self.text_verified,
            "title": self.title,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> SourceReference:
        """Build a validated citation from decoded JSON."""

        expected = {"locator", "machine", "text_verified", "title", "url"}
        require_exact_keys(row, expected, "source", EvidenceError)
        result = cls(
            title=require_string(row["title"], "source title", EvidenceError),
            url=require_string(row["url"], "source url", EvidenceError),
            locator=require_string(row["locator"], "source locator", EvidenceError),
            machine=require_string(row["machine"], "source machine", EvidenceError),
            text_verified=require_bool(
                row["text_verified"],
                "source text_verified",
                EvidenceError,
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, order=True)
class Uncertainty:
    """A closed interval the true value is asserted to lie inside."""

    lower: float
    upper: float
    unit: str

    def validate(self) -> None:
        """Reject an inverted, non-finite or unitless interval."""

        lower = require_float(self.lower, "uncertainty lower bound", EvidenceError)
        upper = require_float(self.upper, "uncertainty upper bound", EvidenceError)
        if upper < lower:
            raise EvidenceError(
                f"uncertainty upper bound {upper} precedes lower bound {lower}"
            )
        _trimmed(self.unit, "uncertainty unit")

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies inside the closed interval."""

        return self.lower <= value <= self.upper

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {"lower": self.lower, "unit": self.unit, "upper": self.upper}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Uncertainty:
        """Build a validated interval from decoded JSON."""

        expected = {"lower", "unit", "upper"}
        require_exact_keys(row, expected, "uncertainty", EvidenceError)
        result = cls(
            lower=require_float(row["lower"], "uncertainty lower bound", EvidenceError),
            upper=require_float(row["upper"], "uncertainty upper bound", EvidenceError),
            unit=require_string(row["unit"], "uncertainty unit", EvidenceError),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class EvidenceRecord:
    """Evidence state, provenance and shot support for one field."""

    path: str
    evidence: FieldEvidence
    first_shot: int
    last_shot: int
    statement: str
    assumptions: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    source: SourceReference | None = None
    uncertainty: Uncertainty | None = None
    blocks_axisymmetric_forward_model: bool = False

    @property
    def sort_key(self) -> tuple[str, int, int, str]:
        """Return the canonical ordering key."""

        return (self.path, self.first_shot, self.last_shot, str(self.evidence))

    def supports(self, shot: int) -> bool:
        """Return whether this record applies to ``shot``."""

        return self.first_shot <= shot <= self.last_shot

    def validate(self) -> None:
        """Reject a record whose provenance does not support its state."""

        _trimmed(self.path, "record path")
        _trimmed(self.statement, f"statement for {self.path!r}")
        if not isinstance(self.evidence, FieldEvidence):
            raise EvidenceError(f"unknown evidence state {self.evidence!r}")
        first = require_int(self.first_shot, "record first shot", EvidenceError)
        last = require_int(self.last_shot, "record last shot", EvidenceError)
        if first < 0:
            raise EvidenceError("record first shot must be non-negative")
        if last < first:
            raise EvidenceError("record last shot must not precede first shot")
        _trimmed_sequence(self.assumptions, f"assumptions for {self.path!r}")
        candidates = _trimmed_sequence(self.candidates, f"candidates for {self.path!r}")
        if candidates and tuple(sorted(candidates)) != candidates:
            raise EvidenceError(f"candidates for {self.path!r} must be sorted")
        require_bool(
            self.blocks_axisymmetric_forward_model,
            f"forward-model flag for {self.path!r}",
            EvidenceError,
        )
        if self.source is not None:
            self.source.validate()
        if self.uncertainty is not None:
            self.uncertainty.validate()
        self._validate_state()

    def _validate_state(self) -> None:
        """Require the provenance each evidence state depends on."""

        needs_source = {FieldEvidence.MEASURED, FieldEvidence.PUBLISHED}
        needs_interval = {FieldEvidence.GENERATED, FieldEvidence.FITTED}
        if self.evidence in needs_source and self.source is None:
            raise EvidenceError(
                f"{self.evidence} field {self.path!r} must cite a source"
            )
        if self.evidence in needs_interval:
            if self.uncertainty is None:
                raise EvidenceError(
                    f"{self.evidence} field {self.path!r} must carry an uncertainty"
                )
            if not self.assumptions:
                raise EvidenceError(
                    f"{self.evidence} field {self.path!r} must state its assumptions"
                )
        if self.evidence is FieldEvidence.UNRESOLVED:
            if self.uncertainty is not None:
                raise EvidenceError(
                    f"unresolved field {self.path!r} cannot bound a value it lacks"
                )
            if not self.assumptions:
                raise EvidenceError(
                    f"unresolved field {self.path!r} must state what is missing"
                )
            if len(self.candidates) == 1:
                raise EvidenceError(
                    f"unresolved field {self.path!r} cannot offer one candidate"
                )
        elif self.candidates:
            raise EvidenceError(
                f"{self.evidence} field {self.path!r} cannot keep open candidates"
            )
        if self.blocks_axisymmetric_forward_model and (
            self.evidence is not FieldEvidence.UNRESOLVED
        ):
            raise EvidenceError(
                f"{self.evidence} field {self.path!r} cannot block the forward model"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "assumptions": list(self.assumptions),
            "blocks_axisymmetric_forward_model": (
                self.blocks_axisymmetric_forward_model
            ),
            "candidates": list(self.candidates),
            "evidence": str(self.evidence),
            "first_shot": self.first_shot,
            "last_shot": self.last_shot,
            "path": self.path,
            "source": None if self.source is None else self.source.as_dict(),
            "statement": self.statement,
            "uncertainty": (
                None if self.uncertainty is None else self.uncertainty.as_dict()
            ),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> EvidenceRecord:
        """Build a validated record from decoded JSON."""

        expected = {
            "assumptions",
            "blocks_axisymmetric_forward_model",
            "candidates",
            "evidence",
            "first_shot",
            "last_shot",
            "path",
            "source",
            "statement",
            "uncertainty",
        }
        require_exact_keys(row, expected, "evidence record", EvidenceError)
        state = require_string(row["evidence"], "evidence state", EvidenceError)
        try:
            evidence = FieldEvidence(state)
        except ValueError as error:
            raise EvidenceError(f"unknown evidence state {state!r}") from error
        source = row["source"]
        uncertainty = row["uncertainty"]
        if source is not None and not isinstance(source, Mapping):
            raise EvidenceError("source must be an object or null")
        if uncertainty is not None and not isinstance(uncertainty, Mapping):
            raise EvidenceError("uncertainty must be an object or null")
        result = cls(
            path=require_string(row["path"], "record path", EvidenceError),
            evidence=evidence,
            first_shot=require_int(
                row["first_shot"],
                "record first shot",
                EvidenceError,
            ),
            last_shot=require_int(row["last_shot"], "record last shot", EvidenceError),
            statement=require_string(
                row["statement"],
                "record statement",
                EvidenceError,
            ),
            assumptions=_trimmed_sequence(row["assumptions"], "assumptions"),
            candidates=_trimmed_sequence(row["candidates"], "candidates"),
            source=None if source is None else SourceReference.from_dict(source),
            uncertainty=(
                None if uncertainty is None else Uncertainty.from_dict(uncertainty)
            ),
            blocks_axisymmetric_forward_model=require_bool(
                row["blocks_axisymmetric_forward_model"],
                "forward-model flag",
                EvidenceError,
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class EvidenceLedger:
    """The complete, canonically ordered evidence set for one artifact."""

    records: tuple[EvidenceRecord, ...] = ()

    def validate(self) -> None:
        """Reject unordered records or two evidence states for one field."""

        keys = [record.sort_key for record in self.records]
        if keys != sorted(keys):
            raise EvidenceError("evidence records must be canonically ordered")
        spans: dict[str, list[tuple[int, int]]] = {}
        for record in self.records:
            record.validate()
            for first, last in spans.setdefault(record.path, []):
                if record.first_shot <= last and first <= record.last_shot:
                    raise EvidenceError(
                        f"field {record.path!r} carries two evidence states over "
                        f"shots {max(first, record.first_shot)}-"
                        f"{min(last, record.last_shot)}"
                    )
            spans[record.path].append((record.first_shot, record.last_shot))

    @classmethod
    def create(cls, records: Iterable[EvidenceRecord]) -> EvidenceLedger:
        """Return a canonically ordered validated ledger."""

        ledger = cls(records=tuple(sorted(records, key=lambda row: row.sort_key)))
        ledger.validate()
        return ledger

    def state_counts(self) -> dict[str, int]:
        """Count records by evidence state."""

        counts = {str(state): 0 for state in FieldEvidence}
        for record in self.records:
            counts[str(record.evidence)] += 1
        return counts

    def paths_with_state(self, evidence: FieldEvidence) -> tuple[str, ...]:
        """Return the fields carrying one evidence state, in canonical order."""

        return tuple(
            record.path for record in self.records if record.evidence is evidence
        )

    def forward_model_blockers(self) -> tuple[str, ...]:
        """Return the unresolved fields that stop an axisymmetric forward model."""

        return tuple(
            record.path
            for record in self.records
            if record.blocks_axisymmetric_forward_model
        )

    def for_shot(self, shot: int) -> EvidenceLedger:
        """Select the records supporting one shot."""

        return EvidenceLedger(
            records=tuple(record for record in self.records if record.supports(shot))
        )

    def as_list(self) -> list[dict[str, Any]]:
        """Return the canonical JSON representation."""

        return [record.as_dict() for record in self.records]

    def canonical_bytes(self) -> bytes:
        """Serialize the validated ledger to byte-stable JSON."""

        self.validate()
        return canonical_json({"records": self.as_list()})

    @property
    def digest(self) -> str:
        """Return the content address of the canonical ledger."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()[:16]

    @classmethod
    def from_list(cls, rows: Any) -> EvidenceLedger:
        """Parse a canonical record array into a validated ledger."""

        if isinstance(rows, str) or not isinstance(rows, Iterable):
            raise EvidenceError("evidence records must be an array")
        ledger = cls(
            records=tuple(
                EvidenceRecord.from_dict(row)
                if isinstance(row, Mapping)
                else _raise_record_error()
                for row in rows
            )
        )
        ledger.validate()
        return ledger


def _raise_record_error() -> Any:
    raise EvidenceError("evidence record entry must be an object")
