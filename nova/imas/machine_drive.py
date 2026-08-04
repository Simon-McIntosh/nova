"""Which measured channel drives which described conductor, and how hard.

A machine description says where every conductor sits and how many turns it
carries.  That is not enough to build a forward operator, because the campaign
that recorded a shot did not publish a current per conductor: it published a set
of named channels, and what one ampere in a channel means differs from channel to
channel.  One channel measures the current in a single conductor, so the ampere
turns it drives are the conductor's turn count.  Another has already been
multiplied by that count before publication, so the ampere turns it drives are
one per ampere and multiplying by the turn count again inflates the column by the
turn count squared.  A third feeds two parallel circuits, so it drives half its
amperes through each.

A :class:`ChannelDrive` states that conversion once, per channel, as data.  Its
weight is the total ampere turns the named conductor carries per ampere of the
named channel, and it **supersedes** the conductor's ``turns_with_sign`` for that
channel -- a consumer applies one or the other, never both.  Every drive carries
the same provenance apparatus as a field value, because a weight is a claim about
the machine exactly as a turn count is, and a fitted weight must stay
distinguishable from a measured one.

A conductor whose current no channel measures simply has no drive.  Absence is
the statement, and why it is absent belongs in the evidence ledger beside it,
never as a weight of one standing in for a measurement nobody made.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from nova.imas.machine_evidence import (
    FieldEvidence,
    MachineDescriptionError,
    Uncertainty,
    canonical_json,
    require_exact_keys,
    require_float,
    require_int,
    require_string,
)


class DriveError(MachineDescriptionError):
    """Raised when a channel-drive record is malformed or inadmissible."""


DRIVE_CONTAINERS = frozenset({"pf_active", "pf_passive"})
"""IDSs whose conductors a measured current channel may drive."""

SINGLE_ELEMENT = "single"
"""The conductor is one element and takes the whole weight."""

SECTION_AREA = "section_area"
"""The weight splits across the listed elements in proportion to section area.

This is what a uniform current density in one connected conductor does, and it is
the only split admissible without a per-element measurement: a group of plates
that share an enclosure carry one induced current between them, and how much each
takes follows from its cross-section rather than from a fit.
"""

DRIVE_DISTRIBUTIONS = frozenset({SINGLE_ELEMENT, SECTION_AREA})

_DRIVEABLE_EVIDENCE = frozenset(
    {
        FieldEvidence.MEASURED,
        FieldEvidence.PUBLISHED,
        FieldEvidence.GENERATED,
        FieldEvidence.FITTED,
    }
)
"""An unresolved drive is an absent drive, so it is never carried as a record."""

_NEEDS_INTERVAL = frozenset({FieldEvidence.GENERATED, FieldEvidence.FITTED})


def _trimmed(value: Any, context: str) -> str:
    text = require_string(value, context, DriveError)
    if not text or text.strip() != text:
        raise DriveError(f"{context} must be non-empty trimmed text")
    return text


@dataclass(frozen=True, order=True)
class ChannelDrive:
    """One measured channel, the conductor it drives, and the ampere turns it drives.

    ``elements`` names the element indices inside ``conductor`` the weight applies
    to, so a channel can drive part of a container's structure without the
    container being cut up to say so.  ``path`` points at the evidence record
    carrying this drive's provenance, which is what keeps a machine-readable
    weight and its justification from drifting apart.
    """

    channel: str
    container: str
    conductor: str
    elements: tuple[int, ...]
    circuit: str
    ampere_turns_per_ampere: float
    distribution: str
    evidence: FieldEvidence
    path: str
    uncertainty: Uncertainty | None = None

    @property
    def sort_key(self) -> str:
        """Return the canonical ordering key."""

        return self.channel

    def validate(self) -> None:
        """Reject a drive whose weight, target or provenance is inadmissible."""

        for value, context in (
            (self.channel, "drive channel"),
            (self.conductor, "drive conductor"),
            (self.path, "drive path"),
        ):
            _trimmed(value, context)
        if self.circuit and self.circuit.strip() != self.circuit:
            raise DriveError(f"drive circuit for {self.channel!r} must be trimmed text")
        if self.container not in DRIVE_CONTAINERS:
            raise DriveError(
                f"drive {self.channel!r} names unknown container {self.container!r}"
            )
        if not isinstance(self.evidence, FieldEvidence):
            raise DriveError(f"unknown evidence state {self.evidence!r}")
        if self.evidence not in _DRIVEABLE_EVIDENCE:
            raise DriveError(
                f"drive {self.channel!r} cannot carry evidence state {self.evidence}: "
                "a channel with no admissible weight carries no drive at all"
            )
        if not self.elements:
            raise DriveError(f"drive {self.channel!r} must name at least one element")
        context = f"element index for {self.channel!r}"
        for index in self.elements:
            if require_int(index, context, DriveError) < 0:
                raise DriveError(f"{context} must be non-negative")
        if tuple(sorted(set(self.elements))) != self.elements:
            raise DriveError(f"elements for {self.channel!r} must be sorted and unique")
        weight = require_float(
            self.ampere_turns_per_ampere,
            f"weight for {self.channel!r}",
            DriveError,
        )
        if weight == 0.0:
            raise DriveError(
                f"drive {self.channel!r} carries a zero weight, which describes a "
                "conductor the channel does not drive rather than one it does"
            )
        if self.distribution not in DRIVE_DISTRIBUTIONS:
            raise DriveError(
                f"drive {self.channel!r} names unknown distribution "
                f"{self.distribution!r}"
            )
        if (self.distribution == SINGLE_ELEMENT) != (len(self.elements) == 1):
            raise DriveError(
                f"drive {self.channel!r} distributes {len(self.elements)} elements as "
                f"{self.distribution!r}"
            )
        if self.uncertainty is not None:
            self.uncertainty.validate()
        if self.evidence in _NEEDS_INTERVAL and self.uncertainty is None:
            raise DriveError(
                f"{self.evidence} drive {self.channel!r} must bound its weight"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "ampere_turns_per_ampere": float(self.ampere_turns_per_ampere),
            "channel": self.channel,
            "circuit": self.circuit,
            "conductor": self.conductor,
            "container": self.container,
            "distribution": self.distribution,
            "elements": list(self.elements),
            "evidence": str(self.evidence),
            "path": self.path,
            "uncertainty": (
                None if self.uncertainty is None else self.uncertainty.as_dict()
            ),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ChannelDrive:
        """Build a validated drive from decoded JSON."""

        expected = {
            "ampere_turns_per_ampere",
            "channel",
            "circuit",
            "conductor",
            "container",
            "distribution",
            "elements",
            "evidence",
            "path",
            "uncertainty",
        }
        require_exact_keys(row, expected, "channel drive", DriveError)
        state = require_string(row["evidence"], "drive evidence state", DriveError)
        try:
            evidence = FieldEvidence(state)
        except ValueError as error:
            raise DriveError(f"unknown evidence state {state!r}") from error
        elements = row["elements"]
        if isinstance(elements, str) or not isinstance(elements, Iterable):
            raise DriveError("drive elements must be an array")
        uncertainty = row["uncertainty"]
        if uncertainty is not None and not isinstance(uncertainty, Mapping):
            raise DriveError("drive uncertainty must be an object or null")
        result = cls(
            channel=require_string(row["channel"], "drive channel", DriveError),
            container=require_string(row["container"], "drive container", DriveError),
            conductor=require_string(row["conductor"], "drive conductor", DriveError),
            elements=tuple(
                require_int(index, "drive element index", DriveError)
                for index in elements
            ),
            circuit=require_string(row["circuit"], "drive circuit", DriveError),
            ampere_turns_per_ampere=require_float(
                row["ampere_turns_per_ampere"],
                "drive weight",
                DriveError,
            ),
            distribution=require_string(
                row["distribution"],
                "drive distribution",
                DriveError,
            ),
            evidence=evidence,
            path=require_string(row["path"], "drive path", DriveError),
            uncertainty=(
                None if uncertainty is None else Uncertainty.from_dict(uncertainty)
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class DriveMap:
    """Every channel a campaign publishes that drives a described conductor."""

    drives: tuple[ChannelDrive, ...] = ()

    def validate(self) -> None:
        """Reject unordered drives or one channel carrying two weights."""

        keys = [drive.sort_key for drive in self.drives]
        if keys != sorted(keys):
            raise DriveError("channel drives must be canonically ordered")
        if len(set(keys)) != len(keys):
            raise DriveError("one channel cannot carry two drive weights")
        for drive in self.drives:
            drive.validate()

    @classmethod
    def create(cls, drives: Iterable[ChannelDrive]) -> DriveMap:
        """Return a canonically ordered validated drive map."""

        drive_map = cls(drives=tuple(sorted(drives, key=lambda row: row.sort_key)))
        drive_map.validate()
        return drive_map

    def channels(self) -> tuple[str, ...]:
        """Return every driven channel, in canonical order."""

        return tuple(drive.channel for drive in self.drives)

    def columns(self) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
        """Return the distinct conductor sets driven, and what identifies each.

        A column is a container, a component inside it and the elements the drive
        reaches, and it is the element set that makes the count right in both
        directions.  Two channels publishing the same coil at different scales --
        a conductor current and its ampere-turn product -- reach the same elements
        and are one column.  Several channels reaching disjoint element sets of
        one component -- a case family whose enclosures are measured separately --
        are that many columns, without the component being cut up to say so.
        """

        seen = []
        for drive in self.drives:
            column = (drive.container, drive.conductor, drive.elements)
            if column not in seen:
                seen.append(column)
        return tuple(sorted(seen))

    def for_channel(self, channel: str) -> ChannelDrive:
        """Return the drive one channel carries."""

        for drive in self.drives:
            if drive.channel == channel:
                return drive
        raise KeyError(f"no drive for channel {channel!r}")

    def select(self, channels: Iterable[str]) -> DriveMap:
        """Return the drives a campaign's channel set can use."""

        wanted = set(channels)
        return DriveMap(
            drives=tuple(drive for drive in self.drives if drive.channel in wanted)
        )

    def as_list(self) -> list[dict[str, Any]]:
        """Return the canonical JSON representation."""

        return [drive.as_dict() for drive in self.drives]

    def canonical_bytes(self) -> bytes:
        """Serialize the validated drive map to byte-stable JSON."""

        self.validate()
        return canonical_json({"drives": self.as_list()})

    @property
    def digest(self) -> str:
        """Return the content address of the canonical drive map."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()[:16]

    @classmethod
    def from_list(cls, rows: Any) -> DriveMap:
        """Parse a canonical drive array into a validated map."""

        if isinstance(rows, str) or not isinstance(rows, Iterable):
            raise DriveError("channel drives must be an array")
        drive_map = cls(
            drives=tuple(
                ChannelDrive.from_dict(row)
                if isinstance(row, Mapping)
                else _raise_drive_error()
                for row in rows
            )
        )
        drive_map.validate()
        return drive_map


def _raise_drive_error() -> Any:
    raise DriveError("channel drive entry must be an object")
