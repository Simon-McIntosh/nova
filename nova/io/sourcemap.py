"""What one facility channel means, as data a reader can check.

A machine description says where the conductors and sensors are.  It does not say
which of an archive's named channels measures which of them, in what unit, or in
whose coordinate convention, and none of those three is recoverable from the
channel name.  A :class:`SourceSignal` states all of it for one channel: the
group and channel it is read from, the unit it arrives in, the described sensor or
conductor it belongs to, the Data Dictionary path it fills, the transformation
type that fixes its convention factor, and the evidence the conversion rests on.

The conversion is three named factors, never one opaque number:

``unit_factor``
    the arithmetic of the unit strings alone -- kiloamperes to amperes.
``channel_factor``
    what the archive did to the quantity before publishing it.  A channel
    already multiplied by a turn count, or one feeding several parallel
    circuits, differs from the conductor quantity by this factor and by nothing
    else.
``convention_factor``
    derived from the source and target conventions by
    :mod:`~nova_cocos`, never written down.

Keeping them apart is what makes a wrong one findable.  A unit error is a power of
ten, a channel error is a turn count, and a convention error is a sign or a 2*pi;
collapsed into a single scale they are indistinguishable, and the round trip
passes either way because it inverts whatever it applied.

A channel the description cannot receive is not given a weight of one and quietly
served.  It becomes a :class:`BlockedSignal` carrying the condition that is
unmet, so the count of what a shot can and cannot supply is readable off the map
instead of being discovered by a consumer at solve time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import xarray

from nova.imas.machine_evidence import (
    FieldEvidence,
    canonical_json,
    require_exact_keys,
    require_float,
    require_int,
    require_string,
)
from nova_cocos import (
    TRANSFORMATIONS,
    B0_LIKE,
    DODPSI_LIKE,
    IP_LIKE,
    PSI_LIKE,
    Q_LIKE,
    convention,
    convention_transform,
    identify_convention,
)
from nova.io.ingest import PROVISIONAL_NAMESPACE
from nova.io.standardname import StandardNameResolver

ACCEPTED = "accepted"
"""The standard name is published in the governed catalogue."""

PROPOSAL = "proposal"
"""The standard name is a recorded proposal, absent from the governed catalogue."""

CATALOG_STATES = frozenset({ACCEPTED, PROPOSAL})

_SERVABLE_EVIDENCE = frozenset(
    {
        FieldEvidence.MEASURED,
        FieldEvidence.PUBLISHED,
        FieldEvidence.GENERATED,
        FieldEvidence.FITTED,
    }
)
"""An unresolved conversion is a blocked channel, so it is never a served row."""


class SourceMapError(ValueError):
    """Raised when a map row is malformed or inadmissible."""


def _trimmed(value: Any, context: str) -> str:
    text = require_string(value, context, SourceMapError)
    if not text or text.strip() != text:
        raise SourceMapError(f"{context} must be non-empty trimmed text")
    return text


@dataclass(frozen=True, order=True)
class SourceSignal:
    """One facility channel, the described thing it measures, and the conversion.

    ``target_index`` is the position inside the Data Dictionary array of structures
    the channel fills, and it is the join a consumer cannot make for itself: an
    archive orders its channels by its own naming and a machine description orders
    its sensors by geometry, so which sensor a channel reads is a statement that
    has to be measured and then written down.

    ``time_base`` names the clock the samples sit on.  Two channels acquired on
    different clocks are not resampled onto one here, because interpolating a
    measurement is a modelling choice and a map is not the place to make it.
    """

    standard_name: str
    catalog_status: str
    source_group: str
    source_channel: str
    source_unit: str
    target_path: str
    target_unit: str
    target_index: int | None
    transformation: str
    source_convention: int
    target_convention: int
    unit_factor: float
    channel_factor: float
    time_base: str
    evidence: FieldEvidence
    statement: str

    @property
    def sort_key(self) -> tuple[str, str, str]:
        """Return the canonical ordering key."""

        return (self.source_group, self.source_channel, self.target_path)

    @property
    def convention_factor(self) -> float:
        """Return the factor the two conventions impose on this quantity."""

        transform = convention_transform(
            source=self.source_convention,
            target=self.target_convention,
        )
        return transform.factor(self.transformation)

    @property
    def factor(self) -> float:
        """Return the whole conversion: units, channel semantics and convention."""

        return (
            float(self.unit_factor)
            * float(self.channel_factor)
            * (self.convention_factor)
        )

    def apply(self, values: Any) -> np.ndarray:
        """Convert source samples to the target unit and convention."""

        return np.asarray(values, dtype=float) * self.factor

    def invert(self, values: Any) -> np.ndarray:
        """Convert target samples back to the source unit and convention."""

        return np.asarray(values, dtype=float) / self.factor

    def validate(self) -> None:
        """Reject a row whose conversion, target or provenance is inadmissible."""

        for value, context in (
            (self.standard_name, "standard name"),
            (self.source_group, "source group"),
            (self.source_channel, "source channel"),
            (self.source_unit, "source unit"),
            (self.target_path, "target path"),
            (self.target_unit, "target unit"),
            (self.time_base, "time base"),
            (self.statement, "conversion statement"),
        ):
            _trimmed(value, context)
        if self.catalog_status not in CATALOG_STATES:
            raise SourceMapError(
                f"{self.source_channel!r} carries unknown catalogue state "
                f"{self.catalog_status!r}"
            )
        if self.transformation not in TRANSFORMATIONS:
            raise SourceMapError(
                f"{self.source_channel!r} names unknown transformation type "
                f"{self.transformation!r}"
            )
        if not isinstance(self.evidence, FieldEvidence):
            raise SourceMapError(f"unknown evidence state {self.evidence!r}")
        if self.evidence not in _SERVABLE_EVIDENCE:
            raise SourceMapError(
                f"{self.source_channel!r} cannot carry evidence state "
                f"{self.evidence}: a channel with no admissible conversion is a "
                "blocked channel and carries no row"
            )
        if self.target_index is not None:
            index = require_int(self.target_index, "target index", SourceMapError)
            if index < 0:
                raise SourceMapError("target index must be non-negative")
        for value, context in (
            (self.unit_factor, f"unit factor for {self.source_channel!r}"),
            (self.channel_factor, f"channel factor for {self.source_channel!r}"),
        ):
            scale = require_float(value, context, SourceMapError)
            if scale == 0.0:
                raise SourceMapError(f"{context} is zero, which erases the signal")
        for value, context in (
            (self.source_convention, "source convention"),
            (self.target_convention, "target convention"),
        ):
            require_int(value, context, SourceMapError)
        # constructing the transform validates both conventions and the type
        self.convention_factor

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "catalog_status": self.catalog_status,
            "channel_factor": float(self.channel_factor),
            "convention_factor": float(self.convention_factor),
            "evidence": str(self.evidence),
            "source_channel": self.source_channel,
            "source_convention": int(self.source_convention),
            "source_group": self.source_group,
            "source_unit": self.source_unit,
            "standard_name": self.standard_name,
            "statement": self.statement,
            "target_convention": int(self.target_convention),
            "target_index": self.target_index,
            "target_path": self.target_path,
            "target_unit": self.target_unit,
            "time_base": self.time_base,
            "transformation": self.transformation,
            "unit_factor": float(self.unit_factor),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> SourceSignal:
        """Build a validated row from decoded JSON.

        ``convention_factor`` is derived rather than read: a stored value that
        disagreed with the algebra would be a second source of truth for a sign.
        """

        expected = {
            "catalog_status",
            "channel_factor",
            "convention_factor",
            "evidence",
            "source_channel",
            "source_convention",
            "source_group",
            "source_unit",
            "standard_name",
            "statement",
            "target_convention",
            "target_index",
            "target_path",
            "target_unit",
            "time_base",
            "transformation",
            "unit_factor",
        }
        require_exact_keys(row, expected, "source signal", SourceMapError)
        state = require_string(row["evidence"], "evidence state", SourceMapError)
        try:
            evidence = FieldEvidence(state)
        except ValueError as error:
            raise SourceMapError(f"unknown evidence state {state!r}") from error
        index = row["target_index"]
        signal = cls(
            standard_name=require_string(
                row["standard_name"], "standard name", SourceMapError
            ),
            catalog_status=require_string(
                row["catalog_status"], "catalogue state", SourceMapError
            ),
            source_group=require_string(
                row["source_group"], "source group", SourceMapError
            ),
            source_channel=require_string(
                row["source_channel"], "source channel", SourceMapError
            ),
            source_unit=require_string(
                row["source_unit"], "source unit", SourceMapError
            ),
            target_path=require_string(
                row["target_path"], "target path", SourceMapError
            ),
            target_unit=require_string(
                row["target_unit"], "target unit", SourceMapError
            ),
            target_index=(
                None
                if index is None
                else require_int(index, "target index", SourceMapError)
            ),
            transformation=require_string(
                row["transformation"], "transformation type", SourceMapError
            ),
            source_convention=require_int(
                row["source_convention"], "source convention", SourceMapError
            ),
            target_convention=require_int(
                row["target_convention"], "target convention", SourceMapError
            ),
            unit_factor=require_float(
                row["unit_factor"], "unit factor", SourceMapError
            ),
            channel_factor=require_float(
                row["channel_factor"], "channel factor", SourceMapError
            ),
            time_base=require_string(row["time_base"], "time base", SourceMapError),
            evidence=evidence,
            statement=require_string(row["statement"], "statement", SourceMapError),
        )
        signal.validate()
        stored = require_float(
            row["convention_factor"], "convention factor", SourceMapError
        )
        if not np.isclose(stored, signal.convention_factor, rtol=1e-12, atol=0.0):
            raise SourceMapError(
                f"stored convention factor {stored!r} for "
                f"{signal.source_channel!r} disagrees with the factor the two "
                f"conventions give, {signal.convention_factor!r}"
            )
        return signal


@dataclass(frozen=True, order=True)
class BlockedSignal:
    """One channel the description cannot receive, and what would unblock it."""

    source_group: str
    source_channel: str
    target_path: str
    reason: str
    unmet: str

    @property
    def sort_key(self) -> tuple[str, str]:
        """Return the canonical ordering key."""

        return (self.source_group, self.source_channel)

    def validate(self) -> None:
        """Reject a blocked channel whose obstruction is not stated."""

        for value, context in (
            (self.source_group, "blocked source group"),
            (self.source_channel, "blocked source channel"),
            (self.reason, "blocked reason"),
            (self.unmet, "unmet condition"),
        ):
            _trimmed(value, context)
        if self.target_path.strip() != self.target_path:
            raise SourceMapError("blocked target path must be trimmed text")

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "reason": self.reason,
            "source_channel": self.source_channel,
            "source_group": self.source_group,
            "target_path": self.target_path,
            "unmet": self.unmet,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> BlockedSignal:
        """Build a validated blocked channel from decoded JSON."""

        expected = {
            "reason",
            "source_channel",
            "source_group",
            "target_path",
            "unmet",
        }
        require_exact_keys(row, expected, "blocked signal", SourceMapError)
        blocked = cls(
            source_group=require_string(
                row["source_group"], "blocked source group", SourceMapError
            ),
            source_channel=require_string(
                row["source_channel"], "blocked source channel", SourceMapError
            ),
            target_path=require_string(
                row["target_path"], "blocked target path", SourceMapError
            ),
            reason=require_string(row["reason"], "blocked reason", SourceMapError),
            unmet=require_string(row["unmet"], "unmet condition", SourceMapError),
        )
        blocked.validate()
        return blocked


@dataclass(frozen=True)
class SourceSignalMap:
    """Every channel a facility publishes for one description, served or blocked."""

    signals: tuple[SourceSignal, ...] = ()
    blocked: tuple[BlockedSignal, ...] = ()

    def validate(self) -> None:
        """Reject unordered rows, or one channel serving one target twice."""

        keys = [signal.sort_key for signal in self.signals]
        if keys != sorted(keys):
            raise SourceMapError("source signals must be canonically ordered")
        if len(set(keys)) != len(keys):
            raise SourceMapError("one channel cannot fill one target twice")
        blocked_keys = [row.sort_key for row in self.blocked]
        if blocked_keys != sorted(blocked_keys):
            raise SourceMapError("blocked channels must be canonically ordered")
        served = {
            (signal.source_group, signal.source_channel) for signal in self.signals
        }
        overlap = served & set(blocked_keys)
        if overlap:
            raise SourceMapError(
                f"channels {sorted(overlap)} are both served and blocked"
            )
        for signal in self.signals:
            signal.validate()
        for row in self.blocked:
            row.validate()

    @classmethod
    def create(
        cls,
        signals: Iterable[SourceSignal],
        blocked: Iterable[BlockedSignal] = (),
    ) -> SourceSignalMap:
        """Return a canonically ordered validated map."""

        source_map = cls(
            signals=tuple(sorted(signals, key=lambda row: row.sort_key)),
            blocked=tuple(sorted(blocked, key=lambda row: row.sort_key)),
        )
        source_map.validate()
        return source_map

    def channels(self) -> tuple[str, ...]:
        """Return every served channel, in canonical order."""

        return tuple(signal.source_channel for signal in self.signals)

    def for_channel(self, channel: str) -> tuple[SourceSignal, ...]:
        """Return every row a channel fills."""

        return tuple(
            signal for signal in self.signals if signal.source_channel == channel
        )

    def for_target(self, target_path: str) -> tuple[SourceSignal, ...]:
        """Return every row filling one target path."""

        return tuple(
            signal for signal in self.signals if signal.target_path == target_path
        )

    def select(self, channels: Iterable[str]) -> SourceSignalMap:
        """Return the rows a campaign's channel set can use."""

        wanted = set(channels)
        return SourceSignalMap(
            signals=tuple(
                signal for signal in self.signals if signal.source_channel in wanted
            ),
            blocked=self.blocked,
        )

    def time_bases(self) -> tuple[str, ...]:
        """Return every clock the served rows sit on."""

        return tuple(sorted({signal.time_base for signal in self.signals}))

    def proposals(self) -> tuple[str, ...]:
        """Return the standard names still waiting on catalogue publication."""

        return tuple(
            sorted(
                {
                    signal.standard_name
                    for signal in self.signals
                    if signal.catalog_status == PROPOSAL
                }
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "blocked": [row.as_dict() for row in self.blocked],
            "signals": [signal.as_dict() for signal in self.signals],
        }

    def canonical_bytes(self) -> bytes:
        """Serialize the validated map to byte-stable JSON."""

        self.validate()
        return canonical_json(self.as_dict())

    @property
    def digest(self) -> str:
        """Return the content address of the canonical map."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceSignalMap:
        """Parse a canonical map payload into a validated map."""

        require_exact_keys(
            payload, {"blocked", "signals"}, "source map", SourceMapError
        )
        for key in ("signals", "blocked"):
            rows = payload[key]
            if isinstance(rows, str) or not isinstance(rows, Iterable):
                raise SourceMapError(f"{key} must be an array")
        source_map = cls(
            signals=tuple(
                SourceSignal.from_dict(row) for row in _rows(payload["signals"])
            ),
            blocked=tuple(
                BlockedSignal.from_dict(row) for row in _rows(payload["blocked"])
            ),
        )
        source_map.validate()
        return source_map


def _rows(rows: Any) -> Iterable[Mapping[str, Any]]:
    for row in rows:
        if not isinstance(row, Mapping):
            raise SourceMapError("map entries must be objects")
        yield row


@dataclass(frozen=True)
class SignalGroup:
    """The rows sharing one standard name and one clock, in target order.

    A description carries many sensors of a kind, and the archive publishes one
    channel per sensor, so the tensorized form of a family is one variable with a
    channel axis rather than seventy-five variables with the same name.  The axis
    is ordered by the described target so the array's row and the description's
    sensor are the same index by construction.
    """

    standard_name: str
    time_base: str
    signals: tuple[SourceSignal, ...]

    @property
    def key(self) -> tuple[str, str]:
        """Return the grouping key."""

        return (self.standard_name, self.time_base)


def group_signals(signals: Iterable[SourceSignal]) -> tuple[SignalGroup, ...]:
    """Group rows by standard name and clock, ordering each by its target."""

    grouped: dict[tuple[str, str], list[SourceSignal]] = {}
    for signal in signals:
        grouped.setdefault((signal.standard_name, signal.time_base), []).append(signal)
    return tuple(
        SignalGroup(
            standard_name=name,
            time_base=base,
            signals=tuple(
                sorted(
                    members,
                    key=lambda row: (
                        row.target_index if row.target_index is not None else -1,
                        row.target_path,
                    ),
                )
            ),
        )
        for (name, base), members in sorted(grouped.items())
    )


def tensorize(
    source_map: SourceSignalMap,
    samples: Mapping[str, Any],
    clocks: Mapping[str, Any],
    *,
    resolver: StandardNameResolver | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> xarray.Dataset:
    """Return the converted signals as one standard-name-keyed dataset.

    Every variable carries the source channels it was built from, the three
    conversion factors, both conventions and the described targets it fills, so a
    number in the store can be traced to a channel and a conversion without the
    map beside it.  A name the installed catalogue does not carry is served under
    the provisional namespace exactly as :mod:`~nova.io.ingest` serves one, which
    keeps a candidate contribution visibly distinct from a governed name.
    """

    resolver = resolver or StandardNameResolver()
    coordinates: dict[str, Any] = {}
    for base in source_map.time_bases():
        if base not in clocks:
            raise SourceMapError(f"no clock supplied for time base {base!r}")
        coordinates[f"{base}_time"] = np.asarray(clocks[base], dtype=float)
    data_vars: dict[str, xarray.DataArray] = {}
    provisional: list[str] = []
    for group in group_signals(source_map.signals):
        missing = [
            signal.source_channel
            for signal in group.signals
            if signal.source_channel not in samples
        ]
        if missing:
            raise SourceMapError(f"no samples supplied for channels {sorted(missing)}")
        resolution = resolver.resolve(group.standard_name)
        time_name = f"{group.time_base}_time"
        length = coordinates[time_name].size
        columns = []
        for signal in group.signals:
            values = signal.apply(samples[signal.source_channel])
            if values.shape != (length,):
                raise SourceMapError(
                    f"channel {signal.source_channel!r} has {values.shape} samples "
                    f"against {length} on clock {group.time_base!r}"
                )
            columns.append(values)
        channel_dim = f"{group.standard_name}_channel"
        variable = group.standard_name
        if resolution.provisional:
            variable = f"{PROVISIONAL_NAMESPACE}/{group.standard_name}"
            provisional.append(group.standard_name)
        array = xarray.DataArray(
            np.stack(columns, axis=1),
            dims=(time_name, channel_dim),
            coords={
                channel_dim: [signal.source_channel for signal in group.signals],
            },
            attrs={
                "standard_name": group.standard_name,
                "catalog_status": group.signals[0].catalog_status,
                "channel_factor": [
                    float(signal.channel_factor) for signal in group.signals
                ],
                "convention_factor": [
                    float(signal.convention_factor) for signal in group.signals
                ],
                "evidence": [str(signal.evidence) for signal in group.signals],
                "resolution_source": resolution.source.value,
                "resolution_status": resolution.status,
                "source_convention": [
                    int(signal.source_convention) for signal in group.signals
                ],
                "source_group": [signal.source_group for signal in group.signals],
                "source_unit": [signal.source_unit for signal in group.signals],
                "target_convention": [
                    int(signal.target_convention) for signal in group.signals
                ],
                "target_index": [
                    -1 if signal.target_index is None else int(signal.target_index)
                    for signal in group.signals
                ],
                "target_path": [signal.target_path for signal in group.signals],
                "transformation": [signal.transformation for signal in group.signals],
                "unit_factor": [float(signal.unit_factor) for signal in group.signals],
                "units": group.signals[0].target_unit,
            },
        )
        data_vars[variable] = array
    dataset = xarray.Dataset(data_vars, coords=coordinates)
    dataset.attrs = {
        "map_digest": source_map.digest,
        "blocked_channels": [row.source_channel for row in source_map.blocked],
        "provisional_names": sorted(set(provisional)),
        **dict(attrs or {}),
    }
    return dataset


def round_trip_residual(
    source_map: SourceSignalMap,
    dataset: xarray.Dataset,
    samples: Mapping[str, Any],
) -> dict[str, float]:
    """Return each channel's relative error after converting back to the source.

    The test this supports is narrow on purpose.  Inverting a conversion cannot
    show that the conversion is right -- it divides by whatever it multiplied by --
    so what a residual here proves is only that nothing was lost or reordered
    between the store, the dataset and the described target.  Whether the factors
    themselves are right is settled by evidence outside the round trip: measured
    channel ratios, a held-out response fit, and the sign cohorts.
    """

    residuals: dict[str, float] = {}
    for group in group_signals(source_map.signals):
        variable = group.standard_name
        if variable not in dataset:
            variable = f"{PROVISIONAL_NAMESPACE}/{group.standard_name}"
        array = dataset[variable]
        channel_dim = f"{group.standard_name}_channel"
        names = [str(name) for name in array.coords[channel_dim].values]
        for signal in group.signals:
            column = np.asarray(
                array.isel({channel_dim: names.index(signal.source_channel)}).values,
                dtype=float,
            )
            recovered = signal.invert(column)
            raw = np.asarray(samples[signal.source_channel], dtype=float)
            residuals[signal.source_channel] = _relative_error(recovered, raw)
    return residuals


def _relative_error(recovered: np.ndarray, raw: np.ndarray) -> float:
    """Return the largest absolute difference, scaled by the raw signal's size."""

    finite = np.isfinite(recovered) & np.isfinite(raw)
    if not finite.any():
        return float("nan")
    scale = float(np.max(np.abs(raw[finite])))
    if scale == 0.0:
        return float(np.max(np.abs(recovered[finite] - raw[finite])))
    return float(np.max(np.abs(recovered[finite] - raw[finite])) / scale)


DD3_CONVENTION = 11
"""The COCOS the Data Dictionary major version 3 publishes equilibrium data in."""

DD4_CONVENTION = 17
"""The COCOS the Data Dictionary major version 4 publishes equilibrium data in."""

NOVA_CONVENTION = DD4_CONVENTION
"""The convention nova works in internally, so a DD4 read passes through."""

EQUILIBRIUM_GROUP = "equilibrium"
"""The source group every equilibrium quantity row belongs to."""

_DD_EQUILIBRIUM_QUANTITIES: tuple[tuple[str, str, str], ...] = (
    ("profiles_2d/psi", PSI_LIKE, "Wb"),
    ("profiles_1d/psi", PSI_LIKE, "Wb"),
    ("profiles_1d/dpressure_dpsi", DODPSI_LIKE, "Pa/Wb"),
    ("profiles_1d/f_df_dpsi", DODPSI_LIKE, "T.m/Wb"),
    ("profiles_1d/q", Q_LIKE, "1"),
    ("global_quantities/ip", IP_LIKE, "A"),
    ("global_quantities/psi_axis", PSI_LIKE, "Wb"),
    ("global_quantities/psi_boundary", PSI_LIKE, "Wb"),
    ("global_quantities/q_axis", Q_LIKE, "1"),
    ("global_quantities/q_95", Q_LIKE, "1"),
    ("boundary_separatrix/psi", PSI_LIKE, "Wb"),
    ("vacuum_toroidal_field/b0", B0_LIKE, "T"),
)
"""One row per equilibrium quantity a convention rescales, with its DD unit.

The transformation type is the Data Dictionary's own tag for the quantity, so
the factor follows from the two conventions rather than from a table here.
"""


MAST_SOURCE_CONVENTION = 3
"""The COCOS the MAST level-1 EFIT archive publishes its flux in.

The archive is not a Data Dictionary read, so its convention is not implied by
a DD version and has to be declared.  It is measured rather than assumed: the
MAST EFIT reconstruction on the level-1 store is COCOS 3.
"""

MAST_FLUX_TRANSFORMATION = PSI_LIKE
"""The transformation every flux row of the MAST archive carries."""


def mast_flux_factor() -> float:
    """Return the factor the MAST archive's flux rows take into COCOS 17.

    COCOS 3 and COCOS 17 share ``sigma_Bp`` and ``sigma_RphiZ`` and differ in
    ``e_Bp``, so a flux-like row converts by ``2*pi`` with no sign change.  The
    MAST frame reader applies this factor and nothing else, so the 2*pi is
    applied exactly once on every flux row of that route.
    """

    return convention_transform(
        source=MAST_SOURCE_CONVENTION, target=NOVA_CONVENTION
    ).factor(MAST_FLUX_TRANSFORMATION)


def dd_equilibrium_convention(dd_version: str) -> int:
    """Return the COCOS nova reads a Data Dictionary version's equilibrium data as.

    DD3 data is COCOS 11 and DD4 data is COCOS 17.  Any other major version
    names no convention nova reads, so it is refused rather than guessed.
    """

    text = require_string(dd_version, "DD version", SourceMapError)
    major = text.split(".", 1)[0].strip()
    if major == "3":
        return DD3_CONVENTION
    if major == "4":
        return DD4_CONVENTION
    raise SourceMapError(
        f"no COCOS is declared for Data Dictionary version {text!r}; nova reads "
        f"DD3 as COCOS {DD3_CONVENTION} and DD4 as COCOS {DD4_CONVENTION}"
    )


def dd_equilibrium_factor(dd_version: str, transformation: str) -> float:
    """Return the factor carrying nova's own convention into a write's convention.

    The translation above runs the other way, from a source's convention into
    COCOS 17; a writer that stores nova's data into a Data Dictionary entry
    needs the inverse direction.  Which convention a version carries is
    :func:`dd_equilibrium_convention`'s to state, and which factor a quantity
    type takes follows from the two conventions through
    :func:`nova_cocos.convention_transform`, so a DD3 target stores a psi-like
    row with a sign flip and a DD4 target passes it through.
    """

    return convention_transform(
        source=NOVA_CONVENTION, target=dd_equilibrium_convention(dd_version)
    ).factor(transformation)


def determine_digits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-slice sign rows to the four COCOS digits.

    ``sigma_Bp`` comes from Sauter Eq. 23, ``sign(psi_boundary - psi_axis) =
    sigma_Ip * sigma_Bp``, cross-checked against ``sign(dp/dpsi) = -sigma_Ip *
    sigma_Bp``.  ``sigma_rho_theta_phi`` comes from ``sign(q) = sigma_Ip *
    sigma_B0 * sigma_rho_theta_phi``.  ``e_Bp`` and ``sigma_RphiZ`` are not
    derivable from scalar equilibrium quantities and are supplied by the caller:
    a Data Dictionary read takes both from its DD version, and the archive
    reader measures ``e_Bp`` with a flux-loop ratio.  A digit no row pins is
    reported as ``0`` -- undetermined -- rather than defaulted to a sign.
    """

    signed = [r for r in rows if r.get("has_signs")]
    if not signed:
        return {"n_shots": 0}

    sigma_ip = np.array([np.sign(r["ip"]) for r in signed])
    sigma_psi = np.array([np.sign(r["psi_boundary"] - r["psi_axis"]) for r in signed])
    sigma_bp = sigma_psi * sigma_ip

    b0 = np.array([r["b0"] for r in signed])
    q95 = np.array([r["q_95"] for r in signed])
    have_q = np.isfinite(b0) & np.isfinite(q95) & (q95 != 0.0)
    sigma_rtp = np.sign(q95[have_q]) * sigma_ip[have_q] * np.sign(b0[have_q])

    dp = np.array([r["dp_dpsi"] for r in signed])
    have_dp = np.isfinite(dp) & (dp != 0.0)
    dp_expected = -sigma_ip[have_dp] * sigma_bp[have_dp]
    dp_agrees = np.sign(dp[have_dp]) == dp_expected

    irod = np.array([r["irod"] for r in signed])
    have_rod = np.isfinite(irod) & np.isfinite(b0) & (irod != 0.0) & (b0 != 0.0)
    rod_agrees = np.sign(irod[have_rod]) == np.sign(b0[have_rod])

    def unanimous(a: np.ndarray) -> tuple[int, float]:
        if a.size == 0:
            return 0, 0.0
        vals, counts = np.unique(a, return_counts=True)
        best = int(vals[int(np.argmax(counts))])
        return best, float(counts.max() / a.size)

    bp, bp_frac = unanimous(sigma_bp[np.isfinite(sigma_bp)])
    rtp, rtp_frac = unanimous(sigma_rtp[np.isfinite(sigma_rtp)])
    return {
        "n_shots": len(signed),
        "sigma_bp": bp,
        "sigma_bp_unanimity": bp_frac,
        "sigma_rho_theta_phi": rtp,
        "sigma_rho_theta_phi_unanimity": rtp_frac,
        "ip_positive_fraction": float(np.mean(sigma_ip > 0)),
        "b0_negative_fraction": float(np.mean(b0[np.isfinite(b0)] < 0))
        if np.isfinite(b0).any()
        else float("nan"),
        "axis_is_flux_maximum_fraction": float(
            np.mean([r["psi_axis_minus_boundary"] > 0 for r in signed])
        ),
        "dp_dpsi_cross_check_pass_fraction": float(np.mean(dp_agrees))
        if dp_agrees.size
        else float("nan"),
        "dp_dpsi_cross_check_n": int(dp_agrees.size),
        "rod_field_consistency_pass_fraction": float(np.mean(rod_agrees))
        if rod_agrees.size
        else float("nan"),
        "rod_field_consistency_n": int(rod_agrees.size),
    }


_UNSET_MAGNITUDE = 1.0e30
"""Above this a Data Dictionary float is its 'unset' sentinel, not a measurement."""


def _observed(value: float) -> float:
    """Return a Data Dictionary float, with an unset sentinel read as absent."""

    number = float(value)
    if not np.isfinite(number) or abs(number) >= _UNSET_MAGNITUDE:
        return float("nan")
    return number


def _digits(row: Mapping[str, float]) -> dict[str, Any]:
    """The four digits one equilibrium slice's observables pin.

    A slice whose flux samples are absent or unset pins nothing, so it reports
    no digit rather than a sign derived from the sentinel.
    """

    ip = _observed(row["ip"])
    psi_axis = _observed(row["psi_axis"])
    psi_boundary = _observed(row["psi_boundary"])
    has_ip = bool(np.isfinite(ip))
    return determine_digits(
        [
            {
                "has_signs": has_ip,
                "ip": ip,
                "psi_axis": psi_axis,
                "psi_boundary": psi_boundary,
                "psi_axis_minus_boundary": psi_axis - psi_boundary,
                "b0": _observed(row["b0"]),
                "q_95": _observed(row["q"]),
                "dp_dpsi": np.nan,
                "irod": np.nan,
            }
        ]
    )


def translate_equilibrium(
    *,
    source_name: str,
    dd_version: str,
    ip: float,
    psi_axis: float,
    psi_boundary: float,
    q: float,
    b0: float,
) -> SourceSignalMap:
    """Translate one Data Dictionary equilibrium read into COCOS 17, or refuse it.

    The Data Dictionary fixes a read's COCOS by its major version: DD3 data is
    COCOS 11 and DD4 data is COCOS 17.  nova works in COCOS 17, so a DD4 read
    passes through with a factor of one on every quantity and a DD3 read is
    converted.  ``e_Bp = 1`` and ``sigma_RphiZ = +1`` are fixed by the DD
    version, since COCOS 11 and COCOS 17 share both; ``sigma_Bp`` and
    ``sigma_rho_theta_phi`` come from the data -- ``ip``, ``psi_axis``,
    ``psi_boundary``, ``q`` and ``b0`` are the observables that pin them.

    A read whose measured digits name a convention other than its own DD
    version's is refused through this map's blocked path, by name, stating the
    DD version, the expected COCOS and the measured digits.  A digit the slice
    does not pin -- an unpopulated ``q`` or ``b0`` leaves
    ``sigma_rho_theta_phi`` undetermined -- leaves that digit unchecked rather
    than guessed, so ``sigma_Bp`` alone still decides conformance.
    """

    expected = dd_equilibrium_convention(dd_version)
    digits = _digits(
        {"ip": ip, "psi_axis": psi_axis, "psi_boundary": psi_boundary, "q": q, "b0": b0}
    )
    sigma_bp = int(digits.get("sigma_bp", 0))
    sigma_rtp = int(digits.get("sigma_rho_theta_phi", 0))
    reference = convention(expected)
    measured: int | None = None
    if sigma_rtp != 0 and sigma_bp != 0:
        measured = identify_convention(
            sigma_bp=sigma_bp,
            e_bp=1,
            sigma_r_phi_z=+1,
            sigma_rho_theta_phi=sigma_rtp,
        )
    conforming = sigma_bp in (0, reference.sigma_bp) and sigma_rtp in (
        0,
        reference.sigma_rho_theta_phi,
    )
    if not conforming:
        stated_bp = (
            "not derivable (psi unpopulated)" if sigma_bp == 0 else f"{sigma_bp:+d}"
        )
        stated_rtp = (
            "not derivable (q or B0 unpopulated)"
            if sigma_rtp == 0
            else f"{sigma_rtp:+d}"
        )
        measured_text = "none" if measured is None else f"COCOS {measured}"
        return SourceSignalMap.create(
            (),
            (
                BlockedSignal(
                    source_group=EQUILIBRIUM_GROUP,
                    source_channel=source_name,
                    target_path=EQUILIBRIUM_GROUP,
                    reason=(
                        f"{source_name} is written at DD {dd_version}, which nova "
                        f"reads as COCOS {expected}, but its measured digits are "
                        f"sigma_Bp={stated_bp}, e_Bp=1, sigma_RphiZ=+1, "
                        f"sigma_rho_theta_phi={stated_rtp}, which name "
                        f"{measured_text} -- the data does not conform to its own "
                        "Data Dictionary version"
                    ),
                    unmet=(
                        f"a conforming DD {dd_version} equilibrium carries COCOS "
                        f"{expected}: sigma_Bp={reference.sigma_bp:+d} and "
                        f"sigma_rho_theta_phi={reference.sigma_rho_theta_phi:+d}"
                    ),
                ),
            ),
        )
    signals = tuple(
        SourceSignal(
            standard_name=f"{EQUILIBRIUM_GROUP}/{path}",
            catalog_status=ACCEPTED,
            source_group=EQUILIBRIUM_GROUP,
            source_channel=f"{EQUILIBRIUM_GROUP}/{path}",
            source_unit=unit,
            target_path=f"{EQUILIBRIUM_GROUP}/{path}",
            target_unit=unit,
            target_index=None,
            transformation=transformation,
            source_convention=expected,
            target_convention=NOVA_CONVENTION,
            unit_factor=1.0,
            channel_factor=1.0,
            time_base=f"{EQUILIBRIUM_GROUP}_time",
            evidence=FieldEvidence.PUBLISHED,
            statement=(
                f"{source_name} publishes {path} in the convention DD {dd_version} "
                f"carries, COCOS {expected}; the target is COCOS {NOVA_CONVENTION} and "
                "the factor is the two conventions' composition with no unit or "
                "channel term"
            ),
        )
        for path, transformation, unit in _DD_EQUILIBRIUM_QUANTITIES
    )
    return SourceSignalMap.create(signals)


def served_targets(source_map: SourceSignalMap) -> dict[str, Sequence[int]]:
    """Return the described target indices each Data Dictionary container receives.

    The container is the path with its named element and its leaf signal removed,
    so a consumer can ask which described sensors of a kind a shot filled without
    knowing which leaf under each of them the samples went to.
    """

    served: dict[str, list[int]] = {}
    for signal in source_map.signals:
        if signal.target_index is None:
            continue
        container = signal.target_path.split("(")[0].removesuffix("/data")
        served.setdefault(container, [])
        if signal.target_index not in served[container]:
            served[container].append(signal.target_index)
    return {name: sorted(rows) for name, rows in sorted(served.items())}
