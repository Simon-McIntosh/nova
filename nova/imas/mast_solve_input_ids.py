"""Load a shot's solve inputs into the described machine and read them back.

The map says what each channel means; this is where a shot is actually served.
The description is opened at its own pinned dictionary version, the converted
samples are written into the dynamic fields of the same IDSs that carry the
geometry, the pulse is written and reopened, and every number is converted back to
the channel it came from.  Nothing about the description is touched on the way
through, and that is checked rather than assumed: a load that quietly moved a
probe would otherwise be indistinguishable from one that did not.

What a round trip does and does not establish is worth being exact about.
Inverting a conversion divides by whatever it multiplied by, so a residual at
floating-point level proves only that the samples reached the right described
sensor and came back in the right order -- it says nothing about whether the
factors are right.  The factors rest on evidence gathered elsewhere: measured
channel ratios, the held-out response fit that produced the turn counts, and the
two field polarities.  The redundancy check here is the one exception, because a
coil publishing both a conductor current and its ampere-turn product gives two
independent routes to the same column, and their disagreement is a measurement of
the turn count the description carries.

One field of the dictionary takes one value, so a coil publishing two channels has
to have one of them chosen.  The conductor-current channel wins: its conversion to
a conductor current is exactly one whatever the turn count is, while the
ampere-turn channel's conversion is the reciprocal of a fitted number.  Choosing
the route that does not pass through a fit is what keeps the fit available as a
check rather than spending it as an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import imas
import numpy as np

from nova.imas.machine_artifact import (
    VerifiedMachineArtifact,
    pinned_dd_version,
    resolve_machine_artifact,
)
from nova.imas.mast_solve_inputs import (
    COIL_CURRENT_NAME,
    CONDUCTOR_UNIT,
    LOOP_FLUX_NAME,
    PLASMA_CURRENT_NAME,
    PROBE_FIELD_NAME,
    SHOT_STORE,
    DescribedMachine,
    ShotSignals,
    SolveInputError,
    describe_machine,
    field_polarity,
    loop_target_indices,
    read_solve_inputs,
    reconstruction_loop_positions,
    solve_input_map,
)
from nova.io.ingest import PROVISIONAL_NAMESPACE
from nova.io.sourcemap import (
    SourceSignal,
    SourceSignalMap,
    group_signals,
    round_trip_residual,
)

DESCRIPTION_IDS = ("pf_active", "pf_passive", "magnetics", "wall", "tf")
"""The machine-description IDSs a solve input load opens."""

DYNAMIC_IDS = ("pf_active", "magnetics")
"""The IDSs a per-slice solve input writes into."""

PLASMA_CURRENT_METHOD = "rogowski"
"""What produced the plasma-current channel, carried beside the samples."""

CONDUCTOR_CURRENT_UNIT = CONDUCTOR_UNIT
"""Source unit of a channel measuring one conductor rather than a turn product."""


@dataclass(frozen=True)
class OpenedDescription:
    """One opened machine description and what the map needs to read from it."""

    ids: Mapping[str, Any]
    machine: DescribedMachine
    dd_version: str
    artifact_digest: str


def open_description(
    cache_directory: Path | str,
    digest: str,
    *,
    drives: Any = None,
    allow_incomplete: bool = True,
) -> OpenedDescription:
    """Resolve, verify and open one content-addressed description.

    ``allow_incomplete`` defaults to accepting an incomplete artifact because the
    description carries unresolved fields by construction, and refusing to open it
    would refuse the served inputs along with the blocked ones.  The blocked rows
    are how the incompleteness reaches a consumer.
    """

    artifact = resolve_machine_artifact(
        cache_directory,
        digest,
        allow_incomplete=allow_incomplete,
    )
    return open_verified_description(artifact, drives=drives)


def open_verified_description(
    artifact: VerifiedMachineArtifact,
    *,
    drives: Any = None,
) -> OpenedDescription:
    """Open an already-verified artifact at the dictionary version it pins."""

    dd_version = pinned_dd_version(artifact)
    entry = imas.DBEntry(
        f"imas:hdf5?path={artifact.directory}", "r", dd_version=dd_version
    )
    try:
        ids = {name: entry.get(name) for name in DESCRIPTION_IDS}
    finally:
        entry.close()
    machine = describe_machine(
        ids,
        dd_version=dd_version,
        drives=artifact.manifest.drive_map if drives is None else drives,
    )
    return OpenedDescription(
        ids=ids,
        machine=machine,
        dd_version=dd_version,
        artifact_digest=artifact.digest,
    )


def build_solve_input_map(
    description: OpenedDescription,
    probe_families: Any,
    *,
    shot: int,
    store: Path | str | None = None,
) -> SourceSignalMap:
    """Return the map for one description, joining the loops through one shot.

    The loop join needs a measured loop position, and the only trustworthy source
    of one is the reconstruction's own static array, so the shot the join is taken
    from is named rather than defaulted.  The join is a property of the
    configuration and not of the shot; that it is the same for any shot of the
    range is a test, not an assumption made here.
    """

    positions = (
        reconstruction_loop_positions(shot)
        if store is None
        else reconstruction_loop_positions(shot, store=store)
    )
    targets = loop_target_indices(description.machine, positions)
    return solve_input_map(description.machine, probe_families, targets)


def preferred_signals(source_map: SourceSignalMap) -> tuple[SourceSignal, ...]:
    """Return one row per described target, preferring a direct measurement."""

    chosen: dict[tuple[str, int | None], SourceSignal] = {}
    for signal in source_map.signals:
        key = (signal.target_path, signal.target_index)
        held = chosen.get(key)
        if held is None:
            chosen[key] = signal
            continue
        if (
            held.source_unit != CONDUCTOR_CURRENT_UNIT
            and signal.source_unit == CONDUCTOR_CURRENT_UNIT
        ):
            chosen[key] = signal
    return tuple(sorted(chosen.values(), key=lambda row: row.sort_key))


def _column(dataset: Any, signal: SourceSignal) -> np.ndarray:
    """Return one channel's converted samples out of the tensorized dataset."""

    variable = signal.standard_name
    if variable not in dataset:
        variable = f"{PROVISIONAL_NAMESPACE}/{signal.standard_name}"
    array = dataset[variable]
    dimension = f"{signal.standard_name}_channel"
    names = [str(name) for name in array.coords[dimension].values]
    return np.asarray(
        array.isel({dimension: names.index(signal.source_channel)}).values, dtype=float
    )


def fill_solve_inputs(
    ids: Mapping[str, Any],
    signals: ShotSignals,
    source_map: SourceSignalMap,
) -> tuple[SourceSignal, ...]:
    """Write one shot's converted samples into the description's dynamic fields.

    Each signal carries the clock it was measured on rather than being resampled
    onto a shared one, which the dictionary's heterogeneous time base is there to
    allow.  The excitation and the diagnostics were acquired on different clocks at
    different rates, and choosing an interpolation is the consumer's decision.
    """

    written = preferred_signals(source_map)
    magnetics = ids["magnetics"]
    pf_active = ids["pf_active"]
    plasma_rows = [row for row in written if row.standard_name == PLASMA_CURRENT_NAME]
    if plasma_rows:
        magnetics.ip.resize(len(plasma_rows))
    for signal in written:
        clock = signals.clocks[signal.time_base]
        values = _column(signals.dataset, signal)
        if signal.standard_name == COIL_CURRENT_NAME:
            node = pf_active.coil[signal.target_index].current
        elif signal.standard_name == PROBE_FIELD_NAME:
            node = magnetics.b_field_pol_probe[signal.target_index].field
        elif signal.standard_name == LOOP_FLUX_NAME:
            node = magnetics.flux_loop[signal.target_index].flux
        elif signal.standard_name == PLASMA_CURRENT_NAME:
            node = magnetics.ip[signal.target_index]
            node.method_name = PLASMA_CURRENT_METHOD
        else:
            raise SolveInputError(
                f"no dictionary node is wired for {signal.standard_name!r}"
            )
        node.data = values
        node.time = clock
    for name in DYNAMIC_IDS:
        ids[name].validate()
    return written


def write_and_reopen(
    ids: Mapping[str, Any],
    path: Path | str,
    *,
    dd_version: str,
) -> dict[str, Any]:
    """Write the served description to a pulse and read it back at the same pin."""

    uri = f"imas:hdf5?path={Path(path)}"
    entry = imas.DBEntry(uri, "x", dd_version=dd_version)
    try:
        for name in DYNAMIC_IDS:
            entry.put(ids[name])
    finally:
        entry.close()
    reopened = imas.DBEntry(uri, "r", dd_version=dd_version)
    try:
        result = {name: reopened.get(name) for name in DYNAMIC_IDS}
    finally:
        reopened.close()
    for name, served in result.items():
        served.validate()
        written = str(served.ids_properties.version_put.data_dictionary)
        if written != dd_version:
            raise SolveInputError(
                f"reopened {name} carries dictionary version {written}, "
                f"expected {dd_version}"
            )
    return result


def read_back_residuals(
    reopened: Mapping[str, Any],
    signals: ShotSignals,
    written: tuple[SourceSignal, ...],
) -> dict[str, float]:
    """Return each channel's error after a write, reopen and inverse conversion."""

    magnetics = reopened["magnetics"]
    pf_active = reopened["pf_active"]
    residuals: dict[str, float] = {}
    for signal in written:
        if signal.standard_name == COIL_CURRENT_NAME:
            node = pf_active.coil[signal.target_index].current
        elif signal.standard_name == PROBE_FIELD_NAME:
            node = magnetics.b_field_pol_probe[signal.target_index].field
        elif signal.standard_name == LOOP_FLUX_NAME:
            node = magnetics.flux_loop[signal.target_index].flux
        else:
            node = magnetics.ip[signal.target_index]
        recovered = signal.invert(np.asarray(node.data, dtype=float))
        raw = signals.samples[signal.source_channel]
        scale = max(float(np.max(np.abs(raw))), 1.0e-30)
        residuals[signal.source_channel] = float(
            np.max(np.abs(recovered - raw)) / scale
        )
        clock = np.asarray(node.time, dtype=float)
        if not np.array_equal(clock, signals.clocks[signal.time_base]):
            raise SolveInputError(
                f"{signal.source_channel!r} came back on a different clock"
            )
    return residuals


def description_fingerprint(ids: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise the described geometry the solve inputs were loaded against.

    Compared before and after a load so a served pulse cannot silently carry a
    moved sensor or a changed turn count: the inputs are dynamic data and the
    description is not, and a load that altered one of them would be a defect no
    residual on the data itself would reveal.
    """

    magnetics = ids["magnetics"]
    pf_active = ids["pf_active"]
    return {
        "coil_names": [str(coil.name) for coil in pf_active.coil],
        "turns": [
            [float(element.turns_with_sign) for element in coil.element]
            for coil in pf_active.coil
        ],
        "probe_poses": [
            [
                float(probe.position.r),
                float(probe.position.z),
                float(probe.poloidal_angle),
            ]
            for probe in magnetics.b_field_pol_probe
        ],
        "loop_positions": [
            [float(point.r), float(point.z), float(point.phi)]
            for loop in magnetics.flux_loop
            for point in loop.position
        ],
    }


@dataclass(frozen=True)
class ShotRoundTrip:
    """What one shot's solve inputs did on the way to the description and back."""

    shot: int
    served_channels: tuple[str, ...]
    absent_channels: tuple[str, ...]
    misaligned_channels: tuple[str, ...]
    sample_counts: Mapping[str, int]
    dropped_samples: Mapping[str, int]
    dataset_residual: float
    read_back_residual: float
    redundancy: Mapping[str, float]
    polarity: Mapping[str, float]
    description_preserved: bool
    provisional_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON representation."""

        return {
            "absent_channels": list(self.absent_channels),
            "dataset_residual": float(self.dataset_residual),
            "description_preserved": bool(self.description_preserved),
            "dropped_samples": {
                k: int(v) for k, v in sorted(self.dropped_samples.items())
            },
            "misaligned_channels": list(self.misaligned_channels),
            "polarity": {k: float(v) for k, v in sorted(self.polarity.items())},
            "provisional_names": list(self.provisional_names),
            "read_back_residual": float(self.read_back_residual),
            "redundancy": {k: float(v) for k, v in sorted(self.redundancy.items())},
            "sample_counts": {k: int(v) for k, v in sorted(self.sample_counts.items())},
            "served_channels": list(self.served_channels),
            "shot": int(self.shot),
        }


def channel_redundancy(
    signals: ShotSignals,
    source_map: SourceSignalMap,
) -> dict[str, float]:
    """Return how far apart two channels of one conductor put its current.

    A coil publishing both a conductor current and an ampere-turn product gives two
    routes to one column, and the second is the first scaled by an integer the
    archive applied.  Their ratio therefore compares that integer against the turn
    count the description carries, so the disagreement here is a measurement of the
    described turn count and not a self-consistency check.
    """

    disagreement: dict[str, float] = {}
    for group in group_signals(source_map.signals):
        if group.standard_name != COIL_CURRENT_NAME:
            continue
        by_target: dict[int | None, list[SourceSignal]] = {}
        for signal in group.signals:
            by_target.setdefault(signal.target_index, []).append(signal)
        for rows in by_target.values():
            if len(rows) < 2:
                continue
            columns = [_column(signals.dataset, row) for row in rows]
            scale = max(float(np.max(np.abs(columns[0]))), 1.0e-30)
            spread = max(
                float(np.max(np.abs(columns[index] - columns[0])))
                for index in range(1, len(columns))
            )
            disagreement[rows[0].target_path] = spread / scale
    return disagreement


def round_trip_shot(
    description: OpenedDescription,
    source_map: SourceSignalMap,
    shot: int,
    *,
    work_directory: Path | str,
    store: Path | str | None = None,
    resolver: Any = None,
) -> ShotRoundTrip:
    """Serve one shot through the description and report what survived."""

    root = SHOT_STORE if store is None else store
    signals = read_solve_inputs(source_map, shot, store=root, resolver=resolver)
    selected = signals.source_map
    dataset_residuals = round_trip_residual(selected, signals.dataset, signals.samples)
    before = description_fingerprint(description.ids)
    written = fill_solve_inputs(description.ids, signals, selected)
    reopened = write_and_reopen(
        description.ids,
        Path(work_directory) / f"solve_inputs_{shot}",
        dd_version=description.dd_version,
    )
    after = description_fingerprint(description.ids)
    residuals = read_back_residuals(reopened, signals, written)
    return ShotRoundTrip(
        shot=int(shot),
        served_channels=tuple(row.source_channel for row in written),
        absent_channels=signals.absent_channels,
        misaligned_channels=signals.misaligned_channels,
        sample_counts=signals.sample_counts,
        dropped_samples=signals.dropped_samples,
        dataset_residual=max(dataset_residuals.values()),
        read_back_residual=max(residuals.values()),
        redundancy=channel_redundancy(signals, selected),
        polarity=field_polarity(shot, store=root),
        description_preserved=before == after,
        provisional_names=tuple(signals.dataset.attrs["provisional_names"]),
    )
