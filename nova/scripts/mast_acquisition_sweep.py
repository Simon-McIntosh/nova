"""Pin the shots at which a probe channel's acquisition range setting changed.

The settings themselves were measured on the calibration cohort's training split, and
that is enough to say which channels stepped and by what factor.  It is not enough to
say where.  Only a few dozen of that split's shots give any one channel a far-field
reading, and they arrive in clusters, so a switch that happened between two clusters
is known only to lie somewhere in a gap five thousand shots wide.  A correction with
a bracket that wide can be applied to the cohort and to almost nothing else.

Four passes narrow it, and they differ in what each is allowed to conclude.

``sweep`` reads every plasma-free shot the cohort classifier found an excitation in,
not only the ones the split trained on.  A plasma-free shot is one the response model
predicts completely, so the ratio it returns is measurement-grade and this pass is
what the promoted settings rest on -- but only over the binding training split, and
the pass reports why.  Measured on that split this sweep reproduces the adjudicated
result exactly: the same nineteen stepping channels, thirty-seven steps, twenty-nine
of them on the declared ladder.  Re-running the block finder over the held-out and
unsplit readings as well returns fifty-five stepping channels and drops the ladder
share to 57 percent, because the block finder splits on their scatter and a boundary
found that way is indistinguishable from a switch once it is in the table.  So those
readings never set a value; they challenge the table and they place boundaries.

``array`` measures a second observable that needs no description -- each channel's
amplitude relative to the array median, which no drive, coil or plasma can move
because it is in the numerator and denominator alike -- and checks it against the
fitted route on the shots both read.  That check is a gate, not a result, and on this
archive **it refuses the observable**: over the binding split the array route places
ten to eighteen boundaries per channel on all thirty-nine channels the fitted route
measures steady, and matches only 41 of 75 real steps.  The field-pattern confound the
observable was declared to carry is what dominates -- a different coil lights up a
different part of the array, and between excitation classes that moves a channel's
ratio by more than the 1.41 a step must reach.  The passes are kept because the
refusal is the finding: without the gate this route reports a median bracket of 24
shots where the fitted route leaves 4942, and that 200-fold narrowing is an artifact.

``pin`` spends reads inside whatever brackets remain, using that model-free
observable, so it can read plasma shots the fitted route refuses.  It bisects the
widest bracket still open, converging in reads growing with the logarithm of a
bracket's width, and one read serves every channel at once because every channel's
setting is recorded on the same shot.  It may move a boundary and may never set a
value -- and while the gate refuses it, it does not move one either: what it would
have placed is recorded beside the gate that rejected it.

``promote`` assembles the table the read path applies: it refers each block to the
block reading nearest the described field, snaps each step to the declared ladder, and
writes the pinning summary beside it so a reader sees which switches the archive pins
and which it cannot.

Every pass reads the archive as published.  Passing the promoted table into the very
measurement that produced it would divide out the setting and then measure that it was
gone, so the raw path is named explicitly here rather than defaulted.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from nova.calibrate.correction_model import (
    ChannelCorrection,
    CorrectionKind,
    CorrectionSet,
    CorrectionStatus,
    Provenance,
    ValidityInterval,
)
from nova.calibrate.correction_set import replace_correction_kind
from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_acquisition_scale import (
    acquisition_record,
    channel_histories,
    step_concurrency,
    stepping_channels,
)
from nova.imas.mast_array_amplitude import (
    agreement,
    agreement_summary,
    channel_amplitudes,
    narrow_bracket,
    narrowing_summary,
)
from nova.imas.mast_block_scale import (
    PROMOTED_ROUTE,
    BlockScaleTable,
    bracket_probe,
    pinning_summary,
)
from nova.imas.mast_channel_drive import channel_drives
from nova.imas.mast_error_field_screen import (
    ChannelCoupling,
    ErrorFieldScreen,
    read_error_field_drive,
)
from nova.imas.mast_fitted_parameters import MIS_SCALED_SHOTS
from nova.imas.mast_probe_calibration import shot_gains, standoff_table
from nova.imas.mast_vacuum_cohort import RAW_ARCHIVE, probe_channels
from nova.imas.mast_vacuum_response import MINIMUM_STANDOFF, ResponseModel

CACHE = Path.home() / ".cache" / "nova-mast"
"""Where the cohort's own records live and this one is written beside them."""

MAST_CORRECTION_DOCUMENT = (
    Path(__file__).parents[1] / "calibrate" / "corrections" / "mast" / "magnetics.yaml"
)
"""The versioned correction document the measured acquisition record enters."""

PROMOTION_RECORD = CACHE / "mast_block_scale_sweep.json"
"""Evidence record written by the promotion pass before the document is updated."""

PROMOTION_PRODUCER = "nova.scripts.mast_acquisition_sweep"
"""Stable producer stamped on the document and every promoted correction."""

REGISTRY_SHOT = 11766
"""The shot the machine geometry selection is taken at, as the cohort took it."""

STRIDE = 4
"""Sample stride, matching the fit the settings were first measured with."""

EXCITATION_CLASSES = (
    "sustained_single_coil",
    "sustained_symmetric_pair",
    "sustained_coil_group",
    "pulsed_excitation",
)
"""Cohort classes carrying poloidal excitation, so a ratio can be measured at all.

The toroidal-field-only and quiescent classes drive no poloidal coil, so the
described field is zero and the ratio is undefined rather than large.
"""


def log(message: str) -> None:
    """Report progress on a stream the batch log keeps in order."""

    print(message, flush=True)


def load_screen() -> ErrorFieldScreen:
    """Rebuild the error-field screen the sensor adjudication measured.

    Read back rather than re-measured so the sweep applies the same thresholds the
    settings were first measured under, and so a channel's refusal on a shot does not
    change because this pass happened to run later.
    """

    record = json.loads((CACHE / "mast_error_field_screen.json").read_text())
    return ErrorFieldScreen(
        couplings=tuple(
            ChannelCoupling(
                channel=str(row["channel"]),
                driver=str(row["driver"]),
                shot_count=int(row["shot_count"]),
                response=float(row["response"]),
                scatter=float(row["scatter"]),
                noise_floor=float(row["noise_floor"]),
                neighbour_response=float(row["neighbour_response"]),
            )
            for row in record["screen"]["couplings"]
        )
    )


def build_model() -> tuple[ResponseModel, dict[str, float]]:
    """Return the response model and the promoted drive weights the fits read."""

    selection = MachineGeometryRegistry.default().select(REGISTRY_SHOT)
    geometry = selection.configuration.geometry
    probes = geometry["magnetics"]["poloidal_probes"]
    model = ResponseModel.build(geometry, probes, probe_channels(probes))
    weights = {
        row.conductor: float(row.ampere_turns_per_ampere)
        for row in channel_drives(geometry).drives
        if row.channel.endswith(
            ("feed_current", "sol_current", "p6l_current", "p6u_current")
        )
    }
    for family in model.families:
        weights.setdefault(family, 1.0)
    return model, weights


def far_field_pairs(model: ResponseModel) -> set[tuple[str, str]]:
    """Return the channel and family pairs standing clear of the excited coil."""

    return {
        key
        for key, standoff in standoff_table(model).items()
        if standoff >= MINIMUM_STANDOFF
    }


def measure_shot(
    shot: int,
    model: ResponseModel,
    weights: Mapping[str, float],
    screen: ErrorFieldScreen,
    far: set[tuple[str, str]],
) -> dict[str, list[float]]:
    """Return each channel's far-field ratios on one shot, one per coil family.

    Every family's ratio is kept rather than pooled here, because how many families a
    reading rests on turns out to decide whether the reading is usable at all: a
    single-family ratio on a shot that lit several coils is one projection carrying
    that coil's own model error, and admitting those chops a channel's history into
    blocks on scatter.  Keeping the list lets the admission rule be chosen against the
    measurement instead of the archive being read again for every candidate rule.

    The rigid scale-and-rotation fit is not asked for: a range setting is an amplitude
    and the pair would double the cost of a pass whose whole purpose is breadth.  The
    error-field screen and the amplitude refusal apply exactly as they do in the fit
    the settings came from, so a shot excluded there stays excluded here.
    """

    if shot in MIS_SCALED_SHOTS:
        return {}
    try:
        drive = read_error_field_drive(shot)
    except Exception as error:  # noqa: BLE001
        log(f"  {shot}: error-field read failed ({error})")
        return {}
    refused = screen.refused(drive)
    try:
        waveforms = RAW_ARCHIVE.read_shot_waveforms(shot)
    except Exception as error:  # noqa: BLE001
        log(f"  {shot}: unreadable ({error})")
        return {}
    gains, _, _ = shot_gains(
        model, waveforms, weights, refused_channels=refused, stride=STRIDE
    )
    rows: dict[str, list[float]] = collections.defaultdict(list)
    for row in gains:
        if (row.channel, row.family) in far:
            rows[row.channel].append(row.gain)
    return dict(rows)


def _cohort() -> dict[str, Any]:
    return json.loads((CACHE / "mast_calibration_cohort.json").read_text())


def excitation_shots() -> list[int]:
    """Return every plasma-free shot the classifier found an excitation in."""

    by_class = _cohort()["cohort"]["by_class"]
    shots: set[int] = set()
    for name in EXCITATION_CLASSES:
        shots |= {int(shot) for shot in by_class.get(name, ())}
    return sorted(shots)


def plasma_shots() -> list[int]:
    """Return the plasma shots whose pre-breakdown window a boundary can use."""

    return sorted(int(shot) for shot in _cohort()["cohort"]["by_class"]["plasma"])


def training_shots() -> set[int]:
    """Return the binding training split, which is what may set a value.

    The classifier declared this split before any of these settings were fitted and it
    is binding on every later fit, so the block structure and the settings themselves
    are measured on it and nothing else.  That is not a quality judgement about the
    other shots: measured on the training split alone this sweep reproduces the
    adjudicated result exactly -- the same nineteen stepping channels, thirty-seven
    steps, twenty-nine on the ladder -- while pooling the held-out and unsplit shots
    into the same block finder returns fifty-five stepping channels and sends the share
    of steps landing on the declared ladder from 78 percent to 57.  Those extra
    readings are real; re-running the block finder over them is what is not allowed,
    because a boundary found by splitting on scatter is indistinguishable from a switch
    once it is in the table.
    """

    return {int(shot) for shot in _cohort()["cohort"]["training"]} - set(
        MIS_SCALED_SHOTS
    )


def held_out_shots() -> set[int]:
    """Return the shots the table is challenged on rather than measured from."""

    return {int(shot) for shot in _cohort()["cohort"]["held_out"]} - set(
        MIS_SCALED_SHOTS
    )


def held_out_check(
    table: BlockScaleTable,
    series: Mapping[str, Mapping[int, Sequence[float]]],
    shots: Iterable[int],
) -> dict[str, Any]:
    """Test the table on readings it was not measured from.

    A held-out reading lands inside some block's span or between two blocks.  Where it
    lands inside one, the block asserts a setting for it, and dividing the reading by
    that block's rung has to bring it near the channel's reference level -- if it does
    not, the block is wrong about shots it was not fitted on.  Where it lands in a
    bracket the table refuses to assert anything, and those are counted separately
    rather than scored, because a refusal cannot be wrong.
    """

    import numpy as np

    wanted = {int(shot) for shot in shots}
    agreed = missed = refused = 0
    worst: list[tuple[float, str, int]] = []
    for channel, rows in sorted(series.items()):
        blocks = table.blocks.get(channel, ())
        if not blocks:
            continue
        reference = next(
            (block.scale for block in blocks if block.unchanged), blocks[0].scale
        )
        for shot, values in sorted(rows.items()):
            if shot not in wanted:
                continue
            correction = table.correction(channel, shot)
            if not correction.applied:
                refused += 1
                continue
            reading = float(np.median(values)) / correction.scale
            gap = abs(reading / reference - 1.0) if reference else math.inf
            if gap <= 0.25:
                agreed += 1
            else:
                missed += 1
                worst.append((gap, channel, shot))
    worst.sort(reverse=True)
    scored = agreed + missed
    return {
        "agreed": agreed,
        "missed": missed,
        "refused": refused,
        "share_agreeing": agreed / scored if scored else math.nan,
        "worst": [
            {"channel": channel, "gap": gap, "shot": shot}
            for gap, channel, shot in worst[:12]
        ],
    }


def _series_path(name: str) -> Path:
    return CACHE / f"mast_acquisition_{name}.json"


def _load_series(name: str) -> dict[str, dict[int, list[float]]]:
    path = _series_path(name)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        channel: {int(shot): list(values) for shot, values in rows.items()}
        for channel, rows in raw["series"].items()
    }


def _write_series(
    name: str, series: Mapping[str, Mapping[int, Sequence[float]]]
) -> None:
    _series_path(name).write_text(
        json.dumps(
            {
                "route": name,
                "series": {
                    channel: {
                        str(shot): [float(value) for value in values]
                        for shot, values in sorted(rows.items())
                    }
                    for channel, rows in sorted(series.items())
                },
            },
            indent=1,
            sort_keys=True,
        )
    )


def _merge(
    series: dict[str, dict[int, list[float]]],
    shot: int,
    ratios: Mapping[str, Sequence[float] | float],
) -> None:
    for channel, value in ratios.items():
        values = [float(value)] if isinstance(value, (int, float)) else list(value)
        series.setdefault(channel, {})[int(shot)] = [float(item) for item in values]


def _every_shot(series: Mapping[str, Mapping[int, Sequence[float]]]) -> set[int]:
    return {int(shot) for rows in series.values() for shot in rows}


def shot_set_rule(
    series: Mapping[str, Mapping[int, Sequence[float]]],
    label: str,
    shots: set[int],
) -> dict[str, Any]:
    """Report the block structure one shot set produces.

    Coverage and coherence pull opposite ways, and the ladder breaks the tie: a set
    that admits more readings but sends the share of steps landing on the declared
    ladder down has bought coverage by turning scatter into blocks.  The ladder was
    declared before any of these settings were classified, so it is evidence about the
    shot set rather than a target the set can be tuned onto.
    """

    rows = {
        channel: {shot: list(values) for shot, values in inner.items() if shot in shots}
        for channel, inner in series.items()
    }
    rows = {channel: inner for channel, inner in rows.items() if inner}
    histories = channel_histories(rows)
    stepping = stepping_channels(histories)
    steps = [step for row in stepping for step in row.steps]
    counts = sorted(len(inner) for inner in rows.values()) or [0]
    return {
        "channels": len(rows),
        "label": label,
        "median_shots": counts[len(counts) // 2],
        "on_ladder": sum(1 for step in steps if step.on_ladder),
        "readings": sum(len(inner) for inner in rows.values()),
        "steps": len(steps),
        "stepping": len(stepping),
        "switches": len(steps),
    }


def run_sweep(arguments: argparse.Namespace) -> None:
    """Measure the far-field ratio of every channel on every plasma-free shot."""

    model, weights = build_model()
    screen = load_screen()
    far = far_field_pairs(model)
    shots = excitation_shots()
    if arguments.limit:
        shots = shots[: arguments.limit]
    log(f"plasma-free excitation shots to read: {len(shots)}")
    series = _load_series("plasma_free") if arguments.resume else {}
    done = {shot for rows in series.values() for shot in rows}
    measured = 0
    for index, shot in enumerate(shots):
        if shot in done:
            continue
        ratios = measure_shot(shot, model, weights, screen, far)
        if ratios:
            measured += 1
        _merge(series, shot, ratios)
        if index % 25 == 0:
            log(f"  {index}/{len(shots)} shot {shot} channels {len(ratios)}")
    _write_series("plasma_free", series)
    log(f"shots yielding a reading: {measured}/{len(shots)}")
    log(f"channels with a series: {len(series)}")
    counts = sorted(len(rows) for rows in series.values())
    if counts:
        middle = counts[len(counts) // 2]
        log(f"shots per channel: min {counts[0]} median {middle} max {counts[-1]}")


def _table_from(
    series: Mapping[str, Mapping[int, Sequence[float]]], route: str
) -> BlockScaleTable:
    histories = channel_histories(series)
    return BlockScaleTable.from_histories(
        histories,
        {channel: sorted(rows) for channel, rows in series.items()},
        route=route,
    )


def acquisition_corrections(
    table: BlockScaleTable,
    *,
    evidence_uri: Path | str,
    fitted_at: date,
) -> list[ChannelCorrection]:
    """Translate every measured acquisition block into a schema correction."""

    evidence = str(evidence_uri)
    rows = []
    for channel in table.channels:
        for block in table.blocks[channel]:
            promoted = block.on_ladder
            rows.append(
                ChannelCorrection(
                    channel=block.channel,
                    kind=CorrectionKind.acquisition_scale,
                    status=(
                        CorrectionStatus.promoted
                        if promoted
                        else CorrectionStatus.refused
                    ),
                    value=float(block.rung) if promoted else None,
                    measured_value=float(block.scale),
                    ladder="acquisition_range",
                    validity=[
                        ValidityInterval(
                            pulse_start=block.first_shot,
                            pulse_end=block.last_shot,
                            measured_pulses=list(block.shots),
                        )
                    ],
                    provenance=Provenance(
                        method=block.route or table.route or PROMOTED_ROUTE,
                        evidence_uri=evidence,
                        fitted_at=fitted_at,
                        fitted_by=PROMOTION_PRODUCER,
                        statement=(
                            f"measured response ratio {block.scale:.16g}; "
                            + (
                                f"promoted acquisition rung {block.rung:.16g}"
                                if promoted
                                else "refused because it misses the acquisition ladder"
                            )
                        ),
                    ),
                )
            )
    return rows


def promote_acquisition_corrections(
    table: BlockScaleTable,
    path: Path | str = MAST_CORRECTION_DOCUMENT,
    *,
    evidence_uri: Path | str = PROMOTION_RECORD,
    fitted_at: date | None = None,
) -> CorrectionSet:
    """Write a measured table into the versioned MAST correction document."""

    stamp = fitted_at or date.today()
    rows = acquisition_corrections(table, evidence_uri=evidence_uri, fitted_at=stamp)
    return replace_correction_kind(
        path,
        CorrectionKind.acquisition_scale,
        rows,
        generated_by=PROMOTION_PRODUCER,
        generated_at=stamp,
    )


def array_shot(shot: int, screen: ErrorFieldScreen) -> dict[str, float]:
    """Return each channel's amplitude relative to the array on one shot.

    No response model and no drive weights enter, so this reads on shots the fitted
    route refuses -- a plasma shot's whole record included.  The error-field screen and
    the amplitude refusal still apply: a channel the screen removes is removed from the
    array too, because leaving it in would let the excitation it couples to move the
    reference.
    """

    if shot in MIS_SCALED_SHOTS:
        return {}
    try:
        drive = read_error_field_drive(shot)
        refused = set(screen.refused(drive))
        waveforms = RAW_ARCHIVE.read_shot_waveforms(shot)
    except Exception as error:  # noqa: BLE001
        log(f"  {shot}: unreadable ({error})")
        return {}
    probes = {
        channel: values
        for channel, values in waveforms.probes.items()
        if channel not in refused and values.shape == waveforms.time.shape
    }
    return channel_amplitudes(probes, baseline=waveforms.baseline_mask)


def run_array(arguments: argparse.Namespace) -> None:
    """Measure the model-free amplitude on the shots the fitted route also read.

    This is the gate rather than a result: the two routes have to tell the same story
    about which channels stepped and where, on the shots they share, before the
    model-free one is allowed to place a boundary where nothing else can look.
    """

    screen = load_screen()
    shots = excitation_shots()
    if arguments.limit:
        shots = shots[: arguments.limit]
    series = _load_series("array_plasma_free") if arguments.resume else {}
    done = {shot for rows in series.values() for shot in rows}
    for index, shot in enumerate(shots):
        if shot in done:
            continue
        _merge(series, shot, array_shot(shot, screen))
        if index % 50 == 0:
            log(f"  {index}/{len(shots)} shot {shot}")
    _write_series("array_plasma_free", series)

    fitted = _load_series("plasma_free")
    if not fitted:
        raise SystemExit("run the sweep pass before checking the model-free route")
    flat = {
        channel: {shot: rows[shot][0] for shot in rows}
        for channel, rows in series.items()
    }
    rows = agreement(fitted, flat)
    summary = agreement_summary(rows)
    log(f"channels compared: {summary['channels']}")
    log(
        f"stepping channels reproduced: {summary['stepping_reproduced']}"
        f"/{summary['stepping_channels']}  steps matched "
        f"{summary['matched_steps']}/{summary['total_steps']}"
    )
    log(
        f"steady channels reproduced: {summary['steady_reproduced']}"
        f"/{summary['steady_channels']}  boundaries invented "
        f"{summary['invented_on_steady']}"
    )
    (CACHE / "mast_array_amplitude_agreement.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True)
    )
    log("wrote mast_array_amplitude_agreement.json")


def run_pin(arguments: argparse.Namespace) -> None:
    """Bisect the brackets left open, using the model-free amplitude on any shot."""

    screen = load_screen()
    base = _load_series("plasma_free")
    if not base:
        raise SystemExit("run the sweep pass before pinning its boundaries")
    extra = _load_series("array_pinning") if arguments.resume else {}
    candidates = sorted(set(plasma_shots()) | set(excitation_shots()))
    attempted: set[int] = {shot for rows in extra.values() for shot in rows}
    log(f"candidate shots: {len(candidates)}  already read: {len(attempted)}")

    brackets = _table_from(base, PROMOTED_ROUTE).brackets()
    for step in range(arguments.reads):
        target = bracket_probe(brackets, candidates, measured=attempted)
        if target is None:
            log("every bracket is pinned as tightly as the archive allows")
            break
        window = [
            shot
            for shot in candidates
            if abs(shot - target) <= arguments.reach and shot not in attempted
        ]
        window.sort(key=lambda shot: (abs(shot - target), shot))
        for shot in window[: arguments.attempts]:
            attempted.add(shot)
            ratios = array_shot(shot, screen)
            if ratios:
                _merge(extra, shot, ratios)
                log(f"  read {step}: shot {shot} gave {len(ratios)} channels")
                break
        else:
            log(f"  read {step}: no readable shot near {target}")
        if step % 10 == 0:
            _write_series("array_pinning", extra)
    _write_series("array_pinning", extra)
    log(f"model-free readings for pinning: {sum(len(rows) for rows in extra.values())}")


def run_promote(arguments: argparse.Namespace) -> None:
    """Assemble the table the read path applies, and report what pins it."""

    measured = _load_series("plasma_free")
    if not measured:
        raise SystemExit("run the sweep pass before promoting a table")
    training = training_shots()
    rules = [
        shot_set_rule(measured, "every plasma-free shot read", _every_shot(measured)),
        shot_set_rule(measured, "the binding training split", training),
    ]
    for report in rules:
        log(
            f"  {report['label']:32s}: {report['channels']:2d} channels, "
            f"{report['readings']:5d} readings, median {report['median_shots']:3.0f} "
            f"shots/channel, {report['stepping']:2d} stepping, "
            f"{report['switches']:3d} switches, "
            f"{report['on_ladder']}/{report['steps']} on the ladder"
        )
    combined = {
        channel: {
            shot: list(values) for shot, values in rows.items() if shot in training
        }
        for channel, rows in measured.items()
    }
    combined = {channel: rows for channel, rows in combined.items() if rows}
    log(f"settings measured on the binding training split ({len(training)} shots)")
    histories = channel_histories(combined)
    table = _table_from(combined, PROMOTED_ROUTE)
    stepping = stepping_channels(histories)
    steps = [step for row in stepping for step in row.steps]
    concurrency = step_concurrency(combined, steps)
    archive = sorted(set(excitation_shots()) | set(plasma_shots()))
    summary = pinning_summary(table, archive)

    log(f"channels measured: {len(table.channels)}")
    log(f"stepping channels: {len(table.stepping)} {list(table.stepping)}")
    log(f"channels a read corrects: {len(table.corrected)} {list(table.corrected)}")
    log(f"switch brackets: {summary['switch_count']}  pinned: {summary['pinned']}")
    log(
        f"bracket width: median {summary['median_width']:.0f} "
        f"widest {summary['widest_width']}"
    )
    refused = [
        (block.channel, block.scale)
        for channel in table.channels
        for block in table.blocks[channel]
        if not block.on_ladder
    ]
    log(f"blocks refused as off ladder: {len(refused)} {refused}")

    challenge = held_out_check(table, measured, held_out_shots())
    log(
        f"held-out readings agreeing with the block they fall in: "
        f"{challenge['agreed']}/{challenge['agreed'] + challenge['missed']} "
        f"({100 * challenge['share_agreeing']:.0f}%), {challenge['refused']} refused"
    )

    record = acquisition_record(histories, concurrency)
    record["held_out_challenge"] = challenge
    record["shot_set_rules"] = rules
    record["training_shots"] = sorted(training)
    record["pinning"] = summary
    record["table"] = table.as_dict()

    array_series = _load_series("array_plasma_free")
    pinning = _load_series("array_pinning")
    if array_series or pinning:
        flat: dict[str, dict[int, float]] = {}
        for source in (array_series, pinning):
            for channel, rows in source.items():
                flat.setdefault(channel, {}).update(
                    {shot: values[0] for shot, values in rows.items()}
                )
        gate = agreement_summary(
            agreement(
                combined,
                {
                    channel: {
                        shot: value for shot, value in rows.items() if shot in training
                    }
                    for channel, rows in flat.items()
                },
            )
        )
        passes = (
            gate["invented_on_steady"] == 0
            and gate["stepping_reproduced"] == gate["stepping_channels"]
        )
        log(
            f"model-free gate on the training split: "
            f"{gate['stepping_reproduced']}/{gate['stepping_channels']} stepping "
            f"channels reproduced, {gate['matched_steps']}/{gate['total_steps']} steps "
            f"matched, boundaries invented on {gate['invented_on_steady']}"
            f"/{gate['steady_channels']} steady channels -> "
            f"{'PASS' if passes else 'REFUSED'}"
        )
        record["model_free_gate"] = {**gate, "passes": passes}
        narrowed = [
            row
            for bracket in table.brackets()
            if (
                row := narrow_bracket(
                    bracket.channel,
                    bracket.before_shot,
                    bracket.after_shot,
                    bracket.ratio,
                    flat.get(bracket.channel, {}),
                )
            )
            is not None
        ]
        report = narrowing_summary(narrowed)
        report["admissible"] = passes
        log(
            f"boundaries the model-free route would place: {report['placed']}"
            f"/{summary['switch_count']}  narrowing {report['narrowed']}"
        )
        log(
            f"width they would leave: median {report['median_width']:.0f} "
            f"widest {report['widest_width']} (fitted median "
            f"{report['median_fitted_width']:.0f})"
        )
        if not passes:
            log(
                "the gate refused the route, so none of that narrowing is adopted: an "
                "instrument that places boundaries on channels measured steady is "
                "reading its own confound, and a narrower bracket built from it would "
                "be a worse statement than the wide one it replaced"
            )
        record["narrowing"] = report

    PROMOTION_RECORD.write_text(json.dumps(record, indent=1, sort_keys=True))
    log("wrote mast_block_scale_sweep.json")
    if arguments.promote_to:
        document = promote_acquisition_corrections(
            table,
            arguments.promote_to,
            evidence_uri=PROMOTION_RECORD,
        )
        log(
            f"promoted {sum(len(rows) for rows in table.blocks.values())} acquisition "
            f"records into {arguments.promote_to} as set {document.set_version}"
        )


RUNG_COLOURS = {
    2.0: "#b04a4a",
    1.4142135623730951: "#d08a3a",
    1.0: "#1a6b3c",
    0.7071067811865476: "#3a7ab0",
    0.5: "#4a4ab0",
}
"""One colour per ladder rung, so a block map reads as a setting rather than a level."""


def _rung_colour(rung: float | None) -> str:
    """Return the colour of the rung a block sits at, grey where it has none."""

    if rung is None:
        return "#999999"
    nearest = min(RUNG_COLOURS, key=lambda value: abs(value - rung))
    return RUNG_COLOURS[nearest]


def draw_figures(arguments: argparse.Namespace) -> None:
    """Draw the block map, the admission-rule trade-off and the pinning it bought."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    record = json.loads(arguments.record.read_text())
    table = record["table"]
    directory = arguments.figures
    directory.mkdir(parents=True, exist_ok=True)

    blocks = [row for row in table["blocks"]]
    corrected = sorted(
        {
            row["channel"]
            for row in blocks
            if row["rung"] is not None and row["rung"] != 1.0
        }
    )
    rows = [row for row in blocks if row["channel"] in corrected]
    order = {channel: index for index, channel in enumerate(corrected)}

    figure, axis = plt.subplots(figsize=(11.0, max(4.0, 0.26 * len(corrected))))
    for row in rows:
        height = order[row["channel"]]
        axis.plot(
            [row["first_shot"], row["last_shot"]],
            [height, height],
            color=_rung_colour(row["rung"]),
            lw=4.0,
            solid_capstyle="butt",
        )
        axis.plot(
            row["shots"],
            np.full(len(row["shots"]), height),
            "|",
            color="#222",
            ms=4,
            mew=0.6,
        )
    for bracket in record["pinning"]["brackets"]:
        if bracket["channel"] not in order:
            continue
        height = order[bracket["channel"]]
        axis.plot(
            [bracket["before_shot"], bracket["after_shot"]],
            [height, height],
            color="#bbbbbb",
            lw=1.0,
            ls=":",
        )
    axis.set_yticks(range(len(corrected)))
    axis.set_yticklabels(corrected, fontsize=6)
    axis.set_xlabel("shot")
    axis.set_title(
        "Where each channel held which range setting: bars are blocks coloured by "
        "rung,\nticks are the shots that measured them, dots are the switch brackets"
    )
    handles = [
        plt.Line2D([], [], color=colour, lw=4.0, label=f"x{rung:.3g}")
        for rung, colour in sorted(RUNG_COLOURS.items())
    ]
    handles.append(plt.Line2D([], [], color="#999999", lw=4.0, label="off ladder"))
    axis.legend(handles=handles, fontsize=7, ncol=6, loc="upper center")
    figure.tight_layout()
    path = directory / "scale_block_map.svg"
    figure.savefig(path)
    plt.close(figure)
    log(f"wrote {path}")

    rules = record.get("shot_set_rules", [])
    if rules:
        figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
        labels = [row["label"].replace(" ", "\n", 1) for row in rules]
        places = np.arange(len(rules))
        axes[0].bar(
            places, [row["readings"] for row in rules], color="#3a7ab0", width=0.55
        )
        axes[0].set_ylabel("channel-shot readings")
        axes[0].set_title("Coverage")
        axes[1].bar(
            places, [row["stepping"] for row in rules], color="#b04a4a", width=0.55
        )
        axes[1].set_ylabel("channels called stepping")
        axes[1].set_title("Blocks the finder returns")
        share = [
            row["on_ladder"] / row["steps"] if row["steps"] else np.nan for row in rules
        ]
        axes[2].bar(places, share, color="#1a6b3c", width=0.55)
        axes[2].set_ylim(0.0, 1.02)
        axes[2].set_ylabel("share of steps on the declared ladder")
        axes[2].set_title("Coherence")
        for axis in axes:
            axis.set_xticks(places)
            axis.set_xticklabels(labels, fontsize=7)
        for place, value, count in zip(places, share, [row["steps"] for row in rules]):
            axes[2].annotate(
                f"{count} steps",
                (place, value),
                textcoords="offset points",
                xytext=(0, 5),
                fontsize=7,
                ha="center",
            )
        figure.suptitle(
            "Why the settings are measured on the binding split alone: the wider "
            "set triples the channels called stepping\nand sends the share of steps "
            "landing on the ladder -- declared before any of this was classified -- "
            "from 78% to 57%"
        )
        figure.tight_layout()
        path = directory / "scale_shot_set.svg"
        figure.savefig(path)
        plt.close(figure)
        log(f"wrote {path}")

    falsification = arguments.falsification
    if falsification.exists():
        verdict = json.loads(falsification.read_text())
        rows = sorted(verdict["reversal"].items())
        figure, axis = plt.subplots(figsize=(8.0, 4.2))
        places = np.arange(len(rows))
        axis.bar(
            places - 0.18,
            [-1.0 if row["raw"] else 1.0 for _, row in rows],
            width=0.34,
            color="#b04a4a",
            label="as published",
        )
        axis.bar(
            places + 0.18,
            [-1.0 if row["normalised"] else 1.0 for _, row in rows],
            width=0.34,
            color="#1a6b3c",
            label="range setting divided out",
        )
        axis.axhline(0.0, color="#333", lw=0.8)
        axis.set_xticks(places)
        axis.set_xticklabels([name for name, _ in rows])
        axis.set_yticks([-1.0, 1.0])
        axis.set_yticklabels(
            ["train and held-out\ndisagree in sign", "train and held-out\nagree"]
        )
        axis.legend(fontsize=8, loc="center right")
        axis.set_title(
            "The pre-registered falsification: the displacement a coil's near probes "
            "want\nreversed between train and held-out on three coils, and on two of "
            "them it stops"
        )
        figure.tight_layout()
        path = directory / "scale_falsification.svg"
        figure.savefig(path)
        plt.close(figure)
        log(f"wrote {path}")

    narrowing = record.get("narrowing")
    if narrowing and narrowing["brackets"]:
        figure, axis = plt.subplots(figsize=(9.0, 4.4))
        fitted = np.asarray(
            [row["fitted_width"] for row in narrowing["brackets"]], dtype=float
        )
        placed = np.asarray(
            [row["width"] for row in narrowing["brackets"]], dtype=float
        )
        keep = fitted > 0
        axis.loglog(
            fitted[keep],
            np.clip(placed[keep], 1.0, None),
            "o",
            color="#1a6b3c",
            ms=5,
            alpha=0.8,
        )
        limit = [1.0, max(fitted.max(), 1.0)]
        axis.plot(limit, limit, color="#999", ls="--", lw=1.0, label="no narrowing")
        axis.set_xlabel("bracket width the fitted route left [shots]")
        axis.set_ylabel("width the model-free route would leave [shots]")
        gate = record.get("model_free_gate", {})
        admissible = bool(narrowing.get("admissible"))
        axis.set_title(
            "What the model-free observable would have bought, and did not: each point "
            "is one switch"
            + (
                ""
                if admissible
                else "\nREFUSED by its own gate -- it places boundaries on "
                f"{gate.get('invented_on_steady', '?')} of "
                f"{gate.get('steady_channels', '?')} channels measured steady, so "
                "it is reading its field-pattern confound"
            )
        )
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = directory / "scale_bracket_pinning.svg"
        figure.savefig(path)
        plt.close(figure)
        log(f"wrote {path}")

    if arguments.gains.exists():
        ledger = json.loads(arguments.gains.read_text())["calibration"]["gains"]
        base = json.loads(arguments.raw_gains.read_text())["calibration"]["gains"]
        shared = sorted(set(ledger) & set(base))
        figure, axis = plt.subplots(figsize=(11.0, 4.4))
        places = np.arange(len(shared))
        axis.plot(
            places,
            [base[channel] for channel in shared],
            "o",
            color="#b04a4a",
            ms=4,
            label="as published",
        )
        axis.plot(
            places,
            [ledger[channel] for channel in shared],
            "o",
            color="#1a6b3c",
            ms=4,
            label="range setting divided out",
        )
        for place, channel in zip(places, shared):
            axis.plot(
                [place, place],
                [base[channel], ledger[channel]],
                color="#bbb",
                lw=0.7,
                zorder=0,
            )
        axis.axhspan(0.95, 1.05, color="#1a6b3c", alpha=0.08)
        axis.axhline(1.0, color="#333", lw=0.8)
        axis.set_xticks(places)
        axis.set_xticklabels(shared, rotation=90, fontsize=5)
        axis.set_ylabel("far-field gain")
        axis.set_title(
            "The far-field gain ledger: channels that had been handed a static gain "
            "spanning a switch\ncome back to unity once the setting is divided out "
            "instead"
        )
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = directory / "scale_gain_ledger.svg"
        figure.savefig(path)
        plt.close(figure)
        log(f"wrote {path}")


def main(argv: Iterable[str] | None = None) -> None:
    """Run one pass of the boundary-pinning sweep."""

    parser = argparse.ArgumentParser(description=__doc__)
    passes = parser.add_subparsers(dest="pass_name", required=True)

    sweep = passes.add_parser("sweep", help="read every plasma-free excitation shot")
    sweep.add_argument(
        "--limit", type=int, default=0, help="stop after this many shots"
    )
    sweep.add_argument("--resume", action="store_true", help="keep an earlier series")
    sweep.set_defaults(handler=run_sweep)

    array = passes.add_parser(
        "array", help="measure the model-free amplitude and check it against the fit"
    )
    array.add_argument(
        "--limit", type=int, default=0, help="stop after this many shots"
    )
    array.add_argument("--resume", action="store_true", help="keep an earlier series")
    array.set_defaults(handler=run_array)

    pin = passes.add_parser("pin", help="bisect the brackets on plasma shots")
    pin.add_argument("--reads", type=int, default=120, help="how many shots to read")
    pin.add_argument(
        "--reach",
        type=int,
        default=400,
        help="how far from a bisection target a substitute shot may be taken",
    )
    pin.add_argument(
        "--attempts",
        type=int,
        default=6,
        help="how many shots near one target to try before giving up on it",
    )
    pin.add_argument("--resume", action="store_true", help="keep an earlier series")
    pin.set_defaults(handler=run_pin)

    promote = passes.add_parser("promote", help="assemble and report the table")
    promote.add_argument(
        "--promote-to",
        nargs="?",
        type=Path,
        const=MAST_CORRECTION_DOCUMENT,
        help="promote into this correction document (the MAST document by default)",
    )
    promote.set_defaults(handler=run_promote)

    figures = passes.add_parser("figures", help="draw the block map and what pins it")
    figures.add_argument(
        "--record", type=Path, default=CACHE / "mast_block_scale_sweep.json"
    )
    figures.add_argument(
        "--falsification",
        type=Path,
        default=CACHE / "mast_block_scale_falsification.json",
    )
    figures.add_argument(
        "--gains",
        type=Path,
        default=CACHE / "mast_winding_lattice_block_normalised.json",
    )
    figures.add_argument(
        "--raw-gains", type=Path, default=CACHE / "mast_winding_lattice.json"
    )
    figures.add_argument(
        "--figures", type=Path, default=Path("docs/figures/mast-vacuum-floor")
    )
    figures.set_defaults(handler=draw_figures)

    arguments = parser.parse_args(list(argv) if argv is not None else None)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
