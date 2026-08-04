"""Pin the shots at which a probe channel's acquisition range setting changed.

The settings themselves were measured on the calibration cohort's training split, and
that is enough to say which channels stepped and by what factor.  It is not enough to
say where.  Only a few dozen of that split's shots give any one channel a far-field
reading, and they arrive in clusters, so a switch that happened between two clusters
is known only to lie somewhere in a gap five thousand shots wide.  A correction with
a bracket that wide can be applied to the cohort and to almost nothing else.

Two passes narrow it, and they differ in what they are allowed to conclude.

``sweep`` reads every plasma-free shot the cohort classifier found an excitation in,
not only the ones the split trained on.  These are measurement-grade: the response
model predicts a plasma-free shot completely, so the ratio it returns is a statement
about the channel.  This pass is what the promoted settings rest on.

``pin`` spends its reads inside whatever brackets remain, on the vacuum phase of
plasma shots.  A plasma shot pre-charges the coils before breakdown, and the sample
window that survives the plasma-current test is an ordinary vacuum measurement -- but
one taken on a shot the cohort refused, so this pass may place a boundary and may not
set a setting.  It only has to resolve a factor of two, and it chooses its next shot
by bisecting the widest bracket still open, so it converges on a boundary in reads
growing with the logarithm of the bracket's width.  One read serves every channel
at once, because every channel's setting is recorded on the same shot.

``promote`` assembles the table the read path applies, refers each block to the block
reading nearest the described field, snaps each step to the declared ladder, and
writes the pinning summary beside it so a reader sees which switches the archive pins
and which it cannot.

Every pass reads the archive as published.  Passing the promoted table into the very
measurement that produced it would divide out the setting and then measure that it
was gone, so the raw path is named explicitly here rather than defaulted.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_acquisition_scale import (
    acquisition_record,
    channel_histories,
    step_concurrency,
    stepping_channels,
)
from nova.imas.mast_block_scale import (
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
from nova.imas.mast_vacuum_cohort import probe_channels, read_shot_waveforms
from nova.imas.mast_vacuum_response import MINIMUM_STANDOFF, ResponseModel

CACHE = Path.home() / ".cache" / "nova-mast"
"""Where the cohort's own records live and this one is written beside them."""

RAW = BlockScaleTable()
"""The read path with no correction applied.

Named once and passed explicitly to every read in this driver.  The sweep measures
the setting the archive recorded a channel at, so it has to see what the archive
published; defaulting to the promoted table would divide the setting out first and
then measure that the channel is at unity.
"""

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

MINIMUM_FAR_FIELD_FAMILIES = 1
"""Coil families a shot must give a channel a far-field reading on to count."""


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
) -> dict[str, float]:
    """Return each channel's far-field ratio on one shot, or nothing measurable.

    The rigid scale-and-rotation fit is not asked for: a range setting is an
    amplitude and the pair would double the cost of a pass whose whole purpose is
    breadth.  The error-field screen and the amplitude refusal apply exactly as they
    do in the fit the settings came from, so a shot excluded there stays excluded
    here.
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
        waveforms = read_shot_waveforms(shot, block_scale=RAW)
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
    return {
        channel: float(np.median(values))
        for channel, values in rows.items()
        if len(values) >= MINIMUM_FAR_FIELD_FAMILIES
    }


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
    ratios: Mapping[str, float],
) -> None:
    for channel, value in ratios.items():
        series.setdefault(channel, {})[int(shot)] = [float(value)]


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


def run_pin(arguments: argparse.Namespace) -> None:
    """Bisect the brackets left open, on the vacuum phase of plasma shots."""

    model, weights = build_model()
    screen = load_screen()
    far = far_field_pairs(model)
    base = _load_series("plasma_free")
    if not base:
        raise SystemExit("run the sweep pass before pinning its boundaries")
    extra = _load_series("plasma_vacuum_phase") if arguments.resume else {}
    candidates = plasma_shots()
    attempted: set[int] = {shot for rows in extra.values() for shot in rows}
    log(f"candidate plasma shots: {len(candidates)}  already read: {len(attempted)}")

    for step in range(arguments.reads):
        combined = {
            channel: {**base.get(channel, {}), **extra.get(channel, {})}
            for channel in set(base) | set(extra)
        }
        table = _table_from(combined, "bisected on the plasma vacuum phase")
        target = bracket_probe(table.brackets(), candidates, measured=attempted)
        if target is None:
            log("every bracket is pinned as tightly as the archive allows")
            break
        window = [
            shot
            for shot in candidates
            if abs(shot - target) <= arguments.reach and shot not in attempted
        ]
        window.sort(key=lambda shot: (abs(shot - target), shot))
        ratios: dict[str, float] = {}
        for shot in window[: arguments.attempts]:
            attempted.add(shot)
            ratios = measure_shot(shot, model, weights, screen, far)
            if ratios:
                _merge(extra, shot, ratios)
                log(f"  read {step}: shot {shot} gave {len(ratios)} channels")
                break
        else:
            log(f"  read {step}: no readable shot near {target}")
        if step % 10 == 0:
            _write_series("plasma_vacuum_phase", extra)
    _write_series("plasma_vacuum_phase", extra)
    log(f"plasma-phase readings: {sum(len(rows) for rows in extra.values())}")


def run_promote(arguments: argparse.Namespace) -> None:
    """Assemble the table the read path applies, and report what pins it."""

    base = _load_series("plasma_free")
    if not base:
        raise SystemExit("run the sweep pass before promoting a table")
    extra = _load_series("plasma_vacuum_phase") if arguments.with_plasma_phase else {}
    combined = {
        channel: {**base.get(channel, {}), **extra.get(channel, {})}
        for channel in set(base) | set(extra)
    }
    route = (
        "far-field response ratio on plasma-free shots"
        if not extra
        else "far-field response ratio on plasma-free shots, boundaries bisected on "
        "the vacuum phase of plasma shots"
    )
    histories = channel_histories(combined)
    table = _table_from(combined, route)
    stepping = stepping_channels(histories)
    steps = [step for row in stepping for step in row.steps]
    concurrency = step_concurrency(combined, steps)
    summary = pinning_summary(
        table, sorted(set(excitation_shots()) | set(plasma_shots()))
    )

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

    record = acquisition_record(histories, concurrency)
    record["pinning"] = summary
    record["table"] = table.as_dict()
    (CACHE / "mast_block_scale_sweep.json").write_text(
        json.dumps(record, indent=1, sort_keys=True)
    )
    log("wrote mast_block_scale_sweep.json")
    if arguments.promote_to:
        Path(arguments.promote_to).write_text(
            json.dumps(table.as_dict(), indent=1, sort_keys=True)
        )
        log(f"wrote the promoted table to {arguments.promote_to}")


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
        "--with-plasma-phase",
        action="store_true",
        help="include the boundaries bisected on plasma shots",
    )
    promote.add_argument(
        "--promote-to", default="", help="write the table to this path"
    )
    promote.set_defaults(handler=run_promote)

    arguments = parser.parse_args(list(argv) if argv is not None else None)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
