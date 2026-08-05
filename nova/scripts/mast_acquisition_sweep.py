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
what the promoted settings rest on.  It keeps every coil family's ratio separately,
because how many families a reading rests on decides whether the reading is usable:
admitting single-family ratios more than doubles the shot coverage and then chops the
histories into blocks on their scatter, which is a loss disguised as a gain.
``--families`` reports the block structure each candidate rule produces so the choice
is made against the measurement.

``array`` measures a second observable that needs no description -- each channel's
amplitude relative to the array median, which no drive, coil or plasma can move
because it is in the numerator and denominator alike -- and checks it against the
fitted route on the shots both read.  That check is a gate, not a result.

``pin`` spends reads inside whatever brackets remain, using that model-free
observable, so it can read plasma shots the fitted route refuses.  It bisects the
widest bracket still open, converging in reads growing with the logarithm of a
bracket's width, and one read serves every channel at once because every channel's
setting is recorded on the same shot.  It may move a boundary and may never set a
value.

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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


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

MINIMUM_FAR_FIELD_FAMILIES = 3
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


def admitted(
    series: Mapping[str, Mapping[int, Sequence[float]]], families: int
) -> dict[str, dict[int, list[float]]]:
    """Keep the readings resting on at least this many coil families.

    A reading's family count is the number of independent excitations that agreed on
    it, so it is the only thing separating a pooled ratio from one projection onto a
    single coil's modelled waveform.  Applied here rather than at read time so the
    rule stays visible and its effect on the block structure stays measurable.
    """

    return {
        channel: {
            shot: list(values)
            for shot, values in rows.items()
            if len(values) >= families
        }
        for channel, rows in series.items()
        if any(len(values) >= families for values in rows.values())
    }


def family_rule(
    series: Mapping[str, Mapping[int, Sequence[float]]], families: int
) -> dict[str, Any]:
    """Report the block structure one admission rule produces.

    Coverage and coherence pull opposite ways here, and the ladder is what breaks the
    tie: a rule that admits more shots but sends the share of steps landing on the
    declared ladder down has bought shots by turning scatter into blocks.  The ladder
    was declared before any of these settings were classified, so it is evidence about
    the rule rather than a target the rule can be tuned onto.
    """

    rows = admitted(series, families)
    histories = channel_histories(rows)
    stepping = stepping_channels(histories)
    steps = [step for row in stepping for step in row.steps]
    counts = sorted(len(shots) for shots in rows.values()) or [0]
    return {
        "channels": len(rows),
        "families": families,
        "median_shots": counts[len(counts) // 2],
        "on_ladder": sum(1 for step in steps if step.on_ladder),
        "readings": sum(len(shots) for shots in rows.values()),
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
        waveforms = read_shot_waveforms(shot, block_scale=RAW)
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
    rules = [family_rule(measured, families) for families in range(1, 6)]
    for report in rules:
        log(
            f"  families >= {report['families']}: {report['channels']:2d} channels, "
            f"{report['readings']:5d} readings, median {report['median_shots']:3.0f} "
            f"shots/channel, {report['stepping']:2d} stepping, "
            f"{report['switches']:3d} switches, "
            f"{report['on_ladder']}/{report['steps']} on the ladder"
        )
    combined = admitted(measured, arguments.families)
    log(f"admitting readings resting on {arguments.families} or more coil families")
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

    record = acquisition_record(histories, concurrency)
    record["admitted_families"] = arguments.families
    record["family_rules"] = rules
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
        log(
            f"boundaries the model-free route placed: {report['placed']}"
            f"/{summary['switch_count']}  narrowed {report['narrowed']}"
        )
        log(
            f"width after narrowing: median {report['median_width']:.0f} "
            f"widest {report['widest_width']} (was median "
            f"{report['median_fitted_width']:.0f})"
        )
        record["narrowing"] = report
        record["route_agreement"] = agreement_summary(agreement(combined, flat))
    (CACHE / "mast_block_scale_sweep.json").write_text(
        json.dumps(record, indent=1, sort_keys=True)
    )
    log("wrote mast_block_scale_sweep.json")
    if arguments.promote_to:
        Path(arguments.promote_to).write_text(
            json.dumps(table.as_dict(), indent=1, sort_keys=True)
        )
        log(f"wrote the promoted table to {arguments.promote_to}")


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

    rules = record.get("family_rules", [])
    if rules:
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
        counts = [row["families"] for row in rules]
        axes[0].plot(counts, [row["readings"] for row in rules], "o-", color="#3a7ab0")
        axes[0].set_ylabel("channel-shot readings admitted")
        axes[0].set_xlabel("coil families a reading must rest on")
        axes[0].set_title("Coverage falls with the rule")
        share = [
            row["on_ladder"] / row["steps"] if row["steps"] else np.nan for row in rules
        ]
        axes[1].plot(counts, share, "o-", color="#1a6b3c")
        axes[1].set_ylim(0.0, 1.02)
        axes[1].set_ylabel("share of steps landing on the declared ladder")
        axes[1].set_xlabel("coil families a reading must rest on")
        axes[1].set_title("Coherence rises with it")
        for axis, values in zip(axes, ([row["stepping"] for row in rules], share)):
            for x, y, n in zip(counts, values, [row["stepping"] for row in rules]):
                axis.annotate(
                    f"{n} stepping",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 7),
                    fontsize=6,
                    ha="center",
                )
        figure.suptitle(
            "Choosing the admission rule: the ladder was declared before any of these "
            "settings were classified, so it is evidence about the rule"
        )
        figure.tight_layout()
        path = directory / "scale_family_rule.svg"
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
        axis.set_ylabel("width after the model-free route placed it [shots]")
        axis.set_title(
            "What the model-free observable bought: each point is one switch, and "
            "distance below\nthe dashed line is how much of its bracket was closed"
        )
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = directory / "scale_bracket_pinning.svg"
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
        "--families",
        type=int,
        default=MINIMUM_FAR_FIELD_FAMILIES,
        help="coil families a reading must rest on to be admitted",
    )
    promote.add_argument(
        "--promote-to", default="", help="write the table to this path"
    )
    promote.set_defaults(handler=run_promote)

    figures = passes.add_parser("figures", help="draw the block map and what pins it")
    figures.add_argument(
        "--record", type=Path, default=CACHE / "mast_block_scale_sweep.json"
    )
    figures.add_argument(
        "--figures", type=Path, default=Path("docs/figures/mast-vacuum-floor")
    )
    figures.set_defaults(handler=draw_figures)

    arguments = parser.parse_args(list(argv) if argv is not None else None)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
