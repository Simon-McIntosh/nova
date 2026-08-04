"""Read the archive's designed calibration experiments and close the turn counts.

Five stages, each writing what the next reads.  ``classify`` turns the cached shot
census into experiment classes, an identifiability map and a declared train and
held-out split -- and it is deliberately the only stage that decides the split, so
no later stage can choose one after seeing a result.  ``resurvey`` re-reads the
plasma-free shots for the channels the original census did not record, which is
what closes the search for a non-axisymmetric excitation.  ``scale`` measures each
shot's amplitude against the turn counts the archive itself publishes, which
screens out shots whose magnetics were recorded at the wrong gain before any count
is fitted to them.  ``turns`` fits what is left.  ``figures`` draws it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_calibration_cohort import (
    ExperimentClass,
    ampere_turn_ratio_support,
    calibration_experiments,
    class_counts,
    integer_ampere_turn_ratios,
    select_calibration_cohort,
)
from nova.imas.mast_sensor_noise import (
    measure_noise_envelope,
    measure_repeat_scatter,
    repeat_groups,
)
from nova.imas.mast_vacuum_cohort import (
    SHOT_STORE,
    ShotSurvey,
    probe_channels,
    read_shot_waveforms,
    survey_store,
)
from nova.imas.mast_vacuum_response import (
    ADMISSIBLE_SCALE,
    MINIMUM_STANDOFF,
    ResponseError,
    ResponseModel,
    aggregate_turns,
    per_shot_estimates,
    published_turn_scale,
    score_prediction,
)


def load_surveys(path: Path) -> tuple[ShotSurvey, ...]:
    """Read a cached census back, carrying whichever detail it recorded.

    A census written before the excitation detail was recorded is still readable,
    because the fields it lacks describe channels rather than shots and their
    absence means unmeasured rather than zero.  The classes that depend on them
    say so: a shot with no error-field reading is never reported as having driven
    one.
    """

    payload = json.loads(path.read_text())
    return tuple(
        ShotSurvey(
            shot=int(row["shot"]),
            plasma_current_peak=float(row["plasma_current_peak"]),
            toroidal_current_peak=float(row["toroidal_current_peak"]),
            coil_peaks={k: float(v) for k, v in row["coil_peaks"].items()},
            coil_hold_times={
                k: float(v) for k, v in row.get("coil_hold_times", {}).items()
            },
            turn_multipliers={k: float(v) for k, v in row["turn_multipliers"].items()},
            absent_groups=tuple(row["absent_groups"]),
            absent_channels=tuple(row["absent_channels"]),
            field_channels=tuple(row["field_channels"]),
            current_identity=row.get("current_identity", ""),
            field_identity=row.get("field_identity", ""),
            case_peaks={k: float(v) for k, v in row.get("case_peaks", {}).items()},
            error_field_peaks={
                k: float(v) for k, v in row.get("error_field_peaks", {}).items()
            },
            toroidal_hold_time=float(row.get("toroidal_hold_time", 0.0)),
        )
        for row in payload["surveys"]
    )


REPRESENTATIVE_SHOT = 11766
"""Registry selection whose configuration the calibration is authored against."""

SOLENOID_FAMILY = "sol"
"""Coil whose weight the two measurement routes disagree about."""

VERTICAL_PAIR = ("p6_lower", "p6_upper")
"""Coils the store publishes as ampere-turns and drives only as a pair."""


def _model(standoff: float) -> tuple[ResponseModel, Any]:
    """Build the geometry-derived response once, and return its provenance."""

    registry = MachineGeometryRegistry.default()
    selection = registry.select(REPRESENTATIVE_SHOT)
    geometry = selection.configuration.geometry
    probes = geometry["magnetics"]["poloidal_probes"]
    channels = probe_channels(probes)
    model = ResponseModel.build(geometry, probes, channels, minimum_standoff=standoff)
    return model, selection


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")


def classify(arguments: argparse.Namespace) -> None:
    """Class the census, map identifiability and declare the split."""

    surveys = load_surveys(arguments.census)
    experiments = calibration_experiments(surveys)
    cohort = select_calibration_cohort(
        surveys,
        held_out_fraction=arguments.held_out_fraction,
        noise_limit=arguments.noise_limit,
    )
    counts = class_counts(experiments)
    ratios = integer_ampere_turn_ratios(surveys)
    support = ampere_turn_ratio_support(surveys)

    print(f"classed {len(experiments)} shots")
    for name, count in counts.items():
        print(f"  {name:<26} {count:>6d}")
    print(f"\npublished integer ampere-turn ratios ({len(ratios)} coils):")
    for family, integer in ratios.items():
        print(f"  {family:<16} {integer:>4d} turns on {support[family]:>6d} shots")
    print("\nidentifiability:")
    for row in cohort.identifiability:
        state = (
            "alone"
            if row.identifiable
            else ("pair sum only" if row.sum_only else "UNREACHABLE")
        )
        print(
            f"  {row.family:<16} {state:<14} alone={len(row.alone):>3d} "
            f"sum={len(row.in_sum):>4d} strongest={row.strongest / 1e3:6.1f} kA"
        )
    print(
        f"\nsplit declared: {len(cohort.training)} training, "
        f"{len(cohort.held_out)} held out, {len(cohort.noise_shots)} noise shots"
    )
    _write(
        arguments.cohort,
        {
            "class_counts": counts,
            "cohort": cohort.as_dict(),
            "experiments": [
                row.as_dict()
                for row in experiments
                if row.measures_turns or row.measures_noise
            ],
            "published_ampere_turn_ratios": ratios,
            "published_ratio_support": support,
            "store": str(arguments.store),
        },
    )


def resurvey(arguments: argparse.Namespace) -> None:
    """Re-read the plasma-free shots for the channels the census omitted."""

    surveys = load_surveys(arguments.census)
    shots = sorted(
        row.shot
        for row in surveys
        if row.plasma_current_peak < 5.0e3 and not row.absent_groups
    )
    print(f"re-reading {len(shots)} plasma-free shots for excitation detail")
    rows = survey_store(shots, store=arguments.store, processes=arguments.processes)
    driven = [row for row in rows if row.error_field_driven]
    peaks = {
        channel: max(
            (row.error_field_peaks.get(channel, 0.0) for row in rows), default=0.0
        )
        for row in rows
        for channel in row.error_field_peaks
    }
    print(f"read {len(rows)} shots; {len(driven)} drove a non-axisymmetric coil")
    for channel, peak in sorted(peaks.items()):
        print(f"  {channel:<18} strongest {peak:9.1f} A")
    _write(
        arguments.detail,
        {
            "error_field_driven_shots": [row.shot for row in driven],
            "error_field_peak_by_channel": {
                k: float(v) for k, v in sorted(peaks.items())
            },
            "shots_read": len(rows),
            "surveys": [row.as_dict() for row in rows],
        },
    )


def scale(arguments: argparse.Namespace) -> None:
    """Measure each identifying shot's amplitude against the published turns."""

    surveys = load_surveys(arguments.census)
    published = {
        family: float(integer)
        for family, integer in integer_ampere_turn_ratios(surveys).items()
    }
    experiments = [
        row for row in calibration_experiments(surveys) if row.measures_turns
    ]
    model, selection = _model(arguments.standoff)
    rows: list[dict[str, Any]] = []
    print(f"measuring amplitude on {len(experiments)} identifying shots")
    for experiment in experiments:
        if not set(experiment.identifies) & set(published):
            continue
        try:
            waveforms = read_shot_waveforms(experiment.shot, store=arguments.store)
            measured = published_turn_scale(
                waveforms, model, published, stride=arguments.stride
            )
        except ResponseError as error:
            print(f"  {experiment.shot:6d} unreadable: {error}")
            continue
        except Exception as error:  # noqa: BLE001 - a shot may be absent or corrupt
            print(f"  {experiment.shot:6d} failed: {error}")
            continue
        rows.append(measured.as_dict() | {"experiment": str(experiment.experiment)})
        print(
            f"  {measured.shot:6d} {'+'.join(measured.families):<34} "
            f"scale={measured.scale:7.4f} spread={measured.probe_spread:6.4f} "
            f"probes={measured.probe_count:3d} "
            f"{'' if measured.admissible else 'REFUSED'}"
        )
    admitted = [row for row in rows if row["admissible"]]
    refused = [row for row in rows if not row["admissible"]]
    values = np.asarray([row["scale"] for row in admitted], dtype=float)
    print(f"\n{len(admitted)} shots inside {ADMISSIBLE_SCALE}, {len(refused)} refused")
    if values.size:
        print(
            f"admitted amplitude: median {np.median(values):.4f} "
            f"spread {np.std(values):.4f} "
            f"range [{values.min():.4f}, {values.max():.4f}]"
        )
    if refused:
        print(
            "refused amplitudes: "
            + ", ".join(f"{row['shot']}={row['scale']:.3f}" for row in refused)
        )
    _write(
        arguments.scale_report,
        {
            "admissible_interval": list(ADMISSIBLE_SCALE),
            "admitted_count": len(admitted),
            "admitted_median": float(np.median(values))
            if values.size
            else float("nan"),
            "admitted_spread": float(np.std(values)) if values.size else float("nan"),
            "minimum_standoff": arguments.standoff,
            "physical_digest": selection.configuration.physical_digest,
            "published_ampere_turn_ratios": published,
            "refused_count": len(refused),
            "scales": rows,
        },
    )


def _read_many(shots: Sequence[int], store: Path) -> dict[int, Any]:
    """Read several shots' waveforms, skipping the ones the store cannot serve."""

    waveforms: dict[int, Any] = {}
    for shot in shots:
        try:
            waveforms[shot] = read_shot_waveforms(shot, store=store)
        except Exception as error:  # noqa: BLE001 - a shot may be absent or corrupt
            print(f"  shot {shot} unreadable: {error}")
    return waveforms


def turns(arguments: argparse.Namespace) -> None:
    """Fit the turn counts the admitted shots support, and bound the rest."""

    surveys = load_surveys(arguments.census)
    published = {
        family: float(integer)
        for family, integer in integer_ampere_turn_ratios(surveys).items()
    }
    cohort = select_calibration_cohort(
        surveys, held_out_fraction=arguments.held_out_fraction
    )
    screened = json.loads(arguments.scale_report.read_text())["scales"]
    admitted = {row["shot"] for row in screened if row["admissible"]}
    refused = {row["shot"] for row in screened if not row["admissible"]}
    model, selection = _model(arguments.standoff)
    classed = calibration_experiments(surveys)
    experiments = {row.shot: row for row in classed if row.measures_turns}
    sustained = {
        row.shot: row for row in classed if row.identifies or row.identifies_sum
    }

    def usable(shot: int) -> bool:
        """Return whether a shot may be fitted, screened or unscreenable.

        A shot that drove only the solenoid cannot be screened at all, because the
        archive publishes no turn count for it and the amplitude test needs one.
        Dropping those would discard the cleanest experiments the store holds for
        the one coil whose weight rests entirely on a fit, so an unscreenable shot
        is admitted and counted separately -- the price is that the solenoid's
        interval must carry the shots' own disagreement, which it does.
        """

        return shot not in refused

    training = [shot for shot in cohort.training if usable(shot)]
    held_out = [shot for shot in cohort.held_out if usable(shot)]
    unscreened = [shot for shot in (*training, *held_out) if shot not in admitted]
    print(
        f"fitting on {len(training)} training shots, challenging on "
        f"{len(held_out)} held-out shots; {len(refused)} refused on amplitude, "
        f"{len(unscreened)} admitted unscreenable"
    )
    train_waveforms = _read_many(training, arguments.store)
    test_waveforms = _read_many(held_out, arguments.store)

    estimates = per_shot_estimates(
        list(train_waveforms.values()), model, stride=arguments.stride
    )
    dispositions = aggregate_turns(
        estimates, archive_multipliers={k: v for k, v in published.items()}
    )
    print("\nsigned turn counts on the admitted calibration shots:")
    for row in dispositions:
        if not row.identified:
            print(f"  {row.family:<16} unidentified")
            continue
        note = (
            ""
            if row.archive_multiplier is None
            else f"  published={row.archive_multiplier:g} "
            f"{'agrees' if row.agrees_with_archive else 'DISAGREES'}"
        )
        print(
            f"  {row.family:<16} {row.multiplier:+9.3f} +-{row.half_width:7.3f} "
            f"n={len(row.shots):<3d}{note}"
        )

    solenoid = _solenoid_interval(estimates, experiments)
    vertical = _vertical_pair(sustained, model, arguments)
    scores: dict[str, Any] = {}
    fitted = {row.family: row.multiplier for row in dispositions if row.identified}
    promoted = dict(published)
    promoted.update({k: v for k, v in fitted.items() if k not in promoted})
    for name, values in (
        ("nominal_unit_turns", {family: 1.0 for family in model.families}),
        ("published_and_fitted", promoted),
        ("fitted_only", fitted),
    ):
        try:
            score = score_prediction(
                list(test_waveforms.values()), model, values, stride=arguments.stride
            )
        except Exception as error:  # noqa: BLE001 - a score may be unavailable
            print(f"held-out {name}: unavailable ({error})")
            continue
        scores[name] = score.as_dict()
        print(
            f"held-out {name:<22} rms={score.residual_rms:.4e} T "
            f"explained={score.variance_explained:+.4f} on {len(score.shots)} shots"
        )

    _write(
        arguments.turn_report,
        {
            "held_out_scores": scores,
            "held_out_shots": held_out,
            "per_shot_estimates": [row.as_dict() for row in estimates],
            "physical_digest": selection.configuration.physical_digest,
            "promoted_weights": promoted,
            "published_ampere_turn_ratios": published,
            "refused_on_amplitude": sorted(refused),
            "solenoid": solenoid,
            "training_shots": training,
            "unscreenable_shots": sorted(unscreened),
            "turn_dispositions": [row.as_dict() for row in dispositions],
            "vertical_pair": vertical,
        },
    )


def _solenoid_interval(
    estimates: Sequence[Any],
    experiments: Mapping[int, Any],
) -> dict[str, Any]:
    """Combine the shots that held the solenoid ALONE into one weight.

    The solenoid is the one coil the archive publishes no turn count for, so its
    weight rests entirely on the fit, and the interval has to carry the shots' own
    disagreement rather than the solve's optimism about any single shot.

    Only shots that held the solenoid and nothing else contribute.  A shot driving
    the solenoid beside another coil still reports a solenoid multiplier and still
    passes the leverage and correlation screens, because the solenoid's ampere turns
    outweigh a few-kiloampere neighbour by a factor of forty -- but the number it
    reports has absorbed that neighbour's misfit.  Measured on this store the two
    populations do not overlap: shots that drove the solenoid by itself read 337 to
    366, and shots that drove it alongside another coil read 428 to 477.  Pooling
    them moves the centre seven percent and triples the interval, so a measurement
    would appear to get worse as data was added; the restriction is what makes the
    interval mean anything.
    """

    alone = {
        shot
        for shot, row in experiments.items()
        if tuple(row.identifies) == (SOLENOID_FAMILY,) and not row.identifies_sum
    }
    rows = [
        row
        for row in estimates
        if row.family == SOLENOID_FAMILY and row.identified and row.shot in alone
    ]
    if not rows:
        return {"identified": False, "shots": []}
    values = np.asarray([row.multiplier for row in rows], dtype=float)
    errors = np.asarray([max(row.standard_error, 1.0e-12) for row in rows])
    weight = 1.0 / errors**2
    centre = float((values * weight).sum() / weight.sum())
    spread = float(np.std(values)) if values.size > 1 else 0.0
    solve = float(1.0 / math.sqrt(weight.sum()))
    half = max(spread, solve)
    print(
        f"\nsolenoid weight {centre:.3f} ampere turns per ampere of its channel, "
        f"interval [{centre - half:.3f}, {centre + half:.3f}] from {len(rows)} shots"
    )
    return {
        "identified": True,
        "interval": [centre - half, centre + half],
        "per_shot": {str(row.shot): float(row.multiplier) for row in rows},
        "shot_count": len(rows),
        "shots": [row.shot for row in rows],
        "solve_error": solve,
        "spread": spread,
        "weight": centre,
    }


def _vertical_pair(
    experiments: Mapping[int, Any],
    model: ResponseModel,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Test the published ampere-turn semantics of the vertical-control pair.

    The store publishes this pair's excitation already multiplied by whatever turn
    count it has, so the quantity a fit can reach is one ampere turn per ampere and
    not the count itself.  The dedicated shots that hold the pair alone are what
    turn that from an assumption into a measurement: a fit returning one confirms
    the channel's meaning, and the physical count stays out of reach because the
    published product fixes only that product.
    """

    pair = set(VERTICAL_PAIR)
    reachable = [
        (row.peak_current, shot)
        for shot, row in experiments.items()
        if pair & (set(row.identifies) | set(row.identifies_sum))
        and not (set(row.identifies) | set(row.identifies_sum)) - pair
    ]
    shots = [shot for _, shot in sorted(reachable, reverse=True)][
        : arguments.pair_shots
    ]
    if not shots:
        return {"measured": False, "reason": "no shot holds the pair alone"}
    print(
        f"\n{len(reachable)} shots hold the vertical pair and nothing else; "
        f"reading the {len(shots)} strongest"
    )
    print(f"\ntesting the vertical pair's published semantics on {len(shots)} shots")
    unit = {family: 1.0 for family in VERTICAL_PAIR}
    rows = []
    for shot in shots:
        try:
            waveforms = read_shot_waveforms(shot, store=arguments.store)
            measured = published_turn_scale(
                waveforms, model, unit, stride=arguments.stride
            )
        except Exception as error:  # noqa: BLE001 - a shot may be absent or corrupt
            print(f"  {shot:6d} unavailable: {error}")
            continue
        rows.append(measured)
        print(
            f"  {shot:6d} scale={measured.scale:7.4f} "
            f"explained={measured.variance_explained:+.4f} "
            f"probes={measured.probe_count:3d}"
        )
    if not rows:
        return {"measured": False, "reason": "no shot admitted a readable window"}
    values = np.asarray([row.scale for row in rows], dtype=float)
    centre, spread = float(np.median(values)), float(np.std(values))
    print(
        f"  published ampere-turn semantics confirmed at {centre:.4f} "
        f"+-{spread:.4f} over {len(rows)} shots"
    )
    return {
        "measured": True,
        "families": list(VERTICAL_PAIR),
        "interval": [centre - spread, centre + spread],
        "per_shot": [row.as_dict() for row in rows],
        "scale": centre,
        "shot_count": len(rows),
        "spread": spread,
    }


def noise(arguments: argparse.Namespace) -> None:
    """Measure every sensor's floor on the shots that drove nothing."""

    surveys = load_surveys(arguments.census)
    cohort = select_calibration_cohort(surveys, noise_limit=arguments.noise_limit)
    shots = list(cohort.noise_shots)
    print(f"measuring the sensor floor on {len(shots)} shots")
    waveforms = list(_read_many(shots, arguments.store).values())
    if not waveforms:
        raise SystemExit("no noise shot could be read")
    envelope = measure_noise_envelope(waveforms)
    families = envelope.family_scatter()
    print(f"\npooled sensor floor {envelope.pooled_scatter:.4e} T")
    for family, value in families.items():
        print(f"  {family:<8} {value:.4e} T")
    worst = sorted(envelope.channels, key=lambda row: -row.scatter)[:6]
    print("noisiest channels:")
    for row in worst:
        print(
            f"  {row.channel:<8} {row.scatter:.4e} T  drift {row.drift_rate:.3e} T/s "
            f"on {row.shot_count} shots"
        )

    experiments = [
        row for row in calibration_experiments(surveys) if row.measures_turns
    ]
    refused: set[int] = set()
    if arguments.scale_report.exists():
        refused = {
            row["shot"]
            for row in json.loads(arguments.scale_report.read_text())["scales"]
            if not row["admissible"]
        }
        print(f"\nexcluding {len(refused)} amplitude-refused shots from the repeats")
    repeats = []
    grouped = repeat_groups(experiments, exclude=refused)
    for family, group, peak in grouped[: arguments.repeat_groups]:
        readable = _read_many(group, arguments.store)
        try:
            measured = measure_repeat_scatter(
                family, group, readable, peak_current=peak
            )
        except Exception as error:  # noqa: BLE001 - a group may not repeat cleanly
            print(f"  repeat {family} {group}: unavailable ({error})")
            continue
        repeats.append(measured)
        print(
            f"  repeat {family:<16} {measured.shots} "
            f"relative {measured.relative_scatter:.4f} "
            f"absolute {measured.absolute_scatter:.3e} T"
        )
    _write(
        arguments.noise_report,
        {
            "envelope": envelope.as_dict(),
            "repeat_scatter": [row.as_dict() for row in repeats],
        },
    )


FIGURE_DIRECTORY = Path("docs/figures/mast-vacuum-floor")
"""Where the calibration figures are written."""


def figures(arguments: argparse.Namespace) -> None:
    """Draw the cohort map, the amplitude screen and the noise envelope."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arguments.figures.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(arguments.cohort.read_text())
    _draw_cohort(plt, cohort, arguments.figures / "calibration_cohort.svg")
    if arguments.scale_report.exists():
        _draw_scale(
            plt,
            json.loads(arguments.scale_report.read_text()),
            arguments.figures / "published_turn_amplitude.svg",
        )
    if arguments.noise_report.exists():
        _draw_noise(
            plt,
            json.loads(arguments.noise_report.read_text()),
            arguments.figures / "sensor_noise_envelope.svg",
        )
    if arguments.turn_report.exists():
        _draw_turns(
            plt,
            json.loads(arguments.turn_report.read_text()),
            arguments.figures / "turn_closure.svg",
        )
    print(f"wrote figures to {arguments.figures}")


def _draw_cohort(plt: Any, report: dict[str, Any], path: Path) -> None:
    """Show what the archive holds, and which coils each class can measure."""

    counts = report["class_counts"]
    order = [
        ExperimentClass.SUSTAINED_SINGLE_COIL,
        ExperimentClass.SUSTAINED_SYMMETRIC_PAIR,
        ExperimentClass.SUSTAINED_COIL_GROUP,
        ExperimentClass.PULSED_EXCITATION,
        ExperimentClass.TOROIDAL_FIELD_ONLY,
        ExperimentClass.QUIESCENT,
    ]
    labels = [str(name).replace("_", " ") for name in order]
    values = [counts.get(str(name), 0) for name in order]
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), constrained_layout=True
    )
    colours = ["tab:green"] * 3 + ["tab:orange", "tab:blue", "tab:purple"]
    left.barh(range(len(values)), values, color=colours)
    left.set_yticks(range(len(values)))
    left.set_yticklabels(labels, fontsize=9)
    left.invert_yaxis()
    left.set_xscale("log")
    left.set_xlabel("shots in the store")
    left.set_title(
        "What the archive holds\n"
        "green measures a coil, blue and purple measure the sensors"
    )
    for index, value in enumerate(values):
        left.text(max(value, 1) * 1.1, index, str(value), va="center", fontsize=9)
    left.grid(alpha=0.3, axis="x")

    rows = report["cohort"]["identifiability"]
    names = [row["family"] for row in rows]
    alone = [row["alone_count"] for row in rows]
    in_sum = [row["in_sum_count"] for row in rows]
    positions = np.arange(len(rows))
    right.barh(positions - 0.2, alone, height=0.38, color="tab:green", label="alone")
    right.barh(
        positions + 0.2, in_sum, height=0.38, color="tab:grey", label="pair sum only"
    )
    right.set_yticks(positions)
    right.set_yticklabels(names, fontsize=8)
    right.invert_yaxis()
    right.set_xscale("symlog", linthresh=1)
    right.set_xlabel("shots that can measure this coil")
    right.set_title("Identifiability per coil\nwhich experiments reach which winding")
    right.legend(fontsize=8, loc="lower right")
    right.grid(alpha=0.3, axis="x")
    figure.savefig(path)
    plt.close(figure)


def _draw_scale(plt: Any, report: dict[str, Any], path: Path) -> None:
    """Show the amplitude screen that separates the mis-scaled shots."""

    rows = sorted(report["scales"], key=lambda row: row["shot"])
    shots = [row["shot"] for row in rows]
    values = [row["scale"] for row in rows]
    good = [row["admissible"] for row in rows]
    lower, upper = report["admissible_interval"]
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.4, 1.0]},
    )
    left.axhspan(lower, upper, color="tab:green", alpha=0.12)
    left.axhline(1.0, color="black", lw=0.8, ls="--")
    left.scatter(
        range(len(rows)),
        values,
        c=["tab:blue" if ok else "firebrick" for ok in good],
        s=34,
    )
    for index, row in enumerate(rows):
        if not row["admissible"]:
            left.annotate(
                str(row["shot"]),
                (index, row["scale"]),
                fontsize=7,
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                color="firebrick",
            )
    left.set_xticks(range(len(rows)))
    left.set_xticklabels(shots, rotation=90, fontsize=6)
    left.set_ylabel("measured field / field from published turns")
    left.set_ylim(0.0, 1.4)
    left.set_title(
        "Does the shot read what the archive's own turn counts predict?\n"
        "red shots read about half and cannot measure a count"
    )
    left.grid(alpha=0.3, axis="y")

    right.hist(values, bins=np.linspace(0.35, 1.25, 28), color="tab:blue")
    right.axvspan(lower, upper, color="tab:green", alpha=0.12)
    right.set_xlabel("amplitude ratio")
    right.set_ylabel("shots")
    right.set_title("Nothing sits between\nthe two populations")
    right.grid(alpha=0.3, axis="y")
    figure.savefig(path)
    plt.close(figure)


def _draw_noise(plt: Any, report: dict[str, Any], path: Path) -> None:
    """Show every sensor's measured floor and how it splits by family."""

    channels = report["envelope"]["channels"]
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.6, 1.0]},
    )
    names = [row["channel"] for row in channels]
    values = [row["scatter"] for row in channels]
    family_of = [name.rstrip("0123456789") for name in names]
    palette = {"ccbv": "tab:blue", "obr": "tab:orange", "obv": "tab:green"}
    left.bar(
        range(len(values)),
        values,
        color=[palette.get(name, "grey") for name in family_of],
    )
    pooled = report["envelope"]["pooled_scatter"]
    left.axhline(pooled, color="black", lw=1.0, ls="--")
    left.annotate(
        f"pooled {pooled:.2e} T",
        (len(values) * 0.62, pooled),
        fontsize=9,
        textcoords="offset points",
        xytext=(0, 6),
    )
    left.set_xticks(range(len(names)))
    left.set_xticklabels(names, rotation=90, fontsize=5)
    left.set_yscale("log")
    left.set_ylabel("scatter about the drift ramp [T]")
    left.set_title(
        "The sensor floor, measured where nothing was driven\n"
        "one bar per channel, coloured by probe family"
    )
    left.grid(alpha=0.3, axis="y")

    families = report["envelope"]["family_scatter"]
    keys = list(families)
    right.bar(
        range(len(keys)),
        [families[key] for key in keys],
        color=[palette.get(key, "grey") for key in keys],
    )
    right.set_xticks(range(len(keys)))
    right.set_xticklabels(keys, fontsize=9)
    right.set_yscale("log")
    right.set_ylabel("pooled floor [T]")
    right.set_title("By family")
    right.grid(alpha=0.3, axis="y")
    figure.savefig(path)
    plt.close(figure)


def _draw_turns(plt: Any, report: dict[str, Any], path: Path) -> None:
    """Show each coil's fitted count beside the count the archive publishes."""

    rows = report["turn_dispositions"]
    published = report["published_ampere_turn_ratios"]
    figure, axes = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    names = [row["family"] for row in rows]
    for index, row in enumerate(rows):
        if not row["identified"]:
            axes.text(
                index,
                1.0,
                "not\nidentified",
                ha="center",
                va="center",
                fontsize=7,
                color="firebrick",
            )
            continue
        value = row["multiplier"]
        low, high = row["interval"]
        axes.errorbar(
            index,
            value,
            yerr=[[value - low], [high - value]],
            fmt="o",
            color="tab:blue",
            capsize=4,
            markersize=5,
        )
        reference = published.get(row["family"])
        if reference is not None:
            axes.plot(
                index,
                reference,
                marker="_",
                markersize=18,
                color="tab:green",
                linestyle="none",
            )
    solenoid = report.get("solenoid", {})
    if solenoid.get("identified"):
        index = names.index("sol") if "sol" in names else 0
        low, high = solenoid["interval"]
        axes.annotate(
            f"{solenoid['weight']:.1f} [{low:.0f}, {high:.0f}]",
            (index, solenoid["weight"]),
            fontsize=8,
            textcoords="offset points",
            xytext=(10, 0),
        )
    axes.set_xticks(range(len(names)))
    axes.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes.set_yscale("symlog", linthresh=10)
    axes.set_ylabel("ampere turns per ampere of the measured channel")
    axes.set_title(
        "Turn closure on the admitted calibration shots\n"
        "blue the fit with its interval, green bars the count the archive publishes"
    )
    axes.grid(alpha=0.3)
    figure.savefig(path)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one calibration stage."""

    cache = Path.home() / ".cache/nova-mast"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument(
        "--census", type=Path, default=cache / "mast_vacuum_census.json"
    )
    parser.add_argument(
        "--cohort", type=Path, default=cache / "mast_calibration_cohort.json"
    )
    parser.add_argument(
        "--scale-report", type=Path, default=cache / "mast_published_scale.json"
    )
    parser.add_argument(
        "--noise-report", type=Path, default=cache / "mast_sensor_noise.json"
    )
    parser.add_argument(
        "--turn-report", type=Path, default=cache / "mast_turn_closure.json"
    )
    parser.add_argument(
        "--detail", type=Path, default=cache / "mast_excitation_detail.json"
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--standoff", type=float, default=MINIMUM_STANDOFF)
    parser.add_argument("--held-out-fraction", type=float, default=0.25)
    parser.add_argument("--noise-limit", type=int, default=60)
    stages = parser.add_subparsers(dest="stage", required=True)

    grouped = stages.add_parser(
        "classify", help="class the census and declare the split"
    )
    grouped.set_defaults(handler=classify)

    detail = stages.add_parser("resurvey", help="re-read plasma-free excitation detail")
    detail.add_argument("--processes", type=int, default=8)
    detail.set_defaults(handler=resurvey)

    amplitude = stages.add_parser(
        "scale", help="screen shots on published-turn amplitude"
    )
    amplitude.set_defaults(handler=scale)

    counted = stages.add_parser(
        "turns", help="fit the counts the admitted shots support"
    )
    counted.add_argument("--pair-shots", type=int, default=12)
    counted.set_defaults(handler=turns)

    floor = stages.add_parser("noise", help="measure the sensor floor")
    floor.add_argument("--repeat-groups", type=int, default=10)
    floor.set_defaults(handler=noise)

    drawing = stages.add_parser("figures", help="draw the calibration figures")
    drawing.add_argument("--figures", type=Path, default=FIGURE_DIRECTORY)
    drawing.set_defaults(handler=figures)

    arguments = parser.parse_args(argv)
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
