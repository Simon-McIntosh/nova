"""Refine the MAST machine description against no-plasma shots.

Three stages run in order and each writes what the next one reads, so a long
census is paid for once.  ``survey`` walks the shot store and records what every
shot drove and measured.  ``fit`` builds the cohort from that census, splits it,
identifies the parameter blocks the data supports, and reports residuals against
held-out shots.  ``figures`` draws the comparisons the fit established.

The census is the expensive stage and the only one that touches every shot, so it
is the one that parallelises; the fit reads a few dozen shots and runs in a single
process.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_vacuum_cohort import (
    COIL_DRIVES,
    SHOT_STORE,
    ShotSurvey,
    VacuumCohort,
    probe_channels,
    read_shot_waveforms,
    select_vacuum_cohort,
    store_shots,
    survey_store,
)
from nova.imas.mast_fitted_parameters import RADIAL_PROBE_FAMILY
from nova.imas.mast_passive_response import (
    case_grouping,
    passive_coupling,
    passive_groups,
    survey_decays,
)
from nova.imas.mast_vacuum_response import (
    MINIMUM_STANDOFF,
    ResponseModel,
    aggregate_turns,
    fit_response,
    per_shot_estimates,
    probe_residuals,
    score_axis_assignment,
    score_prediction,
)

REPRESENTATIVE_SHOT = 11766
"""Registry selection whose configuration the refinement is authored against."""

HELD_OUT_FAMILY = "P1+P2+P3+P4+P5+P6"
"""The excitation family withheld from training in full.

Withholding the shots that drive every circuit at once is the hardest available
challenge and the one a wrong turn count fails most visibly: each coil's field
overlaps every other's, so an amplitude that was only ever checked against the
coil that produced it has nowhere to hide.  Training therefore sees individual
and few-circuit pulses only.
"""


def survey(arguments: argparse.Namespace) -> None:
    """Walk the shot store and cache what each shot drove and measured."""

    shots = store_shots(arguments.store)
    if arguments.limit:
        shots = shots[:: max(len(shots) // arguments.limit, 1)]
    surveys = survey_store(shots, store=arguments.store, processes=arguments.processes)
    payload = {
        "store": str(arguments.store),
        "shots_in_store": len(shots),
        "shots_read": len(surveys),
        "surveys": [row.as_dict() for row in surveys],
    }
    arguments.census.parent.mkdir(parents=True, exist_ok=True)
    arguments.census.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"surveyed {len(surveys)} of {len(shots)} shots -> {arguments.census}")


def load_surveys(path: Path) -> tuple[ShotSurvey, ...]:
    """Read a cached census back into survey records."""

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
        )
        for row in payload["surveys"]
    )


def archive_multipliers(surveys: Sequence[ShotSurvey]) -> dict[str, float]:
    """Return the archive's own ampere-turn multiplier per coil, where constant.

    A multiplier that varies across the store is not a statement about the coil
    and is dropped rather than averaged.
    """

    collected: dict[str, set[float]] = {}
    for row in surveys:
        for family, ratio in row.turn_multipliers.items():
            collected.setdefault(family, set()).add(round(ratio, 6))
    return {
        family: next(iter(values))
        for family, values in collected.items()
        if len(values) == 1
    }


def isolating_shots(
    shots: Sequence[int],
    surveys: Sequence[ShotSurvey],
) -> dict[str, list[tuple[float, int]]]:
    """Index the shots that drove one coil alone and held it, by coil.

    Two conditions have to hold together and neither is sufficient.  The coil's
    series partner must be quiet, or the pair's total is all that is measurable.
    And the excitation must be sustained, or the reading is the coil plus every
    induced current its own ramp created.  Shots are keyed by descending peak
    current so the strongest are taken first.
    """

    by_shot = {row.shot: row for row in surveys}
    candidates: dict[str, list[tuple[float, int]]] = {}
    for shot in shots:
        row = by_shot.get(shot)
        if row is None:
            continue
        isolated = set(row.asymmetric_coils()) & set(row.sustained_coils())
        for family in isolated:
            candidates.setdefault(family, []).append(
                (-row.coil_peaks.get(family, 0.0), shot)
            )
    for members in candidates.values():
        members.sort()
    return candidates


def choose_training_shots(
    cohort: VacuumCohort,
    surveys: Sequence[ShotSurvey],
    *,
    per_coil: int,
) -> tuple[int, ...]:
    """Pick the shots that isolate each coil, strongest excitation first.

    Selection is organised around coils rather than excitation families, because
    the turn count of a series pair is identifiable only from shots that drove one
    member alone.  Taking the strongest few per coil keeps the read cost bounded
    while giving every coil independent measurements, which is what makes the
    cross-shot spread a measurement of drift rather than an artefact.
    """

    candidates = isolating_shots(cohort.training, surveys)
    chosen: list[int] = []
    for family in (drive.family for drive in COIL_DRIVES):
        for _, shot in candidates.get(family, [])[:per_coil]:
            if shot not in chosen:
                chosen.append(shot)
    return tuple(sorted(chosen))


def choose_held_out_shots(
    cohort: VacuumCohort,
    surveys: Sequence[ShotSurvey],
    *,
    limit: int,
) -> tuple[int, ...]:
    """Pick held-out shots, sustained excitation first, families balanced.

    The held-out arm has to be readable on the same terms as the training arm or
    the challenge measures the window rather than the parameter, so a shot with a
    sustained coil is preferred over a stronger one that only pulsed.
    """

    by_shot = {row.shot: row for row in surveys}
    grouped: dict[str, list[tuple[int, float, int]]] = {}
    for shot in cohort.held_out:
        row = by_shot.get(shot)
        if row is None:
            continue
        peak = max(row.coil_peaks.values(), default=0.0)
        rank = 0 if row.sustained_coils() else 1
        grouped.setdefault(cohort.families[shot], []).append((rank, -peak, shot))
    chosen: list[int] = []
    families = sorted(grouped)
    for members in grouped.values():
        members.sort()
    while len(chosen) < limit:
        added = False
        for family in families:
            members = grouped[family]
            if members:
                chosen.append(members.pop(0)[2])
                added = True
                if len(chosen) >= limit:
                    break
        if not added:
            break
    return tuple(sorted(chosen))


def fit(arguments: argparse.Namespace) -> None:
    """Build the cohort, identify the response, and challenge it on held-out shots."""

    surveys = load_surveys(arguments.census)
    cohort = select_vacuum_cohort(
        surveys,
        held_out_families=arguments.held_out_families,
        held_out_fraction=arguments.held_out_fraction,
    )
    registry = MachineGeometryRegistry.default()
    selection = registry.select(REPRESENTATIVE_SHOT)
    geometry = selection.configuration.geometry
    probes = geometry["magnetics"]["poloidal_probes"]
    channels = probe_channels(probes)

    training = choose_training_shots(cohort, surveys, per_coil=arguments.per_coil)
    held_out = choose_held_out_shots(cohort, surveys, limit=arguments.held_out_shots)
    print(
        f"cohort {len(cohort.shots)} shots in {len(set(cohort.families.values()))} "
        f"families, {len(cohort.exclusions)} refused; reading "
        f"{len(training)} training and {len(held_out)} held-out shots"
    )

    train_waveforms = [
        read_shot_waveforms(shot, store=arguments.store) for shot in training
    ]
    test_waveforms = [
        read_shot_waveforms(shot, store=arguments.store) for shot in held_out
    ]

    candidates = [
        frozenset({"obr"}),
        frozenset({"obv"}),
        frozenset(),
        frozenset({"obr", "obv"}),
    ]
    axis_scores = score_axis_assignment(
        geometry,
        probes,
        channels,
        train_waveforms,
        candidates,
        stride=arguments.stride,
        minimum_standoff=arguments.standoff,
    )
    best = min(axis_scores, key=lambda row: row.residual_rms)
    runner_up = min(
        (row for row in axis_scores if row.radial_families != best.radial_families),
        key=lambda row: row.residual_rms,
    )
    print("sensitive-axis assignment:")
    for row in axis_scores:
        print(
            f"  radial={row.radial_families or ('none',)} "
            f"rms={row.residual_rms:.4e} T explained={row.variance_explained:.4f}"
        )
    print(
        f"  winner radial={best.radial_families} beats {runner_up.radial_families} "
        f"by {runner_up.residual_rms / best.residual_rms:.2f}x in residual"
    )

    model = ResponseModel.build(
        geometry,
        probes,
        channels,
        radial_families=frozenset(best.radial_families),
        minimum_standoff=arguments.standoff,
    )
    archive = archive_multipliers(surveys)
    estimates = per_shot_estimates(train_waveforms, model, stride=arguments.stride)
    dispositions = aggregate_turns(estimates, archive_multipliers=archive)

    print("signed turn counts:")
    for row in dispositions:
        if not row.identified:
            print(f"  {row.family:<16} unidentified by this cohort")
            continue
        note = (
            ""
            if row.archive_multiplier is None
            else f" archive={row.archive_multiplier:g}"
            f" {'agrees' if row.agrees_with_archive else 'DISAGREES'}"
        )
        print(
            f"  {row.family:<16} {row.multiplier:+9.3f} "
            f"[{row.interval[0]:+8.3f},{row.interval[1]:+8.3f}] "
            f"n={len(row.shots):<3d} spread={row.spread:.3f}{note}"
        )

    refined = {
        row.family: (
            float(row.nearest_integer) if row.resolves_an_integer else row.multiplier
        )
        for row in dispositions
        if row.identified
    }
    fitted = {row.family: row.multiplier for row in dispositions if row.identified}
    nominal = {family: 1.0 for family in model.families}

    scores: dict[str, Any] = {}
    for name, values in (
        ("nominal_unit_turns", nominal),
        ("fitted_multipliers", fitted),
        ("refined_integers", refined),
    ):
        try:
            score = score_prediction(
                test_waveforms, model, values, stride=arguments.stride
            )
        except Exception as error:  # noqa: BLE001 - a score may be unavailable
            print(f"held-out {name}: unavailable ({error})")
            continue
        scores[name] = score.as_dict()
        print(
            f"held-out {name:<20} rms={score.residual_rms:.4e} T "
            f"explained={score.variance_explained:+.4f} on {len(score.shots)} shots"
        )

    training_fit = fit_response(train_waveforms, model, stride=arguments.stride)
    residuals = {
        str(wave.shot): probe_residuals(wave, model, refined, stride=arguments.stride)
        for wave in test_waveforms
    }
    payload = {
        "archive_multipliers": archive,
        "axis_scores": [row.as_dict() for row in axis_scores],
        "cohort": {
            "exclusion_reasons": _reason_counts(cohort),
            "families": sorted(set(cohort.families.values())),
            "held_out_count": len(cohort.held_out),
            "held_out_families": list(cohort.held_out_families),
            "refused_count": len(cohort.exclusions),
            "size": len(cohort.shots),
            "training_count": len(cohort.training),
        },
        "held_out_scores": scores,
        "held_out_shots": list(held_out),
        "held_out_shot_families": {
            str(shot): cohort.families[shot] for shot in held_out
        },
        "joint_training_fit": training_fit.as_dict(),
        "minimum_standoff": arguments.standoff,
        "per_shot_estimates": [row.as_dict() for row in estimates],
        "physical_digest": selection.configuration.physical_digest,
        "probe_residuals": residuals,
        "radial_families": list(best.radial_families),
        "refined_integers": refined,
        "registry_digest": registry.registry_digest,
        "store": str(arguments.store),
        "training_shots": list(training),
        "training_shot_families": {
            str(shot): cohort.families[shot] for shot in training
        },
        "turn_dispositions": [row.as_dict() for row in dispositions],
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {arguments.report}")


def _reason_counts(cohort: VacuumCohort) -> dict[str, int]:
    """Group the cohort's refusals by the kind of reason that produced them."""

    counts: dict[str, int] = {}
    for row in cohort.exclusions:
        key = row.reason.split(":")[0].split(" reaches")[0]
        key = " ".join(word for word in key.split() if not word.isdigit())
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def passive(arguments: argparse.Namespace) -> None:
    """Decompose the free decays and report what the probes can separate."""

    surveys = load_surveys(arguments.census)
    cohort = select_vacuum_cohort(
        surveys, held_out_families=[HELD_OUT_FAMILY], held_out_fraction=0.2
    )
    registry = MachineGeometryRegistry.default()
    geometry = registry.select(REPRESENTATIVE_SHOT).configuration.geometry
    probes = geometry["magnetics"]["poloidal_probes"]
    channels = probe_channels(probes)
    model = ResponseModel.build(
        geometry,
        probes,
        channels,
        radial_families=frozenset({RADIAL_PROBE_FAMILY}),
    )
    groups = passive_groups(geometry)
    coupling = passive_coupling(groups, model.targets)

    by_shot = {row.shot: row for row in surveys}
    ranked = sorted(
        (shot for shot in cohort.shots if by_shot[shot].excited_families),
        key=lambda shot: -max(by_shot[shot].coil_peaks.values(), default=0.0),
    )[: arguments.shots]
    waveforms = [read_shot_waveforms(shot, store=arguments.store) for shot in ranked]
    results = survey_decays(waveforms, model.targets, groups, coupling)

    constants: list[float] = []
    shares: list[float] = []
    rows: list[dict[str, Any]] = []
    decisive = 0
    for spectrum, attributions in results:
        dominant = spectrum.modes[0] if spectrum.modes else None
        if dominant is None or not math.isfinite(dominant.time_constant):
            continue
        constants.append(dominant.time_constant)
        shares.append(dominant.signal_fraction)
        best = next(
            (row for row in attributions if row.mode_index == dominant.index), None
        )
        decisive += 1 if best is not None and best.decisive else 0
        rows.append(
            {
                "attribution": None if best is None else best.as_dict(),
                "spectrum": spectrum.as_dict(),
            }
        )
    if not constants:
        raise SystemExit("no shot in the cohort admitted a readable free decay")

    print(
        f"free decay on {len(rows)} shots: dominant pattern carries "
        f"{min(shares):.3f} to {max(shares):.3f} of the signal, decaying on "
        f"{min(constants) * 1e3:.0f} to {max(constants) * 1e3:.0f} ms"
    )
    print(
        f"one group explains the dominant pattern decisively on {decisive} of "
        f"{len(rows)} shots, so a resistance per family is not supported"
    )
    grouping = case_grouping(geometry)
    print(
        f"coil cases group into {len(grouping)} sets covering "
        f"{sum(len(v) for v in grouping.values())} plates"
    )
    payload = {
        "case_grouping": {name: len(rows_) for name, rows_ in grouping.items()},
        "decay_shots": list(ranked),
        "decisive_attributions": decisive,
        "dominant_share": [min(shares), max(shares)],
        "group_names": [group.name for group in groups],
        "results": rows,
        "time_constant_interval": [min(constants), max(constants)],
    }
    arguments.passive_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.passive_report.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n"
    )
    print(f"wrote {arguments.passive_report}")


FIGURE_DIRECTORY = Path("docs/figures/mast-source-resolution")
"""Where the refinement's comparison figures are written."""


def figures(arguments: argparse.Namespace) -> None:
    """Draw the comparisons the fit established, from the cached report."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(arguments.report.read_text())
    arguments.figures.mkdir(parents=True, exist_ok=True)
    _draw_turn_counts(plt, report, arguments.figures / "fitted_turn_counts.svg")
    _draw_held_out(plt, report, arguments.figures / "held_out_prediction.svg")
    _draw_axis_scores(plt, report, arguments.figures / "probe_axis_assignment.svg")
    print(f"wrote figures to {arguments.figures}")


def _draw_turn_counts(plt: Any, report: dict[str, Any], path: Path) -> None:
    """Show each coil's fitted turn count against the archive's own multiplier."""

    rows = [row for row in report["turn_dispositions"]]
    names = [row["family"] for row in rows]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    for index, row in enumerate(rows):
        if not row["identified"]:
            axes.text(
                index,
                0.5,
                "not\nidentified",
                ha="center",
                va="center",
                fontsize=8,
                color="firebrick",
            )
            continue
        value = row["multiplier"]
        low, high = row["interval"]
        counted = row["resolves_an_integer"]
        axes.errorbar(
            index,
            value,
            yerr=[[value - low], [high - value]],
            fmt="o" if counted else "s",
            color="tab:blue" if counted else "tab:orange",
            capsize=4,
            markersize=6,
        )
        if row["archive_multiplier"] is not None:
            axes.plot(
                index,
                row["archive_multiplier"],
                marker="_",
                markersize=16,
                color="black",
                linestyle="none",
            )
    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes.set_yscale("symlog", linthresh=10)
    axes.set_ylabel("turns per ampere of the measured channel")
    axes.set_title(
        "Signed turn count from the vacuum response\n"
        "circles counted to one integer, squares bounded only, "
        "black bars the archive's own multiplier"
    )
    axes.grid(alpha=0.3)
    figure.savefig(path)
    plt.close(figure)


def _draw_held_out(plt: Any, report: dict[str, Any], path: Path) -> None:
    """Show how much better the fitted response predicts withheld shots."""

    scores = report["held_out_scores"]
    order = ["nominal_unit_turns", "fitted_multipliers", "refined_integers"]
    labels = ["one turn per coil", "fitted multipliers", "authored values"]
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(10.0, 4.2), constrained_layout=True
    )
    values = [scores[key]["variance_explained"] for key in order if key in scores]
    left.bar(range(len(values)), values, color=["grey", "tab:blue", "tab:green"])
    left.set_xticks(range(len(values)))
    left.set_xticklabels(labels[: len(values)], rotation=15, ha="right", fontsize=9)
    left.set_ylabel("held-out variance explained")
    left.set_ylim(0.0, 1.0)
    left.grid(alpha=0.3, axis="y")
    left.set_title("Prediction on shots never fitted")

    per_shot = scores.get("fitted_multipliers", {}).get(
        "per_shot_variance_explained", {}
    )
    if per_shot:
        shots = sorted(per_shot, key=int)
        right.bar(range(len(shots)), [per_shot[s] for s in shots], color="tab:blue")
        right.set_xticks(range(len(shots)))
        right.set_xticklabels(shots, rotation=60, ha="right", fontsize=7)
        right.set_ylabel("variance explained")
        right.set_ylim(0.0, 1.0)
        right.grid(alpha=0.3, axis="y")
        right.set_title("Per held-out shot")
    figure.savefig(path)
    plt.close(figure)


def _draw_axis_scores(plt: Any, report: dict[str, Any], path: Path) -> None:
    """Show that one sensitive-axis assignment predicts the cohort best."""

    rows = report["axis_scores"]
    labels = ["+".join(row["radial_families"]) or "none radial" for row in rows]
    values = [row["residual_rms"] for row in rows]
    best = int(np.argmin(values))
    figure, axes = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    axes.bar(
        range(len(values)),
        values,
        color=["tab:green" if i == best else "grey" for i in range(len(values))],
    )
    axes.set_xticks(range(len(values)))
    axes.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes.set_ylabel("cohort residual [T]")
    axes.set_title(
        "Which probe families measure the radial component\n"
        "lower is better; the winner is not a preference but a refit"
    )
    axes.grid(alpha=0.3, axis="y")
    figure.savefig(path)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one refinement stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument(
        "--census",
        type=Path,
        default=Path.home() / ".cache/nova-mast/mast_vacuum_census.json",
        help="where the shot census is written and read",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path.home() / ".cache/nova-mast/mast_vacuum_refinement.json",
        help="where the fit report is written",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    census = stages.add_parser("survey", help="walk the store and cache the census")
    census.add_argument("--processes", type=int, default=8)
    census.add_argument("--limit", type=int, default=0)
    census.set_defaults(handler=survey)

    identify = stages.add_parser(
        "fit", help="build the cohort and identify the response"
    )
    identify.add_argument("--stride", type=int, default=4)
    identify.add_argument("--per-coil", type=int, default=4)
    identify.add_argument("--held-out-shots", type=int, default=12)
    identify.add_argument("--standoff", type=float, default=MINIMUM_STANDOFF)
    identify.add_argument("--integer-tolerance", type=float, default=0.5)
    identify.add_argument("--held-out-fraction", type=float, default=0.2)
    identify.add_argument("--held-out-families", nargs="*", default=[HELD_OUT_FAMILY])
    identify.set_defaults(handler=fit)

    decay = stages.add_parser("passive", help="decompose the free decays")
    decay.add_argument("--shots", type=int, default=12)
    decay.add_argument(
        "--passive-report",
        type=Path,
        default=Path.home() / ".cache/nova-mast/mast_passive_decay.json",
    )
    decay.set_defaults(handler=passive)

    drawing = stages.add_parser("figures", help="draw the comparisons from a report")
    drawing.add_argument("--figures", type=Path, default=FIGURE_DIRECTORY)
    drawing.set_defaults(handler=figures)

    arguments = parser.parse_args(argv)
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
