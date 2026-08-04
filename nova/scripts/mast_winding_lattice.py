"""Fit the turn layout inside the P4/P5 winding packs and challenge it out of sample.

Three stages, each writing what the next reads.  ``reduce`` reads the archive once
-- the sustained shots that drove one of these coils, screened for a quiescent
error-field bank and for the magnetics-amplitude defect the turn closure refused --
and reduces each to the sums a layout comparison needs.  ``fit`` scans grid shapes
and fill fractions against the training shots' near-field patterns, runs the
displaced-outline control, and then challenges the winner on shots it never saw.
``figures`` draws it.

The split is not decided here.  Which shots train and which are held out was
declared by the cohort classifier before any of this was written, and is read from
its record; this driver refuses a shot for exactly two measured reasons -- a driven
error-field channel, or the amplitude defect -- and both are recorded per shot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import shapely

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_vacuum_cohort import probe_channels, read_shot_waveforms
from nova.imas.mast_vacuum_response import ResponseModel, coil_sections
from nova.imas.mast_winding_lattice import (
    FILL_BOUNDS,
    calibrate_array,
    LATTICE_SHAPES,
    MINIMUM_PATTERN_PROBES,
    MINIMUM_TARGET_SHARE,
    NEAR_FIELD_STANDOFF,
    LatticeError,
    ShotMoments,
    TurnLattice,
    admissible_shapes,
    baseline_columns,
    channel_deltas,
    error_field_quiescent,
    fill_grid,
    lattice_column,
    passes_error_field_screen,
    pooled_residual,
    profile_displacement,
    profile_fill,
    reduce_shot,
    score_hypothesis,
    search_shapes,
    translated_section,
    uniform_column,
)

CACHE = Path.home() / ".cache" / "nova-mast"
"""Where the cohort's own records live and this one is written beside them."""

REPRESENTATIVE_SHOT = 11766
"""Registry selection whose configuration the layout is authored against."""

LATTICE_FAMILIES = ("p4_lower", "p4_upper", "p5_lower", "p5_upper")
"""Coils the archive publishes twenty-three turns for and both sources resolve."""


def _model() -> tuple[ResponseModel, Any, Mapping[str, Any]]:
    """Build the geometry-derived response once, and return its provenance."""

    registry = MachineGeometryRegistry.default()
    selection = registry.select(REPRESENTATIVE_SHOT)
    geometry = selection.configuration.geometry
    probes = geometry["magnetics"]["poloidal_probes"]
    model = ResponseModel.build(geometry, probes, probe_channels(probes))
    return model, selection, geometry


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")


def _load_cohort(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    cohort = payload["cohort"]
    alone = {
        entry["family"]: tuple(int(shot) for shot in entry["alone"])
        for entry in cohort["identifiability"]
    }
    return {
        "alone": alone,
        "held_out": frozenset(int(shot) for shot in cohort["held_out"]),
        "published": {
            family: float(turns)
            for family, turns in cohort["published_ampere_turn_ratios"].items()
        },
        "training": frozenset(int(shot) for shot in cohort["training"]),
    }


def _load_detail(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return {
        int(row["shot"]): {
            "absent_channels": tuple(row.get("absent_channels", ())),
            "error_field_peaks": {
                key: float(value)
                for key, value in row.get("error_field_peaks", {}).items()
            },
        }
        for row in payload["surveys"]
    }


def _load_scatter(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {
        str(row["channel"]): float(row["scatter"])
        for row in payload["envelope"]["channels"]
        if float(row["scatter"]) > 0.0
    }


def _load_weights(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {
        family: float(value) for family, value in payload["promoted_weights"].items()
    }


def _refused_on_amplitude(path: Path) -> frozenset[int]:
    payload = json.loads(path.read_text())
    return frozenset(int(shot) for shot in payload["refused_on_amplitude"])


def reduce_cohort(arguments: argparse.Namespace) -> None:
    """Read the layout cohort once and cache each shot's moments."""

    model, selection, geometry = _model()
    cohort = _load_cohort(arguments.cohort)
    detail = _load_detail(arguments.detail)
    scatter = _load_scatter(arguments.noise)
    weights = _load_weights(arguments.turns)
    refused = _refused_on_amplitude(arguments.turns)

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    screen: list[dict[str, Any]] = []
    for family in LATTICE_FAMILIES:
        for shot in cohort["alone"].get(family, ()):
            state = error_field_quiescent(
                detail.get(shot, {}).get("error_field_peaks", {}),
                detail.get(shot, {}).get("absent_channels", ()),
            )
            row: dict[str, Any] = {
                "error_field": state,
                "family": family,
                "shot": shot,
                "split": "held_out" if shot in cohort["held_out"] else "training",
            }
            if shot in refused:
                row["admitted"], row["reason"] = False, "magnetics amplitude refused"
            elif not passes_error_field_screen(state):
                row["admitted"], row["reason"] = False, "error-field bank not quiescent"
            else:
                row["admitted"], row["reason"] = True, ""
            screen.append(row)
            if not row["admitted"]:
                print(f"  refused {shot} {family}: {row['reason']}")
                continue
            try:
                waveforms = read_shot_waveforms(shot)
                moments = reduce_shot(
                    waveforms, model, family, weights, scatter, stride=arguments.stride
                )
            except (LatticeError, KeyError, OSError, ValueError) as error:
                row["admitted"], row["reason"] = False, f"unreadable: {error}"
                print(f"  refused {shot} {family}: {row['reason']}")
                continue
            key = f"{family}:{shot}"
            arrays[f"{key}:gram"] = moments.drive_gram
            arrays[f"{key}:moment"] = moments.probe_moment
            arrays[f"{key}:square"] = moments.probe_square
            arrays[f"{key}:rows"] = moments.rows
            arrays[f"{key}:standoff"] = moments.standoff
            arrays[f"{key}:screen"] = moments.screen_standoff
            arrays[f"{key}:used"] = moments.samples_used
            arrays[f"{key}:share"] = moments.target_share
            arrays[f"{key}:scatter"] = moments.scatter
            records.append(
                {
                    "channels": list(moments.channels),
                    "families": list(moments.families),
                    "family": family,
                    "near_field_probes": int(moments.near_field().sum()),
                    "sample_count": moments.sample_count,
                    "shot": shot,
                    "split": row["split"],
                }
            )
            print(
                f"  read {shot} {family}: {len(moments.channels)} probes, "
                f"{int(moments.near_field().sum())} in the near field"
            )

    arguments.moments.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.moments, **arrays)
    print(f"wrote {arguments.moments}")
    _write(
        arguments.output,
        {
            "families": list(LATTICE_FAMILIES),
            "minimum_target_share": MINIMUM_TARGET_SHARE,
            "near_field_standoff": NEAR_FIELD_STANDOFF,
            "physical_digest": selection.configuration.physical_digest,
            "promoted_weights": weights,
            "screen": screen,
            "shots": records,
            "stride": arguments.stride,
        },
    )


def _read_moments(path: Path, index: Sequence[Mapping[str, Any]]) -> list[ShotMoments]:
    with np.load(path) as payload:
        return [
            ShotMoments(
                shot=int(row["shot"]),
                family=str(row["family"]),
                families=tuple(row["families"]),
                channels=tuple(row["channels"]),
                rows=payload[f"{row['family']}:{row['shot']}:rows"],
                drive_gram=payload[f"{row['family']}:{row['shot']}:gram"],
                probe_moment=payload[f"{row['family']}:{row['shot']}:moment"],
                probe_square=payload[f"{row['family']}:{row['shot']}:square"],
                standoff=payload[f"{row['family']}:{row['shot']}:standoff"],
                screen_standoff=payload[f"{row['family']}:{row['shot']}:screen"],
                target_share=payload[f"{row['family']}:{row['shot']}:share"],
                samples_used=payload[f"{row['family']}:{row['shot']}:used"],
                sample_count=int(row["sample_count"]),
                scatter=payload[f"{row['family']}:{row['shot']}:scatter"],
            )
            for row in index
        ]


def _scored(
    moments: Sequence[ShotMoments],
    columns: Mapping[str, np.ndarray],
    *,
    probes: str,
    gains: Mapping[str, float] | None = None,
    amplitudes: Mapping[int, float] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Pool a set of shots under one hypothesis on one probe selection."""

    scores = []
    rows = []
    for record in moments:
        mask = (
            record.near_field()
            if probes == "near"
            else record.far_field()
            if probes == "far"
            else np.ones(len(record.channels), dtype=bool)
        )
        if int(mask.sum()) < (MINIMUM_PATTERN_PROBES if probes == "near" else 1):
            continue
        score = score_hypothesis(
            record,
            columns,
            select=mask,
            gains=gains,
            amplitude=None if amplitudes is None else amplitudes.get(record.shot),
        )
        scores.append(score)
        rows.append(
            {
                "amplitude": score.amplitude,
                "probe_count": score.probe_count,
                "residual": score.residual,
                "shot": score.shot,
                "signal": score.signal,
            }
        )
    if not scores:
        raise LatticeError(f"no shot keeps a {probes}-field pattern")
    return pooled_residual(scores), rows


def fit_lattice(arguments: argparse.Namespace) -> None:
    """Scan the layout, run the control, then challenge the winner out of sample."""

    model, selection, geometry = _model()
    cohort = _load_cohort(arguments.cohort)
    index = json.loads(arguments.reduced.read_text())["shots"]
    moments = _read_moments(arguments.moments, index)
    sections = coil_sections(geometry)
    baseline = baseline_columns(model, geometry)
    targets = model.targets

    calibration = calibrate_array(moments, baseline)
    gains = calibration.constrained(arguments.minimum_gain_shots)
    amplitudes = calibration.amplitudes
    print(
        f"array calibration: {calibration.iterations} sweeps, far-field whitened "
        f"residual {calibration.residual:.4g}, "
        f"{sum(1 for value in gains.values() if abs(value - 1.0) > 0.05)} of "
        f"{len(gains)} channels off unity by more than five percent"
    )

    families: dict[str, Any] = {}
    for family in LATTICE_FAMILIES:
        outline = sections[family][0]
        turns = cohort["published"][family]
        offered = admissible_shapes(outline, int(turns))
        searched = search_shapes(outline, int(turns))
        train = [
            record
            for record in moments
            if record.family == family and record.shot in cohort["training"]
        ]
        held = [
            record
            for record in moments
            if record.family == family and record.shot in cohort["held_out"]
        ]
        if not train:
            print(f"{family}: no training shot survived the screen")
            continue

        profiles = {}
        for shape in offered:
            profiles[shape] = profile_fill(
                train,
                targets,
                outline,
                baseline,
                shape=shape,
                fills=fill_grid(
                    bounds=tuple(arguments.fill_bounds), step=arguments.step
                ),
                gains=gains,
                amplitudes=amplitudes,
            )
        chosen = min(profiles, key=lambda shape: profiles[shape].residual)
        profile = profiles[chosen]
        lattice = TurnLattice(chosen[0], chosen[1], profile.fill)
        proposed = dict(baseline)
        proposed[family] = lattice_column(targets, outline, lattice)

        centroid = lattice.centroid(outline)
        polygon = shapely.Polygon(outline).centroid
        offset = (centroid[0] - polygon.x, centroid[1] - polygon.y)
        control = dict(baseline)
        control[family] = uniform_column(targets, translated_section(outline, offset))

        reach = profile_displacement(
            train,
            targets,
            outline,
            baseline,
            reach=arguments.reach,
            steps=arguments.reach_steps,
            gains=gains,
            amplitudes=amplitudes,
        )
        held_reach = None
        if held:
            try:
                held_reach = profile_displacement(
                    held,
                    targets,
                    outline,
                    baseline,
                    reach=arguments.reach,
                    steps=arguments.reach_steps,
                    gains=gains,
                    amplitudes=amplitudes,
                )
            except LatticeError as error:
                print(f"  {family}: no held-out displacement scan ({error})")

        report: dict[str, Any] = {
            "centroid": {"lattice": list(centroid), "outline": [polygon.x, polygon.y]},
            "displacement": {
                "held_out": None if held_reach is None else held_reach.as_dict(),
                "training": reach.as_dict(),
            },
            "displacement_mm": [1.0e3 * offset[0], 1.0e3 * offset[1]],
            "held_out_shots": [record.shot for record in held],
            "profile": profile.as_dict(),
            "published_turns": turns,
            "shape_profiles": {
                f"{shape[0]}x{shape[1]}": value.as_dict()
                for shape, value in profiles.items()
            },
            "shapes_offered": [list(shape) for shape in offered],
            "shapes_searched": [list(shape) for shape in searched],
            "training_shots": [record.shot for record in train],
            "turn_count_derived": lattice.turn_count(outline),
        }
        for label, group in (("training", train), ("held_out", held)):
            if not group:
                continue
            entry: dict[str, Any] = {}
            for probes in ("near", "far", "all"):
                try:
                    uniform_residual, uniform_rows = _scored(
                        group,
                        baseline,
                        probes=probes,
                        gains=gains,
                        amplitudes=amplitudes,
                    )
                    lattice_residual, lattice_rows = _scored(
                        group,
                        proposed,
                        probes=probes,
                        gains=gains,
                        amplitudes=amplitudes,
                    )
                    control_residual, _ = _scored(
                        group,
                        control,
                        probes=probes,
                        gains=gains,
                        amplitudes=amplitudes,
                    )
                    raw_uniform, _ = _scored(group, baseline, probes=probes)
                    raw_lattice, _ = _scored(group, proposed, probes=probes)
                except LatticeError as error:
                    entry[probes] = {"unavailable": str(error)}
                    continue
                entry[probes] = {
                    "control": control_residual,
                    "lattice": lattice_residual,
                    "lattice_gain": 1.0 - lattice_residual / uniform_residual,
                    "control_gain": 1.0 - control_residual / uniform_residual,
                    "per_shot": {
                        "lattice": lattice_rows,
                        "uniform": uniform_rows,
                    },
                    "uncalibrated": {
                        "lattice": raw_lattice,
                        "lattice_gain": 1.0 - raw_lattice / raw_uniform,
                        "uniform": raw_uniform,
                    },
                    "uniform": uniform_residual,
                }
            entry["channels"] = {
                probes: channel_deltas(
                    group,
                    baseline,
                    proposed,
                    select=probes,
                    gains=gains,
                    amplitudes=amplitudes,
                )
                for probes in ("near", "all")
            }
            report[label] = entry
        families[family] = report
        near = report.get("held_out", {}).get("near", {})
        print(
            f"{family}: shape {chosen[0]}x{chosen[1]} fill {profile.fill:.3f} "
            f"[{profile.interval[0]:.3f}, {profile.interval[1]:.3f}] "
            f"train near gain {report['training']['near']['lattice_gain']:+.4f} "
            f"held-out near gain {near.get('lattice_gain', float('nan')):+.4f} "
            f"| best rigid shift {reach.offset[0] * 1e3:+.1f}, "
            f"{reach.offset[1] * 1e3:+.1f} mm for {reach.improvement:+.4f}"
        )

    _write(
        arguments.output,
        {
            "calibration": calibration.as_dict(),
            "families": families,
            "fill_bounds": list(arguments.fill_bounds),
            "minimum_gain_shots": arguments.minimum_gain_shots,
            "minimum_target_share": MINIMUM_TARGET_SHARE,
            "near_field_standoff": NEAR_FIELD_STANDOFF,
            "physical_digest": selection.configuration.physical_digest,
            "shapes_offered": [list(shape) for shape in LATTICE_SHAPES],
        },
    )


def draw_figures(arguments: argparse.Namespace) -> None:
    """Draw the layout, the fill profile and the per-channel held-out deltas."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, _, geometry = _model()
    sections = coil_sections(geometry)
    report = json.loads(arguments.fitted.read_text())
    directory = arguments.figures
    directory.mkdir(parents=True, exist_ok=True)

    families = [name for name in LATTICE_FAMILIES if name in report["families"]]

    figure, axes = plt.subplots(1, len(families), figsize=(4.0 * len(families), 4.6))
    axes = np.atleast_1d(axes)
    for axis, family in zip(axes, families, strict=True):
        entry = report["families"][family]
        outline = sections[family][0]
        shape = tuple(int(value) for value in entry["profile"]["shape"])
        lattice = TurnLattice(shape[0], shape[1], float(entry["profile"]["fill"]))
        closed = np.vstack([outline, outline[:1]])
        axis.plot(
            closed[:, 0], closed[:, 1], color="#333", lw=1.4, label="pack outline"
        )
        grid_r, grid_z = lattice.grid(outline)
        keep = lattice.occupied(outline)
        for vertices in lattice.sections(outline):
            box = np.vstack([vertices, vertices[:1]])
            axis.plot(box[:, 0], box[:, 1], color="#1a6b3c", lw=0.8)
        axis.plot(
            grid_r[~keep], grid_z[~keep], "x", color="#b04a4a", ms=9, label="cross-over"
        )
        axis.plot(*entry["centroid"]["outline"], "o", color="#b04a4a", ms=5)
        axis.plot(*entry["centroid"]["lattice"], "o", color="#1a6b3c", ms=5)
        axis.set_title(
            f"{family}  {shape[0]}x{shape[1]}, fill {lattice.fill:.3f}\n"
            f"{lattice.turn_count(outline)} turns, centroid "
            f"{entry['displacement_mm'][0]:+.2f}, {entry['displacement_mm'][1]:+.2f} mm"
        )
        axis.set_aspect("equal")
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.legend(fontsize=7, loc="upper left")
    figure.suptitle(
        "Turn layout the chamfer derives: one grid position vacated, "
        "twenty-three turns left"
    )
    figure.tight_layout()
    path = directory / "lattice_layout.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path}")

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for family in families:
        entry = report["families"][family]["profile"]
        fills = np.asarray([float(key) for key in entry["profile"]], dtype=float)
        values = np.asarray(list(entry["profile"].values()), dtype=float)
        order = np.argsort(fills)
        axes[0].plot(
            fills[order],
            values[order] / entry["uniform_residual"],
            label=f"{family} ({entry['shape'][0]}x{entry['shape'][1]})",
        )
    axes[0].axhline(1.0, color="#b04a4a", ls="--", lw=1.0, label="uniform density")
    axes[0].set_xlabel("fill fraction (turn span / pack extent)")
    axes[0].set_ylabel("near-field residual / uniform")
    axes[0].set_title("What the near probes prefer")
    axes[0].legend(fontsize=7)

    labels, train_gain, held_gain = [], [], []
    for family in families:
        entry = report["families"][family]
        labels.append(family)
        train_gain.append(100.0 * entry["training"]["near"]["lattice_gain"])
        held = entry.get("held_out", {}).get("near", {})
        held_gain.append(100.0 * held.get("lattice_gain", np.nan))
    position = np.arange(len(labels))
    axes[1].bar(position - 0.18, train_gain, 0.34, label="training", color="#5a7fa8")
    axes[1].bar(position + 0.18, held_gain, 0.34, label="held out", color="#1a6b3c")
    axes[1].axhline(0.0, color="#333", lw=1.0)
    axes[1].set_xticks(position)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("near-field residual removed [%]")
    axes[1].set_title("Layout gain, in sample and out")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    path = directory / "lattice_fill_profile.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path}")

    figure, axes = plt.subplots(
        len(families), 1, figsize=(11.0, 2.9 * len(families)), sharex=False
    )
    axes = np.atleast_1d(axes)
    for axis, family in zip(axes, families, strict=True):
        entry = report["families"][family]
        channels = entry.get("held_out", {}).get("channels", {}).get("all", {})
        if not channels:
            channels = entry["training"]["channels"]["all"]
            axis.set_title(f"{family} — training (no held-out shot survived)")
        else:
            axis.set_title(f"{family} — held out")
        names = sorted(channels, key=lambda name: channels[name]["delta"])
        delta = [1.0e3 * channels[name]["delta"] for name in names]
        colour = ["#1a6b3c" if value <= 0 else "#b04a4a" for value in delta]
        axis.bar(np.arange(len(names)), delta, color=colour)
        axis.axhline(0.0, color="#333", lw=0.8)
        axis.set_xticks(np.arange(len(names)))
        axis.set_xticklabels(names, rotation=90, fontsize=5.5)
        axis.set_ylabel("Δ residual [mT]")
    figure.suptitle(
        "Per-channel residual change from the fitted layout "
        "(green removes misfit, red adds it)"
    )
    figure.tight_layout()
    path = directory / "lattice_channel_deltas.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path}")

    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    targets = model.targets
    for family in families:
        outline = sections[family][0]
        entry = report["families"][family]
        shape = tuple(int(value) for value in entry["profile"]["shape"])
        column = model.response[:, model.families.index(family)]
        lattice = lattice_column(
            targets, outline, TurnLattice(shape[0], shape[1], entry["profile"]["fill"])
        )
        standoff = model.standoff[:, model.families.index(family)]
        keep = np.abs(column) > 1.0e-8
        axis.semilogx(
            standoff[keep],
            1.0e2 * (lattice[keep] - column[keep]) / np.abs(column[keep]),
            "o",
            ms=4,
            label=family,
        )
    axis.axvline(NEAR_FIELD_STANDOFF, color="#b04a4a", ls="--", lw=1.0)
    axis.text(
        NEAR_FIELD_STANDOFF * 1.05,
        axis.get_ylim()[1] * 0.85,
        "turn-fit exclusion",
        fontsize=8,
        color="#b04a4a",
    )
    axis.axhline(0.0, color="#333", lw=0.8)
    axis.set_xlabel("probe standoff [pack widths]")
    axis.set_ylabel("layout − uniform [% of field]")
    axis.set_title(
        "Where a turn layout is visible: the signature dies outside two pack widths"
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = directory / "lattice_near_probe_sensitivity.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run one stage of the winding-layout measurement."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    reduce_parser = sub.add_parser("reduce", help="read the cohort and cache moments")
    reduce_parser.add_argument(
        "--cohort", type=Path, default=CACHE / "mast_calibration_cohort.json"
    )
    reduce_parser.add_argument(
        "--detail", type=Path, default=CACHE / "mast_excitation_detail.json"
    )
    reduce_parser.add_argument(
        "--noise", type=Path, default=CACHE / "mast_sensor_noise.json"
    )
    reduce_parser.add_argument(
        "--turns", type=Path, default=CACHE / "mast_turn_closure.json"
    )
    reduce_parser.add_argument(
        "--moments", type=Path, default=CACHE / "mast_winding_moments.npz"
    )
    reduce_parser.add_argument(
        "--output", type=Path, default=CACHE / "mast_winding_reduced.json"
    )
    reduce_parser.add_argument("--stride", type=int, default=5)
    reduce_parser.set_defaults(handler=reduce_cohort)

    fit_parser = sub.add_parser("fit", help="scan the layout and challenge it")
    fit_parser.add_argument(
        "--cohort", type=Path, default=CACHE / "mast_calibration_cohort.json"
    )
    fit_parser.add_argument(
        "--reduced", type=Path, default=CACHE / "mast_winding_reduced.json"
    )
    fit_parser.add_argument(
        "--moments", type=Path, default=CACHE / "mast_winding_moments.npz"
    )
    fit_parser.add_argument(
        "--output", type=Path, default=CACHE / "mast_winding_lattice.json"
    )
    fit_parser.add_argument(
        "--fill-bounds", type=float, nargs=2, default=list(FILL_BOUNDS)
    )
    fit_parser.add_argument("--step", type=float, default=0.005)
    fit_parser.add_argument("--reach", type=float, default=5.0e-3)
    fit_parser.add_argument("--reach-steps", type=int, default=11)
    fit_parser.add_argument("--minimum-gain-shots", type=int, default=2)
    fit_parser.set_defaults(handler=fit_lattice)

    figure_parser = sub.add_parser("figures", help="draw the layout and its scores")
    figure_parser.add_argument(
        "--fitted", type=Path, default=CACHE / "mast_winding_lattice.json"
    )
    figure_parser.add_argument(
        "--figures", type=Path, default=Path("docs/figures/mast-vacuum-floor")
    )
    figure_parser.set_defaults(handler=draw_figures)

    arguments = parser.parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
