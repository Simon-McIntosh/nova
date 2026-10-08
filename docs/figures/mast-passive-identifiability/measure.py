"""Measure the nominal driven passive spectrum on the recorded vacuum cohort."""

from __future__ import annotations

import argparse
from dataclasses import fields
from collections import Counter
import hashlib
import json
import multiprocessing
import pickle
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import eigh

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_error_field_screen import read_error_field_drive
from nova.scripts.mast_passive_calibration import load_screen
from nova.imas.mast_fitted_parameters import (
    MIS_SCALED_SHOTS,
    SENSOR_FLOOR,
    fitted_turns,
)
from nova.imas.mast_passive_decay_modes import (
    grouped_driven_jacobian,
    projected_spectrum,
)
from nova.imas.mast_passive_inductance import (
    coil_coupling,
    linkage_matrix,
    nominal_resistance,
    passive_turns,
    probe_coupling,
)
from nova.imas.mast_seed_parameters import passive_material
from nova.imas.mast_vacuum_cohort import (
    COIL_DRIVES,
    EXCITATION_CURRENT,
    SHOT_STORE,
    ShotSurvey,
    probe_channels,
    read_shot_waveforms,
    select_vacuum_cohort,
)
from nova.imas.mast_vacuum_response import ResponseModel, coil_sections


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(configuration):
    geometry = configuration.geometry
    families = tuple(sorted(drive.family for drive in COIL_DRIVES))
    probes = geometry["magnetics"]["poloidal_probes"]
    model = ResponseModel.build(
        geometry, probes, probe_channels(probes), families=families
    )
    all_turns = passive_turns(geometry)
    turns = tuple(
        turn for turn in all_turns if passive_material(turn.family) is not None
    )
    linkage = linkage_matrix(turns)
    resistance = nominal_resistance(turns)
    coupling = probe_coupling(turns, model.targets)
    # Match the direct-field owner's area-weighted ampere-turn convention.
    import shapely

    mutual = np.zeros((len(turns), len(families)))
    for column, family in enumerate(families):
        parts = coil_sections(geometry)[family]
        areas = np.array([shapely.Polygon(part).area for part in parts])
        for part, weight in zip(parts, areas / areas.sum(), strict=True):
            _, single = coil_coupling({family: [part]}, turns)
            mutual[:, column] += weight * single[:, 0]
    groups = tuple(turn.family for turn in turns)
    bounds = {}
    materials = {}
    for name in sorted(set(groups)):
        material = passive_material(name)
        bounds[name] = (
            material.resistivity_lower / material.resistivity,
            material.resistivity_upper / material.resistivity,
        )
        materials[name] = {
            "nominal_ohm_m": material.resistivity,
            "interval_ohm_m": [material.resistivity_lower, material.resistivity_upper],
            "circuit_count": groups.count(name),
        }
    scales, intervals, drive_records = [], [], {}
    for family in families:
        row = fitted_turns(family)
        if row.identified:
            ratio = row.turns_per_multiplier
            centre = row.turns / ratio
            lower, upper = row.interval.lower / ratio, row.interval.upper / ratio
            # An exact corroborating integer does not fix an uncertain drive.
            radius = max(centre - lower, upper - centre, abs(centre) * 0.01)
            lower, upper = centre - radius, centre + radius
            source = (
                "fitted_turns.interval in channel-current units; "
                "minimum 1 percent uncertainty"
            )
        else:
            centre, lower, upper = 1.0, 0.1, 10.0
            source = (
                "declared decade bracket around published ampere-turn "
                "interpretation; unsourced sensitivity assumption"
            )
        scales.append(centre)
        intervals.append((lower, upper))
        drive_records[family] = {
            "seed": centre,
            "interval": [lower, upper],
            "basis": source,
        }
    metadata = {
        "physical_digest": configuration.physical_digest,
        "circuits": [turn.name for turn in turns],
        "groups": materials,
        "rodgr": {
            "treatment": "excluded from circuit system and parameter count",
            "reason": (
                "seed owner assigns no material; copper versus vessel steel unresolved"
            ),
            "excluded_circuits": [
                turn.name for turn in all_turns if passive_material(turn.family) is None
            ],
        },
        "drive_scales_ampere_turn_per_channel_ampere": drive_records,
        "reciprocity_residual": linkage.reciprocity_residual,
        "minimum_decay_rate_per_second": float(
            eigh(np.diag(resistance), linkage.matrix, eigvals_only=True).min()
        ),
        "minimum_resistance_ohm": float(resistance.min()),
        "coil_linkage_convention": (
            "area-weighted winding-pack current, matching coil_response_matrix"
        ),
    }
    return (
        model,
        linkage.matrix,
        resistance,
        coupling,
        mutual,
        groups,
        bounds,
        np.array(scales),
        np.array(intervals),
        metadata,
    )


def shot_jacobian(shot, prepared, step, screen):
    (
        model,
        inductance,
        resistance,
        coupling,
        mutual,
        groups,
        bounds,
        scales,
        intervals,
        _,
    ) = prepared
    error_drive = read_error_field_drive(shot, store=SHOT_STORE)
    if error_drive.unmeasured:
        raise ValueError("error-field drive unmeasured")
    wave = read_shot_waveforms(shot)
    driven = [
        family
        for family in model.families
        if np.nanmax(np.abs(wave.drives[family])) >= EXCITATION_CURRENT
    ]
    indices = np.flatnonzero(wave.sample_mask)
    if indices.size < 8 or np.any(np.diff(indices) != 1):
        raise ValueError("vacuum clock has gaps or fewer than eight valid samples")
    # Require a measured quiet prefix so the zero initial passive state has support.
    start, stop = indices[0], indices[-1]
    if not np.all(wave.baseline_mask[start : start + 8]):
        raise ValueError(
            "no eight-sample quiet prefix for zero initial passive current"
        )
    time = np.linspace(
        wave.time[start],
        wave.time[stop],
        int(np.ceil((wave.time[stop] - wave.time[start]) / step)) + 1,
    )
    currents = np.column_stack(
        [np.interp(time, wave.time, wave.drives[name]) for name in model.families]
    )
    keep = model.admissible_probes(driven)
    # These channels carry unresolved pair failures or a documented half gain.
    refused = {"obv03", "obr05", "obr18", "obv10", "obr17"}
    refused.update(screen.refused(error_drive))
    selected = [
        i
        for i, target in enumerate(model.targets)
        if keep[i]
        and target.channel not in refused
        and target.channel in wave.probes
        and np.count_nonzero(np.isfinite(wave.probes[target.channel][indices])) >= 8
    ]
    if len(selected) < 8:
        raise ValueError("fewer than eight finite, far-field, non-refused probes")
    names, passive, nuisance = grouped_driven_jacobian(
        inductance,
        resistance,
        coupling[selected],
        mutual,
        model.response[selected],
        time,
        currents,
        groups,
        bounds,
        scales,
        intervals,
    )
    quiet = np.interp(time, wave.time, wave.baseline_mask.astype(float)) == 1.0
    # Score the drive and the following switch-off tail, not the quiet prefix.
    active = np.any(np.abs(currents) >= EXCITATION_CURRENT, axis=1)
    occupied = np.flatnonzero(active)
    if not occupied.size:
        raise ValueError("resampled drives have no deliberate excitation")
    score = (time >= time[occupied[0]]) & (time <= time[occupied[-1]] + 0.1)
    supported = np.column_stack(
        [
            np.interp(
                time,
                wave.time,
                np.isfinite(wave.probes[model.targets[i].channel]).astype(float),
            )
            == 1.0
            for i in selected
        ]
    )
    scored = supported & score[:, None]
    for column in range(len(selected)):
        baseline = supported[:, column] & quiet
        if np.count_nonzero(baseline) < 8 or np.count_nonzero(scored[:, column]) < 8:
            scored[:, column] = False
            continue
        passive[:, column] -= passive[baseline, column].mean(axis=0)
        nuisance[:, column] -= nuisance[baseline, column].mean(axis=0)
    admitted_columns = np.flatnonzero(scored.sum(axis=0) >= 8)
    if admitted_columns.size < 8:
        raise ValueError("fewer than eight probes with baseline and transient coverage")
    matrix = (
        np.column_stack(
            [
                passive[scored],
                nuisance[scored],
            ]
        )
        / SENSOR_FLOOR
    )
    compact = np.linalg.qr(matrix, mode="r")
    row = {
        "shot": shot,
        "channels": [model.targets[selected[i]].channel for i in admitted_columns],
        "channel_sample_counts": scored.sum(axis=0)[admitted_columns].tolist(),
        "sample_count": int(score.sum()),
        "observation_count": matrix.shape[0],
        "window_seconds": [float(time[score][0]), float(time[score][-1])],
        "integration_step_seconds": float(time[1] - time[0]),
        "drive_peak_ampere": float(np.max(np.abs(currents))),
        "derivative_norm_floor_units": float(np.linalg.norm(matrix[:, : len(names)])),
        "source_identities": sorted({p.group_identity for p in wave.provenance}),
        "physical_digest": prepared[-1]["physical_digest"],
    }
    return names, compact, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("step must be positive")
    payload = json.loads(args.census.read_text())
    keys = {field.name for field in fields(ShotSurvey)}
    surveys = [
        ShotSurvey(**{k: v for k, v in row.items() if k in keys})
        for row in payload["surveys"]
    ]
    cohort = select_vacuum_cohort(surveys, held_out_families=("P1+P2+P3+P4+P5+P6",))
    screen = load_screen(args.screen)
    registry = MachineGeometryRegistry.default()
    cache, records, refused = {}, [], []
    refinements = []
    compact, names, observations = None, None, 0
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    args.progress.write_text("")
    print(f"COHORT selected={len(cohort.shots)}", flush=True)
    for shot in cohort.shots:
        if shot in MIS_SCALED_SHOTS:
            refused.append({"shot": shot, "reason": "acquisition amplitude refusal"})
            continue
        configuration = registry.select(shot).configuration
        key = configuration.physical_digest
        if key not in cache:
            print(f"GEOMETRY building={key}", flush=True)
            import nova.imas.mast_passive_inductance as owner
            from nova.biot.polygon import polygon_greens

            with multiprocessing.get_context("fork").Pool(args.processes) as pool:

                def blocked_greens(radius, height, vertices):
                    if radius.size < 1024:
                        return polygon_greens(radius, height, vertices)
                    blocks = pool.starmap(
                        polygon_greens,
                        [
                            (r, z, vertices)
                            for r, z in zip(
                                np.array_split(radius, args.processes),
                                np.array_split(height, args.processes),
                                strict=True,
                            )
                        ],
                    )
                    return tuple(
                        np.concatenate([block[i] for block in blocks]) for i in range(3)
                    )

                owner.polygon_greens = blocked_greens
                try:
                    cache[key] = prepare(configuration)
                finally:
                    owner.polygon_greens = polygon_greens
            cache_path = args.progress.parent / ("geometry-" + key + ".pickle")
            cache_path.write_bytes(pickle.dumps(cache[key]))
            print(f"GEOMETRY ready={key} cache={cache_path}", flush=True)
        try:
            group_names, block, row = shot_jacobian(shot, cache[key], args.step, screen)
        except (ValueError, KeyError, FileNotFoundError) as error:
            refused.append({"shot": shot, "reason": str(error)})
            print(f"REFUSED shot={shot} reason={error}", flush=True)
            continue
        if names is not None and names != group_names:
            raise ValueError("component families differ across geometry epochs")
        if len(refinements) < 3:
            _, fine_block, fine_row = shot_jacobian(
                shot, cache[key], args.step / 2.0, screen
            )
            coarse_spectrum = projected_spectrum(
                block[:, : len(group_names)],
                block[:, len(group_names) :],
                observation_count=row["observation_count"],
            ).singular_values
            fine_spectrum = projected_spectrum(
                fine_block[:, : len(group_names)],
                fine_block[:, len(group_names) :],
                observation_count=fine_row["observation_count"],
            ).singular_values
            refinements.append(
                {
                    "shot": shot,
                    "coarse_step_seconds": args.step,
                    "fine_step_seconds": args.step / 2.0,
                    "coarse_spectrum": coarse_spectrum.tolist(),
                    "fine_spectrum": fine_spectrum.tolist(),
                    "relative_spectrum_norm_change": float(
                        np.linalg.norm(fine_spectrum - coarse_spectrum)
                        / max(np.linalg.norm(fine_spectrum), np.finfo(float).tiny)
                    ),
                }
            )
        names = group_names
        compact = (
            block
            if compact is None
            else np.linalg.qr(np.vstack([compact, block]), mode="r")
        )
        observations += row["observation_count"]
        records.append(row)
        with args.progress.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(
            f"SHOT {shot} probes={len(row['channels'])} samples={row['sample_count']}",
            flush=True,
        )
    if not records:
        raise ValueError(
            "no shots admitted; no absence or spectrum conclusion is licensed"
        )
    spectrum = projected_spectrum(
        compact[:, : len(names)],
        compact[:, len(names) :],
        observation_count=observations,
    )
    receipt = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source": "nova/imas/mast_passive_decay_modes.py",
        "source_sha256": digest("nova/imas/mast_passive_decay_modes.py"),
        "census_path": str(args.census),
        "census_sha256": digest(args.census),
        "error_field_screen_sha256": digest(args.screen),
        "error_field_screen_path": str(args.screen),
        "store": str(SHOT_STORE),
        "registry_digest": registry.registry_digest,
        "cohort": {
            "selected_count": len(cohort.shots),
            "training": list(cohort.training),
            "held_out": list(cohort.held_out),
            "held_out_families": list(cohort.held_out_families),
            "exclusion_reason_counts": dict(
                Counter(
                    row.reason.split(" reaches")[0].split(" A")[0]
                    if not row.reason.startswith("plasma current")
                    else "plasma current exceeds vacuum threshold"
                    for row in cohort.exclusions
                )
            ),
        },
        "admitted_shots": [
            {
                key: value
                for key, value in row.items()
                if key not in ("channels", "source_identities", "channel_sample_counts")
            }
            for row in records
        ],
        "shot_detail_path": str(args.progress),
        "shot_detail_sha256": digest(args.progress),
        "refused_shots": refused,
        "geometries": {key: value[-1] for key, value in cache.items()},
        "sensor_floor_tesla": SENSOR_FLOOR,
        "observation_count": observations,
        "parameter_groups": names,
        "nuisance_rank": spectrum.nuisance_rank,
        "singular_values_floor_units": spectrum.singular_values.tolist(),
        "fixed_drive_singular_values_floor_units": (
            spectrum.fixed_drive_singular_values.tolist()
        ),
        "right_singular_directions": spectrum.directions.tolist(),
        "identifiable_direction_count": spectrum.identifiable_count,
        "promoted_parameters": [],
        "integration_refinement": refinements,
        "interpretation": (
            "Local interval-scaled RMS sensitivity at nominal seeds "
            "with every coil scale projected out without penalty. Directions are "
            "group combinations, not individually promoted groups. "
            "No held-out fit or global interval certification."
        ),
        "limitations": [
            "rodgr circuits excluded; result conditional on that reduced circuit model",
            "vertical drive scales use an explicitly assumed decade interval",
            "uniform linear current interpolation; refinement recorded separately",
            "each probe is scored only on finite interpolation-supported samples",
            "admitted channels use the recorded pooled systematic sensor floor",
            "zero initial passive current; no unmeasured prior history inferred",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(receipt, indent=2) + "\n"
    if len(encoded.encode()) > 300_000:
        raise ValueError("compact receipt exceeds repository data-file ceiling")
    (args.output / "spectrum.json").write_text(encoded)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nova.media.ink import trace_axes

    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    trace_axes(ax)
    ax.tick_params(labelsize=20, width=1.2)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.2)
    order = np.arange(1, len(names) + 1)
    ax.semilogy(order, spectrum.singular_values, "o-", color="0.15", linewidth=3)
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1.2)
    ax.text(
        len(names) + 0.1, 1.0, "sensor floor", color="0.5", fontsize=20, va="center"
    )
    ax.text(
        order[-1] + 0.1,
        spectrum.singular_values[-1],
        "free drives",
        color="0.15",
        fontsize=20,
        va="center",
    )
    ax.set(
        xlabel="Passive direction",
        ylabel="RMS sensitivity / sensor floor",
        xlim=(0.7, len(names) + 3.5),
    )
    ax.set_xticks(order)
    fig.tight_layout()
    fig.savefig(args.output / "spectrum.svg")
    plt.close(fig)
    print(
        f"SUMMARY selected={len(cohort.shots)} admitted={len(records)} "
        f"refused={len(refused)} groups={len(names)} "
        f"identifiable={spectrum.identifiable_count} "
        f"nuisance_rank={spectrum.nuisance_rank} observations={observations}",
        flush=True,
    )
    print("SPECTRUM " + json.dumps(spectrum.singular_values.tolist()), flush=True)


if __name__ == "__main__":
    main()
