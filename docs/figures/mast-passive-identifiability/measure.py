"""Measure the nominal driven passive spectrum on the recorded vacuum cohort."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import multiprocessing
import os
import pickle
from pathlib import Path
import subprocess

import numpy as np

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_error_field_screen import read_error_field_drive
from nova.scripts.mast_passive_calibration import load_screen
from nova.imas.mast_fitted_parameters import (
    MIS_SCALED_SHOTS,
    SENSOR_FLOOR,
)
from nova.imas.mast_passive_decay_modes import (
    grouped_driven_jacobian,
    projected_spectrum,
)
from nova.imas.mast_vacuum_cohort import (
    EXCITATION_CURRENT,
    SHOT_STORE,
    ShotSurvey,
    read_shot_waveforms,
    select_vacuum_cohort,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def atomic_json(path, value):
    """Publish a complete record without exposing a partially written file."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_factor(path, block, row):
    """Atomically persist one shot's QR factor and its observation provenance."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            factor=block,
            record=np.asarray(json.dumps(row, allow_nan=False)),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_factor(path, identity, columns, shot):
    """Refuse foreign, corrupt or mislabelled factors before they enter a spectrum."""
    with np.load(path, allow_pickle=False) as saved:
        block = saved["factor"]
        row = json.loads(str(saved["record"]))
    if row.get("contract") != identity or row.get("shot") != shot:
        raise ValueError(f"factor identity mismatch: {path}")
    if (
        block.ndim != 2
        or block.shape[1] != columns
        or block.shape[0] > columns
        or not np.isfinite(block).all()
    ):
        raise ValueError(f"invalid QR factor: {path}")
    if row.get("status") == "admitted":
        if block.shape[0] == 0 or row.get("observation_count", 0) < block.shape[0]:
            raise ValueError(f"factor has no supporting observations: {path}")
    elif row.get("status") == "refused":
        if block.shape[0] or row.get("observation_count") != 0 or not row.get("reason"):
            raise ValueError(f"invalid refusal record: {path}")
    else:
        raise ValueError(f"unrecognised factor status: {path}")
    return block, row


def assemble_factors(shots, directory, identity, columns):
    """Reduce only persisted factors, keeping missing selected shots explicit."""
    compact, records, refused, missing = None, [], [], []
    observations = 0
    for shot in shots:
        path = Path(directory) / f"{shot}.npz"
        if not path.exists():
            missing.append(shot)
            continue
        block, row = read_factor(path, identity, columns, shot)
        row["factor_sha256"] = digest(path)
        if row["status"] == "refused":
            refused.append(row)
            continue
        compact = (
            block
            if compact is None
            else np.linalg.qr(np.vstack([compact, block]), mode="r")
        )
        records.append(row)
        observations += row["observation_count"]
    return compact, records, refused, missing, observations


def load_geometry(path):
    """Read the trusted geometry checkpoint supplied by the measurement run."""
    prepared = pickle.loads(Path(path).read_bytes())
    model, inductance, resistance, coupling, mutual, groups, _, scales, _, meta = (
        prepared
    )
    circuits = len(resistance)
    if (
        inductance.shape != (circuits, circuits)
        or coupling.shape != (len(model.targets), circuits)
        or mutual.shape != (circuits, len(scales))
        or len(groups) != circuits
        or set(groups) != set(meta["groups"])
        or not np.isfinite(inductance).all()
        or not np.isfinite(resistance).all()
        or np.any(resistance <= 0.0)
    ):
        raise ValueError("geometry checkpoint dimensions or resistance are invalid")
    np.linalg.cholesky(inductance)
    return prepared


def initialise_scoring(checkpoint, screen, directory, contract):
    """Load shared read-only inputs once per process inside the allocation."""
    global _PREPARED, _SCREEN, _DIRECTORY, _CONTRACT
    _PREPARED = load_geometry(checkpoint)
    _SCREEN = load_screen(screen)
    _DIRECTORY = Path(directory)
    _CONTRACT = contract


def score_task(task):
    """Persist the result before acknowledging completion to the parent process."""
    shot, refined = task
    suffix = "-fine" if refined else ""
    path = _DIRECTORY / f"{shot}{suffix}.npz"
    columns = len(_CONTRACT["groups"]) + len(_CONTRACT["drives"])
    identity = _CONTRACT["identity"]
    if path.exists():
        _, row = read_factor(path, identity, columns, shot)
        return shot, suffix, row["status"], "reused"
    row = {"shot": shot, "contract": identity, "observation_count": 0}
    step = _CONTRACT["step_seconds"] / (2.0 if refined else 1.0)
    try:
        if shot in MIS_SCALED_SHOTS:
            raise ValueError("acquisition amplitude refusal")
        names, block, detail = shot_jacobian(shot, _PREPARED, step, _SCREEN)
        if list(names) != _CONTRACT["groups"]:
            raise RuntimeError("parameter ordering differs from the factor contract")
        row.update(detail, status="admitted")
    except (ValueError, KeyError, FileNotFoundError) as error:
        block = np.empty((0, columns))
        row.update(status="refused", reason=str(error))
    write_factor(path, block, row)
    return shot, suffix, row["status"], "written"


def measurement_context(args):
    """Pin the unchanged 400-shot selection and every input to persisted factors."""
    payload = json.loads(args.census.read_text())
    keys = {field.name for field in fields(ShotSurvey)}
    surveys = [
        ShotSurvey(**{k: v for k, v in row.items() if k in keys})
        for row in payload["surveys"]
    ]
    cohort = select_vacuum_cohort(surveys, held_out_families=("P1+P2+P3+P4+P5+P6",))
    prepared = load_geometry(args.geometry_checkpoint)
    registry = MachineGeometryRegistry.default()
    for shot in cohort.shots:
        if (
            registry.select(shot).configuration.physical_digest
            != prepared[-1]["physical_digest"]
        ):
            raise ValueError(f"shot {shot} needs a different geometry checkpoint")
    contract = {
        "selected_shots": list(cohort.shots),
        "refinement_shots": list(cohort.shots[:3]),
        "groups": sorted(set(prepared[5])),
        "drives": list(prepared[0].families),
        "step_seconds": args.step,
        "sensor_floor_tesla": SENSOR_FLOOR,
        "geometry_sha256": digest(args.geometry_checkpoint),
        "census_sha256": digest(args.census),
        "screen_sha256": digest(args.screen),
        "source_sha256": digest("nova/imas/mast_passive_decay_modes.py"),
        "driver_sha256": digest(__file__),
        "registry_digest": registry.registry_digest,
    }
    contract["identity"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    contract["revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    path = args.factors / "contract.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != contract["identity"]:
            raise ValueError("factor directory belongs to different measurement inputs")
        contract = previous
    elif args.mode == "assemble":
        raise ValueError("no persisted scoring contract to assemble")
    else:
        args.factors.mkdir(parents=True, exist_ok=True)
        atomic_json(path, contract)
    return cohort, prepared, contract


def score_cohort(args, cohort, contract):
    """Score independent shots in one allocation and leave assembly for a resume."""
    tasks = [(shot, False) for shot in cohort.shots]
    tasks += [(shot, True) for shot in contract["refinement_shots"]]
    print(
        f"SCORING selected={len(cohort.shots)} processes={args.processes}", flush=True
    )
    with multiprocessing.get_context("spawn").Pool(
        args.processes,
        initializer=initialise_scoring,
        initargs=(args.geometry_checkpoint, args.screen, args.factors, contract),
    ) as pool:
        for shot, suffix, status, disposition in pool.imap_unordered(
            score_task, tasks, chunksize=1
        ):
            print(
                f"SHOT {shot}{suffix} status={status} factor={disposition}", flush=True
            )
    _, admitted, refused, missing, observations = assemble_factors(
        cohort.shots,
        args.factors,
        contract["identity"],
        len(contract["groups"]) + len(contract["drives"]),
    )
    summary = {
        "selected_count": len(cohort.shots),
        "scored_count": len(admitted) + len(refused),
        "admitted_count": len(admitted),
        "refused_count": len(refused),
        "missing_shots": missing,
        "observation_count": observations,
    }
    atomic_json(args.factors / "scoring-summary.json", summary)
    print("SCORING_COMPLETE " + json.dumps(summary), flush=True)


def assemble_receipt(args, cohort, prepared, contract):
    """Assemble an auditable full or explicitly partial spectrum from disk."""
    names = contract["groups"]
    columns = len(names) + len(contract["drives"])
    compact, records, refused, missing, observations = assemble_factors(
        cohort.shots,
        args.factors,
        contract["identity"],
        columns,
    )
    if compact is None:
        raise ValueError("no admitted factors; no spectrum conclusion is licensed")
    spectrum = projected_spectrum(
        compact[:, : len(names)],
        compact[:, len(names) :],
        observation_count=observations,
    )
    refinements = []
    for shot in contract["refinement_shots"]:
        paths = [args.factors / f"{shot}{suffix}.npz" for suffix in ("", "-fine")]
        if not all(path.exists() for path in paths):
            continue
        arms = [
            read_factor(path, contract["identity"], columns, shot) for path in paths
        ]
        if any(row["status"] != "admitted" for _, row in arms):
            continue
        values = [
            projected_spectrum(
                block[:, : len(names)],
                block[:, len(names) :],
                observation_count=row["observation_count"],
            ).singular_values
            for block, row in arms
        ]
        refinements.append(
            {
                "shot": shot,
                "coarse_step_seconds": args.step,
                "fine_step_seconds": args.step / 2,
                "coarse_spectrum": values[0].tolist(),
                "fine_spectrum": values[1].tolist(),
                "relative_spectrum_norm_change": float(
                    np.linalg.norm(values[1] - values[0])
                    / max(np.linalg.norm(values[1]), np.finfo(float).tiny)
                ),
            }
        )
    detail_path = args.factors / "shot-records.json"
    atomic_json(detail_path, {"admitted": records, "refused": refused})
    receipt = {
        "revision": contract["revision"],
        "factor_contract": contract,
        "source": "nova/imas/mast_passive_decay_modes.py",
        "source_sha256": contract["source_sha256"],
        "census_path": str(args.census),
        "census_sha256": contract["census_sha256"],
        "error_field_screen_path": str(args.screen),
        "error_field_screen_sha256": contract["screen_sha256"],
        "geometry_checkpoint_path": str(args.geometry_checkpoint),
        "geometry_checkpoint_sha256": contract["geometry_sha256"],
        "store": str(SHOT_STORE),
        "registry_digest": contract["registry_digest"],
        "cohort": {
            "selected_count": len(cohort.shots),
            "scored_count": len(records) + len(refused),
            "complete": not missing,
            "missing_shots": missing,
            "training": list(cohort.training),
            "held_out": list(cohort.held_out),
            "held_out_families": list(cohort.held_out_families),
            "excluded_count": len(cohort.exclusions),
        },
        "admitted_shots": [
            {
                k: v
                for k, v in row.items()
                if k not in ("channels", "source_identities", "channel_sample_counts")
            }
            for row in records
        ],
        "refused_shots": refused,
        "shot_detail_path": str(detail_path),
        "shot_detail_sha256": digest(detail_path),
        "factor_directory": str(args.factors),
        "geometries": {prepared[-1]["physical_digest"]: prepared[-1]},
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
            "Local interval-scaled RMS sensitivity at nominal seeds with every coil "
            "scale projected out without penalty. Directions are group combinations, "
            "not individually promoted groups. "
            "No held-out fit or global interval certification."
        ),
        "limitations": [
            "rodgr circuits excluded; result conditional on the reduced circuit model",
            "vertical drive scales use an explicitly assumed decade interval",
            "uniform linear current interpolation; three declared refinement shots",
            "each probe uses finite supported baseline and transient samples",
            "pooled sensor floor treated as a non-averaging RMS threshold",
            "zero initial passive current; no unmeasured prior history inferred",
        ],
    }
    if len(json.dumps(receipt, indent=2).encode()) > 300_000:
        raise ValueError("compact receipt exceeds repository data-file ceiling")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "spectrum.json", receipt)
    write_figure(spectrum, names, args.output)
    print(
        f"SUMMARY selected={len(cohort.shots)} scored={len(records) + len(refused)} "
        f"admitted={len(records)} refused={len(refused)} missing={len(missing)} "
        f"groups={len(names)} identifiable={spectrum.identifiable_count} "
        f"nuisance_rank={spectrum.nuisance_rank} observations={observations}",
        flush=True,
    )
    print("SPECTRUM " + json.dumps(spectrum.singular_values.tolist()), flush=True)


def write_figure(spectrum, names, output):
    """Render the real cohort spectrum, without a permutation control."""
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
    fig.savefig(output / "spectrum.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("score", "assemble"), required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    if not np.isfinite(args.step) or args.step <= 0 or args.processes < 1:
        parser.error("step and processes must be positive and finite")
    cohort, prepared, contract = measurement_context(args)
    if args.mode == "score":
        score_cohort(args, cohort, contract)
    else:
        assemble_receipt(args, cohort, prepared, contract)


if __name__ == "__main__":
    main()
