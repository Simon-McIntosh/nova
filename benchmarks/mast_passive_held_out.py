"""Fit a bounded passive direction and test driven predictions on excluded shots.

The spectrum supplies a fixed, interval-scaled direction without using measured
probe amplitudes. Drive scales are shared across training shots and profiled
within their recorded intervals at every resistance candidate. Held-out targets
never enter either optimization. The nominal comparison gets its own training
nuisance fit, so a resistance improvement cannot just be a drive calibration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import lsq_linear, minimize_scalar

from nova.imas.mast_passive_decay_modes import driven_field, projected_spectrum
from nova.imas.mast_vacuum_cohort import EXCITATION_CURRENT, read_shot_waveforms


@dataclass(frozen=True)
class Transient:
    """Supported, offset-removed readings on the spectrum's declared clock."""

    shot: int
    time: np.ndarray
    currents: np.ndarray
    selected: np.ndarray
    baseline: np.ndarray
    scored: np.ndarray
    observed: np.ndarray


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verified_inputs(spectrum_path):
    """Verify every persisted factor and recover the exact physical checkpoint."""
    receipt = json.loads(Path(spectrum_path).read_text())
    checkpoint = Path(receipt["geometry_checkpoint_path"])
    if digest(checkpoint) != receipt["geometry_checkpoint_sha256"]:
        raise ValueError("geometry checkpoint digest differs from spectrum")
    prepared = pickle.loads(checkpoint.read_bytes())
    if "nova.catalog.mast_geometry" in sys.modules:
        raise ValueError("geometry load imported the retiring registry")
    contract = receipt["factor_contract"]
    blocks, records = [], []
    for expected in receipt["admitted_shots"] + receipt["refused_shots"]:
        path = Path(receipt["factor_directory"]) / f"{expected['shot']}.npz"
        if digest(path) != expected["factor_sha256"]:
            raise ValueError(f"factor digest mismatch: {path}")
        with np.load(path, allow_pickle=False) as saved:
            block = saved["factor"]
            row = json.loads(str(saved["record"]))
        if row["contract"] != contract["identity"] or row["shot"] != expected["shot"]:
            raise ValueError(f"factor identity mismatch: {path}")
        if row["status"] == "admitted":
            blocks.append(block)
            records.append(row)
    count = len(receipt["parameter_groups"])
    joined = np.vstack(blocks)
    reproduced = projected_spectrum(
        joined[:, :count],
        joined[:, count:],
        observation_count=sum(r["observation_count"] for r in records),
    )
    np.testing.assert_allclose(
        reproduced.singular_values,
        receipt["singular_values_floor_units"],
        rtol=1e-8,
        atol=1e-10,
    )
    if reproduced.identifiable_count != 1:
        raise ValueError("measurement requires exactly one identifiable direction")
    return receipt, prepared, sorted(records, key=lambda row: row["shot"])


def direction_parameters(names, bounds, direction):
    """Map a normalized spectrum coordinate to bounded log resistances."""
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (len(names),) or not np.isfinite(vector).all():
        raise ValueError("invalid passive direction")
    radii = np.array([min(-np.log(bounds[n][0]), np.log(bounds[n][1])) for n in names])
    slope = vector * radii
    lower, upper = -np.inf, np.inf
    for name, coefficient in zip(names, slope, strict=True):
        if coefficient == 0:
            continue
        limits = np.sort(np.log(bounds[name]) / coefficient)
        lower, upper = max(lower, limits[0]), min(upper, limits[1])
    if not np.isfinite([lower, upper]).all() or not lower < 0 < upper:
        raise ValueError("direction has no finite interval around its seed")
    return slope, (float(lower), float(upper))


def read_transient(row, model, step):
    """Read only probes admitted by the pinned per-shot spectrum factors."""
    wave = read_shot_waveforms(row["shot"])
    indices = np.flatnonzero(wave.sample_mask)
    if indices.size < 8 or np.any(np.diff(indices) != 1):
        raise ValueError("vacuum clock lost its contiguous supported interval")
    start, stop = indices[0], indices[-1]
    if not np.all(wave.baseline_mask[start : start + 8]):
        raise ValueError("quiet prefix no longer supports zero initial current")
    time = np.linspace(
        wave.time[start],
        wave.time[stop],
        int(np.ceil((wave.time[stop] - wave.time[start]) / step)) + 1,
    )
    currents = np.column_stack(
        [np.interp(time, wave.time, wave.drives[name]) for name in model.families]
    )
    channels = [target.channel for target in model.targets]
    selected = np.array([channels.index(name) for name in row["channels"]])
    quiet = np.interp(time, wave.time, wave.baseline_mask.astype(float)) == 1
    occupied = np.flatnonzero(np.any(np.abs(currents) >= EXCITATION_CURRENT, axis=1))
    score = (time >= time[occupied[0]]) & (time <= time[occupied[-1]] + 0.1)
    baseline, scored, readings = [], [], []
    for name in row["channels"]:
        values = wave.probes[name]
        finite = np.isfinite(values)
        supported = np.interp(time, wave.time, finite.astype(float)) == 1
        base = supported & quiet
        admitted = supported & score
        if min(np.count_nonzero(base), np.count_nonzero(admitted)) < 8:
            raise ValueError(f"probe coverage changed for {row['shot']}/{name}")
        observed = np.interp(time, wave.time[finite], values[finite])
        readings.append(observed - observed[base].mean())
        baseline.append(base)
        scored.append(admitted)
    scored = np.column_stack(scored)
    if scored.sum() != row["observation_count"]:
        raise ValueError(f"observation count changed for shot {row['shot']}")
    if sorted({p.group_identity for p in wave.provenance}) != row["source_identities"]:
        raise ValueError(f"waveform provenance changed for shot {row['shot']}")
    return Transient(
        row["shot"],
        time,
        currents,
        selected,
        np.column_stack(baseline),
        scored,
        np.column_stack(readings),
    )


def permuted_sources(transients):
    """Derange whole measured coil-column histories across shots deterministically."""
    if len(transients) < 2:
        raise ValueError("drive permutation requires at least two shots")
    order = np.random.default_rng(73129).permutation(len(transients))
    sources = np.empty(len(order), dtype=int)
    sources[order] = np.roll(order, 1)
    assert np.all(sources != np.arange(len(order)))
    return sources


def response_factor(transient, donor, prepared, resistance, floor):
    """Compress exact predictions and targets without changing least squares."""
    model, inductance, _, coupling, mutual, _, _, scales, _, _ = prepared
    currents = np.column_stack(
        [
            np.interp(
                transient.time, donor.time, donor.currents[:, column], left=0, right=0
            )
            for column in range(len(scales))
        ]
    )
    columns = []
    for column in range(len(scales)):
        drive = np.zeros_like(currents)
        drive[:, column] = currents[:, column] * scales[column]
        prediction = driven_field(
            inductance,
            resistance,
            coupling[transient.selected],
            mutual,
            model.response[transient.selected],
            transient.time,
            drive,
        )
        for probe in range(len(transient.selected)):
            prediction[:, probe] -= prediction[
                transient.baseline[:, probe], probe
            ].mean()
        columns.append(prediction[transient.scored] / floor)
    design = np.column_stack(columns)
    target = transient.observed[transient.scored] / floor
    return np.linalg.qr(np.column_stack([design, target]), mode="r")


def profile_scales(factors, intervals, scales):
    """Fit all global drive scales, including the nominal seed comparison."""
    joined = np.vstack(factors)
    bounds = np.sort(intervals / scales[:, None], axis=1)
    outcome = lsq_linear(
        joined[:, :-1],
        joined[:, -1],
        bounds=(bounds[:, 0], bounds[:, 1]),
        tol=1e-10,
        max_iter=300,
    )
    if not outcome.success:
        raise ValueError(f"drive nuisance fit did not converge: {outcome.message}")
    return outcome.x, float(np.sum((joined[:, :-1] @ outcome.x - joined[:, -1]) ** 2))


def fit_direction(factor_at, limits, intervals, scales):
    """Profile one bounded physical coordinate using training observations only."""
    cache = {}

    def objective(value):
        key = float(value)
        if key not in cache:
            cache[key] = profile_scales(factor_at(key), intervals, scales)
            print(f"PROFILE coordinate={key:.9g} loss={cache[key][1]:.9g}", flush=True)
        return cache[key][1]

    objective(0.0)
    result = minimize_scalar(
        objective,
        bounds=limits,
        method="bounded",
        options={"xatol": 2e-3, "maxiter": 30},
    )
    if not result.success:
        raise ValueError("bounded passive direction fit did not converge")
    for endpoint in limits:
        objective(endpoint)
    best = min(cache, key=lambda value: cache[value][1])
    return {
        "coordinate": best,
        "drive_multipliers": cache[best][0].tolist(),
        "training_loss": cache[best][1],
        "nominal_drive_multipliers": cache[0.0][0].tolist(),
        "nominal_training_loss": cache[0.0][1],
        "profile": [{"coordinate": a, "loss": cache[a][1]} for a in sorted(cache)],
    }


def score_factors(factors, nominal, fitted, nominal_scales, fitted_scales, floor):
    rows = []
    for transient, base, trial in zip(factors, nominal, fitted, strict=True):
        count = int(transient.scored.sum())
        loss = [
            float(np.sum((matrix[:, :-1] @ scale - matrix[:, -1]) ** 2))
            for matrix, scale in ((base, nominal_scales), (trial, fitted_scales))
        ]
        rows.append(
            {
                "shot": transient.shot,
                "observations": count,
                "nominal_squared_floor_residual": loss[0],
                "fitted_squared_floor_residual": loss[1],
                "nominal_rms_tesla": floor * np.sqrt(loss[0] / count),
                "fitted_rms_tesla": floor * np.sqrt(loss[1] / count),
            }
        )
    base = sum(row["nominal_squared_floor_residual"] for row in rows)
    trial = sum(row["fitted_squared_floor_residual"] for row in rows)
    count = sum(row["observations"] for row in rows)
    return {
        "shots": rows,
        "observations": count,
        "nominal_rms_tesla": float(floor * np.sqrt(base / count)),
        "fitted_rms_tesla": float(floor * np.sqrt(trial / count)),
        "squared_residual_improvement": float(1 - trial / base),
    }


def control_removes_improvement(real, control):
    """Require a real held-out improvement and none under shuffled drives."""
    return bool(np.isfinite([real, control]).all() and real > 0 and control <= 0)


def run_arm(training, held_out, prepared, slope, names, limits, floor, permuted):
    """Train once, then freeze resistance and drive scales for excluded shots."""
    groups, seed, scales, intervals = prepared[5], prepared[2], prepared[7], prepared[8]
    circuit_slope = np.array([slope[names.index(group)] for group in groups])
    train_sources = permuted_sources(training) if permuted else np.arange(len(training))
    test_sources = permuted_sources(held_out) if permuted else np.arange(len(held_out))

    def factors(rows, sources, value):
        resistance = seed * np.exp(value * circuit_slope)
        return [
            response_factor(row, rows[source], prepared, resistance, floor)
            for row, source in zip(rows, sources, strict=True)
        ]

    fit = fit_direction(
        lambda a: factors(training, train_sources, a), limits, intervals, scales
    )
    score = score_factors(
        held_out,
        factors(held_out, test_sources, 0.0),
        factors(held_out, test_sources, fit["coordinate"]),
        fit["nominal_drive_multipliers"],
        fit["drive_multipliers"],
        floor,
    )
    resistance = seed * np.exp(fit["coordinate"] * circuit_slope)
    rates = eigh(np.diag(resistance), prepared[1], eigvals_only=True)
    return {
        "fit": fit,
        "held_out": score,
        "minimum_decay_rate_per_second": float(rates.min()),
        "decay_rates_per_second": rates.tolist(),
        "resistances_ohm": resistance.tolist(),
        "training_drive_sources": [training[i].shot for i in train_sources],
        "held_out_drive_sources": [held_out[i].shot for i in test_sources],
    }


def write_figure(receipt, output):
    """Compare excluded-shot errors with direct labels and neutral line styles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    for ax, label in zip(axes, ("real", "permuted"), strict=True):
        rows = receipt[label]["held_out"]["shots"]
        base = np.array([row["nominal_rms_tesla"] for row in rows]) * 1e6
        trial = np.array([row["fitted_rms_tesla"] for row in rows]) * 1e6
        limits = [
            min(base.min(), trial.min()) * 0.8,
            max(base.max(), trial.max()) * 1.2,
        ]
        ax.loglog(limits, limits, ":", color="0.5", linewidth=1.2)
        ax.scatter(base, trial, color="0.15", s=25)
        ax.set(
            xlim=limits,
            ylim=limits,
            xlabel="Nominal residual [µT]",
            ylabel="Fitted residual [µT]",
        )
        ax.text(
            0.05,
            0.92,
            "Measured drives" if label == "real" else "Permuted drives",
            transform=ax.transAxes,
            fontsize=20,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(Path(output) / "held-out-residuals.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    receipt, prepared, records = verified_inputs(args.spectrum)
    training_ids = set(receipt["cohort"]["training"])
    held_out_ids = set(receipt["cohort"]["held_out"])
    admitted = {r["shot"] for r in records}
    if training_ids & held_out_ids or admitted - (training_ids | held_out_ids):
        raise ValueError(
            "training and held-out split does not partition admitted shots"
        )
    split = {
        "spectrum_sha256": digest(args.spectrum),
        "rule": "Reuse the spectrum cohort split before reading probe amplitudes.",
        "training": sorted(admitted & training_ids),
        "held_out": sorted(admitted & held_out_ids),
        "permutation_seed": 73129,
        "score": "pooled squared residual, all supported transient observations",
        "control": "derange drive histories within each split; refit training only",
    }
    write_json(args.output / "split.json", split)
    print("SPLIT " + json.dumps(split), flush=True)
    names = receipt["parameter_groups"]
    slope, limits = direction_parameters(
        names, prepared[6], receipt["right_singular_directions"][0]
    )
    transients = []
    for row in records:
        transients.append(
            read_transient(row, prepared[0], receipt["factor_contract"]["step_seconds"])
        )
        print(
            f"READ shot={row['shot']} observations={row['observation_count']}",
            flush=True,
        )
    training = [row for row in transients if row.shot in training_ids]
    held_out = [row for row in transients if row.shot in held_out_ids]
    floor = receipt["sensor_floor_tesla"]
    result = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": digest(__file__),
        "split": split,
        "spectrum_path": str(args.spectrum),
        "spectrum_sha256": digest(args.spectrum),
        "geometry_sha256": receipt["geometry_checkpoint_sha256"],
        "verified_factor_count": len(receipt["admitted_shots"])
        + len(receipt["refused_shots"]),
        "sensor_floor_tesla": floor,
        "parameter_groups": names,
        "direction_log_slopes": slope.tolist(),
        "coordinate_interval": limits,
        "geometry_registry_imported": "nova.catalog.mast_geometry" in sys.modules,
        "scheduler_job": os.environ.get("SLURM_JOB_ID"),
        "scheduler_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "promoted_parameters": [],
        "limitations": receipt["limitations"],
    }
    for label, permuted in (("real", False), ("permuted", True)):
        print(f"ARM {label}", flush=True)
        result[label] = run_arm(
            training, held_out, prepared, slope, names, limits, floor, permuted
        )
        write_json(args.run_directory / f"{label}-fit.json", result[label])
        print(
            "ARM_COMPLETE "
            + label
            + " "
            + json.dumps(
                {k: v for k, v in result[label]["held_out"].items() if k != "shots"}
            ),
            flush=True,
        )
    meta = prepared[-1]["groups"]
    result["group_resistivities"] = {
        name: dict(
            meta[name],
            fitted_ohm_m=float(
                meta[name]["nominal_ohm_m"]
                * np.exp(result["real"]["fit"]["coordinate"] * slope[index])
            ),
        )
        for index, name in enumerate(names)
    }
    result["circuits"] = [
        {
            "name": name,
            "nominal_ohm": float(seed),
            "interval_ohm": (seed * np.array(prepared[6][group])).tolist(),
            "fitted_ohm": fitted,
        }
        for name, seed, group, fitted in zip(
            prepared[-1]["circuits"],
            prepared[2],
            prepared[5],
            result["real"]["resistances_ohm"],
            strict=True,
        )
    ]
    result["control_passed"] = control_removes_improvement(
        result["real"]["held_out"]["squared_residual_improvement"],
        result["permuted"]["held_out"]["squared_residual_improvement"],
    )
    result["verdict"] = (
        "held-out improvement with control passed"
        if result["control_passed"]
        else "not validated; retain nominal seeds"
    )
    write_json(args.output / "receipt.json", result)
    write_figure(result, args.output)
    print(
        "SUMMARY "
        + json.dumps(
            {
                "training": len(training),
                "held_out": len(held_out),
                "real_improvement": result["real"]["held_out"][
                    "squared_residual_improvement"
                ],
                "permuted_improvement": result["permuted"]["held_out"][
                    "squared_residual_improvement"
                ],
                "minimum_decay_rate": result["real"]["minimum_decay_rate_per_second"],
                "control_passed": result["control_passed"],
                "verdict": result["verdict"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
