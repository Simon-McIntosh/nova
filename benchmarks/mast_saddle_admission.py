"""Measure saddle admission on a shot split declared before reading responses.

Probe fields fit the coil-current waveforms; uncorrected saddle voltages fit
minus their time derivatives through the same spatial flux columns. Each time
bin averages voltage and differences current at its edges, implementing Faraday's
law without accumulating an arbitrary voltage integration constant. All fits use
training shots only. Binary traversal choices are profiled nuisance values and
are never exported as calibrations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import imas
import numpy as np
from scipy.linalg import eigh
import zarr

from benchmarks.mast_saddle_voltage_reach import _resolve_raw, loop_identities
from nova.biot import toroidalharmonic as th
from nova.biot.polygon import polygon_greens
from nova.imas.io_magnetics import Magnetics
from nova.imas.machine_artifact import resolve_machine_artifact
from nova.imas import mast_misfit_harmonics as mh
from nova.imas.mast_vacuum_cohort import (
    COIL_DRIVES,
    ENERGISED_CURRENT,
    PROBE_FAMILIES,
    SHOT_STORE,
    light_census,
    probe_channels,
    read_shot_waveforms,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = "sha256:b41c076e1fb7e16dabe3bada2f5d890125a857c400ce7599dfa488e8ebef90e4"
EXCLUDED = {"obv03", "obr05", "obr18", "obv10", "obr17"}
RIDGE = 1e-6
BINS = 256
PERMUTATION_SEED = 7301
CONTROL_SHARE = 0.5


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def geometry():
    """Read all diagnostic contours and coil cross-sections from the pinned IDSs."""
    artifact = resolve_machine_artifact(
        Path.home() / ".cache/mast-artifact-ef", ARTIFACT, allow_incomplete=True
    )
    with imas.DBEntry(
        f"imas:hdf5?path={artifact.directory}",
        "r",
        dd_version=artifact.manifest.dd_version,
    ) as entry:
        for name in ("magnetics", "pf_active"):
            if 0 not in entry.list_all_occurrences(name):
                raise ValueError(f"missing {name} occurrence zero")
        magnetics = entry.get("magnetics", 0, lazy=False, autoconvert=False)
        active = entry.get("pf_active", 0, lazy=False, autoconvert=False)
    reader = Magnetics(ids=magnetics)
    probes = []
    for sensor in magnetics.b_field_pol_probe:
        family = str(sensor.name).rsplit("_", 1)[0]
        if family in PROBE_FAMILIES:
            probes.append({"family": family, "sensor": sensor})
    joined = probe_channels(probes)
    channels = [row.channel for row in joined]
    sensors = [probes[row.registry_index]["sensor"] for row in joined]
    r = np.array([float(sensor.position.r) for sensor in sensors])
    z = np.array([float(sensor.position.z) for sensor in sensors])
    angle = np.array([float(sensor.poloidal_angle) for sensor in sensors])
    cosine, sine = np.cos(angle), -np.sin(angle)
    sections = []
    for coil in active.coil:
        parts = []
        for element in coil.element:
            if int(element.geometry.geometry_type) != 1:
                raise ValueError("measurement requires the artifact's outline geometry")
            vertices = np.column_stack(
                [element.geometry.outline.r, element.geometry.outline.z]
            )
            area = (
                abs(
                    np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1))
                    - np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1))
                )
                / 2
            )
            parts.append((vertices, area))
        sections.append((str(coil.name), parts))

    def response(radius, height):
        flux = np.zeros((np.size(radius), len(sections)))
        radial, axial = np.zeros_like(flux), np.zeros_like(flux)
        for column, (_, parts) in enumerate(sections):
            total = sum(area for _, area in parts)
            for vertices, area in parts:
                psi, br, bz = polygon_greens(
                    np.asarray(radius), np.asarray(height), vertices
                )
                flux[:, column] += area / total * psi
                radial[:, column] += area / total * br
                axial[:, column] += area / total * bz
        return flux, radial, axial

    _, br, bz = response(r, z)
    described = cosine[:, None] * br + sine[:, None] * bz
    return (
        reader,
        channels,
        (r, z, cosine, sine),
        described,
        response,
        {
            "artifact_digest": ARTIFACT,
            "dd_version": artifact.manifest.dd_version,
            "directory": str(artifact.directory),
            "coil_columns": [name for name, _ in sections],
            "probe_binding": [
                {"raw": row.channel, "ids_name": str(sensor.name)}
                for row, sensor in zip(joined, sensors, strict=True)
            ],
            "saddle_count": int(np.sum(reader["flux_loop"]["type"] == 2)),
        },
    )


def declare_split(reach_path, output, *, minimum_held_out=2):
    """Admit through the cohort owner, then hold out every fourth ordered shot."""
    reach = json.loads(Path(reach_path).read_text())
    admitted, refused, counts = [], [], {}
    for shot in reach["cohort"]["shots"]:
        census = light_census(shot)
        if census is None or not census.admitted():
            refused.append({"shot": shot, "reason": "light census refusal"})
            continue
        store = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")
        if "xmb" not in store:
            continue
        keys = set(store["xmb"].keys())
        count = sum(_resolve_raw(keys, loop) is not None for loop in loop_identities())
        if count:
            admitted.append(shot)
            counts[str(shot)] = count
    admitted.sort()
    held = admitted[::4]
    training = [shot for shot in admitted if shot not in held]
    if min(len(training), len(held)) < 2:
        raise ValueError("not enough independent shots for a held-out comparison")
    split = {
        "declared_at": datetime.now(timezone.utc).isoformat(),
        "rule": "sort xmb-bearing light-census admissions; indices 0,4,8,... held out",
        "training": training,
        "held_out": held,
        "available_loop_counts": counts,
        "refused": refused,
        "reach_sha256": hashlib.sha256(Path(reach_path).read_bytes()).hexdigest(),
        "fixed_before_scoring": {
            "minimum_training_shots": 2,
            "minimum_held_out_shots": minimum_held_out,
            "bins": BINS,
            "ridge": RIDGE,
            "permutation_seed": PERMUTATION_SEED,
            "maximum_control_gain_share": CONTROL_SHARE,
            "spatial_basis": (
                "axisymmetric ring harmonics, inner and outer, order two, "
                "focus (1.0,0.0), plus all described coil columns"
            ),
        },
    }
    write_json(output, split)
    print("SPLIT", json.dumps(split), flush=True)
    return split


def bin_mean(time, values, edges):
    finite = np.isfinite(time) & np.isfinite(values)
    count = np.histogram(time[finite], edges)[0]
    total = np.histogram(time[finite], edges, weights=values[finite])[0]
    return np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)


def raw_voltage(group, loop):
    """Resolve each family's actual key and the signal's declared clock."""
    key = _resolve_raw(set(group.keys()), loop)
    if key is None:
        return None
    signal = group[key]
    metadata = dict(signal.attrs)
    dimensions = metadata.get("_ARRAY_DIMENSIONS", metadata.get("dims"))
    if not isinstance(dimensions, list) or len(dimensions) != 1:
        raise ValueError(f"{key}: no unique declared clock")
    clock = dimensions[0]
    if clock not in group:
        raise ValueError(f"{key}: unavailable declared clock {clock!r}")
    if metadata.get("units", "").lower() not in {"volt", "volts", "v"}:
        raise ValueError(
            f"{key}: unrecognized voltage unit {metadata.get('units')!r} "
            f"with label {metadata.get('label')!r}"
        )
    time, values = (
        np.asarray(group[clock], dtype=float),
        np.asarray(signal, dtype=float),
    )
    if (
        time.shape != values.shape
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError(f"{key}: malformed voltage clock")
    return (
        time,
        values,
        {
            "key": key,
            "clock": clock,
            "units": metadata["units"],
            "uuid": metadata.get("uuid"),
        },
    )


@dataclass
class ShotFactors:
    shot: int
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray
    count: np.ndarray
    floor: np.ndarray
    coupling: np.ndarray
    provenance: list


def shot_factors(shot, probes):
    """Compress each waveform to sufficient statistics, preserving missing rows."""
    wave = read_shot_waveforms(shot)
    xmb = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["xmb"]
    raw = [raw_voltage(xmb, loop) for loop in loop_identities()]
    available = [item for item in raw if item is not None]
    start = max(wave.time[0], max(item[0][0] for item in available))
    stop = min(wave.time[-1], min(item[0][-1] for item in available))
    if stop <= start:
        raise ValueError(f"{shot}: no joint voltage/current clock")
    edges = np.linspace(start, stop, BINS + 1)
    time = (edges[1:] + edges[:-1]) / 2
    currents = np.column_stack(
        [
            np.interp(edges, wave.time, wave.drives[drive.family])
            if drive.family in wave.drives
            else np.zeros(len(edges))
            for drive in COIL_DRIVES
        ]
    )
    driven = np.any(np.abs(currents) >= ENERGISED_CURRENT, axis=1)
    if not driven.any():
        raise ValueError(f"{shot}: no supported drive")
    quiet_stop = edges[np.flatnonzero(driven)[0]]
    current = (currents[1:] + currents[:-1]) / 2 / 1e4
    derivative = -np.diff(currents, axis=0) / np.diff(edges)[:, None] / 1e4
    valid = np.interp(time, wave.time, wave.sample_mask.astype(float)) == 1
    quiet_drive = (wave.time >= start) & (wave.time < quiet_stop) & wave.baseline_mask
    nchannel, ndrive = len(probes) + len(raw), len(COIL_DRIVES)
    xx = np.zeros((nchannel, ndrive, ndrive))
    xy = np.zeros((nchannel, ndrive))
    yy = np.zeros(nchannel)
    count = np.zeros(nchannel, dtype=int)
    floor = np.full(nchannel, np.nan)
    coupling = np.full(nchannel, np.nan)
    provenance = []
    signals = []
    for channel in probes:
        signal = wave.probes.get(channel)
        signals.append(
            None if signal is None else (wave.time, signal, current, quiet_drive)
        )
    for loop, item in zip(loop_identities(), raw, strict=True):
        if item is None:
            signals.append(None)
            continue
        clock, voltage, source = item
        source.update(
            loop=loop["loop"], ids_name=f"saddle_{loop['family']}_{loop['number'] - 1}"
        )
        provenance.append(source)
        signals.append(
            (clock, voltage, derivative, (clock >= start) & (clock < quiet_stop))
        )
    for index, item in enumerate(signals):
        if item is None:
            continue
        clock, signal, design, quiet = item
        quiet = quiet & np.isfinite(signal)
        if quiet.sum() < 8:
            continue
        offset = float(np.mean(signal[quiet]))
        noise = float(np.std(signal[quiet], ddof=1))
        if not np.isfinite(noise) or noise <= 0:
            continue
        observed = bin_mean(clock, signal - offset, edges)
        supported = valid & np.isfinite(observed) & np.isfinite(design).all(axis=1)
        if supported.sum() < 8:
            continue
        x, y = design[supported], observed[supported]
        xx[index] = x.T @ x
        xy[index] = x.T @ y
        yy[index] = y @ y
        count[index] = len(y)
        floor[index] = noise
        strongest = int(np.argmax(np.diag(xx[index])))
        if xx[index, strongest, strongest] > 0:
            coupling[index] = xy[index, strongest] / xx[index, strongest, strongest]
    return ShotFactors(shot, xx, xy, yy, count, floor, coupling, provenance)


def aggregate(factors, indices):
    return (
        sum(f.xx[indices] for f in factors),
        sum(f.xy[indices] for f in factors),
        sum(f.yy[indices] for f in factors),
        sum(f.count[indices] for f in factors),
    )


def fit_bank(design, stats, noise, saddle, *, include_saddles=True):
    """Profile binary signs on training sufficient statistics and a fixed ridge.

    A sign reverses a row's linear term but leaves the normal matrix unchanged.
    Coordinate descent therefore compares both signs exactly at each step;
    deterministic multiple starts expose the conditional nature of the result.
    """
    xx, xy, yy, count = stats
    keep = count > 0
    if not include_saddles:
        keep &= ~saddle
    weight = 1 / noise**2
    weight = np.where(keep, weight, 0)
    hessian = np.einsum("ki,kj,kab,k->iajb", design, design, xx, weight, optimize=True)
    size = design.shape[1] * xy.shape[1]
    hessian = hessian.reshape(size, size)
    diagonal = np.sqrt(np.maximum(np.diag(hessian), np.finfo(float).tiny))
    live = np.diag(hessian) > np.max(np.diag(hessian)) * 1e-24
    scaled = hessian[np.ix_(live, live)] / np.outer(diagonal[live], diagonal[live])
    eigenvalues, vectors = eigh(scaled)
    inverse = (vectors / (np.maximum(eigenvalues, 0) + RIDGE)) @ vectors.T
    contributions = np.einsum("ki,ka,k->kia", design, xy, weight).reshape(
        len(design), size
    )
    contributions = contributions[:, live] / diagonal[live]
    loop_rows = np.flatnonzero(saddle & keep)
    generator = np.random.default_rng(819)
    best = None
    for attempt in range(16 if include_saddles else 1):
        signs = np.ones(len(design))
        if attempt:
            signs[loop_rows] = generator.choice([-1, 1], len(loop_rows))
        rhs = signs @ contributions
        inverse_rhs = inverse @ rhs
        inverse_contributions = contributions @ inverse
        for _ in range(100):
            changed = False
            for row in loop_rows:
                delta = (
                    -4 * signs[row] * contributions[row] @ inverse_rhs
                    + 4 * contributions[row] @ inverse_contributions[row]
                )
                if delta > 1e-10:
                    rhs -= 2 * signs[row] * contributions[row]
                    inverse_rhs -= 2 * signs[row] * inverse_contributions[row]
                    signs[row] *= -1
                    changed = True
            if not changed:
                break
        objective = float(np.sum(yy * weight) - rhs @ inverse @ rhs)
        if best is None or objective < best[0]:
            solution = np.zeros(size)
            solution[live] = (inverse @ rhs) / diagonal[live]
            best = (
                objective,
                solution.reshape(design.shape[1], xy.shape[1]),
                signs.copy(),
            )
    objective, coefficients, signs = best
    if not include_saddles:
        prediction = design @ coefficients
        signs[saddle] = np.where(
            np.sum(prediction[saddle] * xy[saddle], axis=1) < 0, -1, 1
        )
    return (
        coefficients,
        signs,
        {
            "training_objective": objective,
            "normal_rank": int(np.sum(eigenvalues > RIDGE)),
            "parameters": int(live.sum()),
            "sign_search": (
                "16 deterministic coordinate-descent starts; conditional "
                "nuisance, not sign evidence"
            )
            if include_saddles
            else (
                "coefficients use no saddle rows; evaluation signs profiled "
                "on training shots only"
            ),
        },
    )


def score(design, coefficients, signs, stats):
    xx, xy, yy, count = stats
    predicted = signs[:, None] * (design @ coefficients)
    residual = (
        yy
        - 2 * np.sum(predicted * xy, axis=1)
        + np.einsum("ka,kab,kb->k", predicted, xx, predicted)
    )
    return np.maximum(residual, 0), count


def admission_verdict(real_gain, removed_gain, permuted_gain):
    """Compare each control's gain to the measured gain, including positive controls."""
    return bool(
        real_gain > 0
        and removed_gain < CONTROL_SHARE * real_gain
        and permuted_gain < CONTROL_SHARE * real_gain
    )


def residual_figure(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nova.media.ink import trace_axes

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), dpi=100)
    for axis, family in zip(axes, "lmu", strict=True):
        selected = [row for row in rows if row["loop"].startswith(f"saddle_{family}_")]
        x = np.arange(len(selected))
        for name, label, style, shade in [
            ("removed", "saddles removed", "--", "0.55"),
            ("real", "real geometry", "-", "0.1"),
            ("permuted", "permuted geometry", ":", "0.35"),
        ]:
            values = [row["whitened_rms"][name] for row in selected]
            axis.plot(x, values, style, color=shade, linewidth=3, label=label)
        trace_axes(axis)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [
                row["loop"].replace("saddle_", "")
                + f"\n{row['training_shots']}/{row['held_out_shots']}"
                for row in selected
            ],
            fontsize=20,
        )
        axis.tick_params(axis="y", labelsize=20)
        axis.set_ylabel("RMS / voltage floor", fontsize=22)
    axes[0].legend(frameon=False, fontsize=20, ncol=3)
    axes[-1].set_xlabel("IDS loop name and fit / held-out shot counts", fontsize=22)
    fig.subplots_adjust(bottom=0.09, left=0.13, right=0.98, top=0.97, hspace=0.4)
    fig.savefig(path)
    plt.close(fig)


def main():
    from nova.jax.config import configure_dtypes
    import jax

    configure_dtypes()
    assert jax.config.jax_enable_x64
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--allow-single-held-out",
        action="store_true",
        help=(
            "Report a limited one-shot diagnostic when raw-unit refusals "
            "leave only one held-out shot."
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split = declare_split(
        ROOT / "docs/figures/mast-saddle-loops/reach.json",
        args.output / "split.json",
        minimum_held_out=1 if args.allow_single_held_out else 2,
    )
    reader, probes, poses, described, response, provenance = geometry()
    print("GEOMETRY", json.dumps(provenance), flush=True)
    all_factors, errors = [], []
    for shot in sorted(split["training"] + split["held_out"]):
        try:
            factors = shot_factors(shot, probes)
            all_factors.append(factors)
            print(
                "SHOT",
                shot,
                "served",
                int((factors.count > 0).sum()),
                "saddles",
                int((factors.count[len(probes) :] > 0).sum()),
                flush=True,
            )
        except (ValueError, KeyError) as error:
            errors.append({"shot": shot, "error": str(error)})
            print("REFUSED", shot, repr(error), flush=True)
    train = [f for f in all_factors if f.shot in split["training"]]
    test = [f for f in all_factors if f.shot in split["held_out"]]
    write_json(
        args.output / "read-validation.json",
        {
            "served_training": [f.shot for f in train],
            "served_held_out": [f.shot for f in test],
            "refused": errors,
        },
    )
    minimum_held_out = 1 if args.allow_single_held_out else 2
    if len(train) < 2 or len(test) < minimum_held_out:
        raise ValueError(
            "insufficient served training or held-out shots for the declared mode"
        )
    samples = np.array([f.coupling for f in train])
    noise = np.nanmedian([f.floor for f in train], axis=0)
    noise = np.where(np.isfinite(noise) & (noise > 0), noise, 0)
    field = mh.SensorClass(
        tuple(probes),
        *poses,
        False,
        samples[:, : len(probes)],
        described,
        noise[: len(probes)],
    )
    loops = [
        f"saddle_{loop['family']}_{loop['number'] - 1}" for loop in loop_identities()
    ]
    saddle = mh.saddle_sensor_class(
        reader,
        loops,
        samples[:, len(probes) :],
        lambda r, z: response(r, z)[0],
        noise[len(probes) :],
    )
    bank = mh.assemble(
        "vacuum", [field, saddle], excluded=EXCLUDED, shots=[f.shot for f in train]
    )
    complete_names = probes + loops
    indices = np.array([complete_names.index(name) for name in bank.channel])
    noise = noise[indices]
    is_saddle = np.array([path is not None for path in bank.contours])
    if int(is_saddle.sum()) != 36:
        raise ValueError(
            f"only {is_saddle.sum()}/36 saddle loops meet two-training-shot admission"
        )
    basis = th.ToroidalHarmonics(
        th.FocalCircle(1.0, 0.0), order=2, families=(th.INNER, th.OUTER)
    )
    realized = bank.realize_traversal(np.ones(bank.rows, dtype=int))
    design = np.column_stack([mh.harmonic_design(basis, realized), realized.described])
    train_stats = aggregate(train, indices)
    test_stats = aggregate(test, indices)
    permuted = design.copy()
    loop_rows = np.flatnonzero(is_saddle)
    permutation = np.random.default_rng(PERMUTATION_SEED).permutation(loop_rows)
    permuted[loop_rows] = design[permutation]
    changed = float(np.linalg.norm(permuted[loop_rows] - design[loop_rows]))
    if changed <= 0:
        raise ValueError(
            "geometry permutation changed no response: invalid control instrument"
        )
    arms, fits = {}, {}
    for name, matrix, included in [
        ("removed", design, False),
        ("real", design, True),
        ("permuted", permuted, True),
    ]:
        coefficients, signs, detail = fit_bank(
            matrix, train_stats, noise, is_saddle, include_saddles=included
        )
        residual, count = score(matrix, coefficients, signs, test_stats)
        arms[name] = residual
        fits[name] = detail | {
            "traversal_choices_conditional": dict(
                zip(bank.channel, signs.astype(int).tolist(), strict=True)
            )
        }
        print("FIT", name, json.dumps(detail), flush=True)
    common = (test_stats[3] > 0) & is_saddle
    power = {
        name: float(np.sum(residual[common] / noise[common] ** 2))
        for name, residual in arms.items()
    }
    gains = {name: 1 - value / power["removed"] for name, value in power.items()}
    reach = json.loads((ROOT / "docs/figures/mast-saddle-loops/reach.json").read_text())
    available_counts = {
        f"saddle_{loop['family']}_{loop['number'] - 1}": loop["raw_shot_count"]
        for loop in reach["loops"]
    }
    rows = []
    for row in loop_rows:
        counts = {
            "training_shots": int(sum(f.count[indices[row]] > 0 for f in train)),
            "held_out_shots": int(sum(f.count[indices[row]] > 0 for f in test)),
        }
        if test_stats[3][row] == 0:
            raise ValueError(f"{bank.channel[row]} has no held-out observations")
        rows.append(
            {
                "loop": bank.channel[row],
                **counts,
                "raw_available_shots": available_counts[bank.channel[row]],
                "usable_shots": int(
                    sum(f.count[indices[row]] > 0 for f in all_factors)
                ),
                "floor_volts": float(noise[row]),
                "held_out_samples": int(test_stats[3][row]),
                "rms_volts": {
                    name: float(np.sqrt(residual[row] / test_stats[3][row]))
                    for name, residual in arms.items()
                },
                "whitened_rms": {
                    name: float(
                        np.sqrt(residual[row] / test_stats[3][row]) / noise[row]
                    )
                    for name, residual in arms.items()
                },
                "whitened_squared_residual_contribution": {
                    name: float(residual[row] / noise[row] ** 2)
                    for name, residual in arms.items()
                },
                "permuted_geometry": bank.channel[
                    permutation[np.flatnonzero(loop_rows == row)[0]]
                ],
                "traversal_options": [-1, 1],
                "disposition": "unresolved",
            }
        )
    receipt = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "split": split,
        "geometry": provenance,
        "read_errors": errors,
        "served_training": [f.shot for f in train],
        "served_held_out": [f.shot for f in test],
        "loops": rows,
        "fits": fits,
        "gain": gains,
        "held_out_whitened_power": power,
        "held_out_whitened_rms": {
            name: float(np.sqrt(value / np.sum(test_stats[3][common])))
            for name, value in power.items()
        },
        "attributable": admission_verdict(
            gains["real"], gains["removed"], gains["permuted"]
        ),
        "geometry_control_response_delta": changed,
        "signs_promoted": [],
        "full_cohort_measured": not errors,
        "single_held_out_diagnostic": len(test) == 1,
        "cross_shot_uncertainty": None if len(test) == 1 else "not estimated",
        "scope": (
            "axisymmetric vacuum pickup admission; no non-axisymmetric "
            "harmonic inference or sign calibration"
        ),
        "weighting": (
            "per-sensor median training quiet-signal standard deviation; "
            "no held-out value sets fit weights"
        ),
        "raw_provenance": {str(f.shot): f.provenance for f in all_factors},
        "quiet_floor_note": (
            "voltage and field noise measured before first drive reaches "
            "200 A; bin averaging can lower random-noise RMS below this "
            "unaveraged floor"
        ),
        "removed_control": (
            "no saddle amplitude rows in normal equations; binary "
            "evaluation signs alone profiled on training observations"
        ),
        "limitations": (
            "unmodeled passive dynamics and fixed ridge may affect "
            "residuals; conditional sign search is not a proof of a "
            "global discrete optimum"
        ),
    }
    if "nova.catalog.mast_geometry" in sys.modules:
        raise ValueError("retiring geometry registry was imported")
    write_json(args.output / "receipt.json", receipt)
    residual_figure(args.output / "residuals.svg", rows)
    np.savez_compressed(
        args.run_directory / "shot-factors.npz",
        shots=[f.shot for f in all_factors],
        xx=[f.xx for f in all_factors],
        xy=[f.xy for f in all_factors],
        yy=[f.yy for f in all_factors],
        count=[f.count for f in all_factors],
        floor=[f.floor for f in all_factors],
        design=design,
        noise=noise,
        channel=bank.channel,
    )
    print(
        "SUMMARY",
        json.dumps(
            {
                "training_shots": len(train),
                "held_out_shots": len(test),
                "loops": len(rows),
                "gain": gains,
                "whitened_rms": receipt["held_out_whitened_rms"],
                "attributable": receipt["attributable"],
                "signs_promoted": 0,
                "refused_shots": len(errors),
                "single_held_out_diagnostic": len(test) == 1,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
