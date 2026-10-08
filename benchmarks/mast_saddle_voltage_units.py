"""Audit raw saddle voltage units and condition gain inference on known units.

Gain means recorded units per volt, with an arbitrary constant offset removed.
Its magnitude does not adjudicate traversal sign. A coil-only prediction may
confound passive pickup and uncertain drive turns with acquisition gain; it is
usable only when every known-unit control for that loop reproduces both gain
and waveform within two percent. Otherwise missing units remain unknown.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from benchmarks.mast_saddle_voltage_reach import _resolve_raw, loop_identities
from nova.imas.mast_vacuum_cohort import COIL_DRIVES, SHOT_STORE

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = "sha256:b41c076e1fb7e16dabe3bada2f5d890125a857c400ce7599dfa488e8ebef90e4"
TOLERANCE = 0.02


@dataclass(frozen=True)
class GainEstimate:
    """Positive acquisition gain and residual, with traversal left unresolved."""

    raw_units_per_volt: float
    relative_residual: float
    samples: int


def estimate_gain(predicted_voltage, recorded) -> GainEstimate:
    """Fit an offset and a signed slope, returning only the slope magnitude."""
    predicted = np.asarray(predicted_voltage, dtype=float)
    observed = np.asarray(recorded, dtype=float)
    if predicted.ndim != 1 or predicted.shape != observed.shape:
        raise ValueError("prediction and observation must be aligned vectors")
    if len(predicted) < 8 or not np.isfinite([predicted, observed]).all():
        raise ValueError("gain requires at least eight finite aligned samples")
    x, y = predicted - predicted.mean(), observed - observed.mean()
    xx, yy = float(x @ x), float(y @ y)
    if xx <= 0 or yy <= 0:
        raise ValueError("gain requires nonzero prediction and observed variation")
    slope = float(x @ y / xx)
    if not np.isfinite(slope) or slope == 0:
        raise ValueError("gain is not identifiable")
    residual = float(np.linalg.norm(y - slope * x) / np.sqrt(yy))
    return GainEstimate(abs(slope), residual, len(x))


def validate_known_units(estimates, expected_gains) -> dict:
    """Refuse transfer unless all independent known-unit shots meet the bound."""
    estimates, expected = list(estimates), np.asarray(expected_gains, dtype=float)
    if len(estimates) < 2 or expected.shape != (len(estimates),):
        raise ValueError("calibration requires at least two known-unit shots")
    if not np.isfinite(expected).all() or np.any(expected <= 0):
        raise ValueError("known acquisition gains must be finite and positive")
    errors = np.array(
        [
            abs(e.raw_units_per_volt / g - 1)
            for e, g in zip(estimates, expected, strict=True)
        ]
    )
    residuals = np.array([e.relative_residual for e in estimates])
    accepted = bool(
        np.isfinite(errors).all()
        and np.isfinite(residuals).all()
        and np.all(errors <= TOLERANCE)
        and np.all((residuals >= 0) & (residuals <= TOLERANCE))
    )
    return {
        "accepted": accepted,
        "known_shots": len(estimates),
        "maximum_relative_gain_error": float(errors.max()),
        "maximum_relative_residual": float(residuals.max()),
        "tolerance": TOLERANCE,
    }


def voltage_scale(metadata) -> float | None:
    """Read explicit voltage units; an arbitrary label supplies no conversion."""
    return {
        "v": 1.0,
        "volt": 1.0,
        "volts": 1.0,
        "mv": 0.001,
        "millivolt": 0.001,
        "millivolts": 0.001,
    }.get(str(metadata.get("units", "")).strip().lower())


def signal_clock(group, key):
    """Follow each signal's declared clock rather than assuming a time key."""
    signal = group[key]
    attrs = dict(signal.attrs)
    dimensions = attrs.get("_ARRAY_DIMENSIONS", attrs.get("dims"))
    if not isinstance(dimensions, list) or len(dimensions) != 1:
        raise ValueError(f"{key}: no unique declared clock")
    clock = dimensions[0]
    time, values = np.asarray(group[clock], float), np.asarray(signal, float)
    clock_unit = str(group[clock].attrs.get("units", "")).strip().lower()
    if clock_unit not in {"s", "sec", "second", "seconds"}:
        raise ValueError(f"{key}: unknown clock unit {clock_unit!r}")
    if (
        time.ndim != 1
        or time.shape != values.shape
        or len(time) < 8
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError(f"{key}: malformed declared clock")
    return time, values


def drive_scale(unit: str, drive, turns) -> float:
    """Convert declared feed current or ampere-turns to ampere-turns."""
    unit = unit.strip().lower()
    if drive.reports_ampere_turns:
        if unit not in {"ka * turn", "ka*turn", "ka.turn"}:
            raise ValueError(
                f"{drive.channel}: expected kiloampere-turns, got {unit!r}"
            )
        return 1000.0
    if unit not in {"ka", "kiloamp", "kiloamps", "kiloampere", "kiloamperes"}:
        raise ValueError(f"{drive.channel}: unknown feed-current unit {unit!r}")
    count = turns[drive.family]
    if count is None or not np.isfinite(count) or count == 0:
        raise ValueError(f"{drive.family}: no finite winding turn count")
    return 1000 * count * drive.turn_to_channel_current_ratio


def pickup_geometry():
    """Compose saddle line integrals from the verified IDS and vacuum kernel."""
    import imas
    import shapely
    from nova.imas.machine_artifact import resolve_machine_artifact
    from nova.imas.mast_vacuum_response import loop_response_matrix

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
    components, turns = {}, {}
    for coil in active.coil:
        parts, winding = [], []
        for element in coil.element:
            if int(element.geometry.geometry_type) != 1:
                raise ValueError("coil needs outline geometry")
            parts.append(
                shapely.Polygon(
                    np.column_stack(
                        [element.geometry.outline.r, element.geometry.outline.z]
                    )
                )
            )
            winding.append(float(element.turns_with_sign))
        if not winding or not np.allclose(winding, winding[0]):
            raise ValueError("unequal element turns need an explicit winding model")
        shape = parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)
        components[str(coil.name)] = shapely.to_wkb(shape).hex()
        turns[str(coil.name)] = (
            sum(winding) if all(abs(value) < 1e10 for value in winding) else None
        )
    families = tuple(d.family for d in COIL_DRIVES)
    contours = {
        str(loop.name): np.array(
            [[float(p.r), float(p.z), float(p.phi)] for p in loop.position]
        )
        for loop in magnetics.flux_loop
        if str(loop.name).startswith("saddle_")
    }
    responses = {}
    nodes, weights = np.polynomial.legendre.leggauss(8)
    for identity in loop_identities():
        name = f"saddle_{identity['family']}_{identity['number'] - 1}"
        contour = contours[name].copy()
        if not np.allclose(contour[0], contour[-1], atol=1e-12, rtol=0):
            raise ValueError(f"{name}: open contour")
        contour[:, 2] = np.unwrap(contour[:, 2])
        delta = np.diff(contour, axis=0)
        samples = contour[:-1, None] + (nodes[None, :, None] + 1) / 2 * delta[:, None]
        flux = loop_response_matrix(
            {"active_components": components},
            samples[..., :2].reshape(-1, 2),
            families=families,
        )
        # A_phi R = psi/(2 pi), so the contour's oriented dphi links total flux.
        factors = (delta[:, 2, None] * weights[None] / (4 * np.pi)).ravel()
        responses[identity["loop"]] = factors @ flux
    return (
        responses,
        turns,
        {
            "artifact_digest": ARTIFACT,
            "dd_version": artifact.manifest.dd_version,
            "directory": str(artifact.directory),
            "families": list(families),
            "turns": turns,
            "kernel": "nova.imas.mast_vacuum_response.loop_response_matrix",
            "model": "direct coil pickup; no fitted passive or drive correction",
        },
    )


def shot_estimates(root, identities, response, turns, *, bins=256):
    """Compare binned raw voltage to minus the predicted flux edge difference."""
    xmb, currents = root["xmb"], root["amc"]
    drives, sources = [], []
    for drive in COIL_DRIVES:
        time, values = signal_clock(currents, drive.channel)
        unit = str(currents[drive.channel].attrs.get("units", "")).lower()
        factor = drive_scale(unit, drive, turns)
        drives.append((time, values * factor))
        sources.append(
            {
                "channel": drive.channel,
                "units": unit,
                "ampere_turns_per_raw_unit": factor,
            }
        )
    output = {}
    for identity in identities:
        key = _resolve_raw(set(xmb.keys()), identity)
        if key is None:
            continue
        try:
            time, raw = signal_clock(xmb, key)
            start = max(time[0], *(t[0] for t, _ in drives))
            stop = min(time[-1], *(t[-1] for t, _ in drives))
            if stop <= start:
                raise ValueError("no joint current/voltage clock")
            edges = np.linspace(start, stop, bins + 1)
            drive_edges = np.column_stack([np.interp(edges, t, v) for t, v in drives])
            predicted = -np.diff(drive_edges @ response[identity["loop"]]) / np.diff(
                edges
            )
            finite = np.isfinite(raw)
            counts = np.histogram(time[finite], edges)[0]
            total = np.histogram(time[finite], edges, weights=raw[finite])[0]
            means = np.divide(
                total, counts, out=np.full(bins, np.nan), where=counts > 0
            )
            supported = np.isfinite(means) & np.isfinite(predicted)
            estimate = estimate_gain(predicted[supported], means[supported])
            output[identity["loop"]] = asdict(estimate)
        except (ValueError, KeyError) as error:
            output[identity["loop"]] = {"error": str(error)}
    return output, sources


def audit(reach_path: Path, store: Path) -> dict:
    """Re-read the reach cohort and retain all unknown, missing and refused rows."""
    import zarr

    reach = json.loads(reach_path.read_text())
    identities, shots = loop_identities(), []
    for shot in reach["cohort"]["shots"]:
        root = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")
        if "xmb" not in root:
            continue
        xmb, rows = root["xmb"], []
        keys = set(xmb.keys())
        for identity in identities:
            key = _resolve_raw(keys, identity)
            row = {"loop": identity["loop"], "key": key}
            if key is None:
                row.update(status="absent", unit=None, volts_per_raw_unit=None)
            else:
                attrs = dict(xmb[key].attrs)
                scale = voltage_scale(attrs)
                row.update(
                    status="known" if scale is not None else "unknown",
                    unit="V" if scale is not None else None,
                    volts_per_raw_unit=scale,
                    source=f"{shot}.zarr/xmb/{key}/.zattrs",
                    metadata={
                        k: attrs.get(k)
                        for k in ("units", "label", "uuid", "signal_type", "uda_name")
                    },
                )
            rows.append(row)
        shots.append({"shot": shot, "signals": rows, "xmb_metadata": dict(xmb.attrs)})
    known = [s for s in shots if any(r["status"] == "known" for r in s["signals"])]
    unknown = [s for s in shots if any(r["status"] == "unknown" for r in s["signals"])]
    if not known:
        raise ValueError("positive control absent: no known voltage metadata found")
    responses, turns, geometry = pickup_geometry()
    for shot in known:
        root = zarr.open_group(str(store / f"{shot['shot']}.zarr"), mode="r")
        shot["pickup_estimates"], shot["drive_sources"] = shot_estimates(
            root, identities, responses, turns
        )
    validation = {}
    for identity in identities:
        name, estimates, expected = identity["loop"], [], []
        errors = []
        for shot in known:
            row = next(r for r in shot["signals"] if r["loop"] == name)
            fit = shot["pickup_estimates"].get(name, {"error": "missing signal"})
            if row["status"] != "known" or "error" in fit:
                errors.append(
                    {"shot": shot["shot"], "reason": fit.get("error", "unknown units")}
                )
            else:
                estimates.append(GainEstimate(**fit))
                expected.append(1 / row["volts_per_raw_unit"])
        validation[name] = (
            {"accepted": False, "errors": errors}
            if errors
            else validate_known_units(estimates, expected)
        )
    for shot in unknown:
        transferable = [i for i in identities if validation[i["loop"]]["accepted"]]
        fits = {}
        if transferable:
            root = zarr.open_group(str(store / f"{shot['shot']}.zarr"), mode="r")
            fits, shot["drive_sources"] = shot_estimates(
                root, transferable, responses, turns
            )
        for row in shot["signals"]:
            if row["status"] != "unknown":
                continue
            fit = fits.get(row["loop"])
            if fit and "error" not in fit and fit["relative_residual"] <= TOLERANCE:
                row.update(
                    status="inferred",
                    unit="V",
                    volts_per_raw_unit=1 / fit["raw_units_per_volt"],
                    gain_source="known-unit-validated coil pickup",
                    pickup_estimate=fit,
                )
            else:
                row["reason"] = (
                    "known-unit pickup validation refused"
                    if not validation[row["loop"]]["accepted"]
                    else "shot pickup estimate refused"
                )
    unresolved = [
        s["shot"]
        for s in unknown
        if any(r["status"] == "unknown" for r in s["signals"])
    ]
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "reach_source": str(reach_path.relative_to(ROOT)),
        "reach_sha256": hashlib.sha256(reach_path.read_bytes()).hexdigest(),
        "store": str(store),
        "geometry": geometry,
        "calibration_validation": validation,
        "threshold": TOLERANCE,
        "gain_definition": (
            "volts_per_raw_unit converts stored samples to volts; "
            "positive magnitude only"
        ),
        "documented_acquisition_setting": None,
        "metadata_positive_control": [s["shot"] for s in known],
        "summary": {
            "xmb_shots": len(shots),
            "target_shots": len(unknown),
            "known_unit_shots": len(known),
            "unknown_shots": len(unresolved),
            "validated_loops": sum(v["accepted"] for v in validation.values()),
        },
        "unknown_shots": unresolved,
        "shots": shots,
        "sign_promoted": False,
        "admission_fit_rerun": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reach", type=Path, default=ROOT / "docs/figures/mast-saddle-loops/reach.json"
    )
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/figures/mast-saddle-voltage-units/receipt.json",
    )
    args = parser.parse_args()
    receipt = audit(args.reach, args.store)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(json.dumps(receipt["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
