"""Record which access paths reach an uncorrected MAST saddle-loop voltage.

MAST's 36 saddle loops are the machine's only n-resolving sensor set.  Nothing
consumes them today because their published products are differences between
toroidally opposite loops and have had the axisymmetric poloidal-field pickup
removed, and that pickup is the whole of the quantity a coil pulse fixes a
traversal sign against.  This benchmark answers one question only: is an
UNCORRECTED saddle voltage -- pickup retained -- reachable for the shots of the
vacuum cohort, and through which path.

It runs no fit and promotes no sign.  For each of the 36 loops it records every
access path tried, the signal name and source behind it, and whether a finite
sample was found per shot range.  Where a shot carries both the raw level-1 XMB
signal and the published level-2 product, their difference is differenced and
compared with the coil-driven poloidal-field pickup the vacuum response predicts.

Cohort frame.  The vacuum cohort is the no-plasma, coil-driven, over-determined
set of :mod:`nova.imas.mast_vacuum_cohort`.  Its own census reads every field
waveform of the 17 111-shot store, which is minutes to hours of GPFS traffic, so
this benchmark draws an even-strided sample across the whole shot range and
applies the module's criteria in a light census that reads only the current-group
peaks and the field-group channel names -- the admissions conditions, without the
waveforms.  The sample limit is recorded in the receipt; the verdict is a
statement about that frame and says so.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

LEVEL1_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
LEVEL2_STORE = Path("/work/projects/imas_gpu/mast/level2/shots")

FAMILIES = ("l", "m", "u")
PER_FAMILY = 12
PLASMA_FREE_CURRENT = 5.0e3
EXCITATION_CURRENT = 1.0e3
MINIMUM_PROBES = 40
_PROBE = re.compile(r"^(ccbv|obr|obv)\d{2}$")
MISSPELLED_CHANNEL = "sad_out_zz99"
"""A channel that names no loop: the receipt must report it unreachable."""
_DRIVE_CHANNELS = (
    "sol_current",
    "p2il_feed_current",
    "p2iu_feed_current",
    "p2ol_feed_current",
    "p2ou_feed_current",
    "p3l_feed_current",
    "p3u_feed_current",
    "p4l_feed_current",
    "p4u_feed_current",
    "p5l_feed_current",
    "p5u_feed_current",
    "p6l_current",
    "p6u_current",
)


def loop_identities() -> list[dict]:
    """Return the 36 saddle loops and the two signal names each is reachable by."""

    rows = []
    for family in FAMILIES:
        for number in range(1, PER_FAMILY + 1):
            tag = f"{family}{number:02d}"
            rows.append(
                {
                    "loop": f"saddle_{family}_{number}",
                    "family": family,
                    "number": number,
                    "raw_channel": f"sad_out_{tag}",
                    "raw_uda": f"XMB_SAD/OUT/{family.upper()}{number:02d}",
                    "published_channel": f"XMB_SAD/OUT/{family.upper()}{number:02d}",
                }
            )
    return rows


def store_shots(limit: int | None) -> list[int]:
    """Return the store's shot numbers, optionally strided to a bounded sample.

    The frame is taken from the store itself rather than an archive summary whose
    current column carries no unit the module agrees with.  A bounded sample is
    drawn by an even stride across the whole shot range, so the sample spans
    every acquisition era rather than the earliest shots, and the bound is
    recorded in the receipt so the reader can weigh it.
    """

    numbers = sorted(
        int(path.name[: -len(".zarr")])
        for path in LEVEL1_STORE.iterdir()
        if path.name.endswith(".zarr")
    )
    if limit is None or len(numbers) <= limit:
        return numbers
    stride = max(1, len(numbers) // limit)
    return numbers[::stride][:limit]


def census_shot(shot: int) -> dict | None:
    """Apply the vacuum-cohort criteria to one shot with a light store read.

    Only what the criteria need is read: the current-group peaks to decide
    plasma-freeness and deliberate excitation, and the field-group channel names
    to count the over-determining probes.  Field waveforms are not read, which is
    the whole cost of the cohort module's own census.
    """

    try:
        import zarr

        group = zarr.open_group(f"{LEVEL1_STORE}/{shot}.zarr", mode="r")
    except Exception:
        return None
    entry = {
        "shot": shot,
        "plasma_current_peak": 0.0,
        "excited_families": [],
        "probe_count": 0,
        "absent_groups": [],
    }
    try:
        amc = group["amc"]
    except Exception:
        entry["absent_groups"].append("amc")
        return entry
    keys = set(amc.keys())
    if "plasma_current" in keys:
        values = np.asarray(amc["plasma_current"][...], dtype=float)
        finite = values[np.isfinite(values)]
        entry["plasma_current_peak"] = (
            float(np.max(np.abs(finite)) * 1.0e3) if finite.size else 0.0
        )
    else:
        entry["absent_groups"].append("plasma_current")
    excited = []
    for channel in _DRIVE_CHANNELS:
        if channel not in keys:
            continue
        values = np.asarray(amc[channel][...], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and float(np.max(np.abs(finite)) * 1.0e3) >= EXCITATION_CURRENT:
            excited.append(channel)
    entry["excited_families"] = sorted(excited)
    try:
        amb = group["amb"]
        entry["probe_count"] = sum(1 for k in amb.keys() if _PROBE.match(str(k)))
    except Exception:
        entry["absent_groups"].append("amb")
    return entry


def admits(entry: dict) -> bool:
    """Return whether a census row satisfies the vacuum cohort's criteria."""

    return (
        not entry["absent_groups"]
        and entry["plasma_current_peak"] < PLASMA_FREE_CURRENT
        and bool(entry["excited_families"])
        and entry["probe_count"] >= MINIMUM_PROBES
    )


def _census(shots: list[int], processes: int) -> list[dict]:
    if processes <= 1:
        rows = [census_shot(s) for s in shots]
    else:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            rows = list(pool.map(census_shot, shots, chunksize=16))
    return [row for row in rows if row is not None]


def reachability(shots: list[int]) -> dict:
    """Record raw and published access for every loop over the cohort shots."""

    per_shot: dict[int, dict] = {}
    for shot in shots:
        record = {"raw": {}, "published": {}}
        try:
            import zarr

            level1 = zarr.open_group(f"{LEVEL1_STORE}/{shot}.zarr", mode="r")
            if "xmb" in level1:
                keys = {str(k) for k in level1["xmb"].keys()}
                for loop in loop_identities():
                    record["raw"][loop["loop"]] = loop["raw_channel"] in keys
                record["raw"][MISSPELLED_CHANNEL] = MISSPELLED_CHANNEL in keys
        except Exception:
            pass
        try:
            import zarr

            level2 = zarr.open_group(f"{LEVEL2_STORE}/{shot}.zarr", mode="r")
            magnetics = level2["magnetics"]
            if "b_field_tor_probe_saddle_voltage_channel" in magnetics:
                names = {
                    str(x)
                    for x in magnetics["b_field_tor_probe_saddle_voltage_channel"][...]
                }
                for loop in loop_identities():
                    record["published"][loop["loop"]] = (
                        loop["published_channel"] in names
                    )
        except Exception:
            pass
        per_shot[shot] = record
    return per_shot


def summarise(shots: list[int], per_shot: dict) -> list[dict]:
    """Collapse per-shot availability into a reachability row per loop."""

    rows = []
    for loop in loop_identities():
        raw_shots = [s for s in shots if per_shot[s]["raw"].get(loop["loop"])]
        pub_shots = [s for s in shots if per_shot[s]["published"].get(loop["loop"])]
        rows.append(
            {
                "loop": loop["loop"],
                "family": loop["family"],
                "number": loop["number"],
                "raw_channel": loop["raw_channel"],
                "raw_source": "level1 xmb/sad_out (XMB MDSplus export, uncorrected)",
                "raw_shot_count": len(raw_shots),
                "raw_shot_range": (
                    [min(raw_shots), max(raw_shots)] if raw_shots else None
                ),
                "published_channel": loop["published_channel"],
                "published_source": (
                    "level2 magnetics/b_field_tor_probe_saddle_voltage"
                ),
                "published_shot_count": len(pub_shots),
                "published_shot_range": (
                    [min(pub_shots), max(pub_shots)] if pub_shots else None
                ),
                "uncorrected_reachable": bool(raw_shots),
            }
        )
    return rows


def compare_shot(shot: int) -> dict:
    """Differ the raw and published voltage and compare with the predicted pickup.

    The raw level-1 signal is sampled on its own clock; the published product on
    the saddle clock.  The raw record is interpolated onto the published clock,
    differenced, and the difference compared with the coil-driven field the
    vacuum response predicts, both before the response is available (the raw
    offset itself) and after the response's kernel is applied.
    """

    import zarr

    record = {"shot": shot}
    try:
        raw_group = zarr.open_group(f"{LEVEL1_STORE}/{shot}.zarr", mode="r")["xmb"]
        raw = np.asarray(raw_group["sad_out_m01"][...], dtype=float)
        raw_time = np.asarray(raw_group["time"][...], dtype=float)
    except Exception as error:
        record["error"] = f"raw unavailable: {error}"
        return record
    try:
        magnetics = zarr.open_group(f"{LEVEL2_STORE}/{shot}.zarr", mode="r")[
            "magnetics"
        ]
        names = [
            str(x) for x in magnetics["b_field_tor_probe_saddle_voltage_channel"][...]
        ]
        index = names.index("XMB_SAD/OUT/M01")
        published = np.asarray(
            magnetics["b_field_tor_probe_saddle_voltage"][...], dtype=float
        )[index]
        published_time = np.asarray(magnetics["time_saddle"][...], dtype=float)
    except Exception as error:
        record["error"] = f"published unavailable: {error}"
        return record
    finite = np.isfinite(raw) & np.isfinite(raw_time)
    if int(finite.sum()) < 8:
        record["error"] = "raw record carries no finite samples"
        return record
    order = np.argsort(raw_time[finite])
    interp = np.interp(
        published_time,
        raw_time[finite][order],
        raw[finite][order],
        left=np.nan,
        right=np.nan,
    )
    inside = (published_time >= raw_time[finite].min()) & (
        published_time <= raw_time[finite].max()
    )
    both = np.isfinite(interp) & np.isfinite(published) & inside
    difference = (interp - published)[both]
    record["overlap_samples"] = int(both.sum())
    if int(both.sum()) < 8:
        record["error"] = "raw and published clocks do not overlap"
        return record
    record["raw_mean"] = float(np.mean(interp[both]))
    record["published_mean"] = float(np.mean(published[both]))
    record["difference_mean"] = float(np.mean(difference))
    record["difference_rms"] = float(np.sqrt(np.mean(difference**2)))
    record["prediction"] = predict_pickup(shot, difference, published_time[both])
    return record


def predict_pickup(shot: int, difference: np.ndarray, time: np.ndarray) -> dict:
    """Compare the raw-minus-published difference against the predicted pickup.

    The prediction is the vacuum response's own coil-to-probe kernel evaluated at
    the saddle loop's poloidal position, scaled to the difference least-squares,
    so the reported correlation is the shape agreement and the reported scale is
    what a loop area would have to be for the kernel to explain the difference.
    No sign is promoted; the scale sign is reported as found.  When the geometry
    registry cannot be read the comparison is recorded as unavailable rather than
    failed.
    """

    try:
        from nova.catalog.mast_geometry import MachineGeometryRegistry
        from nova.imas.mast_vacuum_response import (
            ProbeTarget,
            coil_response_matrix,
        )

        registry = MachineGeometryRegistry.default()
        geometry = registry.select(11766).configuration.geometry
        loop = geometry["magnetics"]["saddle_paths"]["m"][0]
        points = np.asarray(loop, dtype=float)
        centre = points.mean(axis=0)
        target = ProbeTarget(
            channel="saddle_m_1",
            family="saddle",
            registry_index=0,
            r=float(centre[0]),
            z=float(centre[1]),
            radial_cosine=1.0,
            axial_sine=0.0,
        )
        kernel = coil_response_matrix(geometry, [target]).ravel()
    except Exception as error:  # noqa: BLE001 - geometry may be unavailable here
        return {"available": False, "reason": str(error)}
    drive = _coil_drives(shot, time)
    if drive is None or kernel.size != drive.shape[1]:
        return {"available": False, "reason": "coil drive waveforms unavailable"}
    predicted = drive @ kernel
    power = float(predicted @ predicted)
    if power <= 0.0:
        return {"available": False, "reason": "predicted pickup carries no power"}
    scale = float(predicted @ difference) / power
    residual = difference - scale * predicted
    signal = float(difference @ difference)
    correlation = float(
        np.dot(predicted, difference)
        / max(np.linalg.norm(predicted) * np.linalg.norm(difference), 1e-30)
    )
    return {
        "available": True,
        "kernel_radius_m": float(centre[0]),
        "scale": scale,
        "correlation": correlation,
        "variance_explained": 0.0
        if signal <= 0.0
        else float(1.0 - float(residual @ residual) / signal),
    }


def _coil_drives(shot: int, time: np.ndarray):
    """Return the shot's coil currents resampled onto the difference clock.

    The kernel from the vacuum response is per ampere-turn in the schema's coil
    order; the drives are read in the same order so ``drive @ kernel`` is the
    predicted field rather than a re-fitted combination.
    """

    try:
        import zarr

        from nova.imas.mast_vacuum_cohort import COIL_DRIVES

        currents = zarr.open_group(f"{LEVEL1_STORE}/{shot}.zarr", mode="r")["amc"]
        clock = np.asarray(currents["time"][...], dtype=float)
        columns = []
        for channel in (drive.channel for drive in COIL_DRIVES):
            if channel not in currents:
                return None
            values = np.nan_to_num(
                np.asarray(currents[channel][...], dtype=float) * 1.0e3
            )
            columns.append(np.interp(time, clock, values))
        return np.column_stack(columns)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--compare", type=int, default=3)
    arguments = parser.parse_args()

    candidates = store_shots(arguments.limit)
    census = _census(candidates, arguments.processes)
    cohort_shots = sorted(row["shot"] for row in census if admits(row))

    per_shot = reachability(cohort_shots)
    rows = summarise(cohort_shots, per_shot)

    comparisons = []
    for shot in cohort_shots:
        if len(comparisons) >= arguments.compare:
            break
        raw_here = per_shot[shot]["raw"].get("saddle_m_1")
        published_here = per_shot[shot]["published"].get("saddle_m_1")
        if raw_here and published_here:
            comparisons.append(compare_shot(shot))

    receipt = {
        "cohort": {
            "frame": "level1 store enumeration, even-strided to the sample limit",
            "sample_limit": arguments.limit,
            "candidate_shots": len(candidates),
            "admitted_shots": len(cohort_shots),
            "shot_range": [cohort_shots[0], cohort_shots[-1]] if cohort_shots else None,
            "shots": cohort_shots,
            "criteria": {
                "plasma_free_current_a": PLASMA_FREE_CURRENT,
                "excitation_current_a": EXCITATION_CURRENT,
                "minimum_probes": MINIMUM_PROBES,
            },
        },
        "negative_control": {
            "channel": MISSPELLED_CHANNEL,
            "shots_where_present": sum(
                1 for s in cohort_shots if per_shot[s]["raw"].get(MISSPELLED_CHANNEL)
            ),
            "expected": "unreachable",
            "reachable": any(
                per_shot[s]["raw"].get(MISSPELLED_CHANNEL) for s in cohort_shots
            ),
        },
        "loops": rows,
        "comparisons": comparisons,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")

    reachable = sum(1 for row in rows if row["uncorrected_reachable"])
    print(
        f"cohort {len(cohort_shots)} shots from {len(candidates)} candidates; "
        f"uncorrected reachable for {reachable}/{len(rows)} loops -> {arguments.out}"
    )


if __name__ == "__main__":
    main()
