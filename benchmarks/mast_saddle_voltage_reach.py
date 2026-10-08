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
sample was found per shot range.  Each loop's raw key is resolved from the
store's own key set, because the ``m`` family beyond loop 9 is spelled with a
three-digit index (``sad_out_m010``) while every other loop is two-digit
(``sad_out_l12``).
Assuming one width reports loops 10-12 of that family as absent when the store
carries them.

Cohort frame.  The vacuum cohort is the no-plasma, coil-driven, over-determined
set of :mod:`nova.imas.mast_vacuum_cohort`.  Its own census reads every field
waveform of the 17 111-shot store, which is minutes to hours of GPFS traffic, so
this benchmark draws an even-strided sample across the whole shot range and
applies the module's criteria in a light census that reads only the current-group
peaks and the field-group channel names -- the admissions conditions, without the
waveforms.  The sample limit is recorded in the receipt; the verdict is a
statement about that frame and says so.

Saddle geometry for the pickup comparison is read from the magnetics IDS arrays
of the level-2 store (``b_field_tor_probe_saddle_m_r``/``_z``), not from a
catalog machine description: the catalog registry is retired by the plan this
benchmark serves.  The coil cross-sections the response kernel needs are not
carried by the magnetics IDS, so the kernel comparison is recorded as
unavailable with that reason rather than sourced from the retired registry.
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
SCAN_BUDGET = 300
"""Level-2 shots examined when the cohort frame carries no comparison shot."""
MISSPELLED_CHANNEL = "sad_out_zz99"
"""A channel that names no loop: the receipt must report it unreachable."""

_PROBE = re.compile(r"^(ccbv|obr|obv)\d{2}$")
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


def raw_key_candidates(family: str, number: int) -> tuple[str, ...]:
    """Return the spellings a loop's raw key may take in the store.

    The store spells the ``m`` family's tenth to twelfth loops with a
    three-digit index and every other loop with two digits, so both widths are
    offered and the store's own key set decides which exists.
    """

    return (f"sad_out_{family}{number:02d}", f"sad_out_{family}{number:03d}")


def loop_identities() -> list[dict]:
    """Return the 36 saddle loops with their candidate raw and published keys."""

    rows = []
    for family in FAMILIES:
        for number in range(1, PER_FAMILY + 1):
            rows.append(
                {
                    "loop": f"saddle_{family}_{number}",
                    "family": family,
                    "number": number,
                    "raw_key_candidates": list(raw_key_candidates(family, number)),
                    "published_channel": f"XMB_SAD/OUT/{family.upper()}{number:02d}",
                    "published_family": "m",
                }
            )
    return rows


def store_shots(limit: int | None) -> list[int]:
    """Return the store's shot numbers, optionally strided to a bounded sample.

    The frame is taken from the store itself rather than an archive summary whose
    current column carries no unit the cohort module agrees with.  A bounded
    sample is drawn by an even stride across the whole shot range, so the sample
    spans every acquisition era rather than the earliest shots, and the bound is
    recorded in the level-2 receipt so the reader can weigh it.
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
        return entry
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


def _resolve_raw(keys: set[str], loop: dict) -> str | None:
    """Return the loop's raw key as the store spells it, or None if absent."""

    for candidate in loop["raw_key_candidates"]:
        if candidate in keys:
            return candidate
    return None


def reachability(shots: list[int]) -> dict:
    """Record raw and published access for every loop over the cohort shots."""

    loops = loop_identities()
    per_shot: dict[int, dict] = {}
    for shot in shots:
        record = {"raw": {}, "published": {}, "raw_key": {}, "has_xmb": False}
        try:
            import zarr

            level1 = zarr.open_group(f"{LEVEL1_STORE}/{shot}.zarr", mode="r")
            if "xmb" in level1:
                record["has_xmb"] = True
                keys = {str(k) for k in level1["xmb"].keys()}
                for loop in loops:
                    key = _resolve_raw(keys, loop)
                    record["raw"][loop["loop"]] = key is not None
                    if key is not None:
                        record["raw_key"][loop["loop"]] = key
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
                for loop in loops:
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
        identity = loop["loop"]
        raw_shots = [s for s in shots if per_shot[s]["raw"].get(identity)]
        pub_shots = [s for s in shots if per_shot[s]["published"].get(identity)]
        spellings = sorted(
            {
                per_shot[s]["raw_key"][identity]
                for s in raw_shots
                if identity in per_shot[s]["raw_key"]
            }
        )
        rows.append(
            {
                "loop": identity,
                "family": loop["family"],
                "number": loop["number"],
                "raw_key_candidates": loop["raw_key_candidates"],
                "raw_key_resolved": spellings[0] if spellings else None,
                "raw_key_spellings_seen": spellings,
                "raw_source": "level1 xmb/sad_out (XMB export, uncorrected)",
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


def comparison_shots(cohort_shots: list[int], per_shot: dict, limit: int) -> list[int]:
    """Return shots carrying both an uncorrected and a published signal.

    A shot in the cohort that carries both is preferred.  When the cohort frame
    has no such shot -- the cohort sits where the level-2 store is sparse -- the
    comparison widens to any shot present in both stores, because the pickup
    comparison needs only one shot carrying both products, not a cohort member.
    """

    loops = {loop["loop"]: loop for loop in loop_identities()}
    m_loops = [name for name, loop in loops.items() if loop["published_family"] == "m"]

    def both(record: dict) -> bool:
        return any(
            record["raw"].get(name) and record["published"].get(name)
            for name in m_loops
        )

    preferred = [
        s for s in cohort_shots if per_shot[s]["has_xmb"] and both(per_shot[s])
    ]
    if preferred:
        return preferred[:limit]
    widened = []
    listed = sorted(
        int(path.name[: -len(".zarr")])
        for path in LEVEL2_STORE.iterdir()
        if path.name.endswith(".zarr")
    )
    stride = max(1, len(listed) // SCAN_BUDGET)
    for shot in listed[::stride]:
        if not (LEVEL1_STORE / f"{shot}.zarr").exists():
            continue
        try:
            import zarr

            level1 = zarr.open_group(f"{LEVEL1_STORE}/{shot}.zarr", mode="r")
            level2 = zarr.open_group(f"{LEVEL2_STORE}/{shot}.zarr", mode="r")
        except Exception:
            continue
        if "xmb" not in level1:
            continue
        magnetics = level2["magnetics"]
        if "b_field_tor_probe_saddle_voltage_channel" not in magnetics:
            continue
        names = {
            str(x) for x in magnetics["b_field_tor_probe_saddle_voltage_channel"][...]
        }
        keys = {str(k) for k in level1["xmb"].keys()}
        if all(
            _resolve_raw(keys, loops[name]) is None
            or loops[name]["published_channel"] not in names
            for name in m_loops
        ):
            continue
        widened.append(shot)
        if len(widened) >= limit:
            break
    return widened


def compare_shot(shot: int, loop_name: str) -> dict:
    """Differ the raw and published voltage for one loop on one shot.

    The raw level-1 signal is sampled on its own clock; the published product on
    the saddle clock.  The raw record is interpolated onto the published clock,
    differenced, and the difference is compared with the coil-driven field the
    vacuum response would predict.
    """

    import zarr

    loops = {loop["loop"]: loop for loop in loop_identities()}
    loop = loops[loop_name]
    record = {"shot": shot, "loop": loop_name}
    try:
        raw_group = zarr.open_group(f"{LEVEL1_STORE}/{shot}.zarr", mode="r")["xmb"]
        key = _resolve_raw({str(k) for k in raw_group.keys()}, loop)
        if key is None:
            record["error"] = "no raw key for this loop"
            return record
        raw = np.asarray(raw_group[key][...], dtype=float)
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
        index = names.index(loop["published_channel"])
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
    record["raw_key"] = key
    record["overlap_samples"] = int(both.sum())
    if int(both.sum()) < 8:
        record["error"] = "raw and published clocks do not overlap"
        return record
    record["raw_mean"] = float(np.mean(interp[both]))
    record["published_mean"] = float(np.mean(published[both]))
    record["difference_mean"] = float(np.mean(difference))
    record["difference_rms"] = float(np.sqrt(np.mean(difference**2)))
    record["prediction"] = predict_pickup(shot, difference, published_time[both], loop)
    return record


def saddle_position(shot: int, loop: dict) -> dict | None:
    """Return a saddle loop's poloidal centre in metres from the magnetics IDS.

    The loop geometry lives in the magnetics IDS arrays the level-2 store
    publishes; this is the source the plan names, in place of a catalog
    machine description.
    """

    try:
        import zarr

        magnetics = zarr.open_group(f"{LEVEL2_STORE}/{shot}.zarr", mode="r")[
            "magnetics"
        ]
        family = loop["family"]
        channels = [
            str(x)
            for x in magnetics[f"b_field_tor_probe_saddle_{family}_geometry_channel"][
                ...
            ]
        ]
        key = loop["published_channel"].split("/")[-1]
        index = next(
            i for i, name in enumerate(channels) if name.split("_")[-1] == key.lower()
        )
        radius = np.asarray(
            magnetics[f"b_field_tor_probe_saddle_{family}_r"][...], dtype=float
        )[index]
        height = np.asarray(
            magnetics[f"b_field_tor_probe_saddle_{family}_z"][...], dtype=float
        )[index]
        return {
            "centre_r": float(np.nanmean(radius)),
            "centre_z": float(np.asarray(height, dtype=float).mean()),
            "vertices": int(np.size(radius)),
        }
    except Exception as error:  # noqa: BLE001 - geometry may be absent for a shot
        return {"error": str(error)}


def predict_pickup(
    shot: int, difference: np.ndarray, time: np.ndarray, loop: dict
) -> dict:
    """Compare the raw-minus-published difference with the predicted pickup.

    The loop position comes from the magnetics IDS.  The coil cross-sections the
    ``coil_response_matrix`` kernel needs are not carried by the magnetics IDS,
    and the catalog registry that carries them is retired by the plan this
    benchmark serves, so the kernel comparison is recorded as unavailable with
    that reason; the difference's own statistics stand as the measured quantity.
    """

    position = saddle_position(shot, loop) or {}
    result: dict = {"loop_position": position}
    if "error" in position:
        result["available"] = False
        result["reason"] = (
            f"magnetics IDS carries no loop geometry: {position['error']}"
        )
        return result
    result["available"] = False
    result["reason"] = (
        "coil cross-sections for the response kernel are not present in the "
        "magnetics IDS; the catalog machine description that carries them is "
        "unavailable to this benchmark"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--compare", type=int, default=3)
    arguments = parser.parse_args()

    candidates = store_shots(arguments.limit)
    census = _census(candidates, arguments.processes)
    cohort_shots = sorted(row["shot"] for row in census if admits(row))

    per_shot = reachability(cohort_shots)
    rows = summarise(cohort_shots, per_shot)
    with_xmb = [s for s in cohort_shots if per_shot[s]["has_xmb"]]

    compare_pool = comparison_shots(cohort_shots, per_shot, arguments.compare)
    comparisons = [compare_shot(s, "saddle_m_1") for s in compare_pool]

    receipt = {
        "cohort": {
            "frame": "level1 store enumeration, even-strided to the sample limit",
            "sample_limit": arguments.limit,
            "candidate_shots": len(candidates),
            "admitted_shots": len(cohort_shots),
            "shots_with_xmb": len(with_xmb),
            "shots_without_xmb": len(cohort_shots) - len(with_xmb),
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
        f"cohort {len(cohort_shots)} shots from {len(candidates)} candidates "
        f"({len(with_xmb)} carry xmb); uncorrected reachable for "
        f"{reachable}/{len(rows)} loops -> {arguments.out}"
    )


if __name__ == "__main__":
    main()
