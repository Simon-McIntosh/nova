"""Persist the limited-class census as committable evidence.

The per-frame rows go to a compact npz rather than the working JSON, and the
banding is stored AS COMPUTED alongside the aggregate, because the finding is
that the aggregate hides a drift the bands expose.

The receipt is descriptive by default. The interpretive statements -- what the
crossed table means, which acceptance measure follows, which cells are too
thin to lean on -- are a reading of the corpus as first written and are false
about a corpus relabelled after a repair, so they are emitted only when the
caller asks for the reference reading.

An existing array file is never overwritten without being asked, because that
file is the only pre-repair evidence a later comparison has to judge against;
losing it would leave the comparison resting on restated numbers.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE = Path.home() / "nova-media-logs/limited_class_census.json"
OUT = Path("docs/figures/limited-boundary-census")
DEFAULT_NAME = "limited-class-census"
BANDS = (
    (0.0, 0.05),
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 1.0),
)


def summarise(values: np.ndarray) -> dict | None:
    """Return the distribution summary one cell of the table carries."""
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
        "fraction_below_half": float(np.mean(values < 0.5)),
    }


def _reference_reading() -> dict[str, object]:
    """Return the interpretive statements the corpus as first written carries."""
    return {
        "findings": [
            "a cold start alone is harmless: diverted warm 0.838 against cold 0.865",
            "the limited class is impaired: median 0.285 with 87.7 per cent below half",
            "limited and cold together collapse: median 0.066 over 1228 frames",
            "the limited defect survives time-matching at every epoch, sitting two "
            "to four fold below its own-epoch diverted control",
            "the diverted control is itself not flat: it drifts monotonically from "
            "0.70 early to 1.15 late, so every ratio quoted against a diverted "
            "baseline inherits that drift",
        ],
        "acceptance_measure": (
            "a repair is judged by the matched-epoch ratio of limited to diverted "
            "reaching one, NOT by the limited median reaching unity: the control is "
            "not at unity either, so an absolute target would fail a correct fix in "
            "the early bands where diverted sits at 0.70 and pass an incorrect one "
            "late where the control overshoots"
        ),
        "cautions": [
            "the 0 to 50 ms diverted control carries only 36 frames, so that band's "
            "control is thin and should not be leaned on alone",
            "the control drift is monotonic in time and spans 30 per cent low to 15 "
            "per cent high, a shape suggesting something scaling with the discharge "
            "rather than a defect in one branch, so it should not be assumed to "
            "belong to the same family as the limited defect",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    """Write the arrays, the crossed table, the banding and the receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="basename of the array and receipt files this writes",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="also record the interpretive reading of the corpus as first "
        "written; those statements are false about any other corpus",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="overwrite an existing array file rather than refusing; the "
        "banked arrays are the only pre-repair evidence a comparison has",
    )
    arguments = parser.parse_args(argv)
    census = json.loads(arguments.source.read_text())
    rows = census["rows"]
    shot = np.asarray([r["shot"] for r in rows], dtype=np.int32)
    time_s = np.asarray([r["time_s"] for r in rows], dtype=np.float64)
    diverted = np.asarray([r["diverted"] for r in rows], dtype=bool)
    cold = np.asarray([r["cold"] for r in rows], dtype=bool)
    ratio = np.asarray([r["ratio"] for r in rows], dtype=np.float64)

    out = arguments.output
    out.mkdir(parents=True, exist_ok=True)
    arrays = out / f"{arguments.name}.npz"
    if arrays.exists() and not arguments.replace:
        raise SystemExit(
            f"{arrays} already exists; it may be the only pre-repair evidence a "
            "later comparison can be judged against. Pass --name to write "
            "beside it, or --replace to overwrite it deliberately."
        )
    np.savez_compressed(
        arrays,
        shot=shot,
        time_s=time_s,
        diverted=diverted,
        cold_start=cold,
        span_ratio_nova_over_efit=ratio,
    )
    digest = hashlib.sha256(arrays.read_bytes()).hexdigest()

    crossed = {
        f"{'diverted' if d else 'limited'}/{'cold' if c else 'warm'}": summarise(
            ratio[(diverted == d) & (cold == c)]
        )
        for d in (True, False)
        for c in (False, True)
    }
    banded = []
    for low, high in BANDS:
        window = (time_s >= low) & (time_s < high)
        control = summarise(ratio[window & diverted & ~cold])
        limited_warm = summarise(ratio[window & ~diverted & ~cold])
        limited_cold = summarise(ratio[window & ~diverted & cold])
        banded.append(
            {
                "band_ms": [1e3 * low, 1e3 * high],
                "diverted_warm": control,
                "limited_warm": limited_warm,
                "limited_cold": limited_cold,
                "limited_over_diverted_median": (
                    limited_warm["median"] / control["median"]
                    if control and limited_warm
                    else None
                ),
            }
        )

    receipt = {
        "question": (
            "does nova's axis-to-boundary flux span agree with EFIT's, and does "
            "the answer depend on the topology class or on the start condition"
        ),
        "method": (
            "for every guarded frame of every labelled shot, the ratio of nova's "
            "stored axis-to-boundary flux span to EFIT's on the nearest efm slice; "
            "a frame is a cold start when its predecessor failed the branch guard; "
            "session files and the level-1 efm store only, no solve and no device"
        ),
        "session": census.get("session"),
        "shots_attempted": census.get("shots_attempted"),
        "unreadable_sessions": len(census.get("unreadable", ())),
        # A null session or shot count means the source predates the census
        # recording its own provenance, not that the census read nothing.
        "source_records_its_own_provenance": "session" in census,
        "guarded_frames": int(ratio.size),
        "shots_with_a_limited_cold_start": int(
            len({int(s) for s, d, c in zip(shot, diverted, cold) if not d and c})
        ),
        "crossed_table": crossed,
        "time_banded": banded,
        "arrays": {"path": str(arrays), "sha256": digest},
    }
    if arguments.reference:
        receipt.update(_reference_reading())
    (out / f"{arguments.name}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True)
    )
    megabytes = arrays.stat().st_size / 1048576
    print(f"wrote {arrays} ({megabytes:.2f} MB) sha256 {digest[:16]}")
    print(f"wrote {out / f'{arguments.name}.json'}")


if __name__ == "__main__":
    main()
