"""Judge a relabelled corpus against the banked pre-repair census.

The acceptance measure is the MATCHED-EPOCH ratio of the limited median to
the diverted median, never either median against unity. That is not a
stylistic choice: the banked census established that the diverted control
itself drifts from 0.70 early to 1.15 late, so an absolute target would fail
a correct repair in the early bands where the control sits at 0.70 and pass
an incorrect one late where it overshoots.

Two properties of a small relabelled corpus shape how that measure is read,
and both were measured rather than assumed.

*The classes are time-segregated within a shot.* Limited frames sit early and
diverted frames late, so over seven shots there is no time band holding
twenty warm frames of each class -- measured 25, 25, 16, 3, 0, 1, 0 limited
against 0, 0, 14, 38, 89, 65, 59 diverted. The corpus-wide measure works only
because different shots divert at different times and the bands pool across
them. So the diverted control is estimated from the whole banked census,
which is legitimate precisely because a diverted read is untouched by the
repair: the paired check below verifies that assumption on the very frames
where both corpora overlap, rather than trusting it.

*The same frame appears in both corpora.* That admits a paired comparison,
which is stronger than any comparison of medians because each frame is its
own control -- shot and epoch stop being confounders. It is the primary
measure here, and the banded ratio is what gives it a target.

The comparison is against the committed npz rather than a remembered number,
and the pre-repair figures are read from it rather than restated here, so
this cannot silently drift from the evidence it is judged against.

A band with too few frames to support a median is reported with its counts
rather than skipped in silence, because a thin corpus otherwise produces an
empty comparison that reads exactly like a comparison finding nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BANKED = Path("docs/figures/limited-boundary-census/limited-class-census.npz")
BANDS = (
    (0.0, 0.05),
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 1.0),
)

#: Fewest warm frames a class needs in a band before its median is quoted.
MINIMUM_BAND_FRAMES = 20

#: Whether a shadow is derived for the first iterate of a run. The wall-anchor
#: masking alone leaves that read unmasked, so the cold cell stays put and a
#: cold improvement is a finding; a corpus relabelled after a first-iterate
#: repair lands should show the cell move, and reporting it as unexplained
#: would then be backwards. The caller states which corpus it holds.
COLD_START_ADDRESSED = False

#: How far a diverted frame's ratio may move before it is a finding. The
#: masking gives a private wall anchor a losing score rather than an infinite
#: one, because the wall reader interpolates at fixed shape, so a diverted
#: read is meant to come back unchanged.
DIVERTED_TOLERANCE = 0.01


def _load(path: Path) -> dict[str, np.ndarray]:
    """Return one census as named arrays."""
    with np.load(path) as archive:
        return {
            "shot": np.asarray(archive["shot"], dtype=int),
            "time_s": np.asarray(archive["time_s"], dtype=float),
            "ratio": np.asarray(archive["span_ratio_nova_over_efit"], dtype=float),
            "diverted": np.asarray(archive["diverted"], dtype=bool),
            "cold": np.asarray(archive["cold_start"], dtype=bool),
        }


def _restrict(
    census: dict[str, np.ndarray], shots: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the census frames belonging to one set of shots."""
    keep = np.isin(census["shot"], shots)
    return {name: column[keep] for name, column in census.items()}


def _band_median(
    census: dict[str, np.ndarray],
    low: float,
    high: float,
    diverted: bool,
    minimum: int,
) -> dict[str, object]:
    """Return one class's warm median in one time band, or its count alone."""
    window = (census["time_s"] >= low) & (census["time_s"] < high)
    admitted = window & (census["diverted"] == diverted) & ~census["cold"]
    selected = census["ratio"][admitted]
    if selected.size < minimum:
        return {"quoted": False, "count": int(selected.size)}
    return {
        "quoted": True,
        "count": int(selected.size),
        "median": float(np.median(selected)),
    }


def _control_curve(
    banked: dict[str, np.ndarray], minimum: int
) -> dict[tuple[float, float], dict[str, object]]:
    """Return the diverted warm median per band over the WHOLE banked census.

    The control is taken corpus-wide rather than from the relabelled shots
    because a relabelled handful carries almost no diverted frames early and
    almost no limited frames late. A diverted read is untouched by the
    repair, so the same curve serves both sides of the comparison and the
    change in a ratio is entirely the limited side's movement.
    """
    return {
        band: _band_median(banked, *band, diverted=True, minimum=minimum)
        for band in BANDS
    }


def _paired(
    old: dict[str, np.ndarray], new: dict[str, np.ndarray]
) -> dict[str, object]:
    """Return the per-frame before-and-after change, joined on shot and time.

    Each frame is its own control, so this sees a movement that a comparison
    of medians can only infer. Frames present on one side alone are counted
    rather than dropped quietly: a repair that lets more frames pass the
    branch guard SHOULD add frames, and that is a result, not a join defect.
    """

    def identities(census: dict[str, np.ndarray]) -> list[tuple[int, float]]:
        """Return one frame identity per row: the shot and its stored time."""
        return [
            (int(shot), round(float(time), 9))
            for shot, time in zip(census["shot"], census["time_s"], strict=True)
        ]

    was = {
        identity: (float(ratio), bool(diverted))
        for identity, ratio, diverted in zip(
            identities(old), old["ratio"], old["diverted"], strict=True
        )
    }
    matched, added, reclassified = [], 0, 0
    for identity, ratio, diverted, cold in zip(
        identities(new), new["ratio"], new["diverted"], new["cold"], strict=True
    ):
        if identity not in was:
            added += 1
            continue
        previous, previous_class = was[identity]
        reclassified += int(previous_class != bool(diverted))
        matched.append((previous, float(ratio), bool(diverted), bool(cold)))
    if not matched:
        return {"matched_frame_count": 0, "frames_only_after": added}
    before = np.array([row[0] for row in matched])
    after = np.array([row[1] for row in matched])
    diverted = np.array([row[2] for row in matched])
    cold = np.array([row[3] for row in matched])
    report = {
        "matched_frame_count": len(matched),
        "frames_only_after": added,
        "frames_only_before": int(len(was) - len(matched)),
        "frames_changing_topology_class": reclassified,
    }
    for name, mask in (
        ("limited_warm", ~diverted & ~cold),
        ("limited_cold", ~diverted & cold),
        ("diverted_warm", diverted & ~cold),
    ):
        if not np.any(mask):
            report[name] = {"count": 0}
            continue
        change = after[mask] - before[mask]
        report[name] = {
            "count": int(mask.sum()),
            "before_median": float(np.median(before[mask])),
            "after_median": float(np.median(after[mask])),
            "median_change": float(np.median(change)),
            "maximum_absolute_change": float(np.max(np.abs(change))),
            "fraction_improved": float(np.mean(change > 0.0)),
        }
    return report


def compare(
    before: Path,
    after: Path,
    cold_start_addressed: bool = COLD_START_ADDRESSED,
    minimum: int = MINIMUM_BAND_FRAMES,
) -> dict[str, object]:
    """Return the verdict of a relabelled census against the banked one."""
    banked, new = _load(before), _load(after)
    relabelled = np.unique(new["shot"])
    banked_shots = set(banked["shot"].tolist())
    missing = sorted(int(shot) for shot in relabelled if shot not in banked_shots)
    if missing:
        raise ValueError(
            f"the banked census carries no frames for relabelled shots {missing}, "
            "so those shots have no matched population to be judged against"
        )
    old = _restrict(banked, relabelled)
    control = _control_curve(banked, minimum)

    bands, thin = [], []
    for band in BANDS:
        reference = control[band]
        was = _band_median(old, *band, diverted=False, minimum=minimum)
        now = _band_median(new, *band, diverted=False, minimum=minimum)
        if not (reference["quoted"] and was["quoted"] and now["quoted"]):
            thin.append(
                {
                    "band_ms": [1e3 * band[0], 1e3 * band[1]],
                    "limited_warm_before": was["count"],
                    "limited_warm_after": now["count"],
                    "diverted_control": reference["count"],
                }
            )
            continue
        bands.append(
            {
                "band_ms": [1e3 * band[0], 1e3 * band[1]],
                "diverted_control_median": reference["median"],
                "diverted_control_count": reference["count"],
                "limited_warm_before": was,
                "limited_warm_after": now,
                "matched_epoch_ratio_before": was["median"] / reference["median"],
                "matched_epoch_ratio_after": now["median"] / reference["median"],
                "improved": now["median"] > was["median"],
                "reaches_unity_within_10_percent": (
                    abs(now["median"] / reference["median"] - 1.0) <= 0.10
                ),
            }
        )

    paired = _paired(old, new)
    cold = {
        "before_median": float(np.median(old["ratio"][~old["diverted"] & old["cold"]])),
        "after_median": float(np.median(new["ratio"][~new["diverted"] & new["cold"]])),
        "expected_to_move": cold_start_addressed,
    }
    verdict = {
        "acceptance_measure": (
            "the matched-epoch ratio of the limited median to the diverted "
            "median reaching one, never either median against unity"
        ),
        "primary_measure": (
            "the paired per-frame change, because the same frame appears in "
            "both corpora and is therefore its own control"
        ),
        "shots_compared": [int(s) for s in relabelled],
        "shot_count": int(relabelled.size),
        "frames_compared": {
            "before": int(old["shot"].size),
            "after": int(new["shot"].size),
        },
        "banked_side_restricted_to_relabelled_shots": True,
        "diverted_control_taken_corpus_wide": True,
        "minimum_band_frames": minimum,
        "bands": bands,
        "bands_too_thin_to_quote": thin,
        "band_count": len(bands),
        "bands_improved": sum(1 for band in bands if band["improved"]),
        "bands_at_unity": sum(
            1 for band in bands if band["reaches_unity_within_10_percent"]
        ),
        "paired": paired,
        "limited_cold_cell": cold,
        "unexplained": [],
    }

    if not bands and paired.get("matched_frame_count", 0) == 0:
        verdict["unexplained"].append(
            "neither the banded ratio nor the paired comparison could be read, "
            "so this measures nothing; do not read the empty result as agreement"
        )
    elif not bands:
        verdict["unexplained"].append(
            f"no band carried {minimum} warm limited frames on both sides, so "
            "the banded ratio is unreadable here and the verdict rests on the "
            "paired comparison alone"
        )

    # A diverted frame should be untouched by construction, so any movement
    # there is a finding rather than a success.
    control_movement = paired.get("diverted_warm", {}).get("maximum_absolute_change")
    if control_movement is not None and control_movement > DIVERTED_TOLERANCE:
        verdict["unexplained"].append(
            f"a diverted frame's ratio moved by {control_movement:.3f}, and a "
            "diverted read is meant to come back unchanged; that detail is "
            "where the masking would be reaching frames it should not"
        )
    moved = cold["after_median"] > 2.0 * cold["before_median"]
    if moved and not cold_start_addressed:
        verdict["unexplained"].append(
            "the limited cold-start cell improved, and no first-iterate repair "
            "is declared present in this corpus; report rather than bank it"
        )
    if cold_start_addressed and not moved:
        verdict["unexplained"].append(
            "a first-iterate repair is declared present and the limited "
            "cold-start cell did not move, so it is not reaching those frames"
        )
    return verdict


def main(argv: list[str] | None = None) -> int:
    """Compare a relabelled census against the banked pre-repair one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before", type=Path, default=BANKED)
    parser.add_argument(
        "--minimum-band-frames",
        type=int,
        default=MINIMUM_BAND_FRAMES,
        help="fewest warm frames a class needs in a band before its median is "
        "quoted; a band below the floor is reported with its counts",
    )
    parser.add_argument(
        "--cold-start-addressed",
        action="store_true",
        help="the relabelled corpus carries a first-iterate shadow repair, so "
        "the cold cell is expected to move rather than to stay put",
    )
    arguments = parser.parse_args(argv)
    verdict = compare(
        arguments.before,
        arguments.after,
        cold_start_addressed=arguments.cold_start_addressed,
        minimum=arguments.minimum_band_frames,
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
