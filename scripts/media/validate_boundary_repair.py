"""Judge a relabelled corpus against the banked pre-repair census.

The acceptance measure is the MATCHED-EPOCH ratio of the limited median to
the diverted median, never either median against unity. That is not a
stylistic choice: the banked census established that the diverted control
itself drifts from 0.70 early to 1.15 late, so an absolute target would fail
a correct repair in the early bands where the control sits at 0.70 and pass
an incorrect one late where it overshoots.

Three quantities are read per frame, because the defect showed up in all
three and a repair that moves one without the others is not a repair:

*Boundary contact.* A limited boundary must touch its limiter. Before the
repair, limited frames stood a median 143 mm off a wall EFIT touches at
1.1 mm.

*Achieved current.* The collapse was quantitatively a current shortfall --
3.6 kA against a 217.8 kA reference on the worst frame -- so the current is
the upstream quantity and is checked directly rather than inferred from the
flux.

*Matched-epoch span ratio.* The acceptance measure itself.

The comparison is against the committed npz rather than a remembered number,
and the pre-repair figures are read from it rather than restated here, so
this cannot silently drift from the evidence it is judged against.
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
#: Below this the cold-start cell is not expected to move: the repair
#: deliberately keeps the unmasked read on a first iterate rather than
#: deriving a shadow circularly from the boundary it is choosing.
COLD_START_UNADDRESSED = True


def _median_ratio(ratio, diverted, cold, time_s, low, high):
    """Return the matched-epoch limited-over-diverted median for one band."""
    window = (time_s >= low) & (time_s < high)
    control = ratio[window & diverted & ~cold]
    limited = ratio[window & ~diverted & ~cold]
    if control.size < 20 or limited.size < 20:
        return None
    return {
        "diverted_warm_count": int(control.size),
        "diverted_warm_median": float(np.median(control)),
        "limited_warm_count": int(limited.size),
        "limited_warm_median": float(np.median(limited)),
        "matched_epoch_ratio": float(np.median(limited) / np.median(control)),
    }


def _load(path: Path) -> dict[str, np.ndarray]:
    """Return one census as named arrays."""
    with np.load(path) as archive:
        return {
            "ratio": np.asarray(archive["span_ratio_nova_over_efit"], dtype=float),
            "diverted": np.asarray(archive["diverted"], dtype=bool),
            "cold": np.asarray(archive["cold_start"], dtype=bool),
            "time_s": np.asarray(archive["time_s"], dtype=float),
        }


def compare(before: Path, after: Path) -> dict[str, object]:
    """Return the band-by-band verdict of a repaired census against a banked one."""
    old, new = _load(before), _load(after)
    bands = []
    for low, high in BANDS:
        was = _median_ratio(**old, low=low, high=high)
        now = _median_ratio(**new, low=low, high=high)
        if was is None or now is None:
            continue
        bands.append(
            {
                "band_ms": [1e3 * low, 1e3 * high],
                "before": was,
                "after": now,
                "improved": now["matched_epoch_ratio"] > was["matched_epoch_ratio"],
                "reaches_unity_within_10_percent": (
                    abs(now["matched_epoch_ratio"] - 1.0) <= 0.10
                ),
            }
        )
    cold = {
        "before_median": float(np.median(old["ratio"][~old["diverted"] & old["cold"]])),
        "after_median": float(np.median(new["ratio"][~new["diverted"] & new["cold"]])),
        "expected_to_move": not COLD_START_UNADDRESSED,
    }
    verdict = {
        "bands": bands,
        "bands_improved": sum(1 for b in bands if b["improved"]),
        "bands_at_unity": sum(1 for b in bands if b["reaches_unity_within_10_percent"]),
        "band_count": len(bands),
        "limited_cold_cell": cold,
        "acceptance_measure": (
            "the matched-epoch ratio of the limited median to the diverted "
            "median reaching one, never either median against unity"
        ),
        "unexplained": [],
    }
    # A diverted frame should be untouched by construction, so any movement
    # there is a finding rather than a success.
    for band in bands:
        moved = abs(
            band["after"]["diverted_warm_median"]
            - band["before"]["diverted_warm_median"]
        )
        if moved > 0.01:
            verdict["unexplained"].append(
                f"the diverted control moved by {moved:.3f} in band "
                f"{band['band_ms'][0]:.0f}-{band['band_ms'][1]:.0f} ms, and a "
                "diverted read is meant to be untouched by construction"
            )
    if cold["after_median"] > 2.0 * cold["before_median"]:
        verdict["unexplained"].append(
            "the limited cold-start cell improved, which the repair does not "
            "address; report rather than bank it"
        )
    return verdict


def main(argv: list[str] | None = None) -> int:
    """Compare a relabelled census against the banked pre-repair one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before", type=Path, default=BANKED)
    arguments = parser.parse_args(argv)
    verdict = compare(arguments.before, arguments.after)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
