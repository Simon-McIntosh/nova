"""Check that a limited panel's boundary passes through its contact point.

A limited boundary touches the wall.  If the recorded contact point is not on
the boundary the panel draws, the classification cannot be read off the panel
and the receipt is not evidence.  This reads a boundary-competition receipt and
reports, for every arm it classifies, the distance from its recorded contact to
the boundary polyline it records, refusing any limited arm whose contact stands
more than one grid cell away from the surface it claims to bound.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    start = polyline
    end = np.roll(polyline, -1, axis=0)
    edge = end - start
    offset = point[None, :] - start
    length2 = np.einsum("ij,ij->i", edge, edge)
    fraction = np.einsum("ij,ij->i", offset, edge)
    fraction = np.divide(
        fraction, length2, out=np.zeros_like(fraction), where=length2 > 0
    )
    onto = start + np.clip(fraction, 0.0, 1.0)[:, None] * edge
    return float(np.linalg.norm(point[None, :] - onto, axis=1).min())


def check(receipt: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for frame, branches in receipt["frames"].items():
        for branch, arms in branches.items():
            for arm, entry in arms.items():
                if entry.get("status") == "refused":
                    continue
                # An arm classifies as limited by its recorded class, or — in a
                # receipt written before the surface was recorded — by the
                # bounded-by field the panel was drawn from.  Classification
                # alone is not evidence: the arm is checked either way, and a
                # receipt that records no surface fails rather than passing on
                # an empty loop.
                limited = (
                    entry.get("achieved_class") or entry.get("bounds")
                ) == "limited"
                if not limited:
                    continue
                surface = np.asarray(entry.get("boundary_rz_m") or [], dtype=float)
                surface = surface.reshape(-1, 2)
                surface = surface[np.isfinite(surface).all(axis=1)]
                contact = np.asarray(entry.get("contact_rz_m") or [], dtype=float)
                step = np.asarray(entry.get("grid_step_m") or [np.nan, np.nan])
                cell = float(np.nanmax(step))
                label = f"frame {frame} {branch} {arm}"
                if surface.size == 0:
                    failures.append(f"{label}: no boundary is recorded")
                    continue
                if contact.size != 2 or not np.all(np.isfinite(contact)):
                    failures.append(f"{label}: contact point is not finite")
                    continue
                distance = _polyline_distance(contact, surface)
                verdict = "PASS" if distance <= cell else "FAIL"
                print(
                    f"{verdict} {label:26s} "
                    f"contact=({contact[0]:.3f},{contact[1]:+.3f}) "
                    f"off-boundary={distance:.3f} m cell={cell:.3f} m",
                    flush=True,
                )
                if distance > cell:
                    failures.append(
                        f"{label}: contact is {distance:.3f} m from the boundary "
                        f"it is supposed to bound (more than one {cell:.3f} m cell)"
                    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    failures = check(json.loads(arguments.receipt.read_text()))
    if failures:
        for failure in failures:
            print(f"REFUSED {failure}")
        return 1
    print("READBACK OK: every limited arm's boundary passes through its contact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
