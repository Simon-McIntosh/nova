"""Tabulate the axisymmetric stand-ins against the six-limb brick reference.

Usage: compare.py <evidence dir>
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def main():
    root = Path(sys.argv[1])
    case = json.loads((root / "case.json").read_text())
    vessel = case["vessel_count"]
    probes = np.asarray(case["probes"])
    axisymmetric = dict(rows=[])
    for path in sorted((root / "axisymmetric").glob("*.json")):
        axisymmetric["rows"].extend(json.loads(path.read_text())["rows"])
    bricks = [
        json.loads(path.read_text())
        for path in sorted((root / "bricks").glob("bricks-*.json"))
    ]
    summary = dict(bricks=[], axisymmetric=[], stray_probes=probes[vessel:].tolist())
    reference = {}
    for record in bricks:
        for row in record["rows"]:
            iron, air = np.asarray(row["iron_average_T"]), np.asarray(row["air_T"])
            resolved = np.asarray(row["iron_resolved_T"])
            ripple = np.ptp(np.linalg.norm(resolved + air, axis=2), axis=0)
            entry = dict(
                size=record["size"],
                order=record.get("order", 1),
                bricks=record["bricks"],
                mu=row["mu"],
                column_flux_Wb=row["column_flux_Wb"],
                vessel_iron_rms_T=float(np.sqrt((iron[:vessel] ** 2).sum(1).mean())),
                stray_total_T=np.linalg.norm(iron + air, axis=1)[vessel:].tolist(),
                stray_air_T=np.linalg.norm(air, axis=1)[vessel:].tolist(),
                stray_toroidal_ripple_T=ripple[vessel:].tolist(),
            )
            summary["bricks"].append(entry)
            key = (record["size"], record.get("order", 1))
            best = reference.get(row["mu"])
            # The finest centroid-collocated run is the reference.
            if key[1] == 1 and (best is None or key[0] < best[0][0]):
                reference[row["mu"]] = (key, iron, air, row["column_flux_Wb"])
    for row in axisymmetric["rows"]:
        key, iron, air, flux = reference[row["mu"]]
        model = np.asarray(row["iron_T"])
        difference = np.linalg.norm(model - iron, axis=1)
        scale = np.sqrt((iron[:vessel] ** 2).sum(1).mean())
        summary["axisymmetric"].append(
            dict(
                model=row["model"],
                size=row["size"],
                mu=row["mu"],
                cells=row["cells"],
                reference_size=key[0],
                reference_order=key[1],
                column_flux_Wb=row["column_flux_Wb"],
                column_flux_ratio=row["column_flux_Wb"] / flux,
                vessel_rms_error_over_iron_rms=float(
                    np.sqrt((difference[:vessel] ** 2).mean()) / scale
                ),
                stray_total_T=np.linalg.norm(model + air, axis=1)[vessel:].tolist(),
                stray_total_ratio=(
                    np.linalg.norm(model + air, axis=1)
                    / np.linalg.norm(iron + air, axis=1)
                )[vessel:].tolist(),
            )
        )
    (root / "comparison.json").write_text(json.dumps(summary, indent=1) + "\n")
    for entry in summary["bricks"]:
        print(
            "BRICK size={size:.4g} order={order} mu={mu:g} flux={column_flux_Wb:.4f} "
            "vessel_rms={vessel_iron_rms_T:.5f}".format(**entry),
            np.round(np.asarray(entry["stray_total_T"]) * 1e3, 3),
        )
    for entry in summary["axisymmetric"]:
        print(
            "AXI {model:5s} size={size} mu={mu:g} flux_ratio={column_flux_ratio:.4f} "
            "vessel_err={vessel_rms_error_over_iron_rms:.4f}".format(**entry),
            np.round(entry["stray_total_ratio"], 3),
        )


if __name__ == "__main__":
    main()
