"""Extract both arms' prescribed tables from each frame's EFIT map.

This is the input measurement the solve arms prescribe: the map's own p'/FF'
on the established knots and on finer knots toward the separatrix, with the
reliability diagnostics that decide whether a knot may be carried at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_forward_convergence"))
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_xpoint_profile_extent"))
import arms  # noqa: E402
import measure_branches as probe  # noqa: E402

bench = probe.bench
FRAMES = (137, 146, 179)


def read_row(frame):
    paths = sorted(bench.DEFAULT_DATA.glob("*.parquet"))
    selected = bench.select_frames(
        paths, bench.EXECUTION_FRAME_COUNT, bench.polarity_population()
    )
    match = [item for item in selected if item.frame == frame]
    if not match:
        raise SystemExit(f"frame {frame} is not in the declared cohort")
    columns = (
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
    )
    row = bench._read(match[0].path, columns)
    row["_source_path"] = str(match[0].path)
    return row


def diagnostics(row, frame, knots):
    """Return each knot's reliability diagnostics for an explicit surface set."""

    radius, height = bench.canonical_axes(row)
    label = bench.corpus_flux_to_nova_total(
        np.asarray(row["efit_psirz"][frame], dtype=float)
    ).T
    interpolant = RectBivariateSpline(radius, height, label, kx=3, ky=3, s=0)
    axis_flux = float(
        interpolant.ev(row["efit_r_axis"][frame], row["efit_z_axis"][frame])
    )
    count = int(row["efit_lcfs_n"][frame])
    boundary_flux = float(
        np.median(
            interpolant.ev(
                np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
                np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
            )
        )
    )
    normalised = (label - axis_flux) / (boundary_flux - axis_flux)
    receipt = bench.extract_flux_functions(
        radius,
        height,
        label,
        normalised,
        surfaces=knots,
        plasma_mask=bench._plasma_mask(row, frame, radius, height),
        min_samples=6,
    )
    finite = np.isfinite(
        np.asarray(receipt.p_prime, float) + np.asarray(receipt.ff_prime, float)
    )
    reliable = np.asarray(receipt.reliable, bool) & finite
    return {
        "nodes": [float(item) for item in np.asarray(receipt.psi_norm)],
        "reliable": [bool(item) for item in reliable],
        "sample_count": [int(item) for item in np.asarray(receipt.sample_count)],
        "condition_number": [
            float(item) for item in np.asarray(receipt.condition_number)
        ],
        "uncertainty_inflation": [
            float(item) for item in np.asarray(receipt.uncertainty_inflation)
        ],
        "projection_rms": [float(item) for item in np.asarray(receipt.projection_rms)],
        "minimum_gradient": [
            float(item) for item in np.asarray(receipt.minimum_gradient)
        ],
    }


def table(arm, row, frame):
    """Return the tables a solve on this arm is handed, as the build reads them."""

    _, nodes, p_prime, ff_prime = arms.label_state(bench, arm)(row, frame)
    return {
        "coordinate": [float(item) for item in np.asarray(nodes)],
        "p_prime": [float(item) for item in np.asarray(p_prime)],
        "ff_prime": [float(item) for item in np.asarray(ff_prime)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "base_surfaces": list(arms.surfaces("base")),
        "edge_surfaces": list(arms.surfaces("edge")),
        "edge_extraction_surfaces": list(arms.EDGE_EXTRACTION_SURFACES),
        "rows": {},
    }
    for frame in FRAMES:
        row = read_row(frame)
        base_table = table("base", row, frame)
        edge_table = table("edge", row, frame)
        count = len(base_table["coordinate"])
        interior = (
            edge_table["coordinate"][:count] == base_table["coordinate"]
            and edge_table["p_prime"][:count] == base_table["p_prime"]
            and edge_table["ff_prime"][:count] == base_table["ff_prime"]
        )
        report["rows"][str(frame)] = {
            "source_path": row["_source_path"],
            "source_sha256": hashlib.sha256(
                Path(row["_source_path"]).read_bytes()
            ).hexdigest(),
            "base_table": base_table,
            "edge_table": edge_table,
            "interior_identical": bool(interior),
            "edge_diagnostics": diagnostics(
                row, frame, arms.EDGE_EXTRACTION_SURFACES
            ),
            "base_diagnostics": diagnostics(row, frame, arms.BASE_SURFACES),
        }
        print(
            f"FRAME {frame} base={len(base_table['coordinate'])} "
            f"edge={len(edge_table['coordinate'])} interior_identical={interior} "
            f"last_knot={edge_table['coordinate'][-1]:.3f} "
            f"p_edge={edge_table['p_prime'][-1]:.1f} "
            f"ff_edge={edge_table['ff_prime'][-1]:.6f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()