"""Inspect the analytic reference boundary at omitted carrier cells."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import nova
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from scripts.analytic_oracle_fixtures.measure import _analytic_separatrix
from tests.equilibrium.test_topology_read import _analytic_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"REVISION={revision} TREE={Path.cwd().resolve()} "
        f"COMMAND={sys.executable} {' '.join(sys.argv)}",
        flush=True,
    )
    print(f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()}", flush=True)
    assert jax.config.jax_enable_x64 is True
    oracle, _, _, _, _ = _analytic_inputs("diverted")
    raw = np.asarray(oracle.separatrix(4097), dtype=np.float64)
    cleaned = _analytic_separatrix(oracle, 4097)
    boundaries = {"raw": Polygon(raw), "cleaned": Polygon(cleaned)}
    direct = np.load(args.out / "diverted-10000-cells.npz")
    combined = np.load(args.out / "diverted-10000-combined.npz")
    missing = np.flatnonzero(
        (direct["exact"][0] > 0.001) & (combined["reference"][0] == 0)
    )
    control = int(np.flatnonzero(direct["exact"][0] > 1)[0])
    inspect = np.r_[missing[:5], control]
    records = []
    for index in inspect:
        cell = Polygon(direct["vertices"][index, : direct["vertex_count"][index]])
        records.append(
            {
                "cell": int(index),
                "full_area": float(cell.area),
                "exact_area": float(direct["exact_area"][index]),
                **{
                    f"{name}_intersection_area": float(cell.intersection(boundary).area)
                    for name, boundary in boundaries.items()
                },
            }
        )
    result = {
        "missing_count": int(len(missing)),
        "missing_radial_coordinates": np.unique(direct["centre"][missing, 0]).tolist(),
        "raw_vertices": len(raw),
        "cleaned_vertices": len(cleaned),
        "raw_validity": explain_validity(boundaries["raw"]),
        "cleaned_validity": explain_validity(boundaries["cleaned"]),
        "raw_area": float(boundaries["raw"].area),
        "cleaned_area": float(boundaries["cleaned"].area),
        "records": records,
    }
    np.savez_compressed(args.out / "diverted-boundaries.npz", raw=raw, cleaned=cleaned)
    (args.out / "diverted-boundaries.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
