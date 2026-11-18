"""Record EFIT's p' and FF' on the boundary shell the builder prescribes.

The record this writes is the corpus read behind the finite-edge profile: the
prescribed tables stop at the last interior shell, and the boundary shell at
psi_norm = 1 is extracted from the same normalised map on the same 0.05 spacing
through ``benchmarks.diiid_forward_gs_match._profile_edge_values``. Nothing is fitted.py
here; the numbers are the extractor's own read of the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from nova.jax.config import configure_dtypes  # noqa: E402

# The extractor runs its stencils through JAX, so single precision changes the
# boundary value in the fifth digit; the record is a double-precision read.
configure_dtypes()

from benchmarks import diiid_forward_gs_match as bench  # noqa: E402

ROWS = (
    ("P102", "d3d_shot_00000c4a7b.parquet", 102, "pinned seed frame"),
    ("R137D", "d3d_shot_0040ca9bdc.parquet", 137, "diverted"),
    ("R146D", "d3d_shot_002495e835.parquet", 146, "diverted"),
    ("R146L", "d3d_shot_002495e835.parquet", 146, "limited"),
    ("R179L", "d3d_shot_00000c4a7b.parquet", 179, "limited"),
)


def _sha256(value: float) -> str:
    return hashlib.sha256(repr(value).encode("ascii")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    columns = (
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
    )
    rows = {}
    cache = {}
    for name, filename, frame, branch in ROWS:
        path = bench.DEFAULT_DATA / filename
        if filename not in cache:
            row = bench._read(path, columns)
            row["_source_path"] = str(path)
            cache[filename] = row
        row = cache[filename]
        psi_norm, p_prime, ff_prime = bench._profile_edge_values(row, frame)
        rows[name] = {
            "path": str(path),
            "shot": path.stem,
            "frame": frame,
            "branch": branch,
            "psi_norm": psi_norm,
            "p_prime": p_prime,
            "ff_prime": ff_prime,
            "p_prime_sha256": _sha256(p_prime),
            "ff_prime_sha256": _sha256(ff_prime),
        }
        print(
            "EDGE %s frame %d psi_norm %.17g p' %.17g FF' %.17g"
            % (name, frame, psi_norm, p_prime, ff_prime),
            flush=True,
        )
    record = {
        "corpus": str(bench.DEFAULT_DATA),
        "extraction": (
            "nova.equilibrium.map_extraction.extract_flux_functions on the "
            "0.05-spaced shell centres continued to psi_norm = 1, through "
            "benchmarks.diiid_forward_gs_match._profile_edge_values"
        ),
        "surfaces": [float(value) for value in bench.PROFILE_SURFACES],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
