"""Validate executed-call provenance and the terminal residual decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def require_calls(record):
    assert record["counts"]["inner"] > 0, (
        "the executed fixed-point seam was not observed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--missing-calls-control", action="store_true")
    args = parser.parse_args()
    required = ("frame-179-limited", "frame-137-diverted", "mast-21978-35")
    for name in required:
        source = args.input / name
        record = json.loads((source / "receipt.json").read_text())
        if args.missing_calls_control:
            record["counts"]["inner"] = 0
        require_calls(record)
        assert record["route"] == "newton_krylov"
        assert record["support"] > 0
        assert record["storage_order"] == "radius_height"
        with np.load(source / "terminal.npz") as data:
            residual = data["residual"]
            computed = np.max(np.abs(residual)) / max(
                np.max(np.abs(data["mapped"])), 1e-30
            )
            np.testing.assert_allclose(
                computed, record["recomputed_residual"], rtol=1e-13
            )
            np.testing.assert_allclose(
                data["mapped"] - data["state"], residual, rtol=0, atol=0
            )
            assert data["connected"].sum() == record["support"]
            rows = record["block_region"]
            assert sum(row["count"] for row in rows) == residual.size
            np.testing.assert_allclose(
                sum(row["squared_norm_fraction"] for row in rows), 1, rtol=1e-12
            )
        assert record["constraint_rows"] == 1
        np.testing.assert_array_equal(record["constraint_residual"], [0])
        assert len(record["inner"]) == record["counts"]["inner"]
        stages = json.loads((source / "stages.json").read_text())
        assert (
            sum(row["stage"] == "warmup" for row in stages)
            == record["counts"]["warmup"]
        )
        print(
            f"CAPTURE {name}: inner={record['counts']['inner']} "
            f"ladder={record['counts']['ladder']} "
            f"support={record['support']} residual={record['residual']:.12g}"
        )
    print(
        "CAPTURE CHECK PASSED: all three executed traces and decompositions reconcile"
    )


if __name__ == "__main__":
    main()
