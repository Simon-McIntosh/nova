"""Validate executed-call provenance and the terminal residual decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze import analyze
from nova.jax.config import configure_dtypes


def require_calls(record):
    assert record["counts"]["inner"] > 0, (
        "the executed fixed-point seam was not observed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--missing-calls-control", action="store_true")
    args = parser.parse_args()
    configure_dtypes()
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
        summary = analyze(source)
        assert summary["constraint_relative_sup"] < record["residual"]
        assert len(record["inner"]) == record["counts"]["inner"]
        stages = json.loads((source / "stages.json").read_text())
        assert (
            sum(row["stage"] == "warmup" for row in stages)
            == record["counts"]["warmup"]
        )
        assert (
            sum(row["stage"] == "linearization" for row in stages)
            == record["counts"]["linearization"]
        )
        if record["machine"] == "diiid":
            assert not record["converged"] and record["residual"] > 1e-5
            assert summary["terminal_inner_reason"] == 3
            assert summary["terminal_inner_attempted"] == 0
            assert summary["invalid_images"]
            assert not stages[-1]["finite_state"]
            assert not stages[-1]["finite_image"]
            assert record["changed_mask_entries"][record["counts"]["inner"] - 1] == 0
        else:
            assert record["converged"] and record["residual"] < 1e-5
            assert summary["all_warmup_images_finite"]
            assert record["accepted"] > 0
            assert record["last_ladder"]["accepted"]
        if name == "frame-179-limited":
            assert record["attempted"] == record["counts"]["ladder"] == 0
        elif name == "frame-137-diverted":
            last = record["last_ladder"]
            assert not last["accepted"] and last["distrusted"]
            assert min(last["merits"]) > last["before"]
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
