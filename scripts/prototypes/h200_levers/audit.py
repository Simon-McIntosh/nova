"""Validate receipt provenance, agreement and geometry before wall adjudication."""

import argparse
import json
from pathlib import Path

from scripts.prototypes.h200_levers.run import refuse_inert


def decision(off, on, spread, incumbent):
    """Retain an incumbent unless its alternative wins beyond the margin."""
    margin = max(0.03, spread)
    if on > off * (1 + margin):
        return "off", margin
    if off > on * (1 + margin):
        return "on", margin
    return incumbent, margin


def precision(reference, candidate, radius):
    """Compare every lane to the same physical input's paired reference."""
    import numpy as np

    values = {}
    for key in reference:
        if key.endswith("-axis") or key.endswith("-saddle"):
            bound = 5e-4
            error = np.linalg.norm(candidate[key] - reference[key], axis=-1) / radius
        elif key.endswith("-boundary"):
            bound = 1e-4
            span = np.abs(reference[key.removesuffix("-boundary") + "-span"])
            assert np.all(np.isfinite(span)) and np.all(span > 0)
            error = np.abs(candidate[key] - reference[key]) / span
        elif key == "matrix":
            relative = np.linalg.norm(candidate[key] - reference[key])
            relative /= np.linalg.norm(reference[key])
            assert np.isfinite(relative) and relative <= 1e-12
            values[key] = float(relative)
            continue
        else:
            continue
        assert np.all(np.isfinite(error)) and np.max(error) <= bound, (key, error)
        values[key] = float(np.max(error))
    assert values, "precision guard saw no geometry or matrix data"
    return values


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    if args.negative_control:
        sample = dict(rows=[dict(lowered_sha256="known-program")])
        other = dict(rows=[dict(lowered_sha256="different-program")])
        refuse_inert(sample, other)
        print("POSITIVE_CONTROL different programs admitted", flush=True)
        refuse_inert(sample, sample)
        raise AssertionError("identical-program negative control was admitted")
    import numpy as np

    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["completed"], "job has not completed every comparison"
    assert (run / "job.log").read_text().splitlines()[-1] == "EXIT=0"
    receipts = {}
    for arm in summary["arms"]:
        assert arm["exit_status"] == 0
        assert Path(arm["log"]).read_text().splitlines()[-1] == "EXIT=0"
        path = Path(arm["receipt"])
        value = json.loads(path.read_text())
        assert value["complete"] and value["x64"]
        assert value["device"] == "NVIDIA H200 NVL"
        for row in value["rows"]:
            if "residual" in row:
                assert max(row["residual"]) <= 1e-10
        receipts[json.dumps(value, sort_keys=True)] = path
    verified = []
    for comparison in summary["comparisons"]:
        arms = comparison.get("tiles") or [comparison["off"], comparison["on"]]
        references = arms[0]["records"]
        for arm in arms:
            assert all(spread <= 0.05 for spread in arm["spreads"].values())
            for index, candidate in enumerate(arm["records"]):
                reference = references[index]
                assert candidate["input_sha256"] == reference["input_sha256"]
                assert candidate["bank_sha256"] == reference["bank_sha256"]
                left = receipts[json.dumps(reference, sort_keys=True)].with_suffix(
                    ".npz"
                )
                right = receipts[json.dumps(candidate, sort_keys=True)].with_suffix(
                    ".npz"
                )
                with (
                    np.load(left, allow_pickle=False) as a,
                    np.load(right, allow_pickle=False) as b,
                ):
                    errors = precision(a, b, reference["minor_radius"])
                verified.append(
                    dict(
                        name=comparison["name"],
                        width=comparison["width"],
                        arm=arm["arm"],
                        process=index + 1,
                        errors=errors,
                    )
                )
    (run / "precision-audit.json").write_text(json.dumps(verified, indent=2))
    print(f"SUMMARY {len(verified)} paired process receipts inside precision budget")


if __name__ == "__main__":
    main()
