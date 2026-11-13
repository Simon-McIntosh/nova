"""Validate receipt provenance, agreement and geometry before wall adjudication."""

import argparse
import json
import subprocess
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
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--partial", action="store_true")
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
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    from tests.equilibrium.test_topology_read import _analytic_inputs

    oracle, *_ = _analytic_inputs("diverted")
    radius = float(oracle.minor_radius)
    assert radius > 0
    if args.inventory:
        inventory = json.loads((run / "measurement-jobs.json").read_text())
        items = list(inventory["jobs"])
        repairs = run / "repair-jobs.json"
        if repairs.exists():
            items += json.loads(repairs.read_text())["jobs"]
        jobs = [item["job_id"] for item in items]
        output = subprocess.check_output(
            [
                "sacct",
                "-n",
                "-X",
                "-j",
                ",".join(jobs),
                "-o",
                "JobIDRaw,State%30,ExitCode",
                "-P",
            ],
            text=True,
        )
        observed = {
            fields[0]: fields[1:3]
            for row in output.splitlines()
            if len(fields := row.split("|")) >= 3
        }
        assert all(observed.get(job) == ["COMPLETED", "0:0"] for job in jobs), observed
        retained = json.loads((run / "retained-receipts.json").read_text())
        import hashlib

        for path, digest in retained.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        summaries = [
            json.loads((run / "h200-levers-corrected/summary.json").read_text())
        ]
        for item in items:
            directory = Path(item["output"])
            value = json.loads((directory / "summary.json").read_text())
            assert value["completed"] and not value["smoke"]
            assert (directory / "job.log").read_text().splitlines()[-1] == "EXIT=0"
            summaries.append(value)
    else:
        value = json.loads((run / "summary.json").read_text())
        assert args.partial or value["completed"], (
            "job has not completed every comparison"
        )
        if not args.partial:
            assert (run / "job.log").read_text().splitlines()[-1] == "EXIT=0"
        summaries = [value]
    selected_comparisons = {}
    for value in summaries:
        for comparison in value["comparisons"]:
            key = (
                comparison["name"],
                comparison["width"],
                comparison.get("step", 1e-5),
                comparison.get("population", "diverted"),
            )
            selected_comparisons[key] = comparison
    summary = dict(
        arms=[
            arm
            for value in summaries
            for arm in value["arms"]
            if arm["exit_status"] == 0
        ],
        comparisons=list(selected_comparisons.values()),
    )
    receipts = {}
    for arm in summary["arms"]:
        assert arm["exit_status"] == 0
        assert Path(arm["log"]).read_text().splitlines()[-1] == "EXIT=0"
        path = Path(arm["receipt"])
        value = json.loads(path.read_text())
        assert value["complete"] and value["x64"] and not value.get("smoke", False)
        assert value["device"] == "NVIDIA H200 NVL"
        for row in value["rows"]:
            if "residual" in row:
                assert max(row["residual"]) <= 1e-10
        receipts[json.dumps(value, sort_keys=True)] = path
    verified = []
    provenance_errors = []
    for comparison in summary["comparisons"]:
        arms = comparison.get("tiles") or [comparison["off"], comparison["on"]]
        references = arms[0]["records"]
        for arm in arms:
            assert all(spread <= 0.05 for spread in arm["spreads"].values())
            for index, candidate in enumerate(arm["records"]):
                reference = references[index]
                if candidate["input_sha256"] != reference["input_sha256"]:
                    provenance_errors.append(
                        dict(
                            name=comparison["name"],
                            width=comparison["width"],
                            arm=arm["arm"],
                            process=index + 1,
                            reference=reference["input_sha256"],
                            candidate=candidate["input_sha256"],
                        )
                    )
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
                    errors = precision(a, b, radius)
                verified.append(
                    dict(
                        name=comparison["name"],
                        width=comparison["width"],
                        arm=arm["arm"],
                        process=index + 1,
                        errors=errors,
                    )
                )
    # The decision-free read rebuilds geometry. Its frozen-reading branch is
    # retained for attribution and cannot be the refreshed-work comparison.
    for comparison in summary["comparisons"]:
        if comparison["name"] != "carried-decisions":
            continue
        for reference, candidate in zip(
            comparison["off"]["records"], comparison["on"]["records"], strict=True
        ):
            left = receipts[json.dumps(reference, sort_keys=True)].with_suffix(".npz")
            right = receipts[json.dumps(candidate, sort_keys=True)].with_suffix(".npz")
            with (
                np.load(left, allow_pickle=False) as a,
                np.load(right, allow_pickle=False) as b,
            ):
                full = {
                    key.removeprefix("full-"): a[key]
                    for key in a
                    if key.startswith("full-")
                }
                carried = {
                    key.removeprefix("carried-"): b[key]
                    for key in b
                    if key.startswith("carried-")
                }
                errors = precision(
                    {"reading-" + k: v for k, v in full.items()},
                    {"reading-" + k: v for k, v in carried.items()},
                    radius,
                )
            verified.append(
                dict(
                    name="decision-free-full-to-carried",
                    width=comparison["width"],
                    errors=errors,
                )
            )
    (run / "precision-audit.json").write_text(json.dumps(verified, indent=2))
    (run / "input-provenance-errors.json").write_text(
        json.dumps(provenance_errors, indent=2)
    )
    print(
        f"GEOMETRY {len(verified)} paired process receipts inside precision budget",
        flush=True,
    )
    assert not provenance_errors, (
        "paired input fingerprints differ",
        provenance_errors,
    )
    print(f"SUMMARY {len(verified)} paired process receipts inside precision budget")


if __name__ == "__main__":
    main()
