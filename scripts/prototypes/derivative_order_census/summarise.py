"""Combine isolated compile receipts and eager references without importing JAX."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

VARIANTS = ("current", "plain", "without_samples", "masked_samples")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(left, right):
    if left.size == 0 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), 1e-300)))


def row_stem(directories, variant, order):
    for directory in directories:
        for prefix in ("compile-", "", "lower-"):
            stem = directory / f"{prefix}derivative-{variant}-{order}"
            if stem.with_suffix(".json").exists():
                return stem
    return directories[-1] / f"derivative-{variant}-{order}"


def array_path(directories, variant, order, *, reference=False):
    prefixes = ("eager-", "compile-", "") if reference else ("compile-", "", "eager-")
    for directory in directories:
        for prefix in prefixes:
            path = directory / f"{prefix}derivative-{variant}-{order}.npz"
            if path.exists():
                return path
    return None


def comparison(reference_path, observed_path, variant, order):
    reference, observed = np.load(reference_path), np.load(observed_path)
    ref_indices = reference["indices"] if "indices" in reference else np.arange(250)
    obs_indices = observed["indices"] if "indices" in observed else np.arange(250)
    common, left, right = np.intersect1d(ref_indices, obs_indices, return_indices=True)
    assert len(common), "accuracy arrays have no targets in common"
    np.testing.assert_array_equal(
        reference["targets"][left], observed["targets"][right]
    )
    expected, values = reference["values"][left], observed["values"][right]
    assert expected.shape == values.shape
    finite = expected[np.isfinite(expected) & (expected != 0)]
    assert finite.size, "the comparison control has no known-present finite entry"
    control_value = finite[np.abs(finite).argmax()]
    control = float(abs(control_value * 1.01 - control_value) / abs(control_value))
    assert control > 0.009
    generic, corner = common < 200, common >= 200
    return {
        "variant": variant,
        "order": order,
        "state": "compared",
        "generic_targets": int(generic.sum()),
        "corner_targets": int(corner.sum()),
        "generic_max_relative": relative(values[generic], expected[generic]),
        "corner_max_relative": relative(values[corner], expected[corner]),
        "bit_identical": bool(
            np.array_equal(values.view(np.uint64), expected.view(np.uint64))
        ),
        "finite_values": int(np.isfinite(values).sum()),
        "total_values": int(values.size),
        "reference_finite_values": int(np.isfinite(expected).sum()),
        "positive_control_relative": control,
        "target_indices": common.tolist(),
        "array_path": str(observed_path),
        "array_sha256": digest(observed_path),
        "reference_path": str(reference_path),
        "reference_sha256": digest(reference_path),
        "reference_mode": "eager"
        if reference_path.name.startswith("eager-")
        else "compiled",
        "candidate_mode": "eager"
        if observed_path.name.startswith("eager-")
        else "compiled",
    }


def native_outcome(directory, variant, order):
    path = directory / "index.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    matches = [
        row
        for row in rows
        if row.get("action") == "compile"
        and row.get("variant") == variant
        and row.get("order") == order
    ]
    if not matches:
        return None
    result = dict(matches[-1])
    timing = Path(result["time_log"])
    if timing.exists():
        text = timing.read_text()
        peak = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        result["time_peak_rss_kib"] = int(peak[1]) if peak else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--additional-rows", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = ([args.additional_rows] if args.additional_rows else []) + [args.rows]
    records, comparisons = [], []
    for order in range(5):
        reference = array_path(directories, "current", order, reference=order == 2)
        for variant in VARIANTS:
            stem = row_stem(directories, variant, order)
            receipt_path = stem.with_suffix(".json")
            if receipt_path.exists():
                row = json.loads(receipt_path.read_text())
                row["receipt_sha256"] = digest(receipt_path)
                row["receipt_path"] = str(receipt_path)
                row["native_outcome"] = native_outcome(stem.parent, variant, order)
                records.append(row)
            observed = (
                reference
                if order == 2 and variant == "current"
                else array_path(directories, variant, order)
            )
            if reference is None or observed is None:
                comparisons.append(
                    {"variant": variant, "order": order, "state": "unmeasured"}
                )
            else:
                comparisons.append(comparison(reference, observed, variant, order))
    result = {
        "rows": records,
        "comparisons": comparisons,
        "relative_definition": (
            "max(abs(candidate-reference)/max(abs(reference),1e-300))"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in records:
        print(
            row["variant"],
            row["order"],
            row["hlo_instructions"],
            row["packed_kernel_ops"],
            row["packed_kernel_instances"],
            row["temp_bytes"],
            row["phase"],
        )
    for row in comparisons:
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "variant",
                        "order",
                        "state",
                        "generic_targets",
                        "corner_targets",
                        "generic_max_relative",
                        "corner_max_relative",
                        "candidate_mode",
                        "reference_mode",
                    )
                    if key in row
                },
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
