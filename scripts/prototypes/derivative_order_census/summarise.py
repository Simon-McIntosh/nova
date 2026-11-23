"""Compare saved local derivative arrays without importing JAX or running a read."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(left, right):
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), 1e-300)))


def row_stem(directory, variant, order):
    for prefix in ("compile-", "", "lower-"):
        stem = directory / f"{prefix}derivative-{variant}-{order}"
        if stem.with_suffix(".json").exists():
            return stem
    return directory / f"derivative-{variant}-{order}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, comparisons = [], []
    for order in range(5):
        reference_path = row_stem(args.rows, "current", order).with_suffix(".npz")
        reference = np.load(reference_path) if reference_path.exists() else None
        for variant in ("current", "plain", "without_samples", "masked_samples"):
            stem = row_stem(args.rows, variant, order)
            receipt_path, array_path = (
                stem.with_suffix(".json"),
                stem.with_suffix(".npz"),
            )
            if receipt_path.exists():
                row = json.loads(receipt_path.read_text())
                row["receipt_sha256"] = digest(receipt_path)
                records.append(row)
            if reference is None or not array_path.exists():
                comparisons.append(
                    {"variant": variant, "order": order, "state": "unmeasured"}
                )
                continue
            values = np.load(array_path)
            np.testing.assert_array_equal(values["targets"], reference["targets"])
            expected, observed = reference["values"], values["values"]
            assert expected.shape == observed.shape and expected.shape[0] == 250
            altered = expected.copy()
            index = np.unravel_index(np.abs(expected).argmax(), expected.shape)
            altered[index] *= 1.01
            control = relative(altered, expected)
            assert control is not None and control > 0.009
            comparisons.append(
                {
                    "variant": variant,
                    "order": order,
                    "state": "compared",
                    "generic_max_relative": relative(observed[:200], expected[:200]),
                    "corner_max_relative": relative(observed[200:], expected[200:]),
                    "bit_identical": bool(
                        np.array_equal(
                            observed.view(np.uint64), expected.view(np.uint64)
                        )
                    ),
                    "finite_values": int(np.isfinite(observed).sum()),
                    "total_values": int(observed.size),
                    "positive_control_relative": control,
                    "array_sha256": digest(array_path),
                }
            )
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
        print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
