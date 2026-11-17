"""Report value, gradient and Hessian departures through the existing checks."""

import json
from pathlib import Path
import sys

import numpy as np

from tests.biot import test_moment_program_identity as checks


def main():
    output = Path(sys.argv[1])
    result = {"completed": False, "rows": {}, "failures": []}
    original_bytes = checks._byte_differences
    original_relative = checks.relative_error
    active = None

    def bytes_with_rows(actual, expected, points, *, report):
        actual, expected = map(np.ascontiguousarray, (actual, expected))
        changed = actual.view(np.uint64) != expected.view(np.uint64)
        if active == "packed":
            counts = np.count_nonzero(changed, axis=(0, 2))
        else:
            counts = np.count_nonzero(changed, axis=tuple(range(changed.ndim - 1)))
        row = result["rows"].setdefault(active, {})
        previous = row.get("different_elements", [0] * len(counts))
        row["different_elements"] = (np.asarray(previous) + counts).tolist()
        return original_bytes(actual, expected, points, report=report)

    def relative_with_rows(actual, expected):
        error = original_relative(actual, expected)
        axes = (0, 2) if active == "packed" else tuple(range(error.ndim - 1))
        maxima = np.max(error, axis=axes)
        row = result["rows"].setdefault(active, {})
        previous = row.get("derivative_max_relative", [0.0] * len(maxima))
        row["derivative_max_relative"] = np.maximum(previous, maxima).tolist()
        return error

    checks._byte_differences = bytes_with_rows
    checks.relative_error = relative_with_rows
    for active, name in (
        ("packed", "test_packed_moment_primal_and_tangent_identity"),
        ("production", "test_production_source_rows_preserve_jet_identity"),
    ):
        try:
            getattr(checks, name)()
        except AssertionError as error:
            test_id = f"tests/biot/test_moment_program_identity.py::{name}"
            result["failures"].append(test_id)
            print("FAILED", test_id, repr(error), flush=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    result["completed"] = True
    result["labels"] = {
        "packed": [
            "psi",
            "psi_radial_moment",
            "psi_vertical_moment",
            "Br",
            "Br_radial_moment",
            "Br_vertical_moment",
            "Bz",
            "Bz_radial_moment",
            "Bz_vertical_moment",
        ],
        "production_value_gradient": ["psi", "psi_R", "psi_Z"],
        "production_hessian": ["psi_RR", "psi_RZ", "psi_ZZ"],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("PROGRAM_IDENTITY", json.dumps(result), flush=True)
    raise SystemExit(bool(result["failures"]))


if __name__ == "__main__":
    main()
