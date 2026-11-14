"""Publish correctly grouped ratios from a completed, unchanged field receipt."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.iron_limb_production.measure import (
    axisymmetric_drive,
    outside_probe_groups,
    reference_inputs,
    write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    record = json.loads(raw)
    assert "control_passed" in record and len(record["control"]) == 20
    case, references = reference_inputs()
    probes = np.asarray(case["probes"])
    groups = outside_probe_groups(probes)
    air = axisymmetric_drive(case["sources"])(probes)
    record["probe_groups"] = groups
    record["source_receipt"] = str(args.source)
    record["source_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    for row in record["control"]:
        reference = np.asarray(references[row["mu"]]["iron_average_T"])
        field = np.asarray(row["iron_T"])
        ratios = np.linalg.norm(air + field, axis=1) / np.linalg.norm(
            air + reference, axis=1
        )
        for name, indices in groups.items():
            row[f"{name}_ratios"] = ratios[indices].tolist()
    write(args.output, record)
    print("PROBE_GROUPS " + json.dumps(groups), flush=True)
    print(
        f"GROUPED_RECEIPT source_sha256={record['source_receipt_sha256']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
