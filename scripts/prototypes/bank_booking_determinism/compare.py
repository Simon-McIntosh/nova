"""Compare retained booking arrays without treating an absent receipt as equality."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def fingerprint(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def difference(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    finite = np.isfinite(left) & np.isfinite(right)
    delta = np.abs(
        np.subtract(
            left.astype(float),
            right.astype(float),
            out=np.zeros(left.shape),
            where=finite,
        )
    )
    return dict(
        equal=fingerprint(left) == fingerprint(right),
        absolute=float(np.max(np.where(finite, delta, 0))),
        nonfinite_mismatch=int(
            np.sum(~finite & ~((left == right) | (np.isnan(left) & np.isnan(right))))
        ),
    )


def compare(run, arm):
    paths = sorted(run.glob(f"{arm}-[0-4].json"))
    assert paths, f"No {arm} receipts"
    records = [json.loads(path.read_text()) for path in paths]
    arrays = [
        dict(np.load(path.with_suffix(".npz"), allow_pickle=False)) for path in paths
    ]
    control = arrays[0]["initial_moments"].copy()
    slot = np.unravel_index(np.argmax(np.abs(control)), control.shape)
    control[slot] = np.nextafter(control[slot], np.inf)
    observed = difference(arrays[0]["initial_moments"], control)
    assert not observed["equal"] and observed["absolute"] > 0
    pairs = []
    for a, b in itertools.combinations(range(len(records)), 2):
        assert arrays[a].keys() == arrays[b].keys()
        changes = []
        for key in arrays[a]:
            delta = difference(arrays[a][key], arrays[b][key])
            if not delta["equal"]:
                changes.append(dict(key=key, **delta))
        pairs.append(
            dict(
                left=a,
                right=b,
                changes=changes,
                initial=difference(
                    arrays[a]["initial_moments"], arrays[b]["initial_moments"]
                ),
            )
        )
    return dict(
        processes=len(records),
        positive_control=observed,
        pairs=pairs,
        rows=[
            dict(
                process=i,
                sha256=row["initial"]["sha256"],
                calls=row["calls"],
                completed="precision" in row,
            )
            for i, row in enumerate(records)
        ],
        max_initial_difference=max(
            (p["initial"]["absolute"] for p in pairs), default=None
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--arm", default="base")
    args = parser.parse_args()
    result = compare(args.run, args.arm)
    (args.run / f"{args.arm}-comparison.json").write_text(json.dumps(result, indent=2))
    print("COMPARISON", json.dumps(result), flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
