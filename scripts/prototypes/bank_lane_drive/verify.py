"""Require every terminal lane to pass the physical precision budget."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.bank_warm_predictor.report import (
    precision,
    require_budget,
    BOUNDS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    records = {
        arm: json.loads((args.run / f"counts-{arm}-measured.json").read_text())
        for arm in ("main", "head")
    }
    assert records["main"]["inputs"] == records["head"]["inputs"]
    arrays = {
        arm: {
            "terminal-" + key: value
            for key, value in np.load(args.run / f"counts-{arm}-measured.npz").items()
        }
        for arm in records
    }
    ratios = precision(
        arrays["main"], arrays["head"], "terminal", records["main"]["minor_radius"]
    )
    require_budget(ratios)
    refused = []
    for name, bound in BOUNDS.items():
        mutant = {key: value.copy() for key, value in ratios.items()}
        mutant[name][0] = 2 * bound
        try:
            require_budget(mutant)
        except AssertionError:
            refused.append(name)
    assert set(refused) == set(BOUNDS)
    for arm, record in records.items():
        assert record["completed"] and record["certified"]
        assert len(record["residual"]) == 8
        assert np.all(np.isfinite(record["residual"]))
        assert max(record["residual"]) <= 1e-10
        assert record["subsolves"] == len(record["rows"])
        assert record["reads"] == sum(max(row["reads"]) for row in record["rows"])
        assert record["updates"] == sum(max(row["updates"]) for row in record["rows"])
    report = dict(
        inputs=records["main"]["inputs"],
        precision={key: value.tolist() for key, value in ratios.items()},
        refused_controls=refused,
        counts={
            arm: {
                key: record[key]
                for key in ("reads", "updates", "subsolves", "seconds", "residual")
            }
            for arm, record in records.items()
        },
    )
    (args.run / "count-report.json").write_text(json.dumps(report, indent=2))
    print("PRECISION", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
