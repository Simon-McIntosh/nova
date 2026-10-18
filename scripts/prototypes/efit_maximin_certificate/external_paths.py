"""Evaluate delivered escape paths on each record's independent scipy spline."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import nova  # noqa: E402
from scripts.prototypes.efit_maximin_certificate.bounds import bound_polyline  # noqa: E402
from scripts.prototypes.efit_maximin_certificate.certificate import (  # noqa: E402
    load_record,
    spline_of,
)
from scripts.prototypes.efit_maximin_certificate.run import ROWS  # noqa: E402

DELIVERED = pathlib.Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-maximin-certificates"
)
NAMES = (
    "path-mastu-fullPulse-s1-4-5-B.txt",
    "path-synthetic_crossing-A.txt",
    "path-synthetic_crossing-B.txt",
)


def main(output: pathlib.Path):
    print(f"nova.__file__={nova.__file__}", flush=True)
    results = []
    for spec, name in zip(ROWS, NAMES, strict=True):
        path = DELIVERED / name
        if not path.is_file():
            results.append({"case": spec["case"], "file": str(path), "absent": True})
            continue
        record = load_record(pathlib.Path(spec["source"]))
        spline = spline_of(record)
        delivered = np.loadtxt(path)
        points = delivered[:, :2]
        measured = int(record.polarity) * spline.ev(points[:, 0], points[:, 1])
        complete = np.vstack((spec["axis"], points, spec["target"]))
        bound = bound_polyline(spline, int(record.polarity), complete)
        result = {
            "case": spec["case"],
            "file": str(path),
            "sample_count": len(points),
            "first": points[0].tolist(),
            "last": points[-1].tolist(),
            "exact_axis": spec["axis"],
            "exact_target": spec["target"],
            "recorded_sample_min": float(np.min(delivered[:, 2])),
            "scipy_sample_min": float(np.min(measured)),
            "max_field_disagreement": float(np.max(np.abs(delivered[:, 2] - measured))),
            "bound_with_exact_connectors": bound,
            "efit_level": spec["efit_level"],
            "attains_exact_level": bound["lower"] >= spec["efit_level"],
            "refutes_exact_level": bound["upper"] < spec["efit_level"],
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        output.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
