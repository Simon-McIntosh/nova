"""Check a regenerated WEST 62751 solve-inputs pin against committed values.

The pin's value is that it was written before later code changes, so writing
it again proves nothing on its own. This script compares the regenerated
probe positions, CEA B-H table and air-core field against the numbers already
committed in the evidence fragments, the plan and the material test, then
runs west_inputs() against the pin and, with the pin absent in a scratch
copy, without it.

    python scripts/west_iron_inputs/verify.py --pin <pin> --receipt <receipt>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import scripts.iron_graded_refinement.measure as graded  # noqa: E402

# Committed values, each with the record that holds it. The air-core field is
# quoted twice at different precision, and the plan quotes it rounded to
# microtesla, so each is compared at the precision it was written to.
COMMITTED_PROBE_R = [5.52, 6.52, 8.52]
COMMITTED_AIR_MT = [7.02407743131, 4.39682051236, 1.79996152911]
COMMITTED_AIR_UT = [7024.08, 4396.82, 1799.96]
COMMITTED_AIR_PLAN_UT = 7024.0
COMMITTED_B_KNOTS = {0: 0.0, 1: 0.5}
COMMITTED_MU = {0: 215.11019, 1: 1037.9637, -1: 1692.4}
COMMITTED_TIME_S = 38.4395796373
SOLVE_FRAGMENT = (
    "docs/evidence/fragments/iron-core-magnetisation/icm-west-62751-solve.html"
)
PAIR_FRAGMENT = (
    "docs/evidence/fragments/iron-core-magnetisation/icm-west-62751-galerkin-pair.html"
)


def compare(rows, quantity, source, expected, measured, tolerance):
    difference = abs(measured - expected)
    relative = difference / abs(expected) if expected else difference
    passed = relative <= tolerance
    rows.append(
        dict(
            quantity=quantity,
            source=source,
            expected=expected,
            measured=measured,
            absolute_difference=difference,
            relative_difference=relative,
            tolerance=tolerance,
            passed=bool(passed),
        )
    )
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads(args.pin.read_text())
    rows, failures = [], []
    probes = np.asarray(pin["probes_m"])
    air_magnitude = np.asarray(pin["air_magnitude_T"])
    table_b = np.asarray(pin["table_b_T"])
    table_mu = np.asarray(pin["table_mu_relative"])

    for index, radius in enumerate(COMMITTED_PROBE_R):
        if not compare(
            rows,
            f"probe radius {index + 1} [m]",
            PAIR_FRAGMENT,
            radius,
            float(probes[index, 0]),
            0.0,
        ):
            failures.append(rows[-1]["quantity"])
        if not compare(
            rows,
            f"probe height {index + 1} [m]",
            SOLVE_FRAGMENT,
            0.0,
            float(probes[index, 1]),
            0.0,
        ):
            failures.append(rows[-1]["quantity"])
    for index, value in enumerate(COMMITTED_AIR_MT):
        if not compare(
            rows,
            f"air |B| at +{[1, 2, 4][index]} m [mT]",
            SOLVE_FRAGMENT,
            value,
            float(air_magnitude[index]) * 1e3,
            1e-11,
        ):
            failures.append(rows[-1]["quantity"])
    for index, value in enumerate(COMMITTED_AIR_UT):
        if not compare(
            rows,
            f"air |B| at +{[1, 2, 4][index]} m [uT]",
            PAIR_FRAGMENT,
            value,
            float(air_magnitude[index]) * 1e6,
            1e-5,
        ):
            failures.append(rows[-1]["quantity"])
    if not compare(
        rows,
        "air |B| at +1 m, plan section 4 [uT]",
        "plan section 4",
        COMMITTED_AIR_PLAN_UT,
        float(air_magnitude[0]) * 1e6,
        5e-4,
    ):
        failures.append(rows[-1]["quantity"])
    for index, value in COMMITTED_B_KNOTS.items():
        if not compare(
            rows,
            f"table B knot {index} [T]",
            SOLVE_FRAGMENT,
            value,
            float(table_b[index]),
            0.0,
        ):
            failures.append(rows[-1]["quantity"])
    tolerances = {0: 1e-9, 1: 1e-7, -1: 1e-4}
    for index, value in COMMITTED_MU.items():
        source = (
            "tests/test_magnetisation.py west_material"
            if index == 0
            else SOLVE_FRAGMENT
            if index == 1
            else PAIR_FRAGMENT
        )
        if not compare(
            rows,
            f"table mu_relative knot {index}",
            source,
            value,
            float(table_mu[index]),
            tolerances[index],
        ):
            failures.append(rows[-1]["quantity"])
    compare(
        rows,
        "source time [s]",
        SOLVE_FRAGMENT,
        COMMITTED_TIME_S,
        float(pin["time_s"]),
        1e-12,
    )
    if not rows[-1]["passed"]:
        failures.append(rows[-1]["quantity"])

    for row in rows:
        print(
            "COMPARE {quantity}: expected={expected!r} measured={measured!r} "
            "rel={relative_difference:.3e} tol={tolerance:.1e} {verdict}".format(
                verdict="ok" if row["passed"] else "MISMATCH", **row
            ),
            flush=True,
        )
    if failures:
        print("MISMATCH " + json.dumps(failures), flush=True)
        return 2

    graded.REFERENCE = args.pin
    inputs = graded.west_inputs()
    print(
        "WEST_INPUTS_PASS pin_air_drift="
        f"{inputs['air_drift']:.3e} probes={inputs['probes'].tolist()} "
        f"segments={len(inputs['region'].geoms)}",
        flush=True,
    )

    scratch = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp")))
    control_path = scratch / args.pin.name
    shutil.copyfile(args.pin, control_path)
    graded.REFERENCE = control_path
    control_path.unlink()
    control = None
    try:
        graded.west_inputs()
    except FileNotFoundError as error:
        control = dict(outcome="failed", error=repr(error))
        print(f"NEGATIVE_CONTROL_PASS FileNotFoundError {error!r}", flush=True)
    else:
        control = dict(outcome="did-not-fail")
        print(
            "NEGATIVE_CONTROL_FAIL west_inputs() passed with the pin absent", flush=True
        )
    finally:
        scratch.rmdir()
    if control["outcome"] != "failed":
        return 3

    receipt = dict(
        pin=str(args.pin.resolve()),
        pin_sha256=hashlib.sha256(args.pin.read_bytes()).hexdigest(),
        pin_revision=pin["revision"],
        pin_py_blob=pin["pin_py_blob"],
        source_sha256=pin["source_sha256"],
        source_control_max_relative=pin["source_control_max_relative"],
        comparisons=rows,
        all_matched=not failures,
        west_inputs=dict(
            air_drift=inputs["air_drift"],
            probes_m=inputs["probes"].tolist(),
            air_field_T=np.asarray(inputs["air"]).tolist(),
        ),
        negative_control=control,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print("RECEIPT_WRITTEN " + str(args.receipt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
