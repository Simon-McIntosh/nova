"""Check a regenerated WEST 62751 solve-inputs pin against the committed records.

The pin's value is that it was written before later code changes, so writing
it again proves nothing on its own. This script compares the regenerated probe
positions, CEA B-H table and air-core field against the values the committed
records state, reading each out of the record that owns it rather than from a
copy transcribed here, then runs west_inputs() against the pin and, with the
pin absent in a scratch copy, without it.

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
from scripts.west_iron_galerkin_pair.measure import (  # noqa: E402
    REFERENCE as PRODUCTION_REFERENCE,
)
from scripts.west_iron_inputs import committed  # noqa: E402


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

    def check(quantity, source, expected, measured, tolerance):
        if not compare(rows, quantity, source, expected, measured, tolerance):
            failures.append(quantity)

    radii = committed.probe_radii()
    for index, radius in enumerate(radii):
        check(
            f"probe radius {index + 1} [m]",
            committed.PAIR_FRAGMENT,
            radius,
            float(probes[index, 0]),
            0.0,
        )
        check(
            f"probe height {index + 1} [m]",
            committed.SOLVE_FRAGMENT,
            0.0,
            float(probes[index, 1]),
            0.0,
        )
    reach = ["+1", "2", "4"]
    for index, value in enumerate(committed.air_millitesla_solve()):
        check(
            f"air |B| at +{reach[index]} m [mT]",
            committed.SOLVE_FRAGMENT,
            value,
            float(air_magnitude[index]) * 1e3,
            1e-11,
        )
    for index, value in enumerate(committed.air_microtesla_pair()):
        check(
            f"air |B| at +{reach[index]} m [uT]",
            committed.PAIR_FRAGMENT,
            value,
            float(air_magnitude[index]) * 1e6,
            1e-5,
        )
    check(
        "air |B| at +1 m, plan [uT]",
        committed.PLAN,
        committed.air_plan_microtesla(),
        float(air_magnitude[0]) * 1e6,
        5e-4,
    )
    for index, value in enumerate(committed.table_b_knots()):
        check(
            f"table B knot {index} [T]",
            committed.SOLVE_FRAGMENT,
            value,
            float(table_b[index]),
            0.0,
        )
    origin = committed.table_mu_origin()
    check(
        "table mu_relative knot 0",
        committed.SOLVE_FRAGMENT,
        origin[0],
        float(table_mu[0]),
        1e-9,
    )
    check(
        "table mu_relative knot 1",
        committed.SOLVE_FRAGMENT,
        origin[1],
        float(table_mu[1]),
        1e-7,
    )
    check(
        "table mu_relative maximum",
        committed.PAIR_FRAGMENT,
        committed.table_mu_maximum(),
        float(table_mu.max()),
        1e-4,
    )
    check(
        "material test origin permeability",
        committed.MATERIAL_TEST,
        committed.material_test_mu_origin(),
        float(table_mu[0]),
        1e-9,
    )
    check(
        "source time [s]",
        committed.SOLVE_FRAGMENT,
        committed.source_time(),
        float(pin["time_s"]),
        1e-12,
    )

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

    production = dict(
        path=str(PRODUCTION_REFERENCE),
        exists=PRODUCTION_REFERENCE.is_file(),
    )
    print(
        f"PRODUCTION_REFERENCE path={production['path']} exists={production['exists']}",
        flush=True,
    )
    if not production["exists"]:
        print(
            "PRODUCTION_REFERENCE_BLOCKED the tracked constant still resolves "
            "under the retired root, so west_inputs() needs the in-process "
            "override below until the shared-root path change lands",
            flush=True,
        )

    graded.REFERENCE = args.pin
    inputs = graded.west_inputs()
    print(
        "WEST_INPUTS_PASS pin_air_drift="
        f"{inputs['air_drift']:.3f} probes={inputs['probes'].tolist()} "
        f"segments={len(inputs['region'].geoms)}",
        flush=True,
    )

    scratch = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp")))
    control_path = scratch / args.pin.name
    shutil.copyfile(args.pin, control_path)
    graded.REFERENCE = control_path
    control_path.unlink()
    try:
        graded.west_inputs()
    except FileNotFoundError as error:
        control = dict(outcome="failed", error=repr(error))
        print(f"NEGATIVE_CONTROL_PASS FileNotFoundError {error!r}", flush=True)
    else:
        control = dict(outcome="did-not-fail")
        print(
            "NEGATIVE_CONTROL_FAIL west_inputs() passed with the pin absent",
            flush=True,
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
        records_root=str(committed.record_root()),
        production_reference=production,
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
