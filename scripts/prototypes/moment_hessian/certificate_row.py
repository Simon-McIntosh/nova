"""Measure one certificate row while keeping its artifacts in a private run."""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import nova
from nova.biot import polygonanalytic

assert jax.config.jax_enable_x64 is True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmarks import solovev_certificate as certificate

    root = args.output.parent
    certificate.FIGURE_ROOT = root / "figures"
    certificate.PART_ROOT = root / "parts"
    certificate.DIAGNOSTIC_ROOT = root / "diagnostics"
    print(
        f"nova={nova.__file__} measurement_module={polygonanalytic.__file__} "
        f"cwd={Path.cwd().resolve()}",
        flush=True,
    )
    sys.argv = [
        "benchmarks/solovev_certificate.py",
        "--case",
        "diverted-single-null",
        "--requested-cells",
        "-300",
    ]
    certificate.main()
    row = json.loads(certificate._part_path("diverted-single-null", -300).read_text())
    args.output.write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(
        "CERTIFICATE_ROW "
        + json.dumps(
            {
                "residual": row["solver"]["terminal_fixed_point_residual"],
                "trips": row["solver"]["production_telemetry"]["trip_count"],
                "qualification": row["solver"]["qualification"],
                "part": str(certificate._part_path("diverted-single-null", -300)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
