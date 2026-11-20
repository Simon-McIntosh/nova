"""Which projection does imas-efit infer from ``b_field_pol_probe``?

imas-efit negates the stored poloidal angle when it reads a probe
(``src/imas.cpp``: ``poloidalOrientation = -probe->poloidal_angle``), because
IMAS defines the probe sensitive axis as ``cos(theta) Rhat - sin(theta) Zhat``
and EFIT works counter-clockwise.  A pulse whose field values were projected
along ``+sin(theta)`` is therefore inconsistent with the reader by a reflection
of the vertical field.

This script measures the convention on a pulse and its own reconstruction.  It
takes the stored ``poloidal_angle`` and ``field`` rows of a pulse, computes
``B_R`` and ``B_Z`` from the output equilibrium's own flux map at the probe
positions, and reports the root-mean-square difference between the stored row
and the two candidate projections.  The convention with the small residual is
the one the writer used, and it must be the one the reader infers or the rows
are self-contradictory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def probe_rows(pulse: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import imas

    entry = imas.DBEntry(f"imas:hdf5?path={pulse}", "r", dd_version="3.39.0")
    try:
        magnetics = entry.get("magnetics")
        angle = np.array(
            [float(item.poloidal_angle) for item in magnetics.b_field_pol_probe]
        )
        position = np.array(
            [
                [float(item.position.r), float(item.position.z)]
                for item in magnetics.b_field_pol_probe
            ]
        )
        field = np.array(
            [
                float(np.asarray(item.field.data)[0])
                for item in magnetics.b_field_pol_probe
            ]
        )
    finally:
        entry.close()
    return angle, position, field


def flux_map(output: Path):
    import imas

    entry = imas.DBEntry(f"imas:hdf5?path={output}", "r", dd_version="4.1.0")
    try:
        frame = entry.get("equilibrium").time_slice[0]
        radius = np.array(frame.profiles_2d[0].grid.dim1, dtype=float)
        height = np.array(frame.profiles_2d[0].grid.dim2, dtype=float)
        flux = np.array(frame.profiles_2d[0].psi, dtype=float)
    finally:
        entry.close()
    return radius, height, flux


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pulse", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    from scipy.interpolate import RectBivariateSpline

    angle, position, field = probe_rows(Path(args.pulse))
    radius, height, flux = flux_map(Path(args.output))
    spline = RectBivariateSpline(radius, height, flux, kx=3, ky=3)
    radial = -spline.ev(position[:, 0], position[:, 1], dx=0, dy=1)
    vertical = spline.ev(position[:, 0], position[:, 1], dx=1, dy=0)
    radial = radial / (2 * np.pi * position[:, 0])
    vertical = vertical / (2 * np.pi * position[:, 0])
    report = {
        "probes": int(angle.size),
        "field_rms": float(np.sqrt(np.mean(field**2))),
        "rmse_predicted_convention": float(
            np.sqrt(
                np.mean(
                    (radial * np.cos(angle) - vertical * np.sin(angle) - field) ** 2
                )
            )
        ),
        "rmse_prior_convention": float(
            np.sqrt(
                np.mean(
                    (radial * np.cos(angle) + vertical * np.sin(angle) - field) ** 2
                )
            )
        ),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
