"""Measure the COCOS 17 flux-function current relation and each source's digits.

Two measurements, both from stored fields only.

The regression
    On DD4-native data (COCOS 17 by definition, needing no translation) form
    the grid current with :func:`nova.equilibrium.map_extraction.apply_delta_star`
    and the profile current from the stored ``dpressure_dpsi`` and
    ``f_df_dpsi`` under both a ``+2*pi`` and a ``-2*pi`` factor, then regress
    one on the other cell by cell.  A single sign leaves the two fields
    collinear; scatter does not.  The literal negative control feeds the same
    regression the flux map with psi negated, and the slope reverses.

The digits
    For each source, read the COCOS digits from its own stored fields:
    ``sigma_Bp`` and ``sigma_rho_theta_phi`` through
    :func:`nova.io.sourcemap.determine_digits`; ``e_Bp`` from the factor of
    ``2*pi`` the Grad--Shafranov integral needs to reach the recorded current;
    and the sign the Grad--Shafranov relation itself implies between the map
    and the stored derivatives.

Run conditions: IMAS data is read only through ``imas.DBEntry`` at the written
DD version, and every read checks ``homogeneous_time``.  Meant to run on a
batch node, never the login node.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TWO_PI = 2.0 * math.pi


def open_entry(path, dd_version):
    """Open one imasdb pulse read-only at its written DD version."""

    import imas

    return imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version=dd_version)


def read_slice(entry, index):
    """Return the stored fields of one equilibrium slice, in stored units."""

    import numpy as np

    equilibrium = entry.get("equilibrium")
    homogeneous = int(
        np.asarray(equilibrium.ids_properties.homogeneous_time).ravel()[0]
    )
    if homogeneous != 1:
        raise SystemExit(f"equilibrium is homogeneous_time={homogeneous}, not 1")
    slice_ = equilibrium.time_slice[index]
    profiles_2d = slice_.profiles_2d
    profiles_1d = slice_.profiles_1d
    grid = profiles_2d[0].grid
    radius = np.asarray(grid.dim1, dtype=float)
    height = np.asarray(grid.dim2, dtype=float)
    # IMAS stores dim1 = R, dim2 = Z, and the map as [dim2, dim1].
    flux = np.asarray(profiles_2d[0].psi, dtype=float).T
    quantities = slice_.global_quantities
    b0 = np.asarray(equilibrium.vacuum_toroidal_field.b0).ravel()
    psi_1d = np.asarray(profiles_1d.psi, dtype=float)

    def observed(value):
        """Return a stored scalar, with the DD 'unset' sentinel read as absent."""

        number = float(np.asarray(value).ravel()[0])
        return number if np.isfinite(number) and abs(number) < 1.0e30 else float("nan")

    psi_axis = observed(quantities.psi_axis)
    psi_boundary = observed(quantities.psi_boundary)
    if not math.isfinite(psi_axis):
        psi_axis = float(psi_1d[0])
    if not math.isfinite(psi_boundary):
        psi_boundary = float(psi_1d[-1])
    return {
        "radius": radius,
        "height": height,
        "flux": flux,
        "psi_1d": psi_1d,
        "dp_dpsi": np.asarray(profiles_1d.dpressure_dpsi, dtype=float),
        "fdf_dpsi": np.asarray(profiles_1d.f_df_dpsi, dtype=float),
        "ip": float(np.asarray(quantities.ip).ravel()[0]),
        "psi_axis": psi_axis,
        "psi_boundary": psi_boundary,
        "q": observed(quantities.q_95),
        "b0": float(b0[0]) if b0.size else float("nan"),
    }


def resample(radius, height, flux, n_r=65, n_z=129):
    """Resample the stored map onto a uniform grid with a cubic spline.

    A piecewise-linear interpolant has a discontinuous second derivative at
    every stored node, and the Grad--Shafranov stencil differentiates exactly
    that, so the residual grows under refinement instead of falling.  The
    spline is what lets a stored non-uniform ``profiles_2d`` grid reach a
    centred-difference stencil at all.
    """

    import numpy as np
    from scipy.interpolate import RectBivariateSpline

    r = np.asarray(radius, dtype=float)
    z = np.asarray(height, dtype=float)
    f = np.asarray(flux, dtype=float)
    if f.shape != (r.size, z.size):
        f = f.T
    if not np.all(np.diff(r) > 0):
        r, f = r[::-1], f[::-1, :]
    if not np.all(np.diff(z) > 0):
        z, f = z[::-1], f[:, ::-1]
    spline = RectBivariateSpline(r, z, f, kx=3, ky=3)
    uniform_r = np.linspace(r[0], r[-1], n_r)
    uniform_z = np.linspace(z[0], z[-1], n_z)
    return uniform_r, uniform_z, spline(uniform_r, uniform_z)


def normalise_on_grid(psi_1d, values, psi_axis, psi_boundary, psi_norm):
    """Interpolate one stored flux function onto the map's normalised flux."""

    import numpy as np

    span = float(psi_boundary) - float(psi_axis)
    if span == 0.0:
        raise ValueError("axis and boundary flux are equal")
    coordinate = (np.asarray(psi_1d, dtype=float) - float(psi_axis)) / span
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(coordinate) & np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        raise ValueError("fewer than three finite samples in a flux function")
    coordinate, values = coordinate[finite], values[finite]
    order = np.argsort(coordinate)
    coordinate, values = coordinate[order], values[order]
    return np.interp(psi_norm, coordinate, values)


def grid_current(radius, height, flux):
    """Return the toroidal current density the map implies, on ``(R, Z)``."""

    import numpy as np

    from nova.equilibrium.map_extraction import apply_delta_star

    receipt = apply_delta_star(radius, height, flux)
    return np.asarray(receipt.toroidal_current_density, dtype=float)


def regression(grid, profile):
    """Least-squares slope of ``grid`` on ``profile``, with its correlation.

    Both are ``(R, Z)`` arrays on the same uniform grid; the scored set is
    every cell where both are finite and the stencil is complete.
    """

    import numpy as np

    finite = np.isfinite(grid) & np.isfinite(profile)
    if np.count_nonzero(finite) < 20:
        raise RuntimeError("too few finite cells for a regression")
    x = profile[finite]
    y = grid[finite]
    denominator = float(np.dot(x, x) - x.size * x.mean() ** 2)
    slope = float(np.dot(x, y) - x.size * x.mean() * y.mean()) / denominator
    correlation = float(np.corrcoef(x, y)[0, 1])
    residual = x * slope - y
    return {
        "cells": int(x.size),
        "slope": slope,
        "correlation": correlation,
        "residual_over_grid_rms": float(
            np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(y**2))
        ),
    }


def profile_current(radius, dp_dpsi, fdf_dpsi, factor):
    """Return ``factor * (R p' + FF'/(mu0 R))`` on the ``(R, Z)`` grid."""

    import numpy as np
    from scipy.constants import mu_0

    major = np.asarray(radius, dtype=float)[:, np.newaxis]
    return factor * (major * dp_dpsi + fdf_dpsi / (mu_0 * major))


def measure_slice(fields, negate_flux=False):
    """Run both profile signs on one slice, and the negated-map control."""

    import numpy as np

    uniform_r, uniform_z, flux = resample(
        fields["radius"], fields["height"], fields["flux"]
    )
    psi_axis, psi_boundary = fields["psi_axis"], fields["psi_boundary"]
    span = float(psi_boundary) - float(psi_axis)
    operator_flux = -flux if negate_flux else flux
    psi_norm = (flux - psi_axis) / span
    scored = (psi_norm >= 0.0) & (psi_norm <= 1.0)
    dp = normalise_on_grid(
        fields["psi_1d"], fields["dp_dpsi"], psi_axis, psi_boundary, psi_norm
    )
    fdf = normalise_on_grid(
        fields["psi_1d"], fields["fdf_dpsi"], psi_axis, psi_boundary, psi_norm
    )
    grid = grid_current(uniform_r, uniform_z, operator_flux)
    positive = profile_current(uniform_r, dp, fdf, TWO_PI)
    positive = np.where(scored, positive, np.nan)
    grid = np.where(scored, grid, np.nan)
    out = {
        "cells": int(np.count_nonzero(np.isfinite(grid) & np.isfinite(positive))),
        "plus_2pi": regression(grid, positive),
        "minus_2pi": regression(grid, -positive),
    }
    area = float(np.diff(uniform_r).mean() * np.diff(uniform_z).mean())
    finite = np.isfinite(grid)
    out["grid_current_over_ip"] = float(np.sum(grid[finite]) * area / fields["ip"])
    return out


def digits(fields):
    """Return the four COCOS digits this slice's stored fields pin."""

    import numpy as np

    from nova.io.sourcemap import determine_digits

    return determine_digits(
        [
            {
                "has_signs": True,
                "ip": fields["ip"],
                "psi_axis": fields["psi_axis"],
                "psi_boundary": fields["psi_boundary"],
                "psi_axis_minus_boundary": fields["psi_axis"] - fields["psi_boundary"],
                "b0": fields["b0"],
                "q_95": fields["q"],
                "dp_dpsi": float(np.nanmedian(fields["dp_dpsi"])),
                "irod": np.nan,
            }
        ]
    )


def e_bp_from_magnitude(fields):
    """Return the integral current ratio under each ``e_Bp`` reading.

    The map's integral is ``-Delta* psi / (2 pi mu0 R)`` when psi is the
    total flux (``e_Bp = 1``) and ``2 pi`` larger when psi is per radian
    (``e_Bp = 0``), so the hypothesis whose ratio to ``Ip`` is nearest one is
    the one the stored magnitude supports.
    """

    import numpy as np

    uniform_r, uniform_z, flux = resample(
        fields["radius"], fields["height"], fields["flux"]
    )
    from nova.equilibrium.map_extraction import apply_delta_star

    receipt = apply_delta_star(uniform_r, uniform_z, flux)
    density = np.asarray(receipt.toroidal_current_density, dtype=float)
    area = float(np.diff(uniform_r).mean() * np.diff(uniform_z).mean())
    total = float(np.sum(density[np.isfinite(density)]) * area)
    ratio = total / float(fields["ip"])
    return {
        "e_bp_1_ratio": ratio,
        "e_bp_0_ratio": ratio * TWO_PI,
        "nearer": 1 if abs(ratio - 1.0) <= abs(ratio * TWO_PI - 1.0) else 0,
    }


def single_dd4_layer():
    """Return the DD 4.1.0 reference layer converted ready to open."""

    import hashlib

    manifest = Path("tests/data-manifest.json")
    record = json.loads(manifest.read_text())["efitpp_validator_layers"]
    case = record["cases"]["iter_corsica_130506"]
    reference = next(layer for layer in case["layers"] if layer["role"] == "reference")
    layer = Path.home() / ".cache" / "imas-efit-publish-169" / reference["name"]
    if not layer.is_file():
        raise SystemExit(f"pinned DD4 layer not cached at {layer}")
    digest = hashlib.sha256(layer.read_bytes()).hexdigest()
    expected = reference["digest"].removeprefix("sha256:")
    if digest != expected:
        raise SystemExit("pinned DD4 layer digest does not match the manifest")
    from nova.imas.validator_case import netcdf_to_hdf5

    target = Path(__import__("os").environ["TMPDIR"]) / "dd4-layer"
    return netcdf_to_hdf5(layer, target, case["dd_version"], ["equilibrium"]), case[
        "dd_version"
    ]


SOURCES = {
    "CORSICA 130506/6": ("/work/imas/shared/imasdb/ITER/3/130506/6", "3.41.0"),
    "DINA 105052/1": ("/work/imas/shared/imasdb/ITER/3/105052/1", "3.38.1"),
    "DINA 105069/3": ("/work/imas/shared/imasdb/ITER/3/105069/3", "3.39.0"),
    "DINA 105014/1": ("/work/imas/shared/imasdb/ITER/3/105014/1", "3.38.1"),
    "DINA 105013/1": ("/work/imas/shared/imasdb/ITER/3/105013/1", "3.38.1"),
}


def main() -> None:  # pragma: no cover - runs on a batch node
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--slices", type=int, default=3)
    args = parser.parse_args()

    import numpy as np

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax

    assert jax.config.jax_enable_x64 is True

    report = {"module": __import__("nova").__file__}

    # (1b) the DD4-native regression, which is COCOS 17 by definition.
    directory, dd_version = single_dd4_layer()
    import imas

    entry = imas.DBEntry(f"imas:hdf5?path={directory}", "r", dd_version=dd_version)
    equilibrium = entry.get("equilibrium")
    count = len(equilibrium.time_slice)
    sample = sorted(set(np.linspace(0, count - 1, min(count, args.slices)).astype(int)))
    rows = []
    for index in sample:
        fields = read_slice(entry, int(index))
        direct = measure_slice(fields)
        negated = measure_slice(fields, negate_flux=True)
        rows.append(
            {
                "slice": int(index),
                "ip": fields["ip"],
                "plus_2pi": direct["plus_2pi"],
                "minus_2pi": direct["minus_2pi"],
                "grid_current_over_ip": direct["grid_current_over_ip"],
                "negated_map_plus_flux": negated["plus_2pi"],
                "negated_map_over_ip": negated["grid_current_over_ip"],
                "digits": digits(fields),
                "e_bp": e_bp_from_magnitude(fields),
            }
        )
    report["dd4_regression"] = {
        "layer": str(directory),
        "dd_version": dd_version,
        "slices": rows,
    }

    # (2) each source's digits from its own stored fields, no translation.
    report["sources"] = {}
    for name, (path, version) in SOURCES.items():
        entry = open_entry(Path(path), version)
        equilibrium = entry.get("equilibrium")
        count = len(equilibrium.time_slice)
        for index in range(count):
            ip = float(
                np.asarray(equilibrium.time_slice[index].global_quantities.ip).ravel()[
                    0
                ]
            )
            if abs(ip) > 1.0e6:
                break
        else:
            index = 0
        fields = read_slice(entry, index)
        report["sources"][name] = {
            "path": path,
            "dd_version": version,
            "slice": int(index),
            "digits": digits(fields),
            "e_bp": e_bp_from_magnitude(fields),
            "regression": measure_slice(fields),
        }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
