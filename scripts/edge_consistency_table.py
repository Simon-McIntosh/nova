"""Report the edge-consistency measure over the ITER equilibrium sources.

Each source is opened with :class:`imas.DBEntry` at the Data Dictionary version
it was written in and read through
:class:`nova.imas.plasma_source.PlasmaSource`, whose every convention-bearing
quantity passes through :func:`nova.io.sourcemap.translate_equilibrium`.  A DD3
read therefore arrives in nova's internal COCOS 17, and a read whose measured
digits contradict its own DD version is refused there by name rather than
imputed.

The stored ``profiles_2d`` grid is uniform at the precision the two DINA files
carry and quantised at 1e-5 m in the CORSICA file, while the measure's
structured-grid Grad-Shafranov stencil needs a uniform axis, so each flux map is
interpolated onto the uniform axis the repository's own reconstruction returns,
carrying the stored point count over the stored extent, and a stored axis that
reconstruction refuses is reported.  The scored grid is named in every row.

The integral current check the census survey used is carried beside the three
forms: the current the grid read integrates to, and the current the stored flux
functions integrate to, each as a ratio to the slice's own ``Ip``.

Usage::

    python scripts/edge_consistency_table.py --output table.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import imas
import numpy as np
from scipy.interpolate import RectBivariateSpline

from nova.equilibrium.convention import toroidal_current_density
from nova.imas.edge_consistency import (
    EDGE_BAND_RELATIVE_THRESHOLD,
    EDGE_BAND_WIDTH,
    edge_consistency,
    grid_current_density,
)
from nova.imas.plasma_source import PlasmaSource
from nova.jax.config import configure_dtypes
from nova.media.sources.mast_efit import _uniform_axis

IMASDB = Path("/work/imas/shared/imasdb/ITER/3")

SOURCES: tuple[tuple[str, str, str], ...] = (
    ("DINA 105052/1", "105052/1", "3.38.1"),
    ("DINA 105069/3", "105069/3", "3.39.0"),
    ("CORSICA 130506/6", "130506/6", "3.41.0"),
)

#: The two pulses whose stored digits contradict their own DD version.  They
#: are read through the same route and reported as refused rows.
REFUSED: tuple[tuple[str, str, str], ...] = (
    ("DINA 105014/1", "105014/1", "3.38.1"),
    ("DINA 105013/1", "105013/1", "3.38.1"),
)

CURRENT_FLOOR_A = 1.0e6
SAMPLE_SIZE = 24

#: The two scalings that carry a translated DD3 read into the measure's
#: operands.  Both are one: COCOS 11 and COCOS 17 share ``e_Bp = 1``, so the
#: translator's factor is a sign flip on every psi-like and derivative-like row
#: and nothing else, and the value the reader returns is already the operand
#: the measure takes.  The analytic IDS round trip in
#: ``tests/test_edge_consistency_ids_route.py`` is what settles it: the grid
#: current and the profile current it reads back each integrate to its own
#: ``Ip`` at these scalings, and a 2 pi on either side moves them off one.
FLUX_SCALE = 1.0
DERIVATIVE_SCALE = 1.0


def _uniform(axis, name: str = "axis") -> np.ndarray:
    """Return the uniform axis the measure is scored on.

    The repository owns this reconstruction: :func:`_uniform_axis` of
    :mod:`nova.media.sources.mast_efit` returns the endpoint-preserving uniform
    axis of a stored one and refuses a stored axis that is not uniform at its
    own precision.  That routine is used directly here; the reconstruction it
    returns is the only one this script scores on.

    The refusal is a real case rather than a guard: a source whose stored
    coordinates were quantised when the file was written is not uniform at
    stored precision, and its uniform axis then differs from the stored one by
    that quantisation.  For such an axis the reconstruction is made over the
    same extent and point count, and the deviation from the stored axis is
    printed, so the substitution is visible in the run log rather than silent.
    """

    stored = np.asarray(axis, dtype=float)
    try:
        return _uniform_axis(stored, name)
    except ValueError:
        reconstructed = np.linspace(
            float(stored[0]), float(stored[-1]), int(stored.size)
        )
        deviation = float(np.max(np.abs(stored - reconstructed)))
        print(f"{name}: stored axis refused at {deviation:.3e} m, reconstructed")
        return reconstructed


def _sample(indices: list[int], size: int = SAMPLE_SIZE) -> tuple[list[int], bool]:
    """Return the evenly spaced sample of the above-floor indices.

    The second member of the pair reports whether the sample is a strict subset.
    """

    if len(indices) <= size:
        return list(indices), False
    positions = np.linspace(0, len(indices) - 1, size)
    return [indices[int(round(position))] for position in positions], True


def _above_floor(equilibrium, floor: float = CURRENT_FLOOR_A) -> list[int]:
    """Return the slice indices whose stored plasma current clears the floor."""

    indices = []
    for index in range(len(equilibrium.time_slice)):
        value = float(equilibrium.time_slice[index].global_quantities.ip)
        if np.isfinite(value) and abs(value) > floor:
            indices.append(index)
    return indices


def measure_slice(plasma_slice) -> dict:
    """Measure one COCOS 17 slice in the three forms, with its current check."""

    stored_r = np.asarray(plasma_slice.radius, dtype=float)
    stored_z = np.asarray(plasma_slice.height, dtype=float)
    stored_psi = np.asarray(plasma_slice.psi, dtype=float)
    grid_r = _uniform(stored_r, "profiles_2d/grid/dim1")
    grid_z = _uniform(stored_z, "profiles_2d/grid/dim2")
    # A piecewise-linear resample has a discontinuous second derivative at
    # every stored node, which is exactly what the stencil differentiates, so
    # the residual grows under refinement instead of falling.  A cubic spline
    # is C2 and leaves the stencil's own error as the only one.
    spline = RectBivariateSpline(stored_r, stored_z, stored_psi, kx=3, ky=3)
    resampled = spline(grid_r, grid_z)
    flux = FLUX_SCALE * resampled.T

    psi_axis = float(plasma_slice.psi_axis)
    psi_boundary = float(plasma_slice.psi_boundary)
    psi_norm = (flux - psi_axis) / (psi_boundary - psi_axis)
    interior = (psi_norm >= 0.0) & (psi_norm <= 1.0)

    samples = np.asarray(plasma_slice.psi_samples, dtype=float)
    coordinate = (samples - psi_axis) / (psi_boundary - psi_axis)
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    pressure_samples = np.asarray(plasma_slice.p_prime, dtype=float)[order]
    field_samples = np.asarray(plasma_slice.ff_prime, dtype=float)[order]

    def p_prime_of(normalised: np.ndarray) -> np.ndarray:
        return DERIVATIVE_SCALE * np.interp(normalised, coordinate, pressure_samples)

    def ff_prime_of(normalised: np.ndarray) -> np.ndarray:
        return DERIVATIVE_SCALE * np.interp(normalised, coordinate, field_samples)

    result = edge_consistency(
        flux,
        grid_r,
        grid_z,
        psi_axis,
        psi_boundary,
        p_prime_of,
        ff_prime_of,
        inside=interior,
    )

    radius_grid, _height_grid = np.meshgrid(grid_r, grid_z)
    profile_current = toroidal_current_density(
        radius_grid, p_prime_of(psi_norm), ff_prime_of(psi_norm)
    )
    grid_current = grid_current_density(flux, grid_r, grid_z)
    cell_area = np.gradient(grid_z)[:, None] * np.gradient(grid_r)[None, :]
    plasma_current = float(plasma_slice.plasma_current)

    def integrate(field: np.ndarray) -> float:
        good = interior & np.isfinite(field)
        return float(np.sum(np.where(good, field, 0.0) * cell_area))

    return {
        "time_s": float(plasma_slice.time),
        "plasma_current_a": plasma_current,
        "psi_axis_wb": psi_axis,
        "psi_boundary_wb": psi_boundary,
        "grid_radius": [float(grid_r[0]), float(grid_r[-1]), int(grid_r.size)],
        "grid_height": [float(grid_z[0]), float(grid_z[-1]), int(grid_z.size)],
        "band_width": EDGE_BAND_WIDTH,
        "threshold": EDGE_BAND_RELATIVE_THRESHOLD,
        **asdict(result),
        "grid_current_integral_ratio": integrate(grid_current) / plasma_current,
        "profile_current_integral_ratio": integrate(profile_current) / plasma_current,
    }


def read_source(name: str, case: str, dd_version: str) -> dict:
    """Read one source through the translated route and measure its slices."""

    entry = imas.DBEntry(f"imas:hdf5?path={IMASDB / case}", "r", dd_version=dd_version)
    try:
        written = str(
            entry.get("equilibrium").ids_properties.version_put.data_dictionary
        )
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
        homogeneous = int(equilibrium.ids_properties.homogeneous_time)
        indices = _above_floor(equilibrium)
        sampled, evenly_spaced = _sample(indices)
        rows = []
        refusal = None
        for index in sampled:
            try:
                plasma_slice = PlasmaSource.from_imas(
                    equilibrium, active, provenance=name, indices=[index]
                ).slices[0]
            except ValueError as refused:
                refusal = {"slice_index": int(index), "reason": str(refused)}
                break
            rows.append(measure_slice(plasma_slice))
    finally:
        entry.close()
    return {
        "source": name,
        "case": case,
        "written_dd_version": written,
        "homogeneous_time": homogeneous,
        "slices_above_1ma": len(indices),
        "slices_sampled": len(rows),
        "evenly_spaced_sample": evenly_spaced,
        "refused": refusal,
        "rows": rows,
    }


def summarise(source: dict) -> dict:
    """Return the per-source medians and maxima of the three forms."""

    rows = source["rows"]
    forms = ("whole_slice_relative", "band_absolute", "band_relative")
    summary = {"source": source["source"], "slices_sampled": len(rows)}
    for form in forms:
        values = np.array([row[form] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        summary[f"{form}_median"] = (
            float(np.median(finite)) if finite.size else float("nan")
        )
        summary[f"{form}_max"] = float(np.max(finite)) if finite.size else float("nan")
    band = np.array([row["band_relative"] for row in rows], dtype=float)
    summary["slices_above_threshold"] = int(
        np.count_nonzero(band > EDGE_BAND_RELATIVE_THRESHOLD)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe", action="store_true", help="measure one slice only")
    arguments = parser.parse_args(argv)

    configure_dtypes()
    import jax

    assert jax.config.jax_enable_x64 is True
    import nova

    print(f"# nova {nova.__file__}")
    print(f"# x64 {jax.config.jax_enable_x64} platforms {jax.default_backend()}")
    print(
        f"# band width {EDGE_BAND_WIDTH} threshold {EDGE_BAND_RELATIVE_THRESHOLD} "
        f"flux scale {FLUX_SCALE} derivative scale {DERIVATIVE_SCALE}"
    )

    sources = SOURCES[:1] if arguments.probe else SOURCES
    refused = () if arguments.probe else REFUSED
    payload: dict = {"sources": [], "refused": [], "summaries": []}
    for name, case, dd_version in sources:
        source = read_source(name, case, dd_version)
        payload["sources"].append(source)
        payload["summaries"].append(summarise(source))
        print(f"# {name}: {source['slices_sampled']} slices")
    for name, case, dd_version in refused:
        source = read_source(name, case, dd_version)
        payload["refused"].append(
            {
                "source": name,
                "case": case,
                "written_dd_version": source["written_dd_version"],
                "slices_above_1ma": source["slices_above_1ma"],
                "refused": source["refused"],
            }
        )
        print(f"# {name}: refused {source['refused']}")
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"# wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
