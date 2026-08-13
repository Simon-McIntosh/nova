"""Resolve a diverted-reference disagreement on one production MAST slice.

The EFIT catalogue is used only to name and verify the selected diagnostic
slice. Reconstruction uses corrected machine signals and the content-addressed
machine description. A SciPy spline gradient/Hessian search supplies a
topology-independent check of the reconstructed flux map; it never calls the
Nova topology labeler.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root

from nova.imas.mast_chain_factory import build_mast_parity_chain
from nova.imas.mast_efit_referee import (
    _nearest_reference_indices,
    read_efit_referee,
)
from nova.imas.mast_parity_chain import (
    AcceleratorSettings,
    _accelerated_profile_solve,
    _moment_seeds,
    _pack_source_currents,
    _sensor_scales,
)
from nova.imas.mast_solve_inputs import read_corrected_solve_inputs


SHOT = 21978
SLICE_INDEX = 1963
SHOT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
ARTIFACT_CACHE = Path("/run/user/39486/imas-ambix-machine-artifact")


def boundary_termination_reason(read) -> str:
    """Describe which obstruction terminated the boundary push."""

    if not read.found:
        return "no closed axis-connected surface"
    if read.is_diverted:
        if read.boundary_resolved:
            return "x-point saddle bound the separatrix"
        return "x-point saddle remained unresolved"
    if read.x_candidate_count == 0:
        return "wall tangency with no saddle candidate"
    return "wall tangency preceded the saddle candidate"


def _inside_at(radius: float, height: float, rg, zg, inside) -> bool:
    row = int(np.argmin(np.abs(zg - height)))
    column = int(np.argmin(np.abs(rg - radius)))
    return bool(inside[row, column])


def independent_separatrix_saddles(
    psi: np.ndarray,
    rg: np.ndarray,
    zg: np.ndarray,
    inside: np.ndarray,
    boundary_flux: float,
) -> list[tuple[float, float, float]]:
    """Find sub-grid separatrix saddles without Nova topology routines.

    A bicubic interpolant supplies analytic first and second derivatives. Roots
    are seeded from every grid cell, deduplicated in physical space, typed by a
    negative Hessian determinant, and retained at the boundary flux within the
    local one-cell interpolation resolution.
    """

    spline = RectBivariateSpline(zg, rg, psi, kx=3, ky=3, s=0.0)

    def gradient(point):
        height, radius = point
        return np.asarray(
            [
                spline.ev(height, radius, dx=1, dy=0),
                spline.ev(height, radius, dx=0, dy=1),
            ],
            dtype=float,
        )

    roots: list[tuple[float, float]] = []
    for height in 0.5 * (zg[:-1] + zg[1:]):
        for radius in 0.5 * (rg[:-1] + rg[1:]):
            solved = root(gradient, np.asarray([height, radius], dtype=float))
            fitted_height, fitted_radius = solved.x
            supported = (
                solved.success
                and zg[1] <= fitted_height <= zg[-2]
                and rg[1] <= fitted_radius <= rg[-2]
                and _inside_at(fitted_radius, fitted_height, rg, zg, inside)
                and np.linalg.norm(gradient(solved.x)) <= 1.0e-8
            )
            if not supported:
                continue
            if any(
                np.hypot(fitted_radius - prior_r, fitted_height - prior_z) <= 1.0e-4
                for prior_r, prior_z in roots
            ):
                continue
            roots.append((float(fitted_radius), float(fitted_height)))

    saddles = []
    for radius, height in roots:
        radial_curvature = spline.ev(height, radius, dx=0, dy=2)
        vertical_curvature = spline.ev(height, radius, dx=2, dy=0)
        cross_curvature = spline.ev(height, radius, dx=1, dy=1)
        determinant = radial_curvature * vertical_curvature - cross_curvature**2
        if determinant >= 0.0:
            continue
        flux = float(spline.ev(height, radius))
        row = int(np.argmin(np.abs(zg - height)))
        column = int(np.argmin(np.abs(rg - radius)))
        neighbourhood = psi[row - 1 : row + 2, column - 1 : column + 2]
        resolution = float(np.max(np.abs(neighbourhood - flux)))
        if abs(flux - boundary_flux) <= resolution:
            saddles.append((radius, height, flux))
    return saddles


def _artifact_digest(cache: Path, supplied: str | None) -> str:
    if supplied is not None:
        return supplied
    objects = sorted((cache / "sha256").glob("[0-9a-f]" * 64))
    if not objects:
        raise FileNotFoundError(f"no machine artifact under {cache}")
    return f"sha256:{objects[0].name}"


def run(args) -> None:
    """Reconstruct the named slice and print the two-route verdict."""

    inputs = read_corrected_solve_inputs(args.shot, store=args.store)
    if not 0 <= args.slice_index < inputs.slice_count:
        raise IndexError(f"slice index {args.slice_index} is outside the shot")
    reference = read_efit_referee(args.shot, store=args.store)
    reference_index = _nearest_reference_indices(
        reference.time_s, inputs.time_s[[args.slice_index]], None
    )[0]
    if reference_index < 0 or not reference.diverted[reference_index]:
        raise ValueError("selected slice is not aligned to an EFIT-diverted row")

    digest = _artifact_digest(args.artifact_cache, args.artifact_digest)
    components = build_mast_parity_chain(
        args.shot,
        artifact_cache=args.artifact_cache,
        artifact_digest=digest,
        store=args.store,
    )
    selected = replace(
        inputs,
        time_s=inputs.time_s[[args.slice_index]],
        coil_currents_a=inputs.coil_currents_a[[args.slice_index]],
        sensor_signals=inputs.sensor_signals[[args.slice_index]],
        plasma_current_a=inputs.plasma_current_a[[args.slice_index]],
    )
    source = _pack_source_currents(components.profile_solver, selected)
    scale = _sensor_scales(selected.sensor_signals, None)
    _seeds, initial, mask, _vacuum = _moment_seeds(
        components.moment_solver,
        components.profile_solver,
        selected,
        source,
        scale,
    )
    solve = _accelerated_profile_solve(
        components.profile_solver,
        source,
        selected,
        scale,
        mask,
        initial,
        AcceleratorSettings(),
    )
    grid = components.topology_labeler.grid
    psi = solve.flux[0].reshape(grid.zg.size, grid.rg.size)
    read = components.topology_labeler.boundary_reads(psi[None])[0]
    labeler_points = np.asarray(read.xset, dtype=float)
    labeler_points = labeler_points[np.all(np.isfinite(labeler_points), axis=1)]
    independent = independent_separatrix_saddles(
        psi, grid.rg, grid.zg, grid.inside_limiter, read.psi_bnd
    )

    print(f"shot: {args.shot}")
    print(f"corrected_slice_index: {args.slice_index}")
    print(f"solve_time_s: {selected.time_s[0]:.12g}")
    print(f"efit_reference_index: {int(reference_index)}")
    print(f"efit_time_s: {reference.time_s[reference_index]:.12g}")
    print(f"grid_shape: {psi.shape}")
    print(f"independent_saddle_count: {len(independent)}")
    for index, (radius, height, flux) in enumerate(independent):
        print(
            f"independent_saddle_{index}: "
            f"r_m={radius:.12g} z_m={height:.12g} psi_wb={flux:.12g}"
        )
    print(f"labeler_saddle_count: {len(labeler_points)}")
    print(f"boundary_push_termination: {boundary_termination_reason(read)}")

    if independent and not labeler_points.size:
        raise RuntimeError("verdict: labeler defect remains unrepaired")
    if len(independent) != len(labeler_points):
        raise RuntimeError("verdict: independent and labeler saddle counts disagree")
    if independent:
        print("verdict: labeler defect repaired; saddle counts match")
    else:
        print(
            "verdict: genuine reconstruction-versus-EFIT difference; both counts zero"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--shot", type=int, default=SHOT)
    result.add_argument("--slice-index", type=int, default=SLICE_INDEX)
    result.add_argument("--store", type=Path, default=SHOT_STORE)
    result.add_argument("--artifact-cache", type=Path, default=ARTIFACT_CACHE)
    result.add_argument("--artifact-digest")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
