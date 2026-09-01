"""Compare matched-capacity global and LCFS-split fits on measured flux maps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy.interpolate import RectBivariateSpline

from benchmarks.diiid_corpus_conventions import corpus_flux_to_nova_total
from benchmarks.diiid_forward_gs_match import _read, canonical_axes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/hex-cell-single-grid/measured-map-advantage.json"
DEFAULT_FIGURE = ROOT / "docs/figures/hex-cell-single-grid/measured-map-advantage.png"
AMBIX_ROOT = Path("/home/ITER/mcintos/Code/imas-ambix")
MAST_LEVEL_ONE = Path("/work/projects/imas_gpu/mast/level1/shots")
DIIID_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
MAST_SHOT = 22_086
MAST_FRAME = 43
DIIID_SHOT = "d3d_shot_00000c4a7b.parquet"
DIIID_FRAME = 179
DEGREES_OF_FREEDOM = 16
BOUNDARY_BAND_PITCHES = 2.0


@dataclass(frozen=True)
class MeasuredMap:
    """One measured flux map and its own released LCFS."""

    name: str
    radius: np.ndarray
    height: np.ndarray
    psi: np.ndarray
    axis_flux: float
    boundary_flux: float
    lcfs: np.ndarray
    source: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mast_map(
    ambix_root: Path = AMBIX_ROOT,
    shot: int = MAST_SHOT,
    frame: int = MAST_FRAME,
) -> MeasuredMap:
    """Read one MAST map through Ambix's evaluator-only EFIT harness."""

    code = """
import json
import zarr
from imas_ambix.eval.efit_referee import evaluator_context
from scripts.patch_flux_map_report import L1_ROOT, read_efit_slice

shot = int(__import__('sys').argv[1])
frame = int(__import__('sys').argv[2])
store = zarr.open(str(L1_ROOT / f'{shot}.zarr'), mode='r')
time_s = float(store['efm']['all_times'][frame])
with evaluator_context():
    row = read_efit_slice(shot, time_s)
if row is None or int(row['eidx']) != frame:
    raise RuntimeError('the requested MAST EFIT frame did not round-trip')
keys = ('eidx', 'dt_s', 'time_efm_s', 'rg', 'zg', 'psi_zr', 'psi_axis',
        'psi_boundary', 'axis_r', 'axis_z', 'lcfs_r', 'lcfs_z')
print(json.dumps({key: (value.tolist() if hasattr(value, 'tolist') else value)
                  for key, value in row.items() if key in keys},
                 allow_nan=False))
"""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to execute the Ambix evaluator harness")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("PYTHONPATH", None)
    arguments = ["-c", code, str(shot), str(frame)]
    uv_arguments = [
        uv,
        "run",
        "--directory",
        str(ambix_root),
        "--project",
        str(ambix_root),
        "--no-sync",
        "python",
        *arguments,
    ]
    completed = subprocess.run(
        uv_arguments,
        cwd=ambix_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    launcher = "uv --directory with --no-sync"
    if completed.returncode != 0 and "No module named 'torch'" in completed.stderr:
        interpreter = ambix_root / ".venv/bin/python"
        completed = subprocess.run(
            [str(interpreter), *arguments],
            cwd=ambix_root,
            env=environment,
            capture_output=True,
            text=True,
        )
        launcher = str(interpreter)
    if completed.returncode != 0:
        raise RuntimeError(
            "the Ambix evaluator harness failed:\n" + completed.stderr.strip()
        )
    row = json.loads(completed.stdout)
    store = MAST_LEVEL_ONE / f"{shot}.zarr"
    return MeasuredMap(
        name="MAST efm",
        radius=np.asarray(row["rg"], dtype=float),
        height=np.asarray(row["zg"], dtype=float),
        psi=np.asarray(row["psi_zr"], dtype=float).T,
        axis_flux=float(row["psi_axis"]),
        boundary_flux=float(row["psi_boundary"]),
        lcfs=np.c_[row["lcfs_r"], row["lcfs_z"]],
        source={
            "reader": "imas-ambix scripts.patch_flux_map_report.read_efit_slice",
            "firewall": "evaluator_context",
            "launcher": launcher,
            "store": str(store),
            "shot": shot,
            "efm_frame": int(row["eidx"]),
            "time_s": float(row["time_efm_s"]),
            "snap_delta_s": float(row["dt_s"]),
            "flux_crossing": "Wb/radian to total Wb in the Ambix reader",
        },
    )


def _diiid_map(
    data_root: Path = DIIID_DATA,
    shot: str = DIIID_SHOT,
    frame: int = DIIID_FRAME,
) -> MeasuredMap:
    """Read one DIII-D map through the benchmark corpus authorities."""

    path = data_root / shot
    columns = (
        "efit_times",
        "efit_psirz",
        "efit_r_axis",
        "efit_z_axis",
        "efit_lcfs_n",
        "efit_lcfs_r",
        "efit_lcfs_z",
        "efit_grid_R",
        "efit_grid_Z",
    )
    row = _read(path, columns)
    radius, height = canonical_axes(row)
    psi = corpus_flux_to_nova_total(np.asarray(row["efit_psirz"][frame], dtype=float)).T
    interpolant = RectBivariateSpline(radius, height, psi, kx=3, ky=3, s=0)
    axis_flux = float(
        interpolant.ev(row["efit_r_axis"][frame], row["efit_z_axis"][frame])
    )
    count = int(row["efit_lcfs_n"][frame])
    lcfs = np.c_[
        np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
    ]
    boundary_flux = float(np.median(interpolant.ev(lcfs[:, 0], lcfs[:, 1])))
    return MeasuredMap(
        name="DIII-D efit_psirz",
        radius=radius,
        height=height,
        psi=psi,
        axis_flux=axis_flux,
        boundary_flux=boundary_flux,
        lcfs=lcfs,
        source={
            "reader": "benchmarks.diiid_forward_gs_match._read and canonical_axes",
            "path": str(path),
            "sha256": _sha256(path),
            "frame": frame,
            "time_ms": float(row["efit_times"][frame]),
            "flux_crossing": "corpus_flux_to_nova_total exactly once",
        },
    )


def _powers() -> tuple[tuple[int, int], ...]:
    return tuple((i, degree - i) for degree in range(5) for i in range(degree + 1))


COMMON_POWERS = _powers()
GLOBAL_POWERS = COMMON_POWERS + ((5, 0),)


def _polynomial_design(
    radial_coordinate: np.ndarray,
    vertical_coordinate: np.ndarray,
    powers: tuple[tuple[int, int], ...],
) -> np.ndarray:
    return np.stack(
        [radial_coordinate**i * vertical_coordinate**j for i, j in powers], axis=-1
    )


def _polynomial_hessian(
    radial_coordinate: np.ndarray,
    vertical_coordinate: np.ndarray,
    powers: tuple[tuple[int, int], ...],
    radial_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    hessian = np.zeros(radial_coordinate.shape + (len(powers), 2, 2))
    for column, (i, j) in enumerate(powers):
        if i >= 2:
            hessian[..., column, 0, 0] = (
                i
                * (i - 1)
                * radial_coordinate ** (i - 2)
                * vertical_coordinate**j
                / radial_scale**2
            )
        if j >= 2:
            hessian[..., column, 1, 1] = (
                j
                * (j - 1)
                * radial_coordinate**i
                * vertical_coordinate ** (j - 2)
                / vertical_scale**2
            )
        if i >= 1 and j >= 1:
            cross = (
                i
                * j
                * radial_coordinate ** (i - 1)
                * vertical_coordinate ** (j - 1)
                / (radial_scale * vertical_scale)
            )
            hessian[..., column, 0, 1] = cross
            hessian[..., column, 1, 0] = cross
    return hessian


def _point_to_lcfs_distance(points: np.ndarray, lcfs: np.ndarray) -> np.ndarray:
    closed = np.vstack((lcfs, lcfs[0]))
    start = closed[:-1]
    segment = closed[1:] - start
    length_squared = np.sum(segment**2, axis=1)
    offset = points[:, None, :] - start[None, :, :]
    fraction = np.sum(offset * segment[None, :, :], axis=2) / np.maximum(
        length_squared[None, :], np.finfo(float).tiny
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start[None, :, :] + fraction[..., None] * segment[None, :, :]
    return np.sqrt(np.min(np.sum((points[:, None, :] - closest) ** 2, axis=2), axis=1))


def _reference_hessian(
    normalised_flux: np.ndarray, radius: np.ndarray, height: np.ndarray
) -> np.ndarray:
    gradient_r = np.gradient(normalised_flux, radius, axis=0, edge_order=2)
    gradient_z = np.gradient(normalised_flux, height, axis=1, edge_order=2)
    second_rr = np.gradient(gradient_r, radius, axis=0, edge_order=2)
    second_zz = np.gradient(gradient_z, height, axis=1, edge_order=2)
    second_rz = 0.5 * (
        np.gradient(gradient_r, height, axis=1, edge_order=2)
        + np.gradient(gradient_z, radius, axis=0, edge_order=2)
    )
    hessian = np.empty(normalised_flux.shape + (2, 2))
    hessian[..., 0, 0] = second_rr
    hessian[..., 0, 1] = second_rz
    hessian[..., 1, 0] = second_rz
    hessian[..., 1, 1] = second_zz
    return hessian


def _map_metrics(measured: MeasuredMap) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    radius = measured.radius
    height = measured.height
    if measured.psi.shape != (65, 65) or radius.size != 65 or height.size != 65:
        raise ValueError(f"{measured.name} is not the required 65 by 65 map")
    if not np.all(np.isfinite(measured.psi)):
        raise ValueError(f"{measured.name} contains non-finite flux")
    flux_scale = measured.boundary_flux - measured.axis_flux
    if not np.isfinite(flux_scale) or abs(flux_scale) <= np.finfo(float).tiny:
        raise ValueError(f"{measured.name} has an unusable axis-to-boundary flux scale")
    normalised_flux = (measured.psi - measured.axis_flux) / flux_scale
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    radial_midpoint = 0.5 * (radius[0] + radius[-1])
    vertical_midpoint = 0.5 * (height[0] + height[-1])
    radial_scale = 0.5 * (radius[-1] - radius[0])
    vertical_scale = 0.5 * (height[-1] - height[0])
    radial_coordinate = (rr - radial_midpoint) / radial_scale
    vertical_coordinate = (zz - vertical_midpoint) / vertical_scale
    points = np.c_[rr.ravel(), zz.ravel()]
    outside = ~PolygonPath(measured.lcfs).contains_points(points).reshape(rr.shape)

    reference = RectBivariateSpline(radius, height, normalised_flux, kx=3, ky=3, s=0)
    boundary_coordinate = normalised_flux - 1.0
    exterior_curvature = np.where(outside, boundary_coordinate**2, 0.0)
    global_design = _polynomial_design(
        radial_coordinate, vertical_coordinate, GLOBAL_POWERS
    )
    split_design = np.concatenate(
        (
            _polynomial_design(radial_coordinate, vertical_coordinate, COMMON_POWERS),
            exterior_curvature[..., None],
        ),
        axis=-1,
    )
    values = normalised_flux.ravel()
    global_matrix = global_design.reshape(-1, DEGREES_OF_FREEDOM)
    split_matrix = split_design.reshape(-1, DEGREES_OF_FREEDOM)
    global_coefficient = np.linalg.lstsq(global_matrix, values, rcond=None)[0]
    split_coefficient = np.linalg.lstsq(split_matrix, values, rcond=None)[0]

    global_basis_hessian = _polynomial_hessian(
        radial_coordinate,
        vertical_coordinate,
        GLOBAL_POWERS,
        radial_scale,
        vertical_scale,
    )
    split_basis_hessian = _polynomial_hessian(
        radial_coordinate,
        vertical_coordinate,
        COMMON_POWERS,
        radial_scale,
        vertical_scale,
    )
    q_r = reference.ev(points[:, 0], points[:, 1], dx=1, dy=0).reshape(rr.shape)
    q_z = reference.ev(points[:, 0], points[:, 1], dx=0, dy=1).reshape(rr.shape)
    q_rr = reference.ev(points[:, 0], points[:, 1], dx=2, dy=0).reshape(rr.shape)
    q_rz = reference.ev(points[:, 0], points[:, 1], dx=1, dy=1).reshape(rr.shape)
    q_zz = reference.ev(points[:, 0], points[:, 1], dx=0, dy=2).reshape(rr.shape)
    curvature_hessian = np.zeros(rr.shape + (2, 2))
    curvature_hessian[..., 0, 0] = 2.0 * (q_r**2 + boundary_coordinate * q_rr)
    curvature_hessian[..., 0, 1] = 2.0 * (q_r * q_z + boundary_coordinate * q_rz)
    curvature_hessian[..., 1, 0] = curvature_hessian[..., 0, 1]
    curvature_hessian[..., 1, 1] = 2.0 * (q_z**2 + boundary_coordinate * q_zz)
    curvature_hessian[~outside] = 0.0

    global_hessian = np.einsum(
        "...cij,c->...ij", global_basis_hessian, global_coefficient
    )
    split_hessian = (
        np.einsum("...cij,c->...ij", split_basis_hessian, split_coefficient[:-1])
        + split_coefficient[-1] * curvature_hessian
    )
    measured_hessian = _reference_hessian(normalised_flux, radius, height)

    radial_pitch = float(np.mean(np.diff(radius)))
    vertical_pitch = float(np.mean(np.diff(height)))
    cell_pitch = float(np.sqrt(0.5 * (radial_pitch**2 + vertical_pitch**2)))
    distance = _point_to_lcfs_distance(points, measured.lcfs).reshape(rr.shape)
    boundary_band = distance <= BOUNDARY_BAND_PITCHES * cell_pitch
    edge_safe = np.ones(rr.shape, dtype=bool)
    edge_safe[[0, -1], :] = False
    edge_safe[:, [0, -1]] = False
    boundary_band &= edge_safe
    if not np.any(boundary_band):
        raise RuntimeError(f"{measured.name} boundary band contains no cells")

    global_error = global_hessian - measured_hessian
    split_error = split_hessian - measured_hessian
    global_rms = float(np.sqrt(np.mean(global_error[boundary_band] ** 2)))
    split_rms = float(np.sqrt(np.mean(split_error[boundary_band] ** 2)))
    global_scaled = global_rms * cell_pitch**2
    split_scaled = split_rms * cell_pitch**2
    improvement = global_rms / max(split_rms, np.finfo(float).tiny)
    falsifier_fired = bool(split_rms >= global_rms)
    global_fit = (global_matrix @ global_coefficient).reshape(rr.shape)
    split_fit = (split_matrix @ split_coefficient).reshape(rr.shape)
    result = {
        "source": measured.source,
        "grid_shape": [int(radius.size), int(height.size)],
        "lcfs_vertex_count": int(measured.lcfs.shape[0]),
        "axis_flux": measured.axis_flux,
        "boundary_flux": measured.boundary_flux,
        "fit_sample_count": int(values.size),
        "degrees_of_freedom_each": DEGREES_OF_FREEDOM,
        "design_rank": {
            "global_c2": int(np.linalg.matrix_rank(global_matrix)),
            "boundary_split": int(np.linalg.matrix_rank(split_matrix)),
        },
        "design_condition_number": {
            "global_c2": float(np.linalg.cond(global_matrix)),
            "boundary_split": float(np.linalg.cond(split_matrix)),
        },
        "cell_pitch_m": cell_pitch,
        "radial_pitch_m": radial_pitch,
        "vertical_pitch_m": vertical_pitch,
        "boundary_band_half_width_m": BOUNDARY_BAND_PITCHES * cell_pitch,
        "boundary_band_cell_count": int(np.count_nonzero(boundary_band)),
        "boundary_band_second_derivative_rms": {
            "units": "normalised_flux_per_m2",
            "global_c2": global_rms,
            "boundary_split": split_rms,
        },
        "boundary_band_cell_pitch_scaled_second_derivative_rms": {
            "units": "normalised_flux",
            "global_c2": global_scaled,
            "boundary_split": split_scaled,
        },
        "boundary_band_second_derivative_improvement_factor": improvement,
        "fit_value_rms": {
            "units": "normalised_flux",
            "global_c2": float(np.sqrt(np.mean((global_fit - normalised_flux) ** 2))),
            "boundary_split": float(
                np.sqrt(np.mean((split_fit - normalised_flux) ** 2))
            ),
        },
        "falsifier_fired": falsifier_fired,
        "verdict": "FAIL" if falsifier_fired else "PASS",
    }
    plot_data = {
        "radius": radius,
        "height": height,
        "normalised_flux": normalised_flux,
        "lcfs": measured.lcfs,
        "boundary_band": boundary_band,
        "global_error": np.sqrt(np.mean(global_error**2, axis=(-2, -1)))
        * cell_pitch**2,
        "split_error": np.sqrt(np.mean(split_error**2, axis=(-2, -1))) * cell_pitch**2,
    }
    return result, plot_data


def _figure(
    maps: list[MeasuredMap], plot_rows: list[dict[str, np.ndarray]], path: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), constrained_layout=True)
    for row, (measured, plot_data) in enumerate(zip(maps, plot_rows, strict=True)):
        radius = plot_data["radius"]
        height = plot_data["height"]
        map_axis, global_axis, split_axis = axes[row]
        flux_image = map_axis.pcolormesh(
            radius,
            height,
            plot_data["normalised_flux"].T,
            shading="auto",
            cmap="viridis",
        )
        map_axis.plot(measured.lcfs[:, 0], measured.lcfs[:, 1], "w-", lw=1.5)
        map_axis.contour(
            radius,
            height,
            plot_data["boundary_band"].T.astype(float),
            levels=[0.5],
            colors=["#ffcc66"],
            linewidths=1.0,
        )
        figure.colorbar(flux_image, ax=map_axis, label="normalised flux")
        map_axis.set_title(f"{measured.name}: LCFS and two-pitch band")
        maximum = float(
            max(
                np.nanpercentile(plot_data["global_error"], 98),
                np.nanpercentile(plot_data["split_error"], 98),
                np.finfo(float).tiny,
            )
        )
        for axis, key, title in (
            (global_axis, "global_error", "global C2 curvature residual"),
            (split_axis, "split_error", "boundary-split curvature residual"),
        ):
            image = axis.pcolormesh(
                radius,
                height,
                plot_data[key].T,
                shading="auto",
                cmap="magma",
                vmin=0.0,
                vmax=maximum,
            )
            axis.plot(measured.lcfs[:, 0], measured.lcfs[:, 1], "c-", lw=1.0)
            figure.colorbar(image, ax=axis, label="cell-pitch-scaled RMS")
            axis.set_title(title)
        for axis in axes[row]:
            axis.set_xlabel("R [m]")
            axis.set_ylabel("Z [m]")
            axis.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output: Path = DEFAULT_OUTPUT, figure: Path = DEFAULT_FIGURE) -> dict[str, Any]:
    """Run both measured-map comparisons and write the receipt and figure."""

    maps = [_mast_map(), _diiid_map()]
    rows = [_map_metrics(measured) for measured in maps]
    per_map = {
        measured.name: result for measured, (result, _) in zip(maps, rows, strict=True)
    }
    falsifier_fired = any(result["falsifier_fired"] for result in per_map.values())
    receipt = {
        "measurement": (
            "boundary-band second-derivative advantage of an LCFS-split fit "
            "against a matched-capacity single global C2 fit"
        ),
        "method": {
            "fit": (
                "least squares over all 65 by 65 measured samples; 15 shared "
                "degree-at-most-four monomials plus one exterior squared-LCFS "
                "coordinate for the split fit, versus the same 15 monomials "
                "plus one global fifth-degree term"
            ),
            "interface": (
                "each released LCFS polygon; the exterior squared normalised-flux "
                "coordinate is value-and-gradient continuous at that interface"
            ),
            "reference_second_derivative": (
                "centred finite differences of the measured normalised flux map; "
                "outermost grid cells excluded"
            ),
            "boundary_band": (
                "Euclidean distance at most two RMS nearest-neighbour cell pitches "
                "from the released LCFS"
            ),
            "hessian_rms": "RMS over boundary-band cells and all four Hessian entries",
        },
        "degrees_of_freedom_each": DEGREES_OF_FREEDOM,
        "boundary_band_pitches": BOUNDARY_BAND_PITCHES,
        "per_map": per_map,
        "falsifier": (
            "fires if boundary_split second-derivative RMS is not lower than "
            "global_c2 on either measured map"
        ),
        "falsifier_fired": falsifier_fired,
        "verdict": "FAIL" if falsifier_fired else "PASS",
        "artifacts": {"receipt": str(output), "figure": str(figure)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _figure(maps, [plot_data for _, plot_data in rows], figure)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    arguments = parser.parse_args()
    receipt = run(arguments.output, arguments.figure)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
