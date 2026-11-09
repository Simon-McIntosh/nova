"""Report the polished stationary points of each trip's solved field.

Every stationary point ``polish_stationary_points`` returns is reported with its
position, flux, Hessian determinant and eigenvalues, and the winding index of
the gradient on a small loop around it.  The winding index is the topological
test: a genuine extremum or saddle carries +/-1, a point that survives the
polish but is not a stationary point of the interpolated field carries 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import jax.numpy as jnp

from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.linalg.tensor_spline import fit_tensor_spline

from flux_orientation import analysis_field


def _seeds(radius, height, flux, count=24):
    dz, dr = np.gradient(flux, height, radius)
    magnitude = np.abs(dr) + np.abs(dz)
    found = []
    for i in range(2, len(height) - 2):
        for j in range(2, len(radius) - 2):
            block = magnitude[i - 1 : i + 2, j - 1 : j + 2]
            if magnitude[i, j] <= block.min():
                found.append((magnitude[i, j], radius[j], height[i]))
    found.sort()
    return np.asarray([[r, z] for _, r, z in found[:count]], dtype=float)


def _hessian(interp, point, step=1e-3):
    r, z = float(point[0]), float(point[1])
    f = lambda dr, dz: float(interp([[r + dr, z + dz]])[0])  # noqa: E731
    centre = f(0.0, 0.0)
    h_rr = (f(step, 0.0) - 2 * centre + f(-step, 0.0)) / step**2
    h_zz = (f(0.0, step) - 2 * centre + f(0.0, -step)) / step**2
    h_rz = (f(step, step) - f(step, -step) - f(-step, step) + f(-step, -step)) / (
        4 * step**2
    )
    matrix = np.array([[h_rr, h_rz], [h_rz, h_zz]])
    return matrix, np.linalg.det(matrix), np.linalg.eigvalsh(matrix)


def _winding(interp, point, radius=0.02, samples=32):
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    loop = np.c_[point[0] + radius * np.cos(angles), point[1] + radius * np.sin(angles)]
    delta = 1e-4
    gradients = np.empty((samples, 2))
    for index, (r, z) in enumerate(loop):
        dr = (interp([[r + delta, z]])[0] - interp([[r - delta, z]])[0]) / (2 * delta)
        dz = (interp([[r, z + delta]])[0] - interp([[r, z - delta]])[0]) / (2 * delta)
        gradients[index] = (dr, dz)
    phase = np.unwrap(np.arctan2(gradients[:, 1], gradients[:, 0]))
    closure = float(np.angle(np.exp(1j * (phase[0] - phase[-1]))))
    total = (phase[-1] - phase[0]) + closure
    return float(total / (2 * np.pi))


def report(archive: Path, label: str) -> dict:
    with np.load(archive) as data:
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        flux = analysis_field(data["flux"], radius, height)
    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(flux)
    )
    seeds = _seeds(radius, height, flux)
    polished = polish_stationary_points(
        surface, jnp.asarray(seeds), jnp.ones((len(seeds),), dtype=bool)
    )
    position = np.asarray(polished["position_rz"], dtype=float).reshape(-1, 2)
    value = np.asarray(polished["value"], dtype=float).reshape(-1)
    converged = np.asarray(polished["converged"], dtype=bool).reshape(-1)
    kind = np.asarray(polished["hessian_type"]).reshape(-1)
    interp = RegularGridInterpolator(
        (radius, height), flux.T, method="cubic", bounds_error=False, fill_value=np.nan
    )
    points = []
    for index in range(len(position)):
        if not converged[index]:
            continue
        inside = bool(
            radius[0] < position[index][0] < radius[-1]
            and height[0] < position[index][1] < height[-1]
        )
        matrix, determinant, eigenvalues = _hessian(interp, position[index])
        points.append(
            {
                "position_rz": position[index].tolist(),
                "value": float(value[index]),
                "hessian_type": int(kind[index]),
                "hessian_determinant": float(determinant),
                "hessian_eigenvalues": eigenvalues.tolist(),
                "inside_grid": inside,
                "winding_index": _winding(interp, position[index]) if inside else None,
                "gradient_at_point": [
                    float(
                        (
                            interp([[position[index][0] + 1e-4, position[index][1]]])[0]
                            - interp([[position[index][0] - 1e-4, position[index][1]]])[
                                0
                            ]
                        )
                        / 2e-4
                    ),
                    float(
                        (
                            interp([[position[index][0], position[index][1] + 1e-4]])[0]
                            - interp([[position[index][0], position[index][1] - 1e-4]])[
                                0
                            ]
                        )
                        / 2e-4
                    ),
                ],
            }
        )
    return {"archive": str(archive), "label": label, "points": points}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panels", nargs="+", default=["146:1", "146:2"])
    arguments = parser.parse_args()
    records = []
    for item in arguments.panels:
        frame, trip = (int(part) for part in item.split(":"))
        records.append(
            report(
                arguments.run_dir / f"frame-{frame}-trip-{trip}.npz",
                f"frame {frame} trip {trip}",
            )
        )
    arguments.out.write_text(json.dumps({"fields": records}, indent=2) + "\n")
    print(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
