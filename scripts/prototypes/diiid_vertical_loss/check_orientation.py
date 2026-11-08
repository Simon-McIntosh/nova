"""Assert that a trip archive's flux map is read in its declared storage order.

The lattice storage order is decided by ``FluxLattice`` (radius-major), not by
the array shape, and a square lattice cannot be told apart from its transpose
by shape.  This check is the shape test's replacement: it polishes from the
axis the solve reported and requires the displacement to stay under one lattice
spacing, which is what a correctly oriented field gives and what a transposed
one cannot.

``--transpose`` is the negative control.  It transposes the field after it has
been bound to the declared order, reproducing the misread the check exists to
refuse, and the check must then fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.linalg.tensor_spline import fit_tensor_spline

from flux_orientation import ARCHIVE_STORAGE_ORDER, READER_ORDER, analysis_field


def measure(archive: Path, axis, transpose: bool) -> dict:
    """Return the polish displacement of the reported axis on one archive."""
    with np.load(archive) as data:
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        flux = analysis_field(data["flux"], radius, height)

    spacing = {
        "radial": float(radius[1] - radius[0]),
        "vertical": float(height[1] - height[0]),
    }
    bound = min(spacing.values())

    if transpose:
        flux = np.ascontiguousarray(flux.T)
    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(flux)
    )
    seed = np.asarray(axis, dtype=float).reshape(1, 2)
    polished = polish_stationary_points(
        surface, jnp.asarray(seed), jnp.ones((1,), dtype=bool)
    )
    position = np.asarray(polished["position_rz"], dtype=float).reshape(-1, 2)[0]
    converged = bool(np.asarray(polished["converged"], dtype=bool).reshape(-1)[0])
    value = float(np.asarray(polished["value"], dtype=float).reshape(-1)[0])
    hessian = np.asarray(polished["hessian"], dtype=float).reshape(2, 2)
    eigenvalues = np.linalg.eigvalsh(hessian)

    return {
        "archive": str(archive),
        "declared_storage_order": list(ARCHIVE_STORAGE_ORDER),
        "reader_order": list(READER_ORDER),
        "lattice_extent": [int(radius.size), int(height.size)],
        "spacing_m": spacing,
        "one_lattice_spacing_m": bound,
        "transposed": bool(transpose),
        "reported_axis_rz_m": [float(value) for value in seed[0]],
        "polished_rz_m": [float(value) for value in position],
        "shift_m": float(np.hypot(*(position - seed[0]))),
        "converged": converged,
        "flux": value,
        "hessian_eigenvalues": eigenvalues.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--axis", type=float, nargs=2, required=True)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()

    record = measure(arguments.archive, arguments.axis, arguments.transpose)
    record["passed"] = record["shift_m"] < record["one_lattice_spacing_m"]
    payload = json.dumps(record, indent=2)
    print(payload)
    if arguments.out is not None:
        arguments.out.write_text(payload + "\n")
    if not record["passed"]:
        print(
            "ORIENTATION-CHECK FAILED: the reported axis polishes %.6f m, which "
            "is not less than one lattice spacing (%.6f m)"
            % (record["shift_m"], record["one_lattice_spacing_m"])
        )
        raise SystemExit(1)
    print("ORIENTATION-CHECK PASSED")


if __name__ == "__main__":
    sys.exit(main())
