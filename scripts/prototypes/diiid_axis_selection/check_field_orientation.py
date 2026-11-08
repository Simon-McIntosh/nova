"""Check stationary-axis evidence with explicit flux-lattice storage order."""

from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline
from nova.media.poloidal import inside_wall_units


def main():
    configure_dtypes()
    assert jax.config.jax_enable_x64
    archive = Path(__file__).with_name("stationary-field.npz")
    with np.load(archive) as data:
        radius, height = data["radius"], data["height"]
        stored, limiter = data["flux"], data["limiter"]
    lattice = FluxLattice(radius, height)
    reported = np.array([1.6881330007253326, -0.3458483185703583])
    claimed = np.array([1.585, -0.034])
    records = []
    for orientation, values in (
        ("native_radius_height", stored.T),
        ("transposed_height_radius", stored),
    ):
        spline = fit_tensor_spline(
            jnp.asarray(radius), jnp.asarray(height), jnp.asarray(values)
        )
        result = polish_stationary_points(
            spline,
            jnp.asarray(np.vstack((reported, claimed))),
            jnp.ones(2, bool),
        )
        inside = inside_wall_units(lattice.coordinate, limiter).reshape(lattice.shape)
        native_values = values.T
        index = np.unravel_index(
            np.argmax(np.where(inside, native_values, -np.inf)), lattice.shape
        )
        record = {
            "orientation": orientation,
            "argmax_rz": [float(radius[index[0]]), float(height[index[1]])],
        }
        record["seeds"] = []
        for slot, seed in enumerate((reported, claimed)):
            position = np.asarray(result["position_rz"])[slot]
            jet = spline.evaluate(seed[0], seed[1])
            record["seeds"].append(
                {
                    "seed_rz": seed.tolist(),
                    "position_rz": position.tolist(),
                    "shift_m": float(np.linalg.norm(position - seed)),
                    "converged": bool(result["converged"][slot]),
                    "hessian_type": int(result["hessian_type"][slot]),
                    "hessian_eigenvalues": np.linalg.eigvalsh(
                        np.asarray(result["hessian"])[slot]
                    ).tolist(),
                    "seed_gradient": [
                        float(jet.radial_derivative),
                        float(jet.vertical_derivative),
                    ],
                }
            )
        records.append(record)
    native_axis = records[0]["seeds"][0]
    transposed_axis = records[1]["seeds"][0]
    lattice_spacing = min(lattice.radial_step, lattice.vertical_step)
    checks = {
        "native_shift_below_lattice_spacing": native_axis["shift_m"] < lattice_spacing,
        "native_hessian_negative": bool(
            np.all(np.asarray(native_axis["hessian_eigenvalues"]) < 0.0)
        ),
        "transposed_shift_above_0_1_m": transposed_axis["shift_m"] > 0.1,
    }
    print(
        json.dumps(
            {
                "x64": bool(jax.config.jax_enable_x64),
                "lattice_shape": lattice.shape,
                "lattice_spacing_m": lattice_spacing,
                "records": records,
                "checks": checks,
            },
            indent=2,
        ),
        flush=True,
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("FIELD_ORIENTATION_CHECK=FAIL: " + ", ".join(failed))
    print("FIELD_ORIENTATION_CHECK=PASS", flush=True)


if __name__ == "__main__":
    main()
