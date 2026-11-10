"""Check saved nulls against the terminal field and track a reference saddle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline


def diagnose(source, case):
    output = {}
    index = json.loads((source / "comparison.json").read_text())
    paths = {
        r["arm"]: Path(r.get("source_path", source / r["arm"] / case))
        for r in index
        if r["case"] == case
    }
    for arm in ("base", "head"):
        path = paths.get(arm, source / arm / case)
        receipt = json.loads((path / "receipt.json").read_text())
        assert receipt["storage_order"] == "radius_height"
        with np.load(path / "terminal.npz") as archive:
            data = {key: archive[key] for key in archive.files}
        radius, height = data["radius"], data["height"]
        assert data["flux"].shape == (radius.size, height.size)
        reference = np.asarray(receipt["efit_x_point_rz_m"])
        saved = data["saddle"][:2]

        @jax.jit
        def roots(values, seeds):
            field = fit_tensor_spline(
                jnp.asarray(radius), jnp.asarray(height), values.T
            )
            return polish_stationary_points(
                field, seeds, jnp.ones(seeds.shape[0], bool), stationary_steps=32
            )

        tested = roots(jnp.asarray(data["flux"]), jnp.asarray([reference, saved]))
        null_check = {key: np.asarray(value).tolist() for key, value in tested.items()}
        tracked = reference
        continuation = []
        for fraction in np.linspace(0, 1, 21):
            field = (1 - fraction) * data["label"] + fraction * data["flux"]
            result = roots(jnp.asarray(field), jnp.asarray([tracked, saved]))
            position = np.asarray(result["position_rz"])[0]
            continuation.append(
                dict(
                    fraction=float(fraction),
                    position=position.tolist(),
                    converged=bool(np.asarray(result["converged"])[0]),
                    hessian_type=int(np.asarray(result["hessian_type"])[0]),
                    gradient_norm=float(np.asarray(result["gradient_norm"])[0]),
                )
            )
            tracked = position
        rr, zz = np.meshgrid(
            np.linspace(radius[0], radius[-1], 13),
            np.linspace(height[0], height[-1], 17),
        )
        seeds = np.column_stack((rr.ravel(), zz.ravel()))
        census = roots(jnp.asarray(data["flux"]), jnp.asarray(seeds))
        qualified = np.asarray(census["converged"]) & (
            np.asarray(census["hessian_type"]) == -1
        )
        unique = []
        for point in np.asarray(census["position_rz"])[qualified]:
            if not any(np.linalg.norm(point - previous) < 1e-5 for previous in unique):
                unique.append(point)
        assert unique, "known saved saddle not recovered by seed probe"
        field = fit_tensor_spline(
            jnp.asarray(radius), jnp.asarray(height), jnp.asarray(data["flux"].T)
        )
        at_saved = field.evaluate(saved[0], saved[1])
        output[arm] = dict(
            saved_saddle=saved.tolist(),
            saved_gradient=[
                float(at_saved.radial_derivative),
                float(at_saved.vertical_derivative),
            ],
            from_reference_and_saved=null_check,
            homotopy=continuation,
            seed_probe=dict(
                seeds=len(seeds),
                converged_saddle_slots=int(qualified.sum()),
                unique_saddles=np.asarray(unique).tolist(),
            ),
            caveat=(
                "Finite seed probe is not an exhaustive null census; "
                "homotopy interpolates fields, not a solver trajectory."
            ),
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--case", default="frame-137-diverted")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64
    result = diagnose(args.input, args.case)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
