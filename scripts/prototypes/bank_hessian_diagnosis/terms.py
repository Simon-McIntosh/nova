"""Compare individual polygon source terms against their smooth point stencil."""

# ruff: noqa: E402
import json

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

from nova.biot import polygonanalytic as kernel
from nova.equilibrium.topology import BiotMomentCoupling
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT, reference
from scripts.prototypes.read_route.jets import _source_jets


def fields(point, packed):
    dtype = np.longdouble
    args = {k: v.astype(dtype) if v.dtype.kind == "f" else v for k, v in packed.items()}
    point = np.array(point, dtype=dtype)
    original = kernel._conditioned_flux_moments
    kernel._conditioned_flux_moments = lambda xp, r, *unused: (xp.zeros_like(r),) * 3
    try:
        rows = kernel.packed_analytic_moments(
            np,
            np.full(args["norm"].shape, point[0]),
            np.full(args["norm"].shape, point[1]),
            args["edge"],
            args["weight"],
            args["norm"],
            args["centre"].T,
            args["centre"].T,
            args["reflection_axis"],
            args["reflection_partner"],
        )
    finally:
        kernel._conditioned_flux_moments = original
    coefficients = (
        2
        * dtype(np.pi)
        * point[0]
        * np.stack(
            (np.stack(rows[6:9], axis=-1), -np.stack(rows[3:6], axis=-1)), axis=-1
        )
    )
    transform = np.zeros((len(args["norm"]), 3, 3), dtype=dtype)
    transform[:, 0, 0] = 1
    transform[:, 1:, 1:] = args["inverse_second_moment"]
    return np.einsum("cbd,cbm->cmd", coefficients, transform)


def main():
    assert jax.config.jax_enable_x64
    geometry = dict(np.load(ROOT / "h200/geometry.npz"))
    polygons = tuple(
        p[:n]
        for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
    )
    coupling = BiotMomentCoupling.from_polygons(polygons)
    packed = reference().coupling(polygons)
    data = dict(np.load(ROOT / "diagnosis/accuracy-220.npz"))
    point = np.array(data["points"][0], dtype=np.longdouble)
    derivatives = []
    for axis in range(2):
        estimates = []
        for step in (0.001, 0.0005, 0.00025):
            shift = np.zeros(2, dtype=np.longdouble)
            shift[axis] = step
            estimates.append(
                (fields(point + shift, packed) - fields(point - shift, packed))
                / (2 * step)
            )
        fourth = (4 * np.array(estimates[1:]) - np.array(estimates[:-1])) / 3
        derivatives.append((16 * fourth[1] - fourth[0]) / 15)
    hessian = np.stack(derivatives, axis=-1)
    gradient = fields(point, packed)
    np.savez(ROOT / "term-reference.npz", gradient=gradient, hessian=hessian)
    print("TERM_REFERENCE_COMPLETE", flush=True)
    jets = np.asarray(_source_jets(jnp.array(point, dtype=jnp.float64), coupling))
    truth = np.concatenate(
        (
            gradient,
            hessian[:, :, 0, 0, None],
            hessian[:, :, 0, 1, None],
            hessian[:, :, 1, 1, None],
        ),
        axis=-1,
    )
    delta = jets[:, :, 1:] - truth
    weighted = delta * data["moments"][:, :, None]
    ranking = np.argsort(
        np.linalg.norm(weighted[:, :, 2:].reshape(len(polygons), -1), axis=1)
    )[::-1]
    output = []
    for cell in ranking[:12]:
        polygon = polygons[cell]
        direction = np.roll(polygon, -1, axis=0) - polygon
        offset = np.asarray(point, dtype=float) - polygon
        signed_distance = (
            direction[:, 0] * offset[:, 1] - direction[:, 1] * offset[:, 0]
        ) / np.linalg.norm(direction, axis=1)
        item = dict(
            cell=int(cell),
            polygon=polygon.tolist(),
            distance_to_edge_line=signed_distance.tolist(),
            height_difference=(polygon[:, 1] - point[1]).tolist(),
            moments=data["moments"][cell].tolist(),
            route=jets[cell, :, 1:].tolist(),
            reference=truth[cell].tolist(),
            weighted_error=weighted[cell].tolist(),
        )
        output.append(item)
    np.savez(
        ROOT / "term-route.npz", jets=jets, reference=truth, weighted_error=weighted
    )
    (ROOT / "terms.json").write_text(json.dumps(output, indent=2, default=float))
    print("TERM_ERROR_SUM", weighted.sum(axis=(0, 1)).tolist(), flush=True)
    print("TERMS_COMPLETE", [int(i) for i in ranking[:12]], flush=True)


if __name__ == "__main__":
    main()
