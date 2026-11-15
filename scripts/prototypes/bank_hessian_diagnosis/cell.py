"""Resolve a uniform rectangle's Hessian by smooth ring-boundary quadrature.

Translation invariance gives K_Z = -K_b. Integrating that derivative over the
source height gives psi_Z = J integral(K_bottom - K_top) da. Differentiating
this smooth edge integral supplies independent mixed and vertical Hessians,
including for targets inside the material. The radial diagonal then follows
from the differential field equation and the closed-form rectangle gradient.
"""

# ruff: noqa: E402
import json
import argparse

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

from nova.biot import gradedresidual
from nova.biot.greens import MU0, cylinder_greens
from nova.equilibrium.topology import BiotMomentCoupling
from scripts.prototypes.read_route.jets import _source_jets
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT, reference


def boundary(point, digits):
    """Use an elliptic ring integral with no polygon antiderivative or tangent."""
    with mp.workdps(digits):
        r, z = map(lambda value: mp.mpf(float(value)), point)
        mu = mp.mpf(float(MU0))
        density = mp.mpf(200000)

        def ring(radius, height, source_r, source_z):
            parameter = (
                4
                * radius
                * source_r
                / ((radius + source_r) ** 2 + (height - source_z) ** 2)
            )
            return (
                mu
                * mp.sqrt(radius * source_r / parameter)
                * ((2 - parameter) * mp.ellipk(parameter) - 2 * mp.ellipe(parameter))
            )

        def vertical(radius, height):
            return density * mp.quad(
                lambda a: (
                    ring(radius, height, a, mp.mpf("-0.6"))
                    - ring(radius, height, a, mp.mpf("0.6"))
                ),
                [mp.mpf("5.5"), mp.mpf("6.5")],
            )

        dz = vertical(r, z)
        rz = mp.diff(lambda radius: vertical(radius, z), r)
        zz = mp.diff(lambda height: vertical(r, height), z)
        # The separate rectangle corner primitive supplies this first derivative.
        _, _, bz = cylinder_greens(
            np.array([float(r)]), np.array([float(z)]), 6.0, 0.0, 1.0, 1.2
        )
        dr = mp.mpf(float(2 * np.pi * float(r) * bz[0] * 240000))
        inside = mp.mpf("5.5") < r < mp.mpf("6.5") and abs(z) < mp.mpf("0.6")
        forcing = -2 * mp.pi * mu * r * density if inside else mp.mpf(0)
        rr = forcing + dr / r - zz
        values = [dr, dz, rr, rz, zz]
        decimal = [mp.nstr(value, digits - 5) for value in values]
        name = f"cell-independent-{float(r):.12g}-{float(z):.12g}-{digits}.json"
        (ROOT / name).write_text(json.dumps(decimal))
        return np.array(decimal, dtype=np.longdouble)


def source_gradient(point, coupling):
    from nova.biot.polygonanalytic import _packed_analytic_moments_primal

    rows = _packed_analytic_moments_primal(
        jnp,
        jnp.broadcast_to(point[0], coupling.norm.shape),
        jnp.broadcast_to(point[1], coupling.norm.shape),
        coupling.edge,
        coupling.weight,
        coupling.norm,
        coupling.centre.T,
        coupling.centre.T,
        coupling.reflection_axis,
        coupling.reflection_partner,
    )
    return 2 * jnp.pi * point[0] * jnp.stack((rows[6], -rows[3]), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--independent", action="store_true")
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    polygon = np.array([[5.5, -0.6], [6.5, -0.6], [6.5, 0.6], [5.5, 0.6]])
    coupling = BiotMomentCoupling.from_polygons((polygon,))
    ref = reference()
    packed = ref.coupling((polygon,))
    moments = np.array([[240000.0, 0.0, 0.0]])
    point_jet = _source_jets
    ordinary = jax.jacfwd(source_gradient)
    prior = (
        json.loads((ROOT / "cell.json").read_text())
        if (ROOT / "cell.json").exists()
        else []
    )
    points = (
        [np.array([6.5 + 1e-12, 1.3]), np.array([6.5001, 1.3])]
        if args.aligned
        else [np.array([6.15, 0.17]), np.array([7.2, 0.33])]
    )
    if args.independent:
        points = [np.array(row["point"]) for row in prior]
    output = prior if args.aligned or args.independent else []
    for point in points:
        truth = boundary(point, 35)
        refined = boundary(point, 50)
        (ROOT / "cell-independent.json").write_text(
            json.dumps(
                dict(
                    point=point.tolist(),
                    independent=refined.tolist(),
                    precision_change=np.abs(truth - refined).tolist(),
                ),
                default=float,
            )
        )
        print("CELL_INDEPENDENT", point.tolist(), refined.tolist(), flush=True)
        precision_bound = np.maximum(
            abs(truth - refined), 4 * np.finfo(np.longdouble).eps * abs(refined)
        )
        precision_control = float(
            np.linalg.norm(
                (refined * np.longdouble("1.000000000001") - refined) / refined
            )
        )
        assert precision_control > 1e-13
        if args.independent:
            item = next(row for row in output if row["point"] == point.tolist())
            item.update(
                independent=refined.tolist(),
                precision_bound=precision_bound.tolist(),
                precision_control_relative=precision_control,
            )
            (ROOT / "cell.json").write_text(json.dumps(output, indent=2, default=float))
            continue
        bank = np.asarray(point_jet(jnp.array(point), coupling))[0, 0] * moments[0, 0]
        print("CELL_BANK", bank.tolist(), flush=True)
        runtime_moments = tuple(jnp.array(moments).T)

        def direct_gradient(p):
            return BiotMomentCoupling.value_gradient.__wrapped__(
                coupling, p, runtime_moments
            )[1]

        direct_gradient_value = np.asarray(direct_gradient(jnp.array(point)))
        direct_hessian = np.asarray(jax.jacfwd(direct_gradient)(jnp.array(point)))
        gradient = ref.gradient(point[None], packed, moments)[0]
        derivatives = []
        for axis in range(2):
            rows = []
            for step in (0.001, 0.0005, 0.00025):
                shift = np.zeros(2, dtype=np.longdouble)
                shift[axis] = step
                values = ref.gradient(
                    np.array([point + shift, point - shift]), packed, moments
                )
                rows.append((values[0] - values[1]) / (2 * step))
            fourth = (4 * np.array(rows[1:]) - np.array(rows[:-1])) / 3
            derivatives.append((16 * fourth[1] - fourth[0]) / 15)
        hessian = np.array(derivatives).T
        baseline = np.asarray(ordinary(jnp.array(point), coupling))[0] * moments[0, 0]
        saved = gradedresidual._fixed_nodes
        try:
            gradedresidual._fixed_nodes = lambda value, xp: value
            unfrozen = jax.jacfwd(lambda p: source_gradient(p, coupling))
            moving_nodes = np.asarray(unfrozen(jnp.array(point)))[0] * moments[0, 0]
        finally:
            gradedresidual._fixed_nodes = saved
        item = dict(
            point=point.tolist(),
            independent=refined.tolist(),
            precision_change=np.abs(truth - refined).tolist(),
            precision_bound=precision_bound.tolist(),
            precision_control_relative=precision_control,
            bank=bank[1:].tolist(),
            direct=np.r_[
                direct_gradient_value,
                direct_hessian[0],
                direct_hessian[1, 1],
            ].tolist(),
            reference=np.r_[gradient, hessian[0], hessian[1, 1]].tolist(),
            ordinary=baseline.tolist(),
            moving_nodes=moving_nodes.tolist(),
        )
        output.append(item)
        (ROOT / "cell.json").write_text(json.dumps(output, indent=2, default=float))
        print("CELL", json.dumps(item, default=float), flush=True)
    print("CELL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
