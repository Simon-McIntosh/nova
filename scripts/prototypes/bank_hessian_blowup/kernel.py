"""Evaluate one bank node's jet under three derivative arms.

The bank's six booked rows per moment are the flux value, its gradient, and the
symmetric Hessian, contracted against the same inverse second moment the row
builder uses. This script reproduces ``_source_jets`` exactly and swaps only the
row function, so a difference between arms is a difference between derivative
constructions and nothing else.

Arms:

``shipped``
    ``packed_analytic_moments`` as the bank builder calls it, which routes
    through the stable-height custom JVP.
``stable_poles``
    ``_packed_analytic_moments_primal`` with ``stable_poles=True`` under a plain
    ``jacfwd``: the stable pole evaluation without the height substitution.
``plain``
    ``_packed_analytic_moments_primal`` under plain ``jacfwd``, which is the
    pre-repair derivative path.
"""

import argparse
import json

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nova.biot.polygonanalytic import (  # noqa: E402
    _packed_analytic_moments_primal,
    packed_analytic_moments,
)
from nova.equilibrium.topology import BiotMomentCoupling  # noqa: E402

GEOMETRY = "/home/ITER/mcintos/.cache/nova-iter-bank/dina-rerun/bank/geometry.npz"
CERTIFIED = (
    "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261117T110237355670-iter-bank-recertify/h200-head/store/sha256/"
    "81ee8fdf8056855f0846f0e00b7213434451be8e9557369b74ef35a23a62fd4f/bank.npz"
)
PRE_REPAIR = (
    "/home/ITER/mcintos/.cache/nova-dina-flattop/inputs/bank/store/sha256/"
    "fd912554f8e9e7efddce53b119242a46e88f20ac30e029f83781c6253f828425/bank.npz"
)
ARMS = ("shipped", "stable_poles", "plain")


def coupling_from_geometry():
    with np.load(GEOMETRY, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"])
        counts = np.asarray(archive["vertex_count"])
    polygons = tuple(
        vertices[index, : int(count)] for index, count in enumerate(counts)
    )
    return BiotMomentCoupling.from_polygons(polygons)


def row_function(coupling, arm):
    def rows(target):
        arguments = (
            jnp.broadcast_to(target[0], coupling.norm.shape),
            jnp.broadcast_to(target[1], coupling.norm.shape),
            coupling.edge,
            coupling.weight,
            coupling.norm,
            coupling.centre.T,
            coupling.centre.T,
            coupling.reflection_axis,
            coupling.reflection_partner,
        )
        if arm == "shipped":
            return packed_analytic_moments(jnp, *arguments)
        if arm == "stable_poles":
            return _packed_analytic_moments_primal(jnp, *arguments, stable_poles=True)
        return _packed_analytic_moments_primal(jnp, *arguments)

    return rows


def jets(coupling, arm, target):
    rows = row_function(coupling, arm)

    def value_gradient(point):
        stacked = rows(point)
        value = jnp.stack(stacked[:3], axis=-1)
        gradient = (
            2
            * jnp.pi
            * point[0]
            * jnp.stack(
                (
                    jnp.stack(stacked[6:9], axis=-1),
                    -jnp.stack(stacked[3:6], axis=-1),
                ),
                axis=-1,
            )
        )
        return value, gradient

    def gradient_with_value(point):
        value, gradient = value_gradient(point)
        return gradient, (value, gradient)

    hessian, (value, gradient) = jax.jacfwd(gradient_with_value, has_aux=True)(
        jnp.asarray(target)
    )
    coefficients = jnp.concatenate(
        (
            value[..., None],
            gradient,
            hessian[..., 0, 0, None],
            hessian[..., 0, 1, None],
            hessian[..., 1, 1, None],
        ),
        axis=-1,
    )
    inverse = coupling.inverse_second_moment
    count = len(value)
    transform = jnp.stack(
        (
            jnp.stack((jnp.ones(count), jnp.zeros(count), jnp.zeros(count)), axis=-1),
            jnp.stack((jnp.zeros(count), inverse[:, 0, 0], inverse[:, 0, 1]), axis=-1),
            jnp.stack((jnp.zeros(count), inverse[:, 1, 0], inverse[:, 1, 1]), axis=-1),
        ),
        axis=1,
    )
    return jnp.einsum("cbq,cbm->cmq", coefficients, transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--node", type=int, action="append", default=[])
    parser.add_argument("--cell", type=int, action="append", default=[])
    arguments = parser.parse_args()

    coupling = coupling_from_geometry()
    with np.load(CERTIFIED, allow_pickle=False) as archive:
        certified = np.asarray(archive["rows"])
        nodes = np.asarray(archive["nodes"])
    with np.load(PRE_REPAIR, allow_pickle=False) as archive:
        pre_repair = np.asarray(archive["rows"])

    record = {}
    for node, cell in zip(arguments.node, arguments.cell, strict=True):
        point = np.asarray(nodes[node], dtype=np.float64)
        entry = dict(
            node=node,
            cell=cell,
            point=[float(value) for value in point],
            certified=[float(value) for value in certified[node, cell].reshape(-1)],
            pre_repair=[float(value) for value in pre_repair[node, cell].reshape(-1)],
        )
        for arm in ARMS:
            row = np.asarray(jets(coupling, arm, point))
            entry[arm] = [float(value) for value in row[cell].reshape(-1)]
            print(
                "ARM", node, cell, arm, "m1_psiRR", float(row[cell][1][3]), flush=True
            )
        record[f"{node}:{cell}"] = entry

    with open(arguments.output, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print("KERNEL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
