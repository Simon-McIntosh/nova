"""Evaluate one bank node's jet under three derivative arms.

The bank's six booked rows per moment are the flux value, its gradient, and the
symmetric Hessian, contracted against the same inverse second moment the row
builder uses. The jet pipeline is the repository's own
``nova.equilibrium.fixed_points._source_jets``; an arm is that program with its
row function swapped, so a difference between arms is a difference between the
row programs and nothing else.

Arms:

``shipped``
    ``packed_analytic_moments`` as the bank builder calls it, which routes
    through the stable-height custom JVP.
``stable_poles``
    ``_packed_analytic_moments_primal`` with ``stable_poles=True``: the stable
    pole evaluation without the height substitution.
``plain``
    ``_packed_analytic_moments_primal``, which is the pre-repair derivative
    path.
"""

import argparse
import json
from contextlib import contextmanager

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax.numpy as jnp  # noqa: E402

from nova.biot.polygonanalytic import (  # noqa: E402
    _packed_analytic_moments_primal,
    packed_analytic_moments,
)
from nova.equilibrium import fixed_points  # noqa: E402
from nova.equilibrium.fixed_points import _source_jets  # noqa: E402
from nova.equilibrium.topology import BiotMomentCoupling  # noqa: E402

from scripts.prototypes.bank_hessian_blowup import geometry  # noqa: E402

CERTIFIED = (
    "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261117T110237355670-iter-bank-recertify/h200-head/store/sha256/"
    "81ee8fdf8056855f0846e00b7213434451be8e9557369b74ef35a23a62fd4f/bank.npz"
)
PRE_REPAIR = (
    "/home/ITER/mcintos/.cache/nova-dina-flattop/inputs/bank/store/sha256/"
    "fd912554f8e9e7efddce53b119242a46e88f20ac30e029f83781c6253f828425/bank.npz"
)
ARMS = ("shipped", "stable_poles", "plain")


def _stable_poles_primal(xp, *arguments):
    return _packed_analytic_moments_primal(xp, *arguments, stable_poles=True)


ROW_FUNCTIONS = {
    "shipped": packed_analytic_moments,
    "stable_poles": _stable_poles_primal,
    "plain": _packed_analytic_moments_primal,
}


@contextmanager
def row_program(arm):
    """Evaluate the shared jet program with ``arm``'s row function."""
    original = fixed_points.packed_analytic_moments
    fixed_points.packed_analytic_moments = ROW_FUNCTIONS[arm]
    try:
        yield
    finally:
        fixed_points.packed_analytic_moments = original


def coupling_from_geometry():
    return BiotMomentCoupling.from_polygons(geometry.cells())


def jets(coupling, arm, target):
    """The booked rows of every cell at ``target``, under one arm."""
    with row_program(arm):
        return _source_jets(jnp.asarray(target), coupling)


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
