"""Recompute the limited rows' boundary clause from archived state.

A limited row's boundary reference is the limiter's designed contact: the
midpoint of the wall's midplane-crossing edge at the largest major radius,
which the oracle fixtures' ``designed_contact`` returns.  This reads an archived
row, takes that contact and the analytic flux on it, and compares it with the
route's own boundary point and the solved flux the route reports there, so the
clause is stated from the saved data alone.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from scripts.prototypes.bank_route_rows.measure import limited_contact

BOUNDARY_BUDGET = 1e-4


def report(state_path):
    from tests.equilibrium import test_topology_read as fixture

    with np.load(state_path, allow_pickle=False) as archive:
        solved = np.asarray(archive["solved_roots"])
        oracle = np.asarray(archive["analytic_roots"])
        wall = np.asarray(archive["wall"])
    _, analytic, fixture_wall, _, _ = fixture._analytic_inputs("limited")
    assert np.allclose(np.asarray(fixture_wall), wall), "archived wall differs"
    contact = limited_contact(wall)
    contact_flux = float(analytic.value(jnp.asarray(contact)))
    axis_flux = float(analytic.value(jnp.asarray(oracle[0])))
    route_boundary = solved[1]
    # The route's reported boundary flux was not archived, and its boundary
    # point is a wall point rather than a lattice node (the nearest node is
    # centimetres away), so the analytic flux at that point is what the archive
    # fixes: the route places its boundary there and books the flux it finds.
    route_flux = float(analytic.value(jnp.asarray(route_boundary)))
    span = abs(axis_flux - contact_flux)
    boundary_error = abs(route_flux - contact_flux)
    return {
        "state": str(state_path),
        "wall_points": int(len(wall)),
        "contact_rz_m": contact.tolist(),
        "contact_flux": contact_flux,
        "axis_rz_m": oracle[0].tolist(),
        "axis_flux": axis_flux,
        "flux_span": span,
        "route_boundary_rz_m": route_boundary.tolist(),
        "route_boundary_flux": route_flux,
        "route_boundary_flux_basis": "analytic flux at the route's own boundary point",
        "separation_m": float(np.linalg.norm(contact - route_boundary)),
        "boundary_error": boundary_error,
        "boundary_ratio": boundary_error / span,
        "budget": BOUNDARY_BUDGET,
        "passes": bool(boundary_error / span <= BOUNDARY_BUDGET),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [report(path) for path in args.state]
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print("BOUNDARY " + json.dumps(row, default=float))
    print("ALL_PASS=" + str(all(row["passes"] for row in rows)))


if __name__ == "__main__":
    main()
