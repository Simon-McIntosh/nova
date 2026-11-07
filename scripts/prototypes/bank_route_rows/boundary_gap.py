"""Resolve the limited rows' boundary-flux gap from the archived state.

The oracle's limiter contact is the wall node carrying the largest analytic
flux; the route's boundary is its own terminal surface at the wall.  This
reports both points, their separation, and the analytic flux at each, so the
gap is stated from the saved data rather than explained.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def report(state_path):
    from tests.equilibrium import test_topology_read as fixture

    with np.load(state_path, allow_pickle=False) as archive:
        wall = np.asarray(archive["wall"])
        solved = np.asarray(archive["solved_roots"])
        oracle = np.asarray(archive["analytic_roots"])
    _, analytic, fixture_wall, _, _ = fixture._analytic_inputs("limited")
    assert np.allclose(np.asarray(fixture_wall), wall), "archived wall differs"
    flux = np.asarray(jax.vmap(analytic.value)(jnp.asarray(wall)))
    index = int(np.argmax(flux))
    contact = wall[index]
    route_boundary = solved[1]
    separation = float(np.linalg.norm(contact - route_boundary))
    route_flux = float(analytic.value(jnp.asarray(route_boundary)))
    # How fast the analytic flux falls as the contact moves along the wall.
    distances = np.linalg.norm(wall - contact, axis=1)
    order = np.argsort(distances)
    profile = [
        {
            "index": int(order[step]),
            "distance_from_contact": float(distances[order[step]]),
            "flux_drop": float(flux[index] - flux[order[step]]),
        }
        for step in range(1, 7)
    ]
    nearest = order[1]
    return {
        "state": str(state_path),
        "wall_points": int(len(wall)),
        "oracle_contact_index": index,
        "oracle_contact_rz_m": contact.tolist(),
        "oracle_contact_flux": float(flux[index]),
        "route_boundary_rz_m": route_boundary.tolist(),
        "route_boundary_flux": route_flux,
        "separation_m": separation,
        "flux_gap_wb": float(abs(flux[index] - route_flux)),
        "oracle_roots_match": bool(np.allclose(oracle[1], contact)),
        "neighbour": {
            "index": int(nearest),
            "distance_m": float(distances[nearest]),
            "flux_drop_wb": float(flux[index] - flux[nearest]),
        },
        "profile": profile,
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


if __name__ == "__main__":
    main()
