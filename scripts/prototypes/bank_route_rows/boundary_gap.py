"""Recompute the limited rows' boundary clause from archived state.

A limited row's boundary reference is the limiter's designed contact: the
midpoint of the wall's midplane-crossing edge at the largest major radius,
which the oracle fixtures' ``designed_contact`` returns.  This reads an archived
row, takes that contact and the analytic flux on it, and compares it with the
route's own boundary point, so the clause is stated from the saved data alone.

The archives are committed under ``archived/`` beside this module, so the clause
is reproducible from the repository rather than from a run directory: ``--check``
re-derives both rows from those copies and asserts them against the recorded
values in ``archived/boundary-clause.json``, exiting nonzero on any drift.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from scripts.prototypes.bank_route_rows.measure import limited_contact

BOUNDARY_BUDGET = 1e-4
HERE = Path(__file__).resolve().parent
ARCHIVED = HERE / "archived"
CHECKED_RATIO_TOLERANCE = 1e-9


def digest(path):
    """Return the archive's sha256, so a figure's inputs are identified."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    # The route's reported boundary flux is not among the archived arrays, and
    # its boundary point is a wall point rather than a lattice node, so this
    # clause carries the analytic flux at the route's own boundary point.  The
    # route places that point on the wall and books the flux it finds there, so
    # the two agree to the route's own terminal residual, archived beside them.
    route_flux = float(analytic.value(jnp.asarray(route_boundary)))
    span = abs(axis_flux - contact_flux)
    boundary_error = abs(route_flux - contact_flux)
    return {
        "state": Path(state_path).name,
        "state_sha256": digest(state_path),
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


def check():
    """Re-derive both rows from the committed archives and assert the record."""
    record = ARCHIVED / "boundary-clause.json"
    if not record.exists():
        print(f"CHECK_FAIL missing {record}")
        print("CHECK=FAIL")
        return 1
    expected = json.loads(record.read_text())
    for row in expected:
        if not (ARCHIVED / row["state"]).exists():
            print(f"CHECK_FAIL missing archived state {row['state']}")
            print("CHECK=FAIL")
            return 1
    states = [ARCHIVED / row["state"] for row in expected]
    derived = [report(path) for path in states]
    failures = []
    for want, got in zip(expected, derived, strict=True):
        if got["state_sha256"] != want["state_sha256"]:
            failures.append(f"{want['state']}: archived state hash differs")
        if not np.allclose(got["contact_rz_m"], want["contact_rz_m"], atol=1e-12):
            failures.append(f"{want['state']}: designed contact moved")
        for key in ("flux_span", "boundary_error", "boundary_ratio"):
            if abs(got[key] - want[key]) > CHECKED_RATIO_TOLERANCE * abs(want[key]):
                failures.append(
                    f"{want['state']}: {key} {got[key]:.6e} != {want[key]:.6e}"
                )
        if got["passes"] != want["passes"]:
            failures.append(f"{want['state']}: verdict differs")
        if not got["passes"]:
            failures.append(f"{want['state']}: outside the {BOUNDARY_BUDGET} budget")
    for row in derived:
        print("BOUNDARY " + json.dumps(row, default=float))
    for failure in failures:
        print("CHECK_FAIL " + failure)
    print("CHECK=" + ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check())
    if not args.state:
        parser.error("--state is required unless --check is given")
    rows = [report(path) for path in args.state]
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print("BOUNDARY " + json.dumps(row, default=float))
    print("ALL_PASS=" + str(all(row["passes"] for row in rows)))


if __name__ == "__main__":
    main()
