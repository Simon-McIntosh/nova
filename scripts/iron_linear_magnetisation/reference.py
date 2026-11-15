"""Resolve the finite-loop reference independently in spherical harmonics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import binom, eval_legendre  # noqa: E402
from scripts.iron_linear_magnetisation.controls import (  # noqa: E402
    SHELL_MU,
    SHELL_INNER_RADIUS,
    SHELL_OUTER_RADIUS,
    LOOP_RADIUS,
    SHELL_PROBES,
    exact_shielding,
    loop,
    require_clean_script,
)


def exterior_reference(terms):
    """Transmit each odd scalar-potential multipole across both spherical faces.

    The loop's axial field fixes the outgoing coefficients through the series
    of (1 + a²/r²)^(-3/2). Continuity of normal B and tangential H gives the
    degree-dependent shell transmission. The dipole term reproduces the
    existing shielding reference exactly; higher terms retain the finite loop.
    """
    radius = np.linalg.norm(SHELL_PROBES, axis=1)
    cosine, sine = SHELL_PROBES[:, 1] / radius, SHELL_PROBES[:, 0] / radius
    vacuum = np.zeros_like(SHELL_PROBES)
    shielded = np.zeros_like(SHELL_PROBES)
    ratio = SHELL_INNER_RADIUS / SHELL_OUTER_RADIUS
    for index in range(terms):
        degree = 2 * index + 1
        polynomial = eval_legendre(degree, cosine)
        denominator = np.where(sine != 0, cosine**2 - 1, 1.0)
        derivative = (
            degree
            * (cosine * polynomial - eval_legendre(degree - 1, cosine))
            / denominator
        )
        coefficient = (
            (4e-7 * np.pi / 2)
            * binom(-1.5, index)
            * LOOP_RADIUS ** (2 * index + 2)
            / (2 * index + 2)
        )
        radial = (degree + 1) * coefficient * polynomial / radius ** (degree + 2)
        polar = coefficient * sine * derivative / radius ** (degree + 2)
        field = np.column_stack(
            (radial * sine + polar * cosine, radial * cosine - polar * sine)
        )
        mu = SHELL_MU
        shielding = (
            (degree * mu + degree + 1) * ((degree + 1) * mu + degree)
            - degree * (degree + 1) * (mu - 1) ** 2 * ratio ** (2 * degree + 1)
        ) / (mu * (2 * degree + 1) ** 2)
        if degree == 1:
            np.testing.assert_allclose(shielding, exact_shielding(), rtol=2e-15)
        inner, inverse = ratio**degree, ratio ** (-degree - 1)
        # Solve continuity of scalar potential and normal permeability times
        # its derivative at both faces, independently of the reduced formula.
        boundary = np.array(
            [
                [inner, -inner, -inverse, 0],
                [
                    degree * inner / ratio,
                    -mu * degree * inner / ratio,
                    mu * (degree + 1) * inverse / ratio,
                    0,
                ],
                [0, 1, 1, -1],
                [0, mu * degree, -mu * (degree + 1), degree + 1],
            ]
        )
        transmitted = np.linalg.solve(
            boundary, [-inverse, (degree + 1) * inverse / ratio, 0, 0]
        )[-1]
        np.testing.assert_allclose(transmitted, 1 / shielding, rtol=2e-12)
        vacuum += field
        shielded += field / shielding
    return vacuum, shielded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    blob = require_clean_script(__file__, args.blob)
    vacuum, shielded = exterior_reference(10)
    owner = loop(SHELL_PROBES)
    np.testing.assert_allclose(vacuum, owner, rtol=2e-13, atol=1e-25)
    np.testing.assert_allclose(
        shielded, exterior_reference(6)[1], rtol=2e-15, atol=1e-25
    )
    reference = np.linalg.norm(owner, axis=1) / np.linalg.norm(shielded, axis=1)
    rows, previous = {}, None
    records = [
        (path, json.loads(path.read_text()))
        for path in args.rows.rglob("linear-*.json")
    ]
    for path, row in sorted(
        records, key=lambda pair: pair[1]["cell_size_m"], reverse=True
    ):
        if row.get("completed"):
            errors = np.abs(np.asarray(row["shielding"]) / reference - 1)
            rows[path.name] = dict(
                revision=row["revision"],
                cell_size_m=row["cell_size_m"],
                unknowns=row["unknowns"],
                relative_errors=errors.tolist(),
                worst_error=float(errors.max()),
            )
            if previous is not None:
                rows[path.name]["per_probe_order"] = (
                    np.log(previous[1] / errors)
                    / np.log(previous[0] / row["cell_size_m"])
                ).tolist()
            previous = row["cell_size_m"], errors
    receipt = dict(
        revision=args.revision,
        script_blob=blob,
        x64=True,
        completed=True,
        probe_coordinates_m=SHELL_PROBES.tolist(),
        finite_loop_shielding=reference.tolist(),
        dipole_reference=exact_shielding(),
        dipole_relative_offsets=np.abs(reference / exact_shielding() - 1).tolist(),
        vacuum_control_max_relative_error=float(
            np.max(np.abs(vacuum - owner) / np.maximum(np.abs(owner), 1e-30))
        ),
        linear_rows=rows,
    )
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
