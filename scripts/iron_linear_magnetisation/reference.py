"""Resolve the finite-loop reference independently in spherical harmonics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import binom, eval_legendre  # noqa: E402
from scripts.iron_linear_magnetisation.controls import (  # noqa: E402
    SHELL_MU,
    SHELL_PROBES,
    exact_shielding,
    loop,
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
            * 0.05 ** (2 * index + 2)
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
            - degree * (degree + 1) * (mu - 1) ** 2 * 0.9 ** (2 * degree + 1)
        ) / (mu * (2 * degree + 1) ** 2)
        if degree == 1:
            np.testing.assert_allclose(shielding, exact_shielding(), rtol=2e-15)
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
    data = Path(__file__).read_bytes()
    blob = hashlib.sha1(
        b"blob %d\0" % len(data) + data, usedforsecurity=False
    ).hexdigest()
    if blob != args.blob:
        raise SystemExit("executed reference script differs from its committed blob")
    vacuum, shielded = exterior_reference(10)
    owner = loop(SHELL_PROBES)
    np.testing.assert_allclose(vacuum, owner, rtol=2e-13, atol=1e-25)
    np.testing.assert_allclose(
        shielded, exterior_reference(6)[1], rtol=2e-15, atol=1e-25
    )
    reference = np.linalg.norm(owner, axis=1) / np.linalg.norm(shielded, axis=1)
    rows = {}
    for path in sorted(args.rows.glob("linear-*.json")):
        row = json.loads(path.read_text())
        if row.get("completed"):
            rows[path.name] = dict(
                revision=row["revision"],
                unknowns=row["unknowns"],
                relative_errors=np.abs(
                    np.asarray(row["shielding"]) / reference - 1
                ).tolist(),
            )
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
