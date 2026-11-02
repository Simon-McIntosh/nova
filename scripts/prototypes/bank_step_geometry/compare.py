"""Compare every bank input channel's moving geometry with the retained tangent."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import current, support_read
from scripts.prototypes.bank_newton_step.measure import machine
from scripts.prototypes.bank_step_geometry import reference


def emit(name, **values):
    print(name, json.dumps(values, sort_keys=True, default=float), flush=True)


def main():
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        device=str(jax.devices()[0]),
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, trial, _ = machine(
        "diverted"
    )
    with np.load(os.environ["BANK_CONVERGED_STATE"]) as saved:
        converged = tuple(jnp.asarray(part) for part in saved["moments"])
    cells = trial[0].size
    channel_limit = int(os.environ.get("BANK_CHANNEL_LIMIT", str(3 * cells)))
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    for state_name, moments in (("trial", trial), ("converged", converged)):
        support = jax.jit(support_read.read)(
            bank, cache, moments, seeds, roots, convention, policy
        )
        assert bool(support.valid)
        psi = support_read.nodal_jets(bank, moments)[bank.sample_index, 0]
        booked = jax.jit(
            lambda values, reading, geometry: current.book(
                values, reading, geometry, profile, quadrature
            )
        )(psi, support.reading, bank.geometry)
        assert bool(booked.valid)

        def channel(index):
            direction = tuple(
                (jnp.arange(cells) == index % cells).astype(moments[0].dtype)
                * (index // cells == kind)
                for kind in range(3)
            )
            jets = support_read._directional_field(bank, direction)[0]
            closed = support_read._normal_geometry_booking_tangent(
                bank, moments, jets, support, profile, quadrature, booked
            )
            retained = reference._normal_geometry_booking_tangent(
                bank, moments, jets, support, profile, quadrature, booked
            )
            return jnp.stack(closed, -1), jnp.stack(retained, -1)

        compiled = jax.jit(jax.vmap(channel))
        worst = (0.0, -1)
        for start in range(0, channel_limit, 8):
            indices = jnp.arange(start, min(start + 8, channel_limit))
            actual, expected = compiled(indices)
            actual, expected = np.asarray(actual), np.asarray(expected)
            for offset, (observed, oracle) in enumerate(
                zip(actual, expected, strict=True)
            ):
                scale = max(np.linalg.norm(oracle), 1e-20)
                error = np.linalg.norm(observed - oracle) / scale
                if error > worst[0]:
                    worst = (error, start + offset)
            if start % 80 == 0:
                emit(
                    "PROGRESS",
                    state=state_name,
                    channels=min(start + len(indices), channel_limit),
                    worst_relative=worst[0],
                    worst_channel=worst[1],
                )
        emit(
            "PARITY",
            state=state_name,
            channels=channel_limit,
            relative=worst[0],
            worst_channel=worst[1],
            worst_moment=worst[1] // cells,
            worst_cell=worst[1] % cells,
        )
        assert worst[0] <= 1e-10


if __name__ == "__main__":
    main()
