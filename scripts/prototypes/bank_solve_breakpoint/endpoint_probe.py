"""Compare conic endpoint evaluation on the same fresh Newton states."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import inspect
from pathlib import Path
import sys
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from scripts.prototypes.bank_solve_cycle import measure as cycle

sr = cycle.support_read


def main():
    print(
        "HEADER",
        cycle.reference.nova.__file__,
        sr.__file__,
        Path.cwd().resolve(),
        flush=True,
    )
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = (
        cycle.reference.machine("diverted")
    )
    scales = jnp.asarray([2e-5, 2e-5 + 3 * 4e-5 / 15])
    states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    seeds = jnp.broadcast_to(
        roots + 0.03 * jnp.median(bank.geometry.pitch), (2, *roots.shape)
    )
    previous = jnp.broadcast_to(roots, seeds.shape)
    read = jax.jit(
        lambda state: sr.read_batch(
            bank, cache, state, seeds, previous, convention, policy
        )
    )

    def residual(state, support):
        return jax.vmap(
            lambda m, s: sr.step_residual(bank, m, m, s, profile, quadrature)
        )(state, support)

    raw = jax.jit(lambda state, support: residual(state, support))
    original = sr.current._conic_rings

    def interior(*args, **kwargs):
        if "inward_endpoints" in inspect.signature(original).parameters:
            kwargs["inward_endpoints"] = True
        return original(*args, **kwargs)

    with patch.object(sr.current, "_conic_rings", interior):
        inward = (
            jax.jit(lambda state, support: residual(state, support))
            .lower(states, read(states))
            .compile()
        )
    step = jax.jit(
        lambda state, support: jax.vmap(
            lambda m, s: sr.newton_step(
                bank,
                m,
                s,
                profile,
                lambda r, p: jnp.zeros_like(r),
                quadrature,
                fresh_geometry=True,
                condition_diagnostics=False,
            )
        )(state, support)
    )
    receipt = []
    for index in range(6):
        support = read(states)
        a = raw(states, support)
        b = inward(states, support)
        av = np.concatenate(jax.device_get(a.residual), axis=-1)
        bv = np.concatenate(jax.device_get(b.residual), axis=-1)
        error = np.abs(av - bv)
        cells = states[0].shape[-1]
        record = dict(
            update=index,
            raw=np.max(np.abs(av), axis=-1).tolist(),
            inward=np.max(np.abs(bv), axis=-1).tolist(),
            difference=np.max(error, axis=-1).tolist(),
            location=(np.argmax(error, axis=-1) % cells).tolist(),
        )
        for lane, cell in enumerate(record["location"]):
            record[f"normal_{lane}"] = bool(
                support.reading.normal_form_cells[lane, cell]
            )
            record[f"membership_{lane}"] = float(support.reading.membership[lane, cell])
        print("ENDPOINT", json.dumps(record), flush=True)
        receipt.append(record)
        Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2))
        np.savez(
            Path(sys.argv[1]).with_suffix(".npz"),
            **{f"moment_{i}": np.asarray(v) for i, v in enumerate(states)},
        )
        correction = step(states, support)
        states = jax.tree.map(lambda x, d: x + d, states, correction.update)


if __name__ == "__main__":
    main()
