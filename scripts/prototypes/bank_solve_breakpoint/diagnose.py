"""Observe fresh trial norms and discrete selections in the production search."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.prototypes.bank_solve_cycle import measure as cycle
from tests.test_bank_solve_cycle import test_diverted_lanes_leave_near_root_cycle

sr = cycle.support_read
original = sr._damped_scale


def diagnostic(evaluate, state, step, current, auxiliary, **kwargs):
    def report(scale, incumbent, norm, valid, cut, normal, fragments, delta):
        print(
            "TRIAL",
            json.dumps(
                dict(
                    scale=float(scale),
                    incumbent=np.asarray(incumbent).tolist(),
                    residual=np.asarray(norm).tolist(),
                    valid=np.asarray(valid).tolist(),
                    cut_match=np.asarray(cut).tolist(),
                    normal_match=np.asarray(normal).tolist(),
                    fragment_match=np.asarray(fragments).tolist(),
                    membership_delta=np.asarray(delta).tolist(),
                )
            ),
            flush=True,
        )

    reading = auxiliary.reading
    cut = (reading.membership > 0) & (reading.membership < 1 - 1e-14)
    for scale in (1.0, 0.5, 0.25, 0.125):
        trial = jax.tree.map(lambda x, d: x + scale * d, state, step)
        norm, valid, support = evaluate(trial)
        fresh = support.reading
        next_cut = (fresh.membership > 0) & (fresh.membership < 1 - 1e-14)
        jax.debug.callback(
            report,
            scale,
            current,
            norm,
            valid,
            jnp.all(cut == next_cut, axis=-1),
            jnp.all(reading.normal_form_cells == fresh.normal_form_cells, axis=-1),
            jnp.all(
                (reading.fragment_selected == fresh.fragment_selected).reshape(
                    current.size, -1
                ),
                axis=-1,
            ),
            jnp.max(jnp.abs(reading.membership - fresh.membership), axis=-1),
            ordered=True,
        )
    return original(evaluate, state, step, current, auxiliary, **kwargs)


if __name__ == "__main__":
    print(
        "HEADER",
        cycle.reference.nova.__file__,
        sr.__file__,
        Path.cwd().resolve(),
        flush=True,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sr, "_damped_scale", diagnostic)
        test_diverted_lanes_leave_near_root_cycle(patch)
