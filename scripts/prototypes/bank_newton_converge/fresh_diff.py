"""Locate the continuous read change hidden by a tracked Newton solution."""

# ruff: noqa: E402

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def compare(state_path):
    operands, cache, roots, convention, policy, profile, quadrature, seed_moments, _ = (
        machine("diverted")
    )
    with np.load(state_path, allow_pickle=False) as stored:
        moments = tuple(jnp.asarray(part) for part in stored["moments"])
    start = roots + 0.03 * jnp.median(operands.geometry.pitch)
    seed = support_read.read(
        operands, cache, seed_moments, start, roots, convention, policy
    )
    solve_one = jax.jit(
        lambda bank, state, read: support_read.newton_step(
            bank,
            state,
            read,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
        )
    )
    first = solve_one(operands, seed_moments, seed)
    intermediate = tuple(
        value + update for value, update in zip(seed_moments, first.update, strict=True)
    )
    intermediate_read = support_read.step_residual(
        operands, seed_moments, intermediate, seed, profile, quadrature
    )
    carried = seed._replace(reading=intermediate_read.reading)
    second = solve_one(operands, intermediate, carried)
    replayed = tuple(
        value + update
        for value, update in zip(intermediate, second.update, strict=True)
    )
    final_read = support_read.step_residual(
        operands, intermediate, replayed, carried, profile, quadrature
    )
    carried = carried._replace(reading=final_read.reading)
    emit(
        "REPLAY",
        state_gap=float(
            jnp.max(
                jnp.abs(
                    jnp.stack(
                        tuple(a - b for a, b in zip(replayed, moments, strict=True))
                    )
                )
            )
        ),
    )
    moments = replayed
    tracked = support_read.step_residual(
        operands, moments, moments, carried, profile, quadrature
    )
    fresh = support_read.read(
        operands, cache, moments, start, roots, convention, policy
    )
    refreshed = support_read.step_residual(
        operands, moments, moments, fresh, profile, quadrature
    )
    a = tracked.reading
    b = fresh.reading
    masks = {
        "membership": np.asarray(a.membership > 0),
        "clip": np.asarray((a.membership > 0) & (a.membership < 1)),
        "owner": np.asarray(a.fragment_selected),
    }
    other = {
        "membership": np.asarray(b.membership > 0),
        "clip": np.asarray((b.membership > 0) & (b.membership < 1)),
        "owner": np.asarray(b.fragment_selected),
    }
    emit(
        "READ_DIFF",
        axis_shift=float(jnp.linalg.norm(a.axis - b.axis)),
        boundary_shift=float(jnp.linalg.norm(a.boundary - b.boundary)),
        axis_flux_shift=float(a.axis_flux - b.axis_flux),
        boundary_flux_shift=float(a.boundary_flux - b.boundary_flux),
        membership_count=int(
            np.count_nonzero(masks["membership"] != other["membership"])
        ),
        clip_count=int(np.count_nonzero(masks["clip"] != other["clip"])),
        owner_count=int(np.count_nonzero(masks["owner"] != other["owner"])),
        coefficient_sup=float(
            jnp.max(jnp.abs(a.field_coefficients - b.field_coefficients))
        ),
        membership_sup=float(jnp.max(jnp.abs(a.membership - b.membership))),
    )
    delta = jnp.stack(
        tuple(
            x - y for x, y in zip(tracked.booked[:3], refreshed.booked[:3], strict=True)
        ),
        axis=-1,
    )
    emit(
        "BOOK_DIFF",
        tracked_residual=float(jnp.max(jnp.abs(jnp.stack(tracked.residual, axis=-1)))),
        fresh_residual=float(jnp.max(jnp.abs(jnp.stack(refreshed.residual, axis=-1)))),
        current_sup=float(jnp.max(jnp.abs(delta[:, 0]))),
        radial_sup=float(jnp.max(jnp.abs(delta[:, 1]))),
        vertical_sup=float(jnp.max(jnp.abs(delta[:, 2]))),
        worst_current=int(jnp.argmax(jnp.abs(delta[:, 0]))),
        worst_radial=int(jnp.argmax(jnp.abs(delta[:, 1]))),
        worst_vertical=int(jnp.argmax(jnp.abs(delta[:, 2]))),
    )
    for field in (
        "field_coefficients",
        "axis_flux",
        "boundary_flux",
        "membership",
        "fragment_selected",
        "normal_form_cells",
        "saddle_form",
        "edge_interval",
    ):
        candidate = a._replace(**{field: getattr(b, field)})
        from nova.equilibrium import current

        psi = support_read.nodal_jets(operands, moments)[operands.sample_index, 0]
        booked = current.book(psi, candidate, operands.geometry, profile, quadrature)
        result = jnp.stack(
            tuple(m - book for m, book in zip(moments, booked[:3], strict=True)),
            axis=-1,
        )
        emit(
            "SUBSTITUTE",
            field=field,
            residual_sup=float(jnp.max(jnp.abs(result))),
            valid=bool(booked.valid),
        )
    candidate = a._replace(saddle_form=b.saddle_form, edge_interval=b.edge_interval)
    booked = current.book(psi, candidate, operands.geometry, profile, quadrature)
    result = jnp.stack(
        tuple(m - book for m, book in zip(moments, booked[:3], strict=True)), axis=-1
    )
    emit(
        "SUBSTITUTE",
        field="saddle_form_and_edge_interval",
        residual_sup=float(jnp.max(jnp.abs(result))),
        valid=bool(booked.valid),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
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
    )
    compare(args.state)
