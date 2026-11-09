"""Per-update, per-lane observation of the staged bank solve loop.

The staged driver in :mod:`nova.equilibrium.support_read` keeps its search and
its Newton diagnostics in local variables of one ``while_loop`` body, so no
returned receipt carries them. Three module-level names are wrapped instead,
each in the position the driver calls it from:

``_damped_scale``
    the dyadic search; the wrapper sees every trial state and recovers the
    per-lane trial scale from the trial's displacement against the correction,
    and logs the scale and residual the search retained.
``newton_step``
    the linear solve; its factors, condition estimate and Krylov iteration
    count are the linearisation's own receipt.
``read_batch``
    every fresh evaluation; its topology decision is the discrete state the
    step linearises about.

``jax.lax.while_loop`` is wrapped as in ``bank_solve_cycle.observe`` so the
carried per-update arrays are logged as the loop advances. Every event is
written as one JSON line, in execution order, so a reader can line up the
update, the search it ran and the read that produced its residual.
"""

import json
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import support_read

CUT = 1 - 1e-14


def _count(mask):
    """Reduce every trailing axis so the first axis stays the lane count."""
    flat = mask.reshape(mask.shape[0], -1)
    return jnp.sum(flat.astype(jnp.int32), axis=1)


def _emit(write, kind, **values):
    def report(*leaves):
        write(
            json.dumps(
                {
                    "kind": kind,
                    **values,
                    "lanes": [np.asarray(leaf).tolist() for leaf in leaves],
                }
            )
        )

    return report


def install(write):
    """Wrap the driver's three call sites; return an undo callable."""
    started = perf_counter()
    originals = {
        "damped": support_read._damped_scale,
        "newton": support_read.newton_step,
        "read": support_read.read_batch,
        "loop": jax.lax.while_loop,
    }
    sequence = {"search": 0, "step": 0, "read": 0, "update": 0}

    def observed_damped(evaluate, state, step, current, auxiliary, **kwargs):
        index = sequence["search"]
        sequence["search"] += 1

        def observed_evaluate(trial):
            norm, valid, trial_aux = evaluate(trial)
            scale = jnp.max(
                jnp.stack(
                    tuple(
                        jnp.max(
                            jnp.where(
                                jnp.abs(change) > 0,
                                jnp.abs(
                                    (after - before) / jnp.where(change == 0, 1, change)
                                ),
                                0.0,
                            ).reshape(change.shape[0], -1),
                            axis=1,
                        )
                        for after, before, change in zip(
                            trial, state, step, strict=True
                        )
                    )
                ),
                axis=0,
            )
            jax.debug.callback(
                _emit(write, "TRIAL", search=index),
                scale,
                norm,
                valid,
                ordered=True,
            )
            return norm, valid, trial_aux

        result = originals["damped"](
            observed_evaluate, state, step, current, auxiliary, **kwargs
        )
        jax.debug.callback(
            _emit(write, "SEARCH", search=index),
            current,
            result.residual,
            result.scale,
            ordered=True,
        )
        return result

    def observed_newton(*args, **kwargs):
        index = sequence["step"]
        sequence["step"] += 1
        result = originals["newton"](*args, **kwargs)
        jax.debug.callback(
            _emit(write, "STEP", step=index),
            result.refactored.astype(jnp.int32),
            result.krylov_condition,
            result.krylov_residual,
            result.krylov_iterations,
            result.linear_residual,
            result.converged.astype(jnp.int32),
            ordered=True,
        )
        return result

    def observed_read(*args, **kwargs):
        index = sequence["read"]
        sequence["read"] += 1
        support = originals["read"](*args, **kwargs)
        reading = support.reading
        cut = (reading.membership > 0) & (reading.membership < CUT)
        jax.debug.callback(
            _emit(write, "READ", read=index),
            support.valid.astype(jnp.int32),
            reading.reason,
            reading.batch_class
            if hasattr(reading, "batch_class")
            else reading.boundary_class,
            _count(cut),
            _count(reading.normal_form_cells),
            _count(reading.fragment_selected),
            jnp.argmax(reading.fragment_selected, axis=-1),
            _count(reading.x_point_valid),
            reading.qualified.astype(jnp.int32),
            support.stationary.position[:, 0, 0],
            support.stationary.position[:, 0, 1],
            support.axis_flux if hasattr(support, "axis_flux") else reading.axis_flux,
            reading.boundary_flux,
            ordered=True,
        )
        return support

    def observed_loop(condition, body, initial):
        if body.__qualname__ != "solve_batch.<locals>.advance":
            return originals["loop"](condition, body, initial)

        def observed_advance(carried):
            following = body(carried)
            index = sequence["update"]
            sequence["update"] += 1
            jax.debug.callback(
                _emit(write, "UPDATE", update=index),
                following[2],
                following[11],
                following[12],
                following[13],
                following[14],
                following[15],
                (~following[3]).astype(jnp.int32),
                following[16][:, index + 1],
                following[17][:, index],
                ordered=True,
            )
            return following

        return originals["loop"](condition, observed_advance, initial)

    support_read._damped_scale = observed_damped
    support_read.newton_step = observed_newton
    support_read.read_batch = observed_read
    jax.lax.while_loop = observed_loop

    def undo():
        support_read._damped_scale = originals["damped"]
        support_read.newton_step = originals["newton"]
        support_read.read_batch = originals["read"]
        jax.lax.while_loop = originals["loop"]

    return undo, started
