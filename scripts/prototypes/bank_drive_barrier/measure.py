"""Probe the fresh residual at a fixed continuation barrier state."""

# ruff: noqa: E402
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import drive_continuation, support_read
from scripts.prototypes.bank_drive_continuation import measure as predecessor
from scripts.prototypes.bank_warm_start import streams
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture

PERTURBATIONS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=16)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    root = Path(nova.__file__).resolve().parent.parent
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    emit(
        "HEADER",
        revision=revision,
        cwd=str(Path.cwd().resolve()),
        module=support_read.__file__,
        driver=drive_continuation.__file__,
        predecessor=predecessor.__file__,
        prototype=__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        lanes=args.lanes,
    )
    predecessor._MACHINE = machine = streams.build()
    with np.load(args.streams) as archive:
        moments = tuple(
            jnp.asarray(archive[name][: args.lanes]) for name in ("m0", "m1", "m2")
        )
        seeds = jnp.asarray(archive["stationary"][: args.lanes])

    accepted_states = []

    def continuation_solve(
        _operands,
        _cache,
        state,
        seed,
        warm,
        _convention,
        _policy,
        drive,
        _gradient,
        _quadrature,
        _axis_sign=-1.0,
        *,
        max_steps=60,
        tolerance=1e-10,
        **_options,
    ):
        attempt = predecessor.solve(
            drive, state, seed, warm, max_steps=max_steps, tolerance=tolerance
        )
        if (
            bool(jnp.all(attempt.converged))
            and float(jnp.max(attempt.residual)) <= tolerance
        ):
            accepted_states.append((float(drive), attempt.moments))
        return attempt

    reached = drive_continuation.advance_drive(
        None,
        None,
        moments,
        seeds,
        seeds,
        None,
        None,
        lambda drive: drive,
        profile_fixture.gradient,
        None,
        start_drive=1.0,
        target_drive=1.01,
        solve=continuation_solve,
    )
    assert reached.result is not None
    state = reached.result.moments
    drive = reached.drive
    emit(
        "REPLAY",
        drive=drive,
        residual=float(jnp.max(reached.result.residual)),
        substeps=len(reached.substeps),
        updates=reached.updates,
        reads=reached.reads,
        reason=reached.reason,
    )

    @jax.jit
    def evaluate(amplitude):
        profile = streams.scaled_profile(machine.profile, amplitude)
        support = support_read.read_batch(
            machine.operands,
            machine.cache,
            state,
            seeds,
            seeds,
            machine.convention,
            machine.policy,
        )
        residual = jax.vmap(
            lambda one, read: support_read.step_residual(
                machine.operands, one, one, read, profile, machine.quadrature
            )
        )(state, support)
        norm = jnp.max(jnp.abs(jnp.concatenate(residual.residual, axis=-1)), axis=-1)
        return norm, support

    baseline_norm, baseline = evaluate(jnp.asarray(drive))
    baseline_read = baseline.reading
    base_members = np.asarray(baseline_read.membership) > 0
    base_cut = base_members & (np.asarray(baseline_read.membership) < 1 - 1e-14)
    base_normal = np.asarray(baseline_read.normal_form_cells)
    base_fragments = np.asarray(baseline_read.fragment_selected)
    base_class = np.asarray(baseline_read.boundary_class)
    base_qualified = np.asarray(baseline_read.qualified)
    base_xvalid = np.asarray(baseline_read.x_point_valid)
    base_boundary = np.asarray(baseline_read.boundary)
    base_boundary_flux = np.asarray(baseline_read.boundary_flux)
    for epsilon in PERTURBATIONS:
        norm, support = evaluate(jnp.asarray(drive + epsilon))
        reading = support.reading
        membership = np.asarray(reading.membership) > 0
        cut = membership & (np.asarray(reading.membership) < 1 - 1e-14)
        emit(
            "PROBE",
            epsilon=epsilon,
            drive=float(drive + epsilon),
            residual_max=float(jnp.max(norm)),
            residual_delta_max=float(jnp.max(jnp.abs(norm - baseline_norm))),
            valid=int(np.sum(np.asarray(support.valid))),
            membership_count=int(np.sum(membership)),
            membership_flip=int(np.sum(membership != base_members)),
            cut_count=int(np.sum(cut)),
            cut_flip=int(np.sum(cut != base_cut)),
            normal_flip=int(
                np.sum(np.asarray(reading.normal_form_cells) != base_normal)
            ),
            fragment_flip=int(
                np.sum(np.asarray(reading.fragment_selected) != base_fragments)
            ),
            class_flip=int(np.sum(np.asarray(reading.boundary_class) != base_class)),
            qualified_flip=int(np.sum(np.asarray(reading.qualified) != base_qualified)),
            xvalid_flip=int(np.sum(np.asarray(reading.x_point_valid) != base_xvalid)),
            boundary_displacement_max=float(
                np.max(
                    np.linalg.norm(
                        np.asarray(reading.boundary) - base_boundary, axis=-1
                    )
                )
            ),
            boundary_flux_delta_max=float(
                np.max(np.abs(np.asarray(reading.boundary_flux) - base_boundary_flux))
            ),
        )

    @jax.jit
    def singular_values(one_state, amplitude):
        profile = streams.scaled_profile(machine.profile, amplitude)
        support = support_read.read(
            machine.operands,
            machine.cache,
            one_state,
            seeds[0],
            seeds[0],
            machine.convention,
            machine.policy,
        )
        residual = support_read.step_residual(
            machine.operands,
            one_state,
            one_state,
            support,
            profile,
            machine.quadrature,
        )
        matrix = support_read.step_jacobian(
            machine.operands,
            one_state,
            support,
            profile,
            profile_fixture.gradient,
            machine.quadrature,
            booked=residual.booked,
            fresh_geometry=True,
        )
        balanced, _, _ = support_read._equilibrate(matrix, one_state, residual.booked)
        singular = jnp.linalg.svd(matrix, compute_uv=False)
        balanced_singular = jnp.linalg.svd(balanced, compute_uv=False)
        return (
            singular[0],
            singular[-1],
            balanced_singular[0],
            balanced_singular[-1],
            support.valid,
        )

    for accepted_drive, accepted_state in [
        accepted_states[0],
        *accepted_states[-3:],
    ]:
        largest, smallest, balanced_largest, balanced_smallest, valid = singular_values(
            tuple(value[0] for value in accepted_state),
            jnp.asarray(accepted_drive),
        )
        emit(
            "CONDITION",
            drive=accepted_drive,
            largest_singular=float(largest),
            smallest_singular=float(smallest),
            condition_number=float(largest / smallest),
            balanced_largest_singular=float(balanced_largest),
            balanced_smallest_singular=float(balanced_smallest),
            balanced_condition_number=float(balanced_largest / balanced_smallest),
            valid=bool(valid),
        )
    emit("DONE", drive=drive)


if __name__ == "__main__":
    main()
