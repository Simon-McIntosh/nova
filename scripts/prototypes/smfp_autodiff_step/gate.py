"""Measure the section gate on the bank-backed Newton step.

One run covers the fresh-read solve from the diverted 550-cell certificate
seed, its terminal fresh-read residual and analytic-oracle budget, the dense
Jacobian and the Jacobian-vector product against central differences of the
full fresh-read residual at the trial and converged states, the step residual
against its own central difference at the same states, and the present
route's step count and terminal residual from that seed.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle
from scripts.prototypes.bank_route_rows.measure import PRESENT_CELLS, present_route
from scripts.prototypes.smfp_autodiff_step.fixture import Certificate, emit

LENGTHS = (1e-4, 1e-5, 1e-6, 1e-7, 3e-8, 1e-8)


def magnitude(result):
    return float(jnp.max(jnp.abs(jnp.concatenate(result.residual))))


def relative(actual, reference):
    reference = np.asarray(reference)
    return float(
        np.linalg.norm(np.asarray(actual) - reference) / np.linalg.norm(reference)
    )


def balanced_direction(moments):
    index = jnp.arange(moments[0].shape[0])
    shape = jnp.sin(1.7 * index) + jnp.cos(2.3 * index)
    shape = 0.4 * (shape - jnp.mean(shape)) / jnp.max(jnp.abs(shape))
    return (shape, 0.5 * jnp.roll(shape, 17), -0.2 * jnp.roll(shape, 31))


class Gate:
    """The certificate's read, residual, step and derivative programs."""

    def __init__(self, store):
        certificate = self.certificate = Certificate(store)
        bank = certificate.bank
        self.read = jax.jit(
            lambda moments: support_read.read(
                bank,
                certificate.cache,
                moments,
                certificate.seed,
                certificate.roots,
                certificate.convention,
                certificate.policy,
            )
        )
        self.residual = jax.jit(
            lambda base, candidate, read: support_read.step_residual(
                bank,
                base,
                candidate,
                read,
                certificate.profile,
                certificate.quadrature,
            )
        )
        self.step = jax.jit(
            lambda moments, read: support_read.newton_step(
                bank,
                moments,
                read,
                certificate.profile,
                certificate.gradient,
                certificate.quadrature,
                fresh_geometry=True,
                condition_diagnostics=False,
            )
        )
        self.dense = jax.jit(
            lambda moments, read, booked: support_read.step_jacobian(
                bank,
                moments,
                read,
                certificate.profile,
                certificate.gradient,
                certificate.quadrature,
                booked,
                fresh_geometry=True,
            )
        )
        self.applied = jax.jit(
            lambda moments, direction, read, booked: support_read.step_jacobian_vector(
                bank,
                moments,
                direction,
                read,
                certificate.profile,
                certificate.gradient,
                certificate.quadrature,
                booked,
                fresh_geometry=True,
            )
        )

    def fresh(self, moments):
        support = self.read(tuple(moments))
        return support, self.residual(moments, moments, support)

    def derivative_arms(self, label, moments):
        """Fresh-read Jacobian and JVP against central differences."""
        support, result = self.fresh(moments)
        booked = result.booked
        dense_action = np.asarray(
            self.dense(moments, support, booked)
            @ jnp.concatenate(balanced_direction(moments))
        )
        direction = balanced_direction(moments)
        jvp_action = np.asarray(
            jnp.concatenate(self.applied(moments, direction, support, booked))
        )
        state_norm = float(jnp.linalg.norm(jnp.concatenate(moments)))
        best_dense, best_jvp = (float("inf"), None), (float("inf"), None)
        for length in LENGTHS:
            high = tuple(
                value + length * change
                for value, change in zip(moments, direction, strict=True)
            )
            low = tuple(
                value - length * change
                for value, change in zip(moments, direction, strict=True)
            )
            displacement = (
                length * float(jnp.linalg.norm(jnp.concatenate(direction))) / state_norm
            )
            high_support, high_result = self.fresh(high)
            low_support, low_result = self.fresh(low)
            central = (
                jnp.concatenate(high_result.residual)
                - jnp.concatenate(low_result.residual)
            ) / (2.0 * length)
            dense_error = relative(dense_action, central)
            jvp = relative(jvp_action, central)
            emit(
                "JACOBIAN_LENGTH",
                state=label,
                length=length,
                displacement_relative=displacement,
                dense_relative=dense_error,
                jvp_relative=jvp,
                high_valid=bool(high_support.valid),
                low_valid=bool(low_support.valid),
            )
            best_dense = min(best_dense, (dense_error, length))
            best_jvp = min(best_jvp, (jvp, length))
        emit(
            "JACOBIAN_STATE",
            state=label,
            dense_best=best_dense[0],
            dense_length=best_dense[1],
            jvp_best=best_jvp[0],
            jvp_length=best_jvp[1],
            dense_within_1e_6=bool(best_dense[0] <= 1e-6),
            jvp_within_1e_6=bool(best_jvp[0] <= 1e-6),
        )
        return best_dense, best_jvp

    def held_read_arm(self, label, moments):
        """The fixed-support step residual against its central difference."""
        support, _ = self.fresh(moments)
        direction = balanced_direction(moments)
        booked = self.residual(moments, moments, support).booked
        action = np.asarray(
            jnp.concatenate(self.applied(moments, direction, support, booked))
        )
        best = (float("inf"), None)
        for length in LENGTHS:
            high = tuple(
                value + length * change
                for value, change in zip(moments, direction, strict=True)
            )
            low = tuple(
                value - length * change
                for value, change in zip(moments, direction, strict=True)
            )
            high_result = self.residual(moments, high, support)
            low_result = self.residual(moments, low, support)
            central = (
                jnp.concatenate(high_result.residual)
                - jnp.concatenate(low_result.residual)
            ) / (2.0 * length)
            error = relative(action, central)
            emit(
                "HELD_READ_LENGTH",
                state=label,
                length=length,
                jvp_relative=error,
                high_valid=bool(high_result.valid),
                low_valid=bool(low_result.valid),
            )
            best = min(best, (error, length))
        emit(
            "HELD_READ_STATE",
            state=label,
            best=best[0],
            length=best[1],
            within_1e_6=bool(best[0] <= 1e-6),
        )
        return best

    def solve(self, max_steps):
        moments = self.certificate.seed_moments
        support, current = self.fresh(moments)
        before = magnitude(current)
        emit("SEED", residual=before, valid=bool(support.valid))
        index = 0
        for index in range(max_steps):
            if before <= 1e-10 and bool(support.valid):
                break
            started = perf_counter()
            correction = self.step(moments, support)
            jax.block_until_ready(correction)
            emit(
                "LINEAR",
                index=index,
                wall_ms=1e3 * (perf_counter() - started),
                linear_residual=float(correction.linear_residual),
                converged=bool(correction.converged),
                krylov_condition=float(correction.krylov_condition),
                krylov_residual=float(correction.krylov_residual),
                iterations=int(correction.krylov_iterations),
            )
            accepted = False
            for half in range(14):
                length = 2.0**-half
                candidate = tuple(
                    value + length * change
                    for value, change in zip(moments, correction.update, strict=True)
                )
                next_support, refreshed = self.fresh(candidate)
                fresh_norm = magnitude(refreshed)
                if (
                    bool(next_support.valid)
                    and np.isfinite(fresh_norm)
                    and fresh_norm < before
                ):
                    emit(
                        "ACCEPT",
                        index=index,
                        length=length,
                        fresh_residual=fresh_norm,
                    )
                    moments, support, before = candidate, next_support, fresh_norm
                    accepted = True
                    break
            if not accepted:
                emit("NO_DESCENT", index=index, residual=before)
                break
        support, final = self.fresh(moments)
        terminal = magnitude(final)
        converged = bool(support.valid) and terminal <= 1e-10
        emit(
            "FINAL",
            steps=index,
            fresh_residual=terminal,
            converged=converged,
            fresh_valid=bool(support.valid),
            qualified=bool(support.reading.qualified),
        )
        return moments, support, terminal, converged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--state-out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--skip-derivatives", action="store_true")
    parser.add_argument("--skip-present", action="store_true")
    parser.add_argument("--present-root", type=Path, default=Path("present"))
    args = parser.parse_args()
    print("REVISION=" + os.environ.get("SMFP_REVISION", "unknown"), flush=True)
    emit(
        "PROCESS",
        nova=nova.__file__,
        module=support_read.__file__,
        cwd=str(Path.cwd().resolve()),
        backend=jax.default_backend(),
        devices=[str(device) for device in jax.local_devices()],
        x64=bool(jax.config.jax_enable_x64),
    )
    gate = Gate(args.store)
    certificate = gate.certificate
    trial = certificate.seed_moments
    converged_moments, support, terminal, converged = gate.solve(args.max_steps)
    emit("GATE_SOLVE", passed=bool(converged), fresh_residual=terminal)

    if not args.skip_derivatives:
        for label, state in (("trial", trial), ("converged", converged_moments)):
            gate.derivative_arms(label, state)
            gate.held_read_arm(label, state)

    fixture = certificate.fixture
    _, _, wall, _, _ = fixture._analytic_inputs("diverted")
    oracle, _ = analytic_oracle(
        certificate.analytic, certificate.roots, support.reading, wall
    )
    emit("ORACLE", **oracle)
    emit(
        "GATE_ORACLE",
        passed=bool(
            oracle["axis_ratio"] <= 5e-4
            and oracle["saddle_ratio"] <= 5e-4
            and oracle["boundary_ratio"] <= 1e-4
        ),
    )

    np.savez_compressed(
        args.state_out,
        moments=np.stack(tuple(np.asarray(value) for value in converged_moments)),
        converged=np.asarray(converged),
        residual_sup=np.asarray(terminal),
    )
    emit("STATE_SAVED", path=str(args.state_out))

    if not args.skip_present:
        row = present_route("diverted", 550, PRESENT_CELLS[550], root=args.present_root)
        emit(
            "PRESENT_ROUTE",
            converged=row["converged"],
            terminal=row["terminal"],
            newton_steps=row["newton_steps"],
            solve_wall=row["solve_wall"],
            realised_cells=row["realised_cells"],
        )


if __name__ == "__main__":
    main()
