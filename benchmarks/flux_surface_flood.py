"""Measure dilate-by-doubling flux-surface flood equivalence and cost."""

from __future__ import annotations

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.flux_surface_connectivity import flood_fill_core_with_steps
from nova.equilibrium.morphology import _dilate4


def _fixture(nr: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    rg = np.linspace(-1.0, 1.0, nr)
    zg = np.linspace(-1.0, 1.0, nz)
    rr, zz = np.meshgrid(rg, zg)
    confined = (rr / 0.78) ** 2 + (zz / 0.88) ** 2 < 1.0
    pocket = (rr + 0.9) ** 2 + (zz - 0.9) ** 2 < 0.035
    confined |= pocket
    seed = np.zeros_like(confined)
    seed[np.abs(zg).argmin(), np.abs(rg).argmin()] = True
    return confined, seed


def _fixed_iteration_fill(
    confined: jnp.ndarray, seed: jnp.ndarray, n_iter: int
) -> jnp.ndarray:
    def body(_index, core):
        return _dilate4(core) & confined

    return jax.lax.fori_loop(0, n_iter, body, seed & confined).astype(jnp.float32)


def _elapsed(function, confined, seed, n_iter: int, repeats: int):
    compiled = jax.jit(function, static_argnums=(2,))
    output = compiled(confined, seed, n_iter)
    jax.block_until_ready(output)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = compiled(confined, seed, n_iter)
        jax.block_until_ready(output)
        samples.append(time.perf_counter() - started)
    return output, float(np.median(samples))


def measure(nr: int, nz: int, repeats: int) -> dict[str, float | int | bool]:
    confined_host, seed_host = _fixture(nr, nz)
    confined = jnp.asarray(confined_host)
    seed = jnp.asarray(seed_host)
    n_iter = nr + nz

    fixed, fixed_seconds = _elapsed(
        _fixed_iteration_fill, confined, seed, n_iter, repeats
    )

    def doubled_fill(mask, origin, limit):
        return flood_fill_core_with_steps(mask, origin, limit)

    (doubled, steps), doubled_seconds = _elapsed(
        doubled_fill, confined, seed, n_iter, repeats
    )
    exact = bool(np.array_equal(np.asarray(doubled), np.asarray(fixed)))
    return {
        "nr": nr,
        "nz": nz,
        "fixed_iterations": n_iter,
        "doubling_steps": int(steps),
        "log2_nr_plus_nz": math.log2(n_iter),
        "exact_array_equality": exact,
        "fixed_seconds": fixed_seconds,
        "doubling_seconds": doubled_seconds,
        "speedup": fixed_seconds / doubled_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", type=int, default=193)
    parser.add_argument("--nz", type=int, default=129)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    result = measure(args.nr, args.nz, args.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["exact_array_equality"]:
        raise SystemExit("doubling fill differs from fixed-iteration reference")


if __name__ == "__main__":
    main()
