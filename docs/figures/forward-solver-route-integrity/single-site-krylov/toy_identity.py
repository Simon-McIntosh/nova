"""Bitwise comparison of the stream and per-site Krylov steps on toy operators.

Every case is compared against two references: the base per-site body as
committed, and the same body with an optimisation barrier on both sides of
each operator application (the like-for-like fusion boundary). Comparison is
over the raw bit pattern of each field (an unsigned view of the same width,
with shape and dtype checked first), so a signed zero or a NaN payload counts
as a difference; any difference makes the run exit nonzero and prints the
first differing index with both patterns. One JSON row per case records the
first difference of every field that is not bit-identical.
"""

import json
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

assert jax.config.jax_enable_x64 is True
sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import per_site

from nova.equilibrium import fixed_point


_UNSIGNED = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _bit_patterns(value):
    array = np.asarray(value)
    width = array.dtype.itemsize
    if width in _UNSIGNED:
        return array.view(_UNSIGNED[width])
    return array.view(np.uint8).reshape(array.shape + (-1,))


def first_bit_difference(left, right):
    """Return None only when both operands share shape, dtype and every bit.

    Otherwise return a dict describing the mismatch: for a bitwise mismatch it
    carries the first differing index and each operand's raw bit pattern, so a
    signed zero against an unsigned zero and two NaNs with different payloads
    are both reported as differences.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return {"kind": "shape", "left": left.shape, "right": right.shape}
    if left.dtype != right.dtype:
        return {"kind": "dtype", "left": str(left.dtype), "right": str(right.dtype)}
    left_bits = np.ravel(_bit_patterns(left))
    right_bits = np.ravel(_bit_patterns(right))
    differing = np.flatnonzero(left_bits != right_bits)
    if differing.size == 0:
        return None
    index = int(differing[0])
    multi_index = (
        tuple(int(i) for i in np.unravel_index(index, left.shape)) if left.shape else ()
    )
    return {
        "kind": "bits",
        "index": index,
        "multi_index": multi_index,
        "left_pattern:hex": f"{int(left_bits[index]):#018x}",
        "right_pattern:hex": f"{int(right_bits[index]):#018x}",
        "left_value": f"{np.ravel(left)[index]!r}",
        "right_value": f"{np.ravel(right)[index]!r}",
    }


def report_bit_difference(label, left, right):
    """Print the first bit-pattern difference under ``label``; True when identical."""
    difference = first_bit_difference(left, right)
    if difference is None:
        return True
    if difference["kind"] == "bits":
        print(
            f"{label} DIFFERS index={difference['index']} "
            f"multi={difference['multi_index']} "
            f"left_bits={difference['left_pattern:hex']} "
            f"right_bits={difference['right_pattern:hex']} "
            f"left={difference['left_value']} right={difference['right_value']}",
            flush=True,
        )
    else:
        print(f"{label} DIFFERS kind={difference['kind']} {difference}", flush=True)
    return False


def main(argv):
    elementwise = len(argv) > 1 and argv[1] == "elementwise"
    barrier = jax.lax.optimization_barrier
    base = per_site.base_step()
    rng = np.random.default_rng(7)
    rows = []
    all_equal = True
    for size, scale in ((5, 0.3), (12, 1.0), (40, 3.0), (3, 0.0)):
        for iterations in (2, 8):
            matrix = jnp.asarray(
                np.eye(size) + scale * rng.standard_normal((size, size))
            )
            diagonal = jnp.diag(matrix)
            rhs = jnp.asarray(rng.standard_normal(size))
            if elementwise:

                def action(v, diagonal=diagonal, scale=scale):
                    return diagonal * v + 0.3 * scale * jnp.roll(v, 1)
            else:

                def action(v, matrix=matrix):
                    return matrix @ v

            def fenced(v, action=action):
                return barrier(action(barrier(v)))

            kwargs = dict(
                gmres_iterations=iterations,
                condition_ratio_limit=10.0,
                preceding_condition_baseline=jnp.asarray(2.0),
            )
            rest = (rhs, jnp.asarray(0.1))
            new = jax.jit(
                lambda: fixed_point._qualified_krylov_step(action, *rest, **kwargs)
            )()
            row = dict(size=size, iterations=iterations)
            for name, reference in (
                ("base", jax.jit(lambda: base(action, *rest, **kwargs))()),
                ("base-fenced", jax.jit(lambda: base(fenced, *rest, **kwargs))()),
            ):
                differing = {}
                for field, a, b in zip(new._fields, new, reference):
                    if not report_bit_difference(
                        f"{name}/{size}/{iterations}/{field}", a, b
                    ):
                        differing[field] = first_bit_difference(a, b)
                        all_equal = False
                row[name] = differing
            rows.append(row)
            print(json.dumps(row), flush=True)
    for name in ("base", "base-fenced"):
        equal = sum(not row[name] for row in rows)
        print(f"TOY_{name.upper().replace('-', '_')} bit_identical={equal}/{len(rows)}")
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
