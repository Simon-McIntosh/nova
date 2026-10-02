"""Bitwise comparison of the exit-loop and capacity-scan steps on toy operators.

Comparison is over the raw bit pattern of each field (an unsigned view of the
same width, with shape and dtype checked first), so a signed zero or a NaN
payload counts as a difference; a case whose fields are not bit-identical
makes the run exit nonzero and prints the first differing index with both
patterns.
"""

import sys

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import scan_stream

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
            f"DIFFERS {label} index={difference['index']} "
            f"multi={difference['multi_index']} "
            f"left_bits={difference['left_pattern:hex']} "
            f"right_bits={difference['right_pattern:hex']} "
            f"left={difference['left_value']} right={difference['right_value']}",
            flush=True,
        )
    else:
        print(f"DIFFERS {label} kind={difference['kind']} {difference}", flush=True)
    return False


def run(step, seed, size, scale):
    rng = np.random.default_rng(seed)
    matrix = jnp.asarray(np.eye(size) + scale * rng.standard_normal((size, size)))
    rhs = jnp.asarray(rng.standard_normal(size))
    result = jax.jit(
        lambda b: step(
            lambda v: matrix @ v,
            b,
            jnp.asarray(0.1),
            gmres_iterations=8,
            condition_ratio_limit=10.0,
            preceding_condition_baseline=jnp.asarray(2.0),
        )
    )(rhs)
    return [np.asarray(x) for x in result]


def main():
    cases = [(s, n, a) for s in range(4) for n, a in ((12, 0.2), (40, 0.1))]
    exit_rows = [run(fixed_point._qualified_krylov_step, *c) for c in cases]
    scan_stream.install()
    scan_rows = [run(fixed_point._qualified_krylov_step, *c) for c in cases]
    equal = 0
    all_equal = True
    for case, a, b in zip(cases, exit_rows, scan_rows, strict=True):
        same = all(
            report_bit_difference(f"{case}/{field}", x, y)
            for field, (x, y) in enumerate(zip(a, b))
        )
        equal += same
        all_equal = all_equal and same
        print(case, "bitwise" if same else "differs")
    print(f"EXIT_TOY_IDENTITY {equal}/{len(cases)}")
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
