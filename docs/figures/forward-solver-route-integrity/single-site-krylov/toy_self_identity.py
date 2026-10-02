"""Record, or compare against, the stream's own outputs on the toy operators.

``record`` writes every field of every dense and elementwise toy case to an
npz beside this script; ``compare`` recomputes them and requires every field
to match the recording bit for bit. The comparison is over the raw bit
pattern (an unsigned view of the same width, with shape and dtype checked
first), so a signed zero or a NaN payload counts as a difference; a difference
prints the first differing index with both patterns and exits nonzero.
"""

import subprocess
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_point

assert jax.config.jax_enable_x64 is True


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


def main(argv):
    mode = argv[1]
    path = "docs/figures/forward-solver-route-integrity/single-site-krylov/toy-stream-outputs.npz"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} mode={mode}", flush=True)
    outputs = {}
    for arm in ("dense", "elementwise"):
        rng = np.random.default_rng(7)
        for size, scale in ((5, 0.3), (12, 1.0), (40, 3.0), (3, 0.0)):
            for iterations in (2, 8):
                matrix = jnp.asarray(
                    np.eye(size) + scale * rng.standard_normal((size, size))
                )
                diagonal = jnp.diag(matrix)
                rhs = jnp.asarray(rng.standard_normal(size))
                if arm == "elementwise":

                    def action(v, diagonal=diagonal, scale=scale):
                        return diagonal * v + 0.3 * scale * jnp.roll(v, 1)
                else:

                    def action(v, matrix=matrix):
                        return matrix @ v

                result = jax.jit(
                    lambda: fixed_point._qualified_krylov_step(
                        action,
                        rhs,
                        jnp.asarray(0.1),
                        gmres_iterations=iterations,
                        condition_ratio_limit=10.0,
                        preceding_condition_baseline=jnp.asarray(2.0),
                    )
                )()
                for field, value in zip(result._fields, result):
                    outputs[f"{arm}-{size}-{iterations}-{field}"] = np.asarray(value)
    if mode == "record":
        np.savez(path, **outputs)
        print(f"TOY_STREAM_RECORDED fields={len(outputs)}")
        return 0
    recorded = np.load(path)
    differing = [
        key
        for key, value in outputs.items()
        if not report_bit_difference(key, value, recorded[key])
    ]
    print(
        f"TOY_STREAM_SELF_IDENTITY equal={len(outputs) - len(differing)}/{len(outputs)}"
    )
    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
