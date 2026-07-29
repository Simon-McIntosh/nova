"""Coil-construction micro-benchmark for the frame layer.

Times frame construction and current access through the public ``FrameSpace``
entry surface only (``insert`` / attribute + ``loc`` reads), so the identical
script runs on both the columnar store and the legacy pandas-backed frame for
a like-for-like comparison. No persistence (store/load) and no store-internal
calls, so it has no fsspec or backend dependency.

Run:  python -m benchmarks.coil_construction
"""

import timeit

import numpy as np

from nova.frame.framespace import FrameSpace


def build(count: int) -> FrameSpace:
    """Construct a FrameSpace of ``count`` coils via the public insert path.

    Measures the frame layer itself (schema, index build, subspace projection,
    select/multipoint) — geometry is excluded (no poly column) so the number
    reflects store overhead, not shapely polygon construction.
    """
    framespace = FrameSpace(
        required=["x", "z"],
        available=["It"],
        Subspace=["Ic"],
        Array=["Ic"],
        label="PF",
    )
    framespace.insert(
        np.linspace(1.0, 5.0, count),
        np.zeros(count),
        Ic=1.0,
        active=True,
    )
    return framespace


def _best_time_ms(func, *args, repeats: int = 20) -> float:
    """Return the best per-call time in milliseconds over ``repeats`` runs."""
    timer = timeit.Timer(lambda: func(*args))
    return 1e3 * min(timer.repeat(repeats, number=1))


def main() -> None:
    """Report construction and read timings across a range of coil counts."""
    build(10)  # warm import / lazy geometry
    print("coil construction (public FrameSpace.insert surface)")
    for count in (100, 500, 1000, 5000):
        construct_ms = _best_time_ms(build, count)
        print(
            f"  n={count:5d}  construct {construct_ms:8.2f} ms"
            f"  ({1e3 * construct_ms / count:6.2f} us/coil)"
        )

    framespace = build(5000)
    read_loc = _best_time_ms(lambda: framespace.loc[:, "Ic"], repeats=200)
    read_attr = _best_time_ms(lambda: framespace.Ic, repeats=200)
    print(f"  read loc[:, 'Ic'] on n=5000: {read_loc:.3f} ms/call")
    print(f"  read .Ic          on n=5000: {read_attr:.3f} ms/call")


if __name__ == "__main__":
    main()
