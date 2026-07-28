"""Re-derive the brute-force self-flux constants the section-average tests assert on.

:mod:`tests.test_biotsectionaverage` holds the shipped target-section rule against a
four-dimensional quadrature over source section x target section of the coaxial ring
kernel, built from the elliptic integrals and sharing no code with the closed form or
with the rule. That oracle converges algebraically and costs minutes to run to its
Richardson limit, which is a benchmark's price and not a test's -- so the limits are
RECORDED as constants there and re-derived here.

Run this when the rule changes, when a section definition changes, or when a recorded
limit is doubted. It prints the full ladder -- every rung, the step between rungs, the
Richardson limit and the shipped rule's deviation from it -- followed by a paste-ready
block of the constants and the geometry fingerprints that guard them. The oracle
itself is imported from the test module rather than restated, because two copies of a
reference implementation drift and the whole point of this one is that it does not.

Usage::

    python -m benchmarks.section_average_oracle
"""

import time

import numpy as np

from nova.biot.sectionaverage import ORDER, averaged_greens
from tests.test_biotsectionaverage import (
    ASPECT_AREA,
    ASPECT_MAJOR,
    SECTIONS,
    brute_force,
    rectangle,
    section_fingerprint,
)

ORDERS = (10, 12, 14)
"""Rungs of the oracle ladder the recorded Richardson limit is taken over."""

ASPECTS = (0.5, 1.0, 2.0, 2.89, 5.0, 10.0)
"""Height over width of the swept rectangle.

2.89 at this area and major radius reproduces the ITER CS section itself, which is
the aspect the coil element actually meets.
"""


def richardson(value):
    """Return ``(limit, step ratio)`` from three rungs of an algebraic ladder.

    The ratio is reported because it is the extrapolation's conditioning: a ladder in
    its asymptotic regime halves its step and the tail it implies is about as big as
    the last step, while a ratio near one divides by almost nothing and the "limit" is
    then an extrapolation of a rung that has not started converging. A recorded
    constant whose ratio is near one should not be asserted against.
    """
    ratio = (value[1] - value[0]) / (value[2] - value[1])
    return value[2] + (value[2] - value[1]) / (ratio - 1.0), ratio


def ladder(name, vertices):
    """Print the ladder for one section and return its Richardson limit."""
    print(f"\n{name}")
    value, previous = [], None
    for order in ORDERS:
        start = time.perf_counter()
        value.append(brute_force(vertices, vertices, order))
        step = f"  step {abs(value[-1] / previous - 1):.2e}" if previous else ""
        print(
            f"  order {order:2d}  {value[-1]:.12e}{step}"
            f"   {time.perf_counter() - start:6.1f} s"
        )
        previous = value[-1]
    limit, ratio = richardson(value)
    flag = "  WELL CONDITIONED" if ratio > 1.5 else "  STIFF -- do not assert on this"
    print(f"  Richardson limit {limit:.12e}  step ratio {ratio:.2f}{flag}")
    for order in (ORDER, 2 * ORDER):
        shipped = averaged_greens([vertices], vertices, order)[0][0]
        print(f"  shipped fan order {order:2d}  {shipped / limit - 1:+.2e}")
    return limit, ratio


def main():
    """Run both ladders and print the constants block."""
    named = {name: ladder(name, vertices) for name, vertices in SECTIONS.items()}
    swept = {}
    for aspect in ASPECTS:
        width = np.sqrt(ASPECT_AREA / aspect)
        vertices = rectangle(ASPECT_MAJOR, 0.0, width, aspect * width)
        swept[aspect] = ladder(f"aspect {aspect}", vertices)

    print("\n\n--- paste into tests/test_biotsectionaverage.py ---\n")
    print("BRUTE_FORCE_SELF_FLUX = {")
    for name, (limit, ratio) in named.items():
        print(f'    "{name}": {limit:.9e},  # step ratio {ratio:.2f}')
    print("}\n")
    print("SECTION_FINGERPRINT = {")
    for name, vertices in SECTIONS.items():
        area, centre_r, centre_z = section_fingerprint(vertices)
        print(f'    "{name}": ({area:.10f}, {centre_r:.10f}, {centre_z:.10f}),')
    print("}\n")
    print("ASPECT_SELF_FLUX = {")
    for aspect, (limit, ratio) in swept.items():
        print(f"    {aspect}: {limit:.9e},  # step ratio {ratio:.2f}")
    print("}")


if __name__ == "__main__":
    main()
