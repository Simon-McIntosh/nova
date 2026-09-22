"""Test whether exact integer coefficients improve the frozen-patch reference."""

from math import comb
from pathlib import Path
import nova.linalg.interpolant as interpolant
from reference_clip import measure

interpolant._binomial_coefficients = lambda order, extended_precision: tuple(
    float(comb(order, term)) for term in range(order + 1)
)
output = (
    Path(__file__).resolve().parent / "bernstein-reference" / "exact-coefficients-probe"
)
output.mkdir(exist_ok=True)
measure(output)
