"""Paired-fp64 arithmetic for cancellation-sensitive kernel contractions.

Each value is carried as a non-overlapping ``(high, low)`` pair.  The helpers use
error-free fp64 transforms, so callers can retain product and summation residues
without changing array shapes or requiring an extended scalar dtype.  They use
only the array namespace supplied by their operands and therefore trace under the
same NumPy/JAX source graph as the surrounding kernel.
"""

from __future__ import annotations

__all__ = [
    "add",
    "contract",
    "contract_paired",
    "divide",
    "multiply",
    "scale",
    "square_root",
    "subtract",
    "value",
    "where",
    "wrap",
]

_SPLITTER = 134_217_729.0


def _two_sum(left, right):
    total = left + right
    carried = total - left
    error = (left - (total - carried)) + (right - carried)
    return total, error


def _quick_two_sum(left, right):
    total = left + right
    return total, right - (total - left)


def _two_product(left, right):
    product = left * right
    left_split = _SPLITTER * left
    right_split = _SPLITTER * right
    left_high = left_split - (left_split - left)
    right_high = right_split - (right_split - right)
    left_low = left - left_high
    right_low = right - right_high
    error = (
        ((left_high * right_high - product) + left_high * right_low)
        + left_low * right_high
    ) + left_low * right_low
    return product, error


def wrap(value):
    """Return an exact paired representation of one fp64 value."""
    return value, 0.0 * value


def add(left, right):
    """Return the renormalized sum of two paired values."""
    high, high_error = _two_sum(left[0], right[0])
    low, low_error = _two_sum(left[1], right[1])
    correction, correction_error = _two_sum(high_error, low)
    high, low = _quick_two_sum(high, correction)
    low = low + correction_error + low_error
    return _quick_two_sum(high, low)


def subtract(left, right):
    """Return the renormalized difference of two paired values."""
    return add(left, (-right[0], -right[1]))


def multiply(left, right):
    """Return the paired product, retaining the leading product residue."""
    high, error = _two_product(left[0], right[0])
    cross = left[0] * right[1] + left[1] * right[0]
    high, low = _quick_two_sum(high, error + cross)
    low = low + left[1] * right[1]
    return _quick_two_sum(high, low)


def scale(pair, factor):
    """Multiply a paired value by one fp64 factor."""
    return multiply(pair, wrap(factor))


def divide(numerator, denominator):
    """Return a paired quotient using one residual correction."""
    leading = numerator[0] / denominator[0]
    residual = subtract(numerator, multiply(denominator, wrap(leading)))
    correction = (residual[0] + residual[1]) / denominator[0]
    return add(wrap(leading), wrap(correction))


def square_root(pair, xp):
    """Return a paired square root using the exact squared residual."""
    leading = xp.sqrt(pair[0])
    residual = subtract(pair, multiply(wrap(leading), wrap(leading)))
    correction = (residual[0] + residual[1]) / (2.0 * leading)
    return add(wrap(leading), wrap(correction))


def contract(coefficients, moments):
    """Contract two fp64 sequences without discarding product or sum residues."""
    total = wrap(0.0 * moments[0])
    for coefficient, moment in zip(coefficients, moments, strict=False):
        total = add(total, _two_product(coefficient, moment))
    return total


def contract_paired(coefficients, moments):
    """Contract paired coefficients against paired moments."""
    total = wrap(0.0 * moments[0][0])
    for coefficient, moment in zip(coefficients, moments, strict=False):
        total = add(total, multiply(moment, coefficient))
    return total


def value(pair):
    """Round one paired value back to fp64."""
    return pair[0] + pair[1]


def where(condition, selected, alternative, xp):
    """Select paired values component by component."""
    return tuple(
        xp.where(condition, one, other)
        for one, other in zip(selected, alternative, strict=True)
    )
