"""The one terminal classifier for a forward fixed-point descent.

Four readings are distinguished.  ``refusal`` means no finite promotion exists,
so the descent stopped before the fixed point.  ``contracted`` means the
terminal residual met the registered criterion.  ``period-2 cycle`` means the
descent alternates between two levels, and it requires those two levels to
differ by more than the bit-exact tolerance: a tail that repeats one level
exactly alternates between nothing.  ``divergence`` means the residual grew
past twice its own minimum.  Everything else that terminates above tolerance is
a ``contraction stall``.

The classifier lives here alone so the probe and the receipt builder cannot
disagree about what a trajectory is, which is how two classifiers came to read
the same input differently.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TOLERANCE = 1.0e-5
CYCLE_RELATIVE_GAP = 1.0e-3
FLAT_RELATIVE_GAP = 1.0e-12


def classify_terminal(
    residuals: list[float | None],
    terminal: float | None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Return the terminal class of a descent and the evidence for it."""

    if terminal is None:
        return {"class": "refusal", "evidence": "no finite terminal residual"}
    finite = [value for value in residuals if value is not None]
    if not finite:
        return {"class": "refusal", "evidence": "no finite promotion recorded"}
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCE
    if terminal <= tolerance:
        return {
            "class": "contracted",
            "evidence": {"terminal_residual": terminal, "tolerance": tolerance},
            "terminal_residual": terminal,
        }
    if abs(finite[-1] - terminal) > FLAT_RELATIVE_GAP * max(1.0, abs(terminal)):
        finite = finite + [terminal]
    tail = finite[-4:]
    if len(tail) == 4:
        even_gap = abs(tail[2] - tail[0]) / max(abs(tail[2]), 1e-300)
        odd_gap = abs(tail[3] - tail[1]) / max(abs(tail[3]), 1e-300)
        if (
            even_gap > CYCLE_RELATIVE_GAP
            and odd_gap > CYCLE_RELATIVE_GAP
            and tail[2] > tolerance
            and tail[3] > tolerance
        ):
            return {
                "class": "period-2 cycle",
                "evidence": {
                    "alternating_levels": tail,
                    "even_gap_relative": float(even_gap),
                    "odd_gap_relative": float(odd_gap),
                },
            }
    minimum = min(finite)
    index_minimum = finite.index(minimum)
    if index_minimum < len(finite) - 1 and finite[-1] > 2.0 * minimum:
        return {
            "class": "divergence",
            "evidence": {
                "first": finite[0],
                "last": finite[-1],
                "minimum": minimum,
                "index_of_minimum": index_minimum,
            },
        }
    flat = len(finite) >= 2 and abs(finite[-1] - finite[-2]) <= FLAT_RELATIVE_GAP * max(
        1.0, abs(finite[-1])
    )
    return {
        "class": "contraction stall",
        "evidence": {
            "terminal_residual": terminal,
            "first_residual": finite[0],
            "promotions": len(finite),
            "residual_repeats_exactly": bool(flat),
            "note": (
                "a flat tail is a plateau, not a period-2 cycle: the two "
                "alternating levels a cycle requires do not exist here"
            ),
        },
    }