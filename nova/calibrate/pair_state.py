"""Resolve discrete pickup-pair multipliers and their transition sequence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

STATE_MULTIPLIERS = {
    "single_member": 0.5,
    "both_members": 1.0,
    "recovered": 1.5,
}
"""Nominal signal multipliers for the resolved pickup states."""

STATE_TOLERANCE = 0.12
"""Maximum fractional distance from a nominal multiplier."""


class PairStateError(ValueError):
    """Raised when a pair-state sequence cannot be classified."""


@dataclass(frozen=True, order=True)
class PairStateBlock:
    """One contiguous run assigned to the same pickup state."""

    state: str
    multiplier: float
    start: int
    stop: int
    count: int
    measured: float
    maximum_distance: float


@dataclass(frozen=True)
class PairStateSequence:
    """Per-observation state assignments and their contiguous blocks."""

    assignments: tuple[str | None, ...]
    blocks: tuple[PairStateBlock, ...]
    unresolved: tuple[int, ...]

    @property
    def transition_count(self) -> int:
        """Return the number of resolved changes between adjacent blocks."""

        return max(0, len(self.blocks) - 1)

    @property
    def stable(self) -> bool:
        """Return whether one state describes every observation."""

        return not self.unresolved and len(self.blocks) == 1

    @property
    def midlife_step(self) -> bool:
        """Return whether the sequence contains one persistent state change."""

        return not self.unresolved and len(self.blocks) == 2

    @property
    def flips(self) -> bool:
        """Return whether a state recurs after at least one different state."""

        states = [block.state for block in self.blocks]
        return len(states) >= 3 and len(set(states)) < len(states)


def nearest_state(
    measured: float,
    *,
    multipliers: Mapping[str, float] = STATE_MULTIPLIERS,
    tolerance: float = STATE_TOLERANCE,
) -> tuple[str | None, float]:
    """Return the nearest resolved state and its fractional distance."""

    if not math.isfinite(measured) or measured <= 0.0:
        return None, math.inf
    if not multipliers:
        raise PairStateError("at least one state multiplier is required")
    invalid = [
        value
        for value in multipliers.values()
        if value <= 0 or not math.isfinite(value)
    ]
    if invalid:
        raise PairStateError("state multipliers must be finite and positive")
    state = min(
        multipliers,
        key=lambda name: abs(math.log(measured / multipliers[name])),
    )
    distance = abs(measured - multipliers[state]) / multipliers[state]
    return (state if distance <= tolerance else None), distance


def classify_pair_states(
    measured: Sequence[float] | np.ndarray,
    *,
    multipliers: Mapping[str, float] = STATE_MULTIPLIERS,
    tolerance: float = STATE_TOLERANCE,
) -> PairStateSequence:
    """Classify a gain sequence and compress adjacent equal states into blocks."""

    values = np.asarray(measured, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise PairStateError("measured gains must be a non-empty one-dimensional array")
    assigned: list[str | None] = []
    distances: list[float] = []
    for value in values:
        state, distance = nearest_state(
            float(value), multipliers=multipliers, tolerance=tolerance
        )
        assigned.append(state)
        distances.append(distance)

    blocks: list[PairStateBlock] = []
    start = 0
    while start < values.size:
        state = assigned[start]
        stop = start + 1
        while stop < values.size and assigned[stop] == state:
            stop += 1
        if state is not None:
            blocks.append(
                PairStateBlock(
                    state=state,
                    multiplier=float(multipliers[state]),
                    start=start,
                    stop=stop,
                    count=stop - start,
                    measured=float(np.median(values[start:stop])),
                    maximum_distance=float(max(distances[start:stop])),
                )
            )
        start = stop
    unresolved = tuple(index for index, state in enumerate(assigned) if state is None)
    return PairStateSequence(tuple(assigned), tuple(blocks), unresolved)
