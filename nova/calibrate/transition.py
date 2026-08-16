"""Separate coherent pulse response changes from channel-local gain changes.

A scalar gain check is a product of several physical terms.  A response or drive
change shared by all channels in one pulse, a channel's acquisition setting, and a
pickup-pair multiplier all move that scalar.  Simultaneous channels identify the
first term: an alternating-median fit in log amplitude estimates one robust pulse
factor while retaining each channel's absolute scale.  It cannot identify the last
two terms without an independently measured response-state or configuration label.

The functions here take arrays and transition records only.  They quantify both the
common-mode reduction and the observation/state limits that remain, so a sparse or
interleaved archive cannot be reported as an exact transition catalogue.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nova.calibrate.gain import MINIMUM_SHAPE_AGREEMENT
from nova.calibrate.scale_step import scale_blocks


class TransitionError(ValueError):
    """Raised when transition evidence has inconsistent axes or no support."""


@dataclass(frozen=True)
class CommonResponseNormalisation:
    """Gain observations after removing a robust simultaneous pulse factor."""

    shots: np.ndarray
    channels: np.ndarray
    raw_gains: np.ndarray
    corrected_gains: np.ndarray
    pulse_shots: np.ndarray
    pulse_log_factors: np.ndarray
    excluded_shots: int
    excluded_observations: int
    minimum_peers: int
    minimum_shape_agreement: float

    @property
    def observation_count(self) -> int:
        return int(self.shots.size)

    @property
    def shot_count(self) -> int:
        return int(self.pulse_shots.size)


def _axes(
    shots: Sequence[int] | np.ndarray,
    channels: Sequence[str] | np.ndarray,
    gains: Sequence[float] | np.ndarray,
    shape_agreement: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shot = np.asarray(shots, dtype=np.int64)
    channel = np.asarray(channels, dtype=str)
    gain = np.asarray(gains, dtype=float)
    shape = (
        np.ones(gain.shape, dtype=float)
        if shape_agreement is None
        else np.asarray(shape_agreement, dtype=float)
    )
    if shot.ndim != 1:
        raise TransitionError("shot identity must be one-dimensional")
    if (
        channel.shape != shot.shape
        or gain.shape != shot.shape
        or shape.shape != shot.shape
    ):
        raise TransitionError(
            "shot, channel, gain, and quality axes must have equal shape"
        )
    if shot.size == 0:
        raise TransitionError("at least one gain observation is required")
    if np.any(channel == ""):
        raise TransitionError("channel identities must be non-empty")
    if not np.all(np.isfinite(gain) & (gain > 0.0)):
        raise TransitionError("gain observations must be finite and positive")
    return shot, channel, gain, shape


def normalise_common_response(
    shots: Sequence[int] | np.ndarray,
    channels: Sequence[str] | np.ndarray,
    gains: Sequence[float] | np.ndarray,
    *,
    shape_agreement: Sequence[float] | np.ndarray | None = None,
    minimum_shape_agreement: float = MINIMUM_SHAPE_AGREEMENT,
    minimum_peers: int = 3,
    iterations: int = 20,
) -> CommonResponseNormalisation:
    """Remove a pulse-wide multiplicative response using simultaneous channels.

    At least three channels are required by default.  With three, the median pulse
    factor remains unchanged when one channel alone steps; a two-channel ratio would
    split that step between the changed channel and its reference.
    """

    shot, channel, gain, shape = _axes(shots, channels, gains, shape_agreement)
    if not math.isfinite(minimum_shape_agreement) or not (
        0.0 <= minimum_shape_agreement <= 1.0
    ):
        raise TransitionError("minimum shape agreement must lie between zero and one")
    if minimum_peers < 3:
        raise TransitionError("at least three simultaneous channels are required")
    if iterations < 1:
        raise TransitionError("at least one alternating-median iteration is required")

    quality = np.isfinite(shape) & (shape >= minimum_shape_agreement)
    quality_shot = shot[quality]
    quality_channel = channel[quality]
    quality_gain = gain[quality]
    counts = Counter(int(value) for value in quality_shot)
    admitted_shots = {
        value for value, count in counts.items() if count >= minimum_peers
    }
    admitted = np.asarray(
        [int(value) in admitted_shots for value in quality_shot], dtype=bool
    )
    selected_shot = quality_shot[admitted]
    selected_channel = quality_channel[admitted]
    selected_gain = quality_gain[admitted]
    if selected_shot.size == 0:
        raise TransitionError(
            f"no pulse carries {minimum_peers} quality-admitted simultaneous channels"
        )

    logarithm = np.log(selected_gain)
    channel_level = {
        name: float(np.median(logarithm[selected_channel == name]))
        for name in np.unique(selected_channel)
    }
    pulse_level = {int(value): 0.0 for value in np.unique(selected_shot)}
    for _ in range(iterations):
        pulse_level = {
            int(value): float(
                np.median(
                    logarithm[selected_shot == value]
                    - np.asarray(
                        [
                            channel_level[name]
                            for name in selected_channel[selected_shot == value]
                        ]
                    )
                )
            )
            for value in np.unique(selected_shot)
        }
        origin = float(np.median(list(pulse_level.values())))
        pulse_level = {key: value - origin for key, value in pulse_level.items()}
        channel_level = {
            name: float(
                np.median(
                    logarithm[selected_channel == name]
                    - np.asarray(
                        [
                            pulse_level[int(value)]
                            for value in selected_shot[selected_channel == name]
                        ]
                    )
                )
            )
            for name in np.unique(selected_channel)
        }

    pulse_factor = np.asarray(
        [pulse_level[int(value)] for value in selected_shot], dtype=float
    )
    pulse_shots = np.asarray(sorted(pulse_level), dtype=np.int64)
    return CommonResponseNormalisation(
        shots=selected_shot,
        channels=selected_channel,
        raw_gains=selected_gain,
        corrected_gains=np.exp(logarithm - pulse_factor),
        pulse_shots=pulse_shots,
        pulse_log_factors=np.asarray(
            [pulse_level[int(value)] for value in pulse_shots], dtype=float
        ),
        excluded_shots=len(counts) - len(admitted_shots),
        excluded_observations=int(quality_shot.size - selected_shot.size),
        minimum_peers=minimum_peers,
        minimum_shape_agreement=float(minimum_shape_agreement),
    )


def apparent_block_count(
    shots: Sequence[int] | np.ndarray,
    channels: Sequence[str] | np.ndarray,
    gains: Sequence[float] | np.ndarray,
) -> int:
    """Count persistent scalar blocks without treating them as acquisition states."""

    shot, channel, gain, _ = _axes(shots, channels, gains, None)
    grouped: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for pulse, name, value in zip(shot, channel, gain, strict=True):
        grouped[str(name)].setdefault(int(pulse), []).append(float(value))
    return sum(len(scale_blocks(name, values)) for name, values in grouped.items())


def _channel_series(
    shots: np.ndarray, channels: np.ndarray, gains: np.ndarray
) -> dict[str, dict[int, list[float]]]:
    result: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for pulse, channel, gain in zip(shots, channels, gains, strict=True):
        result[str(channel)].setdefault(int(pulse), []).append(float(gain))
    return result


def refine_established_transitions(
    shots: Sequence[int] | np.ndarray,
    channels: Sequence[str] | np.ndarray,
    gains: Sequence[float] | np.ndarray,
    expected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Test established two-level transitions against measured scalar states."""

    shot, channel, gain, _ = _axes(shots, channels, gains, None)
    by_channel = _channel_series(shot, channel, gain)
    rows: list[dict[str, Any]] = []
    for established in expected:
        name = str(established["channel"])
        first = int(established["before_shot"])
        last = int(established["after_shot"])
        before = float(established["before_scale"])
        after = float(established["after_scale"])
        measured = [
            (pulse, float(np.median(values)))
            for pulse, values in sorted(by_channel.get(name, {}).items())
            if first <= pulse <= last and values
        ]
        old: list[int] = []
        changed: list[int] = []
        if before != 0.0 and after != 0.0 and before * after > 0.0:
            for pulse, value in measured:
                if value * before <= 0.0:
                    continue
                before_distance = abs(math.log(abs(value / before)))
                after_distance = abs(math.log(abs(value / after)))
                (old if before_distance <= after_distance else changed).append(pulse)
        before_shot = max(old, default=None)
        after_shot = min(changed, default=None)
        ordered = (
            before_shot is not None
            and after_shot is not None
            and before_shot < after_shot
            and not any(pulse > after_shot for pulse in old)
            and not any(pulse < before_shot for pulse in changed)
        )
        width = after_shot - before_shot if ordered else None
        rows.append(
            {
                "after_scale": after,
                "after_shot": after_shot,
                "before_scale": before,
                "before_shot": before_shot,
                "channel": name,
                "expected_after_shot": last,
                "expected_before_shot": first,
                "measured_in_bracket": len(measured),
                "ordered": bool(ordered),
                "pulse_width": width,
                "ratio": after / before if before != 0.0 else None,
            }
        )
    exact = [row for row in rows if row["pulse_width"] == 1]
    return {
        "exact_count": len(exact),
        "expected_count": len(rows),
        "obv03": next((row for row in rows if row["channel"] == "obv03"), None),
        "ordered_count": sum(1 for row in rows if row["ordered"]),
        "transitions": rows,
    }


def _cause(
    observations: np.ndarray,
    transition: Mapping[str, Any],
) -> tuple[str, int, int | None]:
    gaps = np.diff(observations)
    adjacent = int(np.count_nonzero(gaps == 1))
    minimum_gap = None if gaps.size == 0 else int(gaps.min())
    if observations.size == 0:
        return "no_ratio_observations", adjacent, minimum_gap
    if observations.size == 1:
        return "single_ratio_observation", adjacent, minimum_gap
    if adjacent == 0:
        return "no_adjacent_observation_pair", adjacent, minimum_gap
    if transition["pulse_width"] == 1:
        return "adjacent_transition", adjacent, minimum_gap
    if not transition["ordered"]:
        return "interleaved_or_unclassified_states", adjacent, minimum_gap
    return "ordered_but_nonadjacent", adjacent, minimum_gap


def evaluate_transition_discrimination(
    shots: Sequence[int] | np.ndarray,
    channels: Sequence[str] | np.ndarray,
    gains: Sequence[float] | np.ndarray,
    expected: Sequence[Mapping[str, Any]],
    *,
    shape_agreement: Sequence[float] | np.ndarray | None = None,
    response_states: Sequence[str] | np.ndarray | None = None,
    configuration_states: Sequence[str] | np.ndarray | None = None,
    minimum_shape_agreement: float = MINIMUM_SHAPE_AGREEMENT,
    minimum_peers: int = 3,
) -> dict[str, Any]:
    """Return a falsification report for a simultaneous-channel observable."""

    shot, channel, gain, shape = _axes(shots, channels, gains, shape_agreement)
    normalised = normalise_common_response(
        shot,
        channel,
        gain,
        shape_agreement=shape,
        minimum_shape_agreement=minimum_shape_agreement,
        minimum_peers=minimum_peers,
    )
    quality = np.isfinite(shape) & (shape >= minimum_shape_agreement)
    raw_blocks = apparent_block_count(shot[quality], channel[quality], gain[quality])
    raw_channels = int(np.unique(channel[quality]).size)
    cohort_blocks = apparent_block_count(
        normalised.shots, normalised.channels, normalised.raw_gains
    )
    corrected_blocks = apparent_block_count(
        normalised.shots, normalised.channels, normalised.corrected_gains
    )
    cohort_channels = int(np.unique(normalised.channels).size)
    raw_transitions = raw_blocks - raw_channels
    cohort_transitions = cohort_blocks - cohort_channels
    corrected_transitions = corrected_blocks - cohort_channels
    refinement = refine_established_transitions(
        normalised.shots,
        normalised.channels,
        normalised.corrected_gains,
        expected,
    )
    cause_counts: Counter[str] = Counter()
    coverage: list[dict[str, Any]] = []
    for established, transition in zip(
        expected, refinement["transitions"], strict=True
    ):
        name = str(established["channel"])
        first = int(established["before_shot"])
        last = int(established["after_shot"])
        observations = np.unique(
            normalised.shots[
                (normalised.channels == name)
                & (normalised.shots >= first)
                & (normalised.shots <= last)
            ]
        )
        cause, adjacent, minimum_gap = _cause(observations, transition)
        cause_counts[cause] += 1
        coverage.append(
            {
                "adjacent_pair_count": adjacent,
                "cause": cause,
                "channel": name,
                "minimum_observed_gap": minimum_gap,
                "observations": int(observations.size),
                "expected_after_shot": last,
                "expected_before_shot": first,
            }
        )

    def labelled_count(values: Sequence[str] | np.ndarray | None) -> int:
        if values is None:
            return 0
        labels = np.asarray(values, dtype=str)
        if labels.shape != shot.shape:
            raise TransitionError("state labels must share the gain observation axis")
        return int(np.count_nonzero(labels != ""))

    return {
        "conclusion": (
            "adjacent_precision_supported"
            if refinement["exact_count"] == len(expected)
            else "adjacent_precision_unobtainable_from_banked_vacuum_windows"
        ),
        "corrected_apparent_blocks": corrected_blocks,
        "corrected_apparent_transitions": corrected_transitions,
        "cause_counts": dict(sorted(cause_counts.items())),
        "common_response_blocks_removed": cohort_blocks - corrected_blocks,
        "common_response_transitions_removed": (
            cohort_transitions - corrected_transitions
        ),
        "common_response_reduction_fraction": (
            (cohort_transitions - corrected_transitions) / cohort_transitions
            if cohort_transitions
            else 0.0
        ),
        "configuration_state_labels": labelled_count(configuration_states),
        "coverage": coverage,
        "expected_switches": len(expected),
        "minimum_peers": minimum_peers,
        "minimum_shape_agreement": float(minimum_shape_agreement),
        "normalised_observations": normalised.observation_count,
        "normalised_shots": normalised.shot_count,
        "raw_apparent_blocks": raw_blocks,
        "raw_apparent_transitions": raw_transitions,
        "raw_cohort_apparent_blocks": cohort_blocks,
        "raw_cohort_apparent_transitions": cohort_transitions,
        "response_state_labels": labelled_count(response_states),
        "residual_excess_blocks": corrected_blocks - len(expected),
        "residual_excess_transitions": corrected_transitions - len(expected),
        "refinement": refinement,
        "under_peered_observations": normalised.excluded_observations,
        "under_peered_shots": normalised.excluded_shots,
    }
