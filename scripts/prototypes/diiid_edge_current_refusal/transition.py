"""Resolve the largest observed axis movement along a saved state segment."""

from __future__ import annotations

import json

import numpy as np


def discriminate(output, stages, scale, flux_size, reading):
    """Bisect an observed transition without advancing or modifying the solve."""
    index = max(range(1, len(stages)), key=lambda i: stages[i]["axis_displacement_m"])
    before, after = stages[index - 1 : index + 1]

    def state(row):
        values = np.load(output / f"{row['stage']}-{row['index']}.npz")
        return values["state"][:flux_size] * scale

    initial, terminal = state(before), state(after)
    result = dict(before=before, after=after, samples=[], brackets=[])
    evaluated = []

    def sample(fraction):
        physical, support, current, shadow = reading(
            initial + fraction * (terminal - initial),
            f"transition-read-{len(evaluated)}",
        )
        record = dict(fraction=float(fraction), physical=physical)
        evaluated.append((record, support, current, shadow))
        result["samples"].append(record)
        return evaluated[-1]

    def distance(left, right):
        return float(
            np.linalg.norm(
                np.asarray(left[0]["physical"]["axis_rz_m"])
                - right[0]["physical"]["axis_rz_m"]
            )
        )

    coarse = [sample(fraction) for fraction in np.linspace(0, 1, 9)]
    selected = max(range(8), key=lambda i: distance(coarse[i], coarse[i + 1]))
    left, right = coarse[selected : selected + 2]
    for _ in range(12):
        middle = sample((left[0]["fraction"] + right[0]["fraction"]) / 2)
        if distance(left, middle) >= distance(middle, right):
            right = middle
        else:
            left = middle
        result["brackets"].append(
            dict(
                fraction_low=left[0]["fraction"],
                fraction_high=right[0]["fraction"],
                axis_separation_m=distance(left, right),
                boundary_separation_m=float(
                    np.linalg.norm(
                        np.asarray(left[0]["physical"]["boundary_rz_m"])
                        - right[0]["physical"]["boundary_rz_m"]
                    )
                ),
                support_changes=int(np.sum(left[1] != right[1])),
                shadow_changes=int(np.sum(left[3] != right[3])),
                current_change_l2_a=float(np.linalg.norm(left[2] - right[2])),
                low=left[0]["physical"],
                high=right[0]["physical"],
            )
        )
        (output / "transition.json").write_text(json.dumps(result, indent=2) + "\n")
    print("AXIS_TRANSITION " + json.dumps(result["brackets"][-1]), flush=True)
