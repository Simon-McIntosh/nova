"""Bisect every read verdict and saddle-identity change, and classify the change.

The read at ``kernel.read`` selects the bounding saddle of the axis merge path.
This probe measures the same carrier the exterior-jet sweep uses at arbitrary
offsets and bisects each change to adjacent binary64 values, recording on both
sides the boundary flux, the unnormalised booked current, the live saddle roots
with their levels and determinants, and the computed psi rounding enclosure, so
a change can be classified as a crossing, a fold or an implementation switch.

The enclosure is the bound ``kernel.psi_rounding_enclosure`` states for the
moving field, and the persistence of a max/saddle pair is the difference of
their levels. A pair whose level difference crosses the enclosure across the
bisected interval is a fold, whatever the determinant says.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.fast_map_production import kernel, tie_probe


def build(kind, cells, cache):
    return tie_probe.build(kind, cells, cache)


def sampler_with_enclosure(state):
    """The tie probe's sampler, plus the kernel's own saddle identity and enclosure."""
    base = tie_probe.make_sampler(state)
    geometry = state["geometry"]

    def sampled(shifted, prior=None):
        roots, reading, raw, eigenvalues, live, chosen = base(shifted, prior)
        exterior = state["jets"].exterior + shifted * state["jet_slope"]
        moving = replace(state["jets"], exterior=exterior).field(state["reference"])
        enclosure = kernel.psi_rounding_enclosure(moving.sampled_coefficient(geometry))
        return roots, reading, raw, eigenvalues, live, chosen, enclosure

    return jax.jit(sampled)


def measure(offset, state, sampled, prior=None):
    roots, reading, raw, eigenvalues, live, chosen, enclosure = sampled(
        jnp.asarray(offset * state["scale"]), prior
    )
    jax.block_until_ready(raw)
    eigen = np.asarray(eigenvalues)
    live_np = np.asarray(live)
    determinant = eigen[:, 0] * eigen[:, 1]
    levels = np.asarray(roots.jet.value) * float(state["convention"].sigma)
    position = np.asarray(roots.position)
    form_position = np.asarray(reading.saddle_form.position)
    distances = np.linalg.norm(position - form_position, axis=-1)
    distances = np.where(live_np, distances, np.inf)
    identity = int(np.argmin(distances))
    saddles = [
        {
            "index": int(index),
            "level": float(levels[index]),
            "det": float(determinant[index]),
            "position": position[index].tolist(),
        }
        for index in np.flatnonzero(live_np)
    ]
    return {
        "offset": float(offset),
        "boundary_class": int(reading.boundary_class),
        "saddle_identity": identity,
        "wall_candidate": float(reading.wall_candidate),
        "saddle_candidate": float(reading.saddle_candidate),
        "boundary_flux": float(reading.boundary_flux),
        "raw_current": float(raw),
        "psi_enclosure": float(enclosure),
        "selected_eigenvalues": eigen[identity].tolist(),
        "selected_det": float(determinant[identity]),
        "min_abs_det_live": float(np.min(np.abs(determinant[live_np])))
        if live_np.any()
        else None,
        "live_saddle_count": int(live_np.sum()),
        "valid": bool(reading.valid and reading.qualified),
        "saddles": saddles,
    }


def key_of(row, key):
    return (
        (row["boundary_class"], row["saddle_identity"]) if key == "both" else row[key]
    )


def bisect(left, right, state, sampled, prior, key):
    left_row = measure(left, state, sampled, prior)
    right_row = measure(right, state, sampled, prior)
    if key_of(left_row, key) == key_of(right_row, key):
        return {
            "bracket": [left, right],
            "steps": 0,
            "adjacent": False,
            "unchanged": True,
            "left": left_row,
            "right": right_row,
        }
    steps = 0
    while right != np.nextafter(left, right) and steps < 200:
        middle = (left + right) / 2
        if middle in (left, right):
            break
        middle_row = measure(middle, state, sampled, prior)
        steps += 1
        if key_of(middle_row, key) == key_of(left_row, key):
            left, left_row = middle, middle_row
        else:
            right, right_row = middle, middle_row
    return {
        "bracket": [float(left), float(right)],
        "steps": steps,
        "adjacent": bool(right == np.nextafter(left, right)),
        "unchanged": False,
        "left": left_row,
        "right": right_row,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="diverted")
    parser.add_argument("--cells", type=int, default=250)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scan", default="", help="'lo,hi,count' in offset units")
    parser.add_argument(
        "--bisect", action="append", default=[], help="'lo,hi' bracket; repeatable"
    )
    parser.add_argument(
        "--key", default="both", choices=("boundary_class", "saddle_identity", "both")
    )
    parser.add_argument("--warm", type=float, default=None)
    args = parser.parse_args()

    assert jax.config.jax_enable_x64 is True
    import nova

    from scripts.prototypes.fast_map_production import kernel

    print(
        f"nova={nova.__file__} KERNEL={kernel.__file__} module={__file__} "
        f"cwd={Path.cwd().resolve()} backend={jax.default_backend()}",
        flush=True,
    )
    state = build(args.kind, args.cells, args.cache)
    sampled = sampler_with_enclosure(state)
    prior = None
    warm_count = 0
    if args.warm is not None:
        prior, warm_count = tie_probe.warm(
            state, tie_probe.make_sampler(state), -0.2, args.warm, 0.01
        )
    out = Path(args.output)
    result = {
        "kind": args.kind,
        "cells": len(state["geometry"].centre),
        "pitch": state["pitch"],
        "span": state["span"],
        "warm_states": warm_count,
        "key": args.key,
        "scan": [],
        "changes": [],
        "bisections": [],
    }
    if args.scan:
        lo, hi, count = (float(part) for part in args.scan.split(","))
        for value in np.linspace(lo, hi, int(count)):
            result["scan"].append(measure(value, state, sampled, prior))
        for index in range(1, len(result["scan"])):
            before = result["scan"][index - 1]
            after = result["scan"][index]
            if key_of(before, args.key) != key_of(after, args.key):
                result["changes"].append(
                    {
                        "from": before["offset"],
                        "to": after["offset"],
                        "class_before": before["boundary_class"],
                        "class_after": after["boundary_class"],
                        "identity_before": before["saddle_identity"],
                        "identity_after": after["saddle_identity"],
                    }
                )
        tie_probe.save(out, result)
    for bracket in args.bisect:
        lo, hi = (float(part) for part in bracket.split(","))
        result["bisections"].append(bisect(lo, hi, state, sampled, prior, args.key))
        tie_probe.save(out, result)
    tie_probe.save(out, result)
    print(
        "RESULT "
        + json.dumps(
            {
                "kind": args.kind,
                "cells": result["cells"],
                "key": args.key,
                "changes": len(result["changes"]),
                "bisected": len(result["bisections"]),
            }
        ),
        flush=True,
    )
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
