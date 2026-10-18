"""Scan a moving carrier for census departures and classify each one.

A census count that falls between two states is only a defect if the vanished
root was still hyperbolic where it was last seen: a saddle-node pair merges when
the determinant and the smaller eigenvalue both reach roundoff, and a census
that stops representing a merged root has lost nothing. This runs the repaired
census and the census carrying the exact-zero admission step over the same
window under the same prior, matches roots between adjacent states by nearest
position, and reports for every vanished root the determinant, the eigenvalue
pair and the distance to its nearest neighbour at the last state that held it.
A departure whose vanishing determinant and smaller eigenvalue are both within
the Hessian's own rounding is a fold; one whose determinant is still large is a
defect.
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

from scripts.prototypes.fast_map_production import tie_probe
from scripts.prototypes.read_robustness import census as repaired_module
from scripts.prototypes.read_robustness import census_base_step


def make_arm(census_function, state, prior):
    """One census arm as a jitted function over the offset."""

    def run(offset):
        varied = replace(
            state["jets"], exterior=state["jets"].exterior + offset * state["jet_slope"]
        )
        field = varied.field(state["reference"])
        roots = census_function(
            field, state["geometry"], state["convention"], state["policy"], prior
        ).roots
        eigenvalues = jnp.linalg.eigvalsh(
            state["convention"].sigma * roots.jet.hessian
        )
        determinant = jnp.linalg.det(roots.jet.hessian)
        return roots.position, roots.valid, eigenvalues, determinant

    return jax.jit(run)


def state_row(position, valid, eigenvalues, determinant, scale):
    roots = []
    for index in np.flatnonzero(np.asarray(valid)):
        pair = np.asarray(eigenvalues)[index]
        roots.append(
            {
                "position": [float(value) for value in np.asarray(position)[index]],
                "determinant": float(np.asarray(determinant)[index]),
                "eigenvalues": [float(value) for value in pair],
                "saddle": bool(pair[0] < 0 < pair[1]),
            }
        )
    return {
        "valid": len(roots),
        "saddles": sum(1 for root in roots if root["saddle"]),
        "roots": roots,
        "scale": scale,
    }


def vanishings(before, after):
    """Roots present in one state and not matched in the next, by nearest position."""
    if not after["roots"]:
        return []
    threshold = 0.25 * before["scale"]
    vanished = []
    for root in before["roots"]:
        point = np.asarray(root["position"])
        distances = [
            float(np.linalg.norm(point - np.asarray(other["position"])))
            for other in after["roots"]
        ]
        if min(distances) > threshold:
            vanished.append({**root, "nearest_distance": min(distances)})
    return vanished


def enrich(vanished, state):
    """Add the nearest-neighbour distance within the state that still held the root."""
    for root in vanished:
        point = np.asarray(root["position"])
        others = [
            float(np.linalg.norm(point - np.asarray(other["position"])))
            for other in state["roots"]
            if other["position"] != root["position"]
        ]
        root["nearest_in_state"] = min(others) if others else None
    return vanished


def scan(kind, cells, cache, lo, hi, count, warm_stop, warm_step):
    state = tie_probe.build(kind, cells, cache)
    sampled = tie_probe.make_sampler(state)
    prior, warm_count = tie_probe.warm(state, sampled, -0.2, warm_stop, warm_step)
    pitch = float(np.median(np.asarray(state["geometry"].pitch)))
    arms = {
        "restored": make_arm(census_base_step.base_admission_census(), state, prior),
        "repaired": make_arm(repaired_module.census, state, prior),
    }
    states = {"restored": [], "repaired": []}
    for value in np.linspace(lo, hi, count):
        for name, arm in arms.items():
            position, valid, eigenvalues, determinant = arm(
                jnp.asarray(value * state["scale"])
            )
            states[name].append(
                {
                    "offset": float(value),
                    **state_row(position, valid, eigenvalues, determinant, pitch),
                }
            )
    report = {
        "kind": kind,
        "cells": len(state["geometry"].centre),
        "median_pitch": pitch,
        "warm_count": warm_count,
        "window": [lo, hi, count],
        "arms": {},
    }
    for name, rows in states.items():
        departures = []
        for before, after in zip(rows, rows[1:]):
            if after["valid"] < before["valid"]:
                gone = enrich(vanishings(before, after), before)
                departures.append(
                    {
                        "from": before["offset"],
                        "to": after["offset"],
                        "count": [before["valid"], after["valid"]],
                        "vanished": gone,
                    }
                )
        report["arms"][name] = {"states": rows, "departures": departures}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--kind", default="diverted")
    parser.add_argument("--cells", type=int, default=250)
    parser.add_argument("--low", type=float, default=0.16)
    parser.add_argument("--high", type=float, default=0.18)
    parser.add_argument("--count", type=int, default=41)
    parser.add_argument("--warm-stop", type=float, default=0.16)
    parser.add_argument("--warm-step", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True

    report = scan(
        args.kind,
        args.cells,
        args.cache,
        args.low,
        args.high,
        args.count,
        args.warm_stop,
        args.warm_step,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"kind {report['kind']} cells {report['cells']} "
        f"window {report['window']} backend {jax.default_backend()}",
        flush=True,
    )
    for name in ("restored", "repaired"):
        rows = report["arms"][name]["states"]
        print(f"{name} counts " + " ".join(
            f"{row['offset']:.4g}:{row['valid']}" for row in rows
        ), flush=True)
        for departure in report["arms"][name]["departures"]:
            print(
                f"  {name} departure {departure['from']:.6g} -> {departure['to']:.6g} "
                f"count {departure['count']}",
                flush=True,
            )
            for root in departure["vanished"]:
                print(
                    "    vanished det %.4e eig %.4e %.4e nearest_in_state %s "
                    "nearest_after %.4e"
                    % (
                        root["determinant"],
                        root["eigenvalues"][0],
                        root["eigenvalues"][1],
                        f"{root['nearest_in_state']:.4e}"
                        if root["nearest_in_state"] is not None
                        else "none",
                        root["nearest_distance"],
                    ),
                    flush=True,
                )
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()