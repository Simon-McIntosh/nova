"""Integrate local saddle responses along a declared saved-field homotopy.

The path is the straight line between saved fields, with no forward solve.
Its integrated local Hessian responses are path-dependent diagnostic shares.
The reference-point linear prediction remains in the attribution receipt, so
its error is visible instead of being absorbed by a fitted multiplier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.diiid_xpoint_attribution.attribute import (
    field,
    gradient,
    saddle,
    write,
)


def path_response(data, result, count):
    """Integrate -H(t)^-1 grad(dPhi) without changing the saved currents."""
    radius, height = data["radius"], data["height"]
    reference, solved = data["reference"], data["solved"]
    names = ["profiles", "total_current", "external"] + (
        ["plasma_response", "boundary_shadow"]
        if "boundary_shadow" in data
        else ["redistribution_closure"]
    )
    fields = {name: field(radius, height, data[name]) for name in names}
    nodes, weights = np.polynomial.legendre.leggauss(count)
    nodes, weights = (nodes + 1) / 2, weights / 2
    shares = {name: np.zeros(2) for name in names}
    point = result["reference_saddle_m"]
    trajectory = []
    for fraction, weight in zip(nodes, weights, strict=True):
        flux = reference + fraction * (solved - reference)
        point, hessian = saddle(radius, height, flux, point)
        for name, interpolant in fields.items():
            shares[name] += weight * -np.linalg.solve(
                hessian, gradient(interpolant, point)
            )
        trajectory.append(
            dict(
                fraction=fraction,
                saddle_m=point,
                condition=np.linalg.cond(hessian),
            )
        )
    return dict(
        shifts_m=shares,
        total_m=sum(shares.values()),
        trajectory=trajectory,
        quadrature_nodes=count,
    )


def analyze(folder, frame):
    data = dict(np.load(folder / f"fields-{frame}.npz"))
    result = json.loads((folder / f"result-{frame}.json").read_text())
    paths = [path_response(data, result, count) for count in (16, 32)]
    path = paths[-1]
    change = {
        name: np.asarray(value) - paths[0]["shifts_m"][name]
        for name, value in path["shifts_m"].items()
    }
    closure = path["total_m"] - np.asarray(result["measured_offset_cm"]) / 100
    if max(np.linalg.norm(value) for value in change.values()) > 1e-6:
        raise ValueError("saddle attribution quadrature has not stabilized")
    if np.linalg.norm(closure) > 1e-6:
        raise ValueError("attribution does not close against the measured offset")
    path["quadrature_change_m"] = change
    path["measured_closure_m"] = closure
    write(folder / f"sensitivity-{frame}.json", dict(path=path))
    print(
        "INTEGRATED_SHARES",
        frame,
        json.dumps(
            {name: (100 * value).tolist() for name, value in path["shifts_m"].items()}
        ),
        "closure_cm",
        (100 * closure).tolist(),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    args = parser.parse_args()
    for frame in (137, 146):
        analyze(args.measurement, frame)


if __name__ == "__main__":
    main()
