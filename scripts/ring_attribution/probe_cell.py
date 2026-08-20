"""Reproduce one ring cell's current, centroid, and first moments."""

import argparse
import json
from pathlib import Path

import numpy as np

from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments


parser = argparse.ArgumentParser()
parser.add_argument("cell", type=int)
parser.add_argument(
    "--bank",
    type=Path,
    default=Path("scripts/ring_attribution/results/ring-attribution-fields.npz"),
)
args = parser.parse_args()
with np.load(args.bank) as stored:
    bank = {name: stored[name] for name in stored.files}
cell = args.cell
if not bank["ring_mask"][cell]:
    raise ValueError(f"cell {cell} is not in the ring-incomplete population")
currents, first_fit = padded_polynomial_current_moments(
    bank["support_vertices"],
    bank["support_vertex_count"],
    bank["fit_centres"],
    bank["coordinate_scale"],
    bank["coefficients"],
)
currents = np.asarray(currents)
first_moments = np.asarray(first_fit) + currents[:, None] * (
    bank["fit_centres"] - bank["moment_centres"]
)
current = float(currents[cell])
first = first_moments[cell]
centroid = bank["moment_centres"][cell] + first / current if current else None
print(
    json.dumps(
        {
            "cell": cell,
            "current_a": current,
            "current_centroid_m": None if centroid is None else centroid.tolist(),
            "first_moments_am": first.tolist(),
        },
        indent=2,
    )
)
