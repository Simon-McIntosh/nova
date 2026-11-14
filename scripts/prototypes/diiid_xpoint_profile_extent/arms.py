"""Declare the prescribed p'/FF' tables each profile-extent arm carries.

Both arms read the same EFIT map through the same affine flux-function
extraction. The base arm carries the established nineteen knots on 0.05-0.95,
where the sampled-function end cap takes the profile to zero at the separatrix.
The edge arm keeps those nineteen knots bit-identical and appends knots the map
supports on 0.95-1.0, extracted on shells that lie in that interval alone, so
the comparison changes the edge region and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE_SURFACES = tuple(round(0.05 + 0.05 * index, 3) for index in range(19))
EDGE_SURFACES = tuple(round(0.955 + 0.005 * index, 3) for index in range(10))
EDGE_EXTRACTION_SURFACES = BASE_SURFACES[-1:] + EDGE_SURFACES
ARMS = ("base", "edge")


def surfaces(arm: str) -> tuple[float, ...]:
    """Return the knots the named arm publishes."""

    if arm == "base":
        return BASE_SURFACES
    if arm == "edge":
        return BASE_SURFACES + EDGE_SURFACES
    raise ValueError(f"unknown profile arm {arm!r}")


def pristine(bench):
    """Return the build's own table reader, before any arm replaced it."""

    return getattr(bench, "_profile_extent_pristine_label_state", bench._label_state)


def label_state(bench, arm: str):
    """Return a reader of the arm's prescribed tables for one profile build.

    The base table is the build's own, so the interior stays bit-identical
    between arms. The edge continuation comes from a second extraction whose
    shells all lie in 0.95-1.0, and its first knot is dropped because the base
    table already carries 0.95.
    """

    original_label_state = pristine(bench)
    original_extract = bench.extract_flux_functions

    def edge_extraction(*args, **kwargs):
        return original_extract(*args, surfaces=EDGE_EXTRACTION_SURFACES, **kwargs)

    def read(row, frame):
        label, nodes, p_prime, ff_prime = original_label_state(row, frame)
        if arm == "base":
            return label, nodes, p_prime, ff_prime
        bench.extract_flux_functions = edge_extraction
        try:
            _, edge_nodes, edge_p, edge_ff = original_label_state(row, frame)
        finally:
            bench.extract_flux_functions = original_extract
        keep = np.asarray(edge_nodes) > float(np.max(nodes))
        return (
            label,
            np.r_[nodes, np.asarray(edge_nodes)[keep]],
            np.r_[p_prime, np.asarray(edge_p)[keep]],
            np.r_[ff_prime, np.asarray(edge_ff)[keep]],
        )

    return read


def install(bench, arm: str):
    """Prescribe the arm's tables on the benchmark's profile build."""

    setattr(bench, "_profile_extent_pristine_label_state", pristine(bench))
    bench._label_state = label_state(bench, arm)
    return bench._label_state


def describe(arm: str) -> dict:
    """Return the arm's declared surface set for a receipt."""

    return {"arm": arm, "surfaces": list(surfaces(arm))}


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")