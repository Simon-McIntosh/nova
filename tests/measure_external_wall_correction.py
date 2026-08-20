"""Measure the authored-section correction on external wall rows only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from nova.biot.biotframe import Source, Target
from nova.biot.field import Sample
from nova.biot.solve import Solve
from nova.frame.coil import Coil
from nova.frame.coilset import CoilSet

from measure_exact_section_routes import baseline_elements, load_module


REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
WALL_MULTIPLIERS = {"coarse": 1, "fine": 2}


def evaluate_route(
    reference,
    case,
    wall: np.ndarray,
    polygon_element,
    rectangle_element,
    *,
    qualify_authored_sections: bool,
) -> tuple[Source, dict[str, np.ndarray], float]:
    """Build and couple the external source frame through one section route."""
    original_polygon = Solve.generator["polysection"]
    original_rectangle = Solve.generator["cylinder"]
    original_route = Coil._route_authored_sections
    Solve.generator["polysection"] = polygon_element
    Solve.generator["cylinder"] = rectangle_element
    if not qualify_authored_sections:
        Coil._route_authored_sections = lambda self, index: None
    try:
        coilset = CoilSet(
            dcoil=reference.COIL_FILAMENTS,
            dplasma=10,
            tplasma="hex",
            nwall=reference.WALL_NODES,
        )
        for conductor in case.active:
            coilset.coil.insert(
                *conductor.placement,
                nturn=conductor.turns,
                part="pf",
                name=conductor.name,
            )
        for conductor in case.passive:
            coilset.coil.insert(
                *conductor.placement,
                nturn=conductor.turns,
                part="passive",
                name=conductor.name,
                delta=reference.PASSIVE_ELEMENTS,
            )
        source = Source(coilset.subframe)
        coupling, seconds = wall_coupling(source, wall)
        return source, coupling, seconds
    finally:
        Solve.generator["polysection"] = original_polygon
        Solve.generator["cylinder"] = original_rectangle
        Coil._route_authored_sections = original_route


def wall_coupling(
    source: Source, wall: np.ndarray
) -> tuple[dict[str, np.ndarray], float]:
    """Return exact external coupling components at fixed wall coordinates."""
    target = Target({"x": wall[:, 0], "z": wall[:, 1]}, label="WallCorrection")
    started = time.perf_counter()
    solved = Solve(
        source,
        target,
        reduce=[True, False],
        attrs=["Psi", "Br", "Bz"],
        name="external-wall-correction",
    ).data
    elapsed = time.perf_counter() - started
    return {name: np.asarray(solved[name]) for name in ("Psi", "Br", "Bz")}, elapsed


def component_summary(
    before: np.ndarray,
    after: np.ndarray,
    current: np.ndarray,
    coordinates: np.ndarray,
    unit: str,
) -> dict[str, object]:
    """Summarize matrix and actual-current changes for one field component."""
    difference = after - before
    driven = difference @ current
    index = int(np.argmax(np.abs(driven)))
    return {
        "unit": unit,
        "matrix_sup_per_a": float(np.max(np.abs(difference), initial=0.0)),
        "matrix_changed_entries": int(np.count_nonzero(difference)),
        "actual_current_sup": float(abs(driven[index])),
        "actual_current_signed_at_sup": float(driven[index]),
        "actual_current_argmax_row": index,
        "actual_current_argmax_rz_m": coordinates[index].tolist(),
    }


def route_counts(source: Source) -> dict[str, int]:
    """Return JSON-native source counts by exact section route."""
    labels, counts = np.unique(
        np.asarray(source["segment"], dtype=str), return_counts=True
    )
    return {str(label): int(count) for label, count in zip(labels, counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", choices=tuple(WALL_MULTIPLIERS))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reference = load_module(REFERENCE_PATH, "external_wall_reference_fixture")
    reference.configure_dtypes()
    case = reference.require_reference()
    multiplier = WALL_MULTIPLIERS[args.fixture]
    wall_sample = Sample(case.wall, delta=-(reference.WALL_NODES * multiplier))
    coordinates = np.c_[wall_sample["radius"], wall_sample["height"]]
    baseline_polygon, baseline_rectangle = baseline_elements()
    before_source, before, before_seconds = evaluate_route(
        reference,
        case,
        coordinates,
        baseline_polygon,
        baseline_rectangle,
        qualify_authored_sections=False,
    )
    after_source, after, after_seconds = evaluate_route(
        reference,
        case,
        coordinates,
        Solve.generator["polysection"],
        Solve.generator["cylinder"],
        qualify_authored_sections=True,
    )
    conductors = case.drive(True)
    expected = [conductor.name for conductor in conductors]
    before_order = before_source.subspace.index.to_list()
    after_order = after_source.subspace.index.to_list()
    if before_order != expected or after_order != expected:
        raise AssertionError("external source order differs from fixture drive order")
    current = np.asarray([conductor.current for conductor in conductors])
    components = {
        "flux": component_summary(
            before["Psi"], after["Psi"], current, coordinates, "Wb"
        ),
        "radial_field": component_summary(
            before["Br"], after["Br"], current, coordinates, "T"
        ),
        "vertical_field": component_summary(
            before["Bz"], after["Bz"], current, coordinates, "T"
        ),
    }
    radial = (after["Br"] - before["Br"]) @ current
    vertical = (after["Bz"] - before["Bz"]) @ current
    magnitude = np.hypot(radial, vertical)
    vector_index = int(np.argmax(magnitude))
    report = {
        "fixture": args.fixture,
        "wall_rows": len(coordinates),
        "source_elements": len(after_source),
        "before_route_counts": route_counts(before_source),
        "after_route_counts": route_counts(after_source),
        "wall_seconds": {"before": before_seconds, "after": after_seconds},
        "components": components,
        "poloidal_field_vector": {
            "unit": "T",
            "actual_current_sup": float(magnitude[vector_index]),
            "actual_current_argmax_row": vector_index,
            "actual_current_argmax_rz_m": coordinates[vector_index].tolist(),
        },
        "interpretation": (
            "Only external conductor coupling is evaluated. It is independent of "
            "the plasma source mesh; the fixture name changes only the fixed wall "
            "sampling density."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["components"]["flux"], sort_keys=True))


if __name__ == "__main__":
    main()
