"""Measure ITER winding and passive-ring coupling from its machine description."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imas
import numpy as np

from nova.circuit.conductor import (
    ConductorSet,
    PolygonSection,
    subdivide_passive_thickness,
)
from nova.circuit.linkage import (
    circuit_linkage_matrix,
    guard_positive_definite,
    preserve_series_terminal_linkage,
)
from nova.circuit.passive import terminal_impedance_spectrum
from nova.imas.pulsedesign import ITER


ROOT = Path("/work/imas/shared/imasdb/ITER_MACHINE_DESCRIPTION/3")
SOURCE = (
    (111001, 204, "pf_active"),
    (115004, 6, "pf_passive"),
    (124001, 3, "pf_passive"),
)
FREQUENCY = np.geomspace(0.1, 5000.0, 21)


def read_ids(pulse: int, run: int, name: str):
    with imas.DBEntry(f"imas:hdf5?path={ROOT / str(pulse) / str(run)}", "r") as entry:
        return entry.get(name, autoconvert=False)


def source_conductors() -> tuple[
    ConductorSet, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Read winding sections and each source passive ring in its written DD."""
    active, vessel, cryostat = (read_ids(*source) for source in SOURCE)
    coils = ITER.winding_sections(active)
    columns = {key: [] for key in ConductorSet._columns}
    polygons = []

    def append(r, z, dr, dz, share, circuit):
        for key, value in zip(columns, (r, z, dr, dz, share, circuit), strict=True):
            columns[key].append(value)

    coil_names = []
    coil_owner = []
    for index, coil in enumerate(coils):
        coil_names.append(coil.name)
        for section in coil.elements:
            circuit = len(coil_owner)
            coil_owner.append(index)
            if section.section.name == "rectangle":
                append(
                    *section.centre,
                    section.width,
                    section.height,
                    section.turns,
                    circuit,
                )
            elif section.section.name == "annulus":
                data = section.section.data
                area = np.pi * (
                    float(data["radius_outer"]) ** 2 - float(data["radius_inner"]) ** 2
                )
                side = np.sqrt(area)
                append(*section.centre, side, side, section.turns, circuit)
            else:
                raise ValueError(f"unsupported winding section {section.section.name}")
    passive_thickness = []
    passive_resistance = []
    passive_name = []
    for ids in (vessel, cryostat):
        for loop in ids.loop:
            if len(loop.element) != 1:
                raise ValueError(f"passive loop {loop.name} has multiple elements")
            element = loop.element[0]
            circuit = len(coil_owner) + len(passive_thickness)
            kind = int(element.geometry.geometry_type)
            if kind == 6:
                line = element.geometry.thick_line
                start = np.array([float(line.first_point.r), float(line.first_point.z)])
                end = np.array([float(line.second_point.r), float(line.second_point.z)])
                thickness = float(line.thickness)
                tangent = end - start
                length = float(np.linalg.norm(tangent))
                normal = thickness * np.array([-tangent[1], tangent[0]]) / (2 * length)
                vertices = np.array(
                    [start + normal, end + normal, end - normal, start - normal]
                )
                polygons.append(PolygonSection(circuit, vertices))
                span = np.ptp(vertices, axis=0)
                append(*(start + end) / 2, float(span[0]), float(span[1]), 1.0, circuit)
            elif kind == 5:
                annulus = element.geometry.annulus
                area = np.pi * (
                    float(annulus.radius_outer) ** 2 - float(annulus.radius_inner) ** 2
                )
                side = np.sqrt(area)
                thickness = float(annulus.radius_outer) - float(annulus.radius_inner)
                append(float(annulus.r), float(annulus.z), side, side, 1.0, circuit)
            elif kind == 2:
                rect = element.geometry.rectangle
                thickness = min(abs(float(rect.width)), abs(float(rect.height)))
                append(
                    float(rect.r),
                    float(rect.z),
                    abs(float(rect.width)),
                    abs(float(rect.height)),
                    1.0,
                    circuit,
                )
            else:
                raise ValueError(f"unsupported passive section {kind} in {loop.name}")
            passive_thickness.append(thickness)
            passive_resistance.append(
                float(loop.resistance) if loop.resistance.has_value else np.nan
            )
            passive_name.append(str(loop.name))
    if not np.all(np.isfinite(passive_resistance)):
        raise ValueError("passive source resistance is missing")
    return (
        ConductorSet(**columns, polygon_sections=tuple(polygons)),
        coil_names,
        np.asarray(coil_owner, dtype=np.int64),
        np.asarray(passive_thickness),
        np.asarray(passive_resistance),
        np.asarray(passive_name),
    )


def canonical_coil_reference(active) -> np.ndarray:
    """Compute the whole-coil matrix through Nova's active-coil owner."""
    from nova.imas.machine import PoloidalFieldActive

    converted = imas.convert_ids(
        active, str(active.ids_properties.version_put.data_dictionary)
    )
    for source, target in zip(active.coil, converted.coil, strict=True):
        target.identifier = str(source.name)
        target.name = str(source.name)
    converted.circuit.resize(0)
    converted.supply.resize(0)
    model = PoloidalFieldActive(ids=converted, run=204)
    model.inductance.solve(0)
    return np.asarray(model.inductance.Psi, dtype=np.float64)


def calculate(
    max_thickness: float,
    section_n_max: int,
    *,
    terminal_reference: np.ndarray | None = None,
    raw_linkage: bool = False,
) -> dict:
    conductors, coil_names, owner, thickness, source_resistance, passive_name = (
        source_conductors()
    )
    n_section = len(owner)
    source_passive = np.arange(n_section, n_section + len(thickness))
    refined = subdivide_passive_thickness(
        conductors, source_passive, thickness, max_thickness
    )
    circuits = [*range(n_section), *refined.circuits]
    matrix = circuit_linkage_matrix(
        refined.conductors, circuits, section_n_max=section_n_max
    )
    series = np.zeros((n_section, len(coil_names)))
    series[np.arange(n_section), owner] = 1.0
    raw_coil_matrix = series.T @ matrix[:n_section, :n_section] @ series
    active = read_ids(*SOURCE[0])
    source_coil_names = [
        str(coil.name).rsplit("(", 1)[-1].removesuffix(")") for coil in active.coil
    ]
    if source_coil_names != coil_names:
        raise ValueError("active resistance and section coil order disagree")
    coil_resistance = np.asarray(
        [
            float(coil.resistance) if coil.resistance.has_value else np.nan
            for coil in active.coil
        ]
    )
    if np.any(~np.isfinite(coil_resistance)):
        raise ValueError("active source resistance is missing")
    if not raw_linkage:
        reference = (
            canonical_coil_reference(active)
            if terminal_reference is None
            else np.asarray(terminal_reference, dtype=np.float64)
        )
        matrix[:n_section, :n_section] = preserve_series_terminal_linkage(
            matrix[:n_section, :n_section], series, reference
        )
    coil_matrix = series.T @ matrix[:n_section, :n_section] @ series
    passive_matrix = guard_positive_definite(matrix[n_section:, n_section:])
    resistance = (
        source_resistance[refined.parent - n_section] * refined.resistance_factor
    )
    screened = terminal_impedance_spectrum(
        matrix[:n_section, :n_section],
        matrix[:n_section, n_section:],
        passive_matrix,
        resistance,
        series,
        FREQUENCY,
        terminal_resistance=coil_resistance,
    )
    bare = terminal_impedance_spectrum(
        matrix[:n_section, :n_section],
        np.empty((n_section, 0)),
        np.empty((0, 0)),
        np.empty(0),
        series,
        FREQUENCY,
        terminal_resistance=coil_resistance,
    )
    expected_bare = (
        coil_resistance[None, :]
        + 2j * np.pi * FREQUENCY[:, None] * np.diag(coil_matrix)[None, :]
    )
    bare_error = float(
        np.max(np.abs(np.diagonal(bare, axis1=1, axis2=2) / expected_bare - 1))
    )
    if bare_error > 1e-12:
        raise AssertionError(f"passive removal differs from bare by {bare_error}")
    return {
        "coil_names": coil_names,
        "section_count": n_section,
        "passive_count": len(refined.circuits),
        "source_passive_count": len(thickness),
        "max_source_thickness_m": float(thickness.max()),
        "max_layer_thickness_m": max_thickness,
        "source_resistance_ohm": [
            float(source_resistance.min()),
            float(source_resistance.max()),
        ],
        "coil_inductance_h": coil_matrix.tolist(),
        "raw_coil_inductance_h": raw_coil_matrix.tolist(),
        "coil_resistance_ohm": coil_resistance.tolist(),
        "terminal_reference_applied": not raw_linkage,
        "frequency_hz": FREQUENCY.tolist(),
        "screened_real_ohm": np.diagonal(screened.real, axis1=1, axis2=2).tolist(),
        "screened_imag_ohm": np.diagonal(screened.imag, axis1=1, axis2=2).tolist(),
        "bare_imag_ohm": np.diagonal(bare.imag, axis1=1, axis2=2).tolist(),
        "bare_real_ohm": np.diagonal(bare.real, axis1=1, axis2=2).tolist(),
        "bare_error": bare_error,
        "passive_names": list(sorted(set(passive_name))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-thickness", type=float, required=True)
    parser.add_argument("--section-n-max", type=int, default=3)
    parser.add_argument("--terminal-reference", type=Path)
    parser.add_argument("--raw-linkage", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = None
    if args.terminal_reference is not None:
        reference = np.asarray(
            json.loads(args.terminal_reference.read_text())["coil_inductance_h"]
        )
    result = calculate(
        args.max_thickness,
        args.section_n_max,
        terminal_reference=reference,
        raw_linkage=args.raw_linkage,
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(
        f"SUMMARY sections={result['section_count']} passive={result['passive_count']} "
        f"max_layer_m={args.max_thickness} bare_error={result['bare_error']:.3g}"
    )


if __name__ == "__main__":
    main()
