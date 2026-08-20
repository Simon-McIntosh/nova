"""Build fixed direct-flux matrices for cell centres and hex vertices."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

from nova.frame.polygrid import PolyCell


def load_reference_module():
    path = Path("tests/test_equilibrium_forward_reference.py")
    spec = importlib.util.spec_from_file_location("direct_target_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_dtypes()
    return module


def ordered_offsets(vertices: np.ndarray, centre: np.ndarray) -> np.ndarray:
    offset = vertices - centre
    angle = np.arctan2(offset[:, 1], offset[:, 0])
    return offset[np.argsort(angle)]


def unique_vertices(
    vertices: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    flat = vertices.reshape(-1, 2)
    key = np.rint(flat / tolerance).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return flat[first], inverse.reshape(vertices.shape[:2])


def roundoff_unique(vertices: np.ndarray, epsilon: float) -> np.ndarray:
    unique = []
    for vertex in vertices:
        if not any(np.linalg.norm(vertex - kept) <= epsilon for kept in unique):
            unique.append(vertex)
    return np.asarray(unique)


def tiling_hex_identity(case, cells: int) -> dict[str, object]:
    tiling = PolyCell(case.wall, delta=cells, turn="hex", tile=True)
    exterior = np.asarray(tiling.unitcell.poly.exterior.coords)[:, :2]
    coordinate_scale = max(
        float(np.max(np.abs(exterior))),
        float(np.max(np.abs(tiling.cell_delta))),
        1.0,
    )
    epsilon = 128.0 * np.finfo(np.float64).eps * coordinate_scale
    if np.linalg.norm(exterior[-1] - exterior[0]) > epsilon:
        raise AssertionError("unit-cell exterior lacks its Shapely closing coordinate")
    generator_vertices = roundoff_unique(exterior[:-1], epsilon)
    generator_offsets = ordered_offsets(generator_vertices, np.zeros(2))
    if generator_offsets.shape != (6, 2):
        raise AssertionError(
            "round-off-deduplicated unit-cell exterior must have six vertices"
        )

    width, height = np.asarray(tiling.cell_delta)
    radius = min(width / 2.0, height / np.sqrt(3.0))
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    analytic_vertices = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    analytic_offsets = ordered_offsets(analytic_vertices, np.zeros(2))
    identity_deviation = float(np.max(np.abs(generator_offsets - analytic_offsets)))
    if identity_deviation > epsilon:
        raise AssertionError(
            "deduplicated generator vertices differ from the analytic tiling: "
            f"max deviation {identity_deviation:.17g} m exceeds {epsilon:.17g} m"
        )
    return {
        "offsets": generator_offsets,
        "cell_delta": np.asarray(tiling.cell_delta),
        "raw_vertex_count": len(exterior),
        "generator_vertex_count": len(generator_offsets),
        "deduplication_epsilon": epsilon,
        "identity_max_deviation": identity_deviation,
        "identity_bound": epsilon,
    }


def print_identity(identity: dict[str, object]) -> None:
    print(f"tiling_raw_vertex_count={identity['raw_vertex_count']}")
    print(f"tiling_generator_vertex_count={identity['generator_vertex_count']}")
    print(
        "tiling_vertex_deduplication_epsilon_m="
        f"{identity['deduplication_epsilon']:.17g}"
    )
    print(
        "tiling_vertex_identity_max_deviation_m="
        f"{identity['identity_max_deviation']:.17g}"
    )
    print(f"tiling_vertex_identity_roundoff_bound_m={identity['identity_bound']:.17g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/ring_attribution/inputs/direct-target-matrices.npz"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    reference = load_reference_module()
    case = reference.require_reference()
    identity = tiling_hex_identity(case, reference.SUITE_CELLS)
    print_identity(identity)
    if args.preflight_only:
        print("preflight_identity=pass")
        return
    case, machine = reference._machine(reference.SUITE_CELLS, True)

    six_vertex_rows = np.asarray(
        [
            cell
            for cell, polygon in enumerate(machine.cell_polygons)
            if len(polygon) == 6
        ],
        dtype=np.intp,
    )
    if six_vertex_rows.size == 0:
        raise AssertionError("the authored material grid has no six-vertex rows")
    offsets = np.stack(
        [
            ordered_offsets(machine.cell_polygons[cell], machine.node[cell])
            for cell in six_vertex_rows
        ]
    )
    if offsets.shape[1:] != (6, 2):
        raise AssertionError("six-vertex material rows must carry fixed hex samples")
    measured_canonical = np.median(offsets, axis=0)
    offset_residual = float(np.max(np.abs(offsets - measured_canonical)))
    pitch = float(np.max(np.linalg.norm(measured_canonical, axis=1)))
    roundoff_bound = (
        128.0
        * np.finfo(np.float64).eps
        * max(float(np.max(np.abs(machine.node))), pitch, 1.0)
    )
    offsets_collapsed = offset_residual <= roundoff_bound
    if offsets_collapsed:
        cell_offsets = np.broadcast_to(
            measured_canonical, (len(machine.node), 6, 2)
        ).copy()
        offset_route = "collapsed_six_vertex_material_rows"
        tiling_cell_delta = np.asarray([np.nan, np.nan])
    else:
        tiling_offsets = identity["offsets"]
        cell_offsets = np.broadcast_to(tiling_offsets, (len(machine.node), 6, 2)).copy()
        offset_route = "fixed_per_cell_tiling_parameters"
        tiling_cell_delta = identity["cell_delta"]
    if cell_offsets.shape != (len(machine.node), 6, 2):
        raise AssertionError("tiling offsets must have fixed shape (cells, 6, 2)")
    pitch = float(np.max(np.linalg.norm(cell_offsets, axis=2)))

    cell_vertices = machine.node[:, None, :] + cell_offsets
    vertices, cell_vertex_index = unique_vertices(cell_vertices, roundoff_bound)
    targets = np.vstack([machine.node, vertices])
    cell_sample_index = np.column_stack(
        [
            np.arange(len(machine.node), dtype=np.intp),
            len(machine.node) + cell_vertex_index,
        ]
    )

    machine.coilset.point.attrs = ["Psi", "PsiR", "PsiZ"]
    machine.coilset.point.solve(targets)
    data = machine.coilset.point.data
    source_target = np.asarray(data["Psi"])[:, :-1]
    plasma_target = np.asarray(data["Psi_"])
    plasma_target_r = np.asarray(data["PsiR_"])
    plasma_target_z = np.asarray(data["PsiZ_"])
    combined = np.concatenate(
        [source_target, plasma_target, plasma_target_r, plasma_target_z], axis=1
    )
    if combined.shape[0] != len(targets):
        raise AssertionError("direct-target matrix rows do not span every target")
    if cell_sample_index.shape != (len(machine.node), 7):
        raise AssertionError("every cell must gather one centre and six vertices")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        target_coordinates=targets,
        centre_coordinates=machine.node,
        unique_vertex_coordinates=vertices,
        cell_sample_index=cell_sample_index,
        cell_hex_offsets=cell_offsets,
        combined_target=combined,
        source_current=machine.source_current,
        six_vertex_material_rows=six_vertex_rows,
        canonical_offsets_collapsed=np.asarray(offsets_collapsed),
        offset_route=np.asarray(offset_route),
        tiling_cell_delta_m=tiling_cell_delta,
        tiling_raw_vertex_count=np.asarray(identity["raw_vertex_count"]),
        tiling_generator_vertex_count=np.asarray(identity["generator_vertex_count"]),
        tiling_vertex_deduplication_epsilon_m=np.asarray(
            identity["deduplication_epsilon"]
        ),
        tiling_vertex_identity_max_deviation_m=np.asarray(
            identity["identity_max_deviation"]
        ),
        tiling_vertex_identity_roundoff_bound_m=np.asarray(identity["identity_bound"]),
        canonical_offset_max_deviation_m=np.asarray(offset_residual),
        canonical_offset_roundoff_bound_m=np.asarray(roundoff_bound),
        vertex_tolerance_m=np.asarray(roundoff_bound),
        hex_radius_m=np.asarray(pitch),
    )
    print(f"cells={len(machine.node)}")
    print(f"six_vertex_material_rows={len(six_vertex_rows)}")
    print(f"unique_vertices={len(vertices)}")
    print(f"targets={len(targets)}")
    print(f"samples={cell_sample_index.shape}")
    print(f"combined_matrix={combined.shape}")
    print(f"canonical_offset_max_deviation_m={offset_residual:.17g}")
    print(f"canonical_offset_roundoff_bound_m={roundoff_bound:.17g}")
    print(f"canonical_offsets_collapsed={offsets_collapsed}")
    print(f"offset_route={offset_route}")
    print(f"tiling_cell_delta_m={tiling_cell_delta.tolist()}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
