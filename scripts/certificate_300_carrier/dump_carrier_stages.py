"""Dump every stage of the diverted 300-cell carrier construction to npz.

The carrier is built by ``scripts/analytic_oracle_fixtures/measure.py:build_machine``.
This script replays that build stage by stage, writing each stage's arrays so a
CPU run and an accelerator run can be compared stage by stage.  It is a
diagnostic: it imports the production helpers and never edits them.

Usage: dump_carrier_stages.py <out-dir> [<label>]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    label = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("LABEL", "unknown")
    out.mkdir(parents=True, exist_ok=True)

    from nova.jax.config import configure_dtypes

    configure_dtypes()

    import jax
    import nova

    assert jax.config.jax_enable_x64 is True, "x64 is required for this measurement"

    from scipy.spatial import Delaunay
    from shapely.geometry import LineString

    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium.stencil_mesh import StencilMesh
    from nova.biot.plasmagrid import PlasmaGrid
    from nova.frame.coilset import CoilSet
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture

    revision = os.environ.get("NOVA_REVISION", "unstamped")
    print(
        f"# label={label} revision={revision} nova.__file__={nova.__file__} "
        f"cwd={Path.cwd()}",
        flush=True,
    )
    print(
        f"# jax_platforms={os.environ.get('JAX_PLATFORMS')} "
        f"backend={jax.default_backend()} x64={jax.config.jax_enable_x64}",
        flush=True,
    )

    report: dict[str, object] = {"label": label, "revision": revision,
                                 "backend": jax.default_backend()}

    # --- stage 0: the analytic reference and the wall -----------------------
    case_name = "diverted-single-null"
    requested_cells = -300
    _carrier_case, source_case, exact = certificate._case(case_name)
    wall = certificate._diverted_wall(exact)
    np.savez(
        out / "s0_wall.npz",
        wall=wall,
        coefficients=np.asarray(exact.coefficients, dtype=np.float64),
        magnetic_axis=np.asarray(exact.magnetic_axis, dtype=np.float64),
        x_point=np.asarray(exact.x_point, dtype=np.float64),
        separatrix=np.asarray(exact.separatrix(1441), dtype=np.float64),
    )
    report["s0_wall_nodes"] = int(len(wall))
    print(f"STAGE s0_wall nodes={len(wall)}", flush=True)

    # --- stage 1: the hex CoilSet and the wall insert -----------------------
    coilset = CoilSet(dplasma=requested_cells, tplasma="hex")
    coilset.firstwall.insert(wall, turn="hex")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    centres = np.c_[
        np.asarray(coilset.subframe.loc[plasma, "x"], dtype=np.float64),
        np.asarray(coilset.subframe.loc[plasma, "z"], dtype=np.float64),
    ]
    sections = np.asarray(coilset.aloc["plasma", "section"], dtype=object).astype(str)
    dl = np.asarray(coilset.aloc["plasma", "dl"], dtype=float)
    dt = np.asarray(coilset.aloc["plasma", "dt"], dtype=float)
    np.savez(
        out / "s1_coilset.npz",
        plasma_mask=plasma,
        centres=centres,
        x=np.asarray(coilset.subframe.loc[plasma, "x"], dtype=np.float64),
        z=np.asarray(coilset.subframe.loc[plasma, "z"], dtype=np.float64),
        dl=dl,
        dt=dt,
        sections=sections,
    )
    report["s1_subframe_rows"] = int(len(coilset.subframe))
    report["s1_plasma_cells"] = int(plasma.sum())
    report["s1_dl_unique"] = sorted(set(np.round(dl, 12).tolist()))
    report["s1_dt_unique"] = sorted(set(np.round(dt, 12).tolist()))
    print(
        f"STAGE s1_coilset subframe={len(coilset.subframe)} cells={int(plasma.sum())} "
        f"dl={report['s1_dl_unique']} dt={report['s1_dt_unique']}",
        flush=True,
    )

    # --- stage 2: polygons and area ----------------------------------------
    polygons = tuple(
        np.asarray(item.poly.exterior.coords)[:-1, :2] for item in material
    )
    cleaned = tuple(oracle_fixture._clean_vertices(p) for p in polygons)
    area = np.asarray([item.poly.area for item in material], dtype=np.float64)
    vertex_counts = np.asarray([len(p) for p in cleaned], dtype=np.int64)
    np.savez(
        out / "s2_polygons.npz",
        area=area,
        vertex_counts=vertex_counts,
        polygon_offsets=np.concatenate(
            ([0], np.cumsum(vertex_counts))
        ).astype(np.int64),
        polygon_vertices=np.concatenate(cleaned),
        area_sum=area.sum(),
    )
    report["s2_vertex_counts_hist"] = {
        str(int(k)): int(v) for k, v in zip(*np.unique(vertex_counts, return_counts=True))
    }
    report["s2_area_sum"] = float(area.sum())
    print(
        f"STAGE s2_polygons vertex_hist={report['s2_vertex_counts_hist']} "
        f"area_sum={area.sum():.17g}",
        flush=True,
    )

    # --- stage 3: the Delaunay triangulation -------------------------------
    triangulation = Delaunay(centres)
    np.savez(
        out / "s3_delaunay.npz",
        centres=centres,
        simplices=np.asarray(triangulation.simplices, dtype=np.int64),
        neighbours=np.asarray(triangulation.neighbors, dtype=np.int64),
        vertex_neighbor_indices=np.asarray(
            triangulation.vertex_neighbor_vertices[0], dtype=np.int64
        ),
        vertex_neighbor_values=np.asarray(
            triangulation.vertex_neighbor_vertices[1], dtype=np.int64
        ),
        coplanar=np.asarray(triangulation.coplanar),
    )
    report["s3_simplex_count"] = int(len(triangulation.simplices))
    report["s3_coplanar"] = int(len(triangulation.coplanar))
    print(
        f"STAGE s3_delaunay simplices={len(triangulation.simplices)} "
        f"coplanar={len(triangulation.coplanar)}",
        flush=True,
    )

    # --- stage 4: boundary cells -------------------------------------------
    boundary = LineString(wall)
    boundary_cells = np.asarray(
        [
            position
            for position, item in enumerate(material)
            if item.poly.intersects(boundary)
        ]
    )
    np.savez(out / "s4_boundary_cells.npz", boundary_cells=boundary_cells)
    report["s4_boundary_cells"] = int(len(boundary_cells))
    print(f"STAGE s4_boundary_cells count={len(boundary_cells)}", flush=True)

    # --- stage 5: the neighbour stencil ------------------------------------
    stencil, _ = PlasmaGrid.loop_neighbour_vertices(
        centres, triangulation.vertex_neighbor_vertices, boundary_cells
    )
    ring = oracle_fixture.ring_condition(centres, stencil)
    regular = np.asarray([len(polygon) == 6 for polygon in polygons])
    interior = stencil[(ring < 1.0e3) & regular[stencil].all(axis=1)]
    np.savez(
        out / "s5_stencil.npz",
        stencil=np.asarray(stencil, dtype=np.int64),
        ring_condition=np.asarray(ring, dtype=np.float64),
        regular=regular,
        interior_stencil=np.asarray(interior, dtype=np.int64),
    )
    report["s5_stencil_shape"] = list(np.shape(stencil))
    report["s5_interior_rows"] = int(len(interior))
    print(
        f"STAGE s5_stencil shape={np.shape(stencil)} interior_rows={len(interior)}",
        flush=True,
    )

    # --- stage 6: the mesh and the sampling vertices -----------------------
    mesh = StencilMesh(centres, stencil, area)
    full = np.flatnonzero(sections == "hexagon")
    if len(full) == 0:
        raise ValueError("the oracle mesh has no complete hex generator")
    dimensions = np.c_[dl[full], dt[full]]
    width, height = dimensions[0]
    radius = min(width / 2.0, height / np.sqrt(3.0))
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    sampling = centres[:, None, :] + offsets[None, :, :]
    np.savez(
        out / "s6_sampling.npz",
        dimensions=dimensions,
        radius=np.asarray([radius]),
        offsets=offsets,
        sampling=sampling,
    )
    report["s6_radius"] = float(radius)
    report["s6_sampling_shape"] = list(sampling.shape)
    print(
        f"STAGE s6_sampling shape={sampling.shape} radius={radius:.17g}",
        flush=True,
    )

    # --- stage 7: the moment geometry and its atomic sampling mesh ---------
    from nova.equilibrium.forward_operator import MomentGeometry

    geometry = MomentGeometry.from_cells(mesh, cleaned, sampling_vertices=sampling)
    sample = geometry.sample_node_coordinates
    np.savez(
        out / "s7_moment_geometry.npz",
        sample_coordinates=np.asarray(sample, dtype=np.float64),
        atomic_node_coordinates=np.asarray(
            geometry.atomic_mesh.node_coordinates, dtype=np.float64
        ),
        atomic_centroids=np.asarray(
            geometry.atomic_mesh.centroids, dtype=np.float64
        ),
        atomic_cell_nodes=np.asarray(geometry.atomic_mesh.cell_nodes, dtype=np.int64),
        atomic_cell_vertex_count=np.asarray(geometry.atomic_mesh.cell_vertex_count, dtype=np.int64),
        atomic_tolerance=np.asarray([geometry.atomic_mesh.tolerance]),
    )
    report["s7_sample_coordinates"] = int(len(sample))
    report["s7_atomic_nodes"] = int(len(geometry.atomic_mesh.node_coordinates))
    report["s7_atomic_cells"] = int(len(geometry.atomic_mesh.centroids))
    print(
        f"STAGE s7_moment sample_coordinates={len(sample)} "
        f"atomic_nodes={len(geometry.atomic_mesh.node_coordinates)} "
        f"atomic_cells={len(geometry.atomic_mesh.centroids)}",
        flush=True,
    )

    # --- stage 8: the assembled plasma-to-target blocks --------------------
    builder = oracle_fixture.resolve_builder()
    assembled = oracle_fixture._assemble_families(
        {"grid": centres, "wall": wall, "sample": sample},
        cleaned,
        geometry.atomic_mesh.centroids,
        builder=builder,
    )
    payload: dict[str, np.ndarray] = {}
    for family in ("grid", "wall", "sample"):
        for index, name in enumerate(("plain", "radial", "vertical")):
            payload[f"{family}_{name}"] = np.asarray(assembled[family][index], dtype=np.float64)
    np.savez(out / "s8_assembled.npz", **payload)
    report["s8_builder"] = builder
    report["s8_shapes"] = {k: list(v.shape) for k, v in payload.items()}
    print(f"STAGE s8_assembled builder={builder}", flush=True)

    # --- operator-level node count: grid + wall + sample -------------------
    report["total_nodes"] = (
        int(len(centres)) + int(len(wall)) + int(len(sample))
    )
    print(
        f"TOTAL grid={len(centres)} wall={len(wall)} sample={len(sample)} "
        f"node_number={report['total_nodes']}",
        flush=True,
    )

    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print("EXIT=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())