"""Independently count contour-tree superlevel components on stored fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RectBivariateSpline

from nova.equilibrium.contour_tree import ContourTreeResult, build_contour_tree
from nova.equilibrium.contour_tree_mesh import ContourMesh, build_contour_mesh
from nova.equilibrium.wall_mask import vessel_unit
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.media.sources.mast_efit import read_frame, read_geometry
from nova.media.sources.plasma_mesh import hex_mesh


ROOT = Path(__file__).resolve().parents[1]
PART_ROOT = ROOT / "docs/figures/gs-absolute-accuracy/solovev/production-route-parts"
FIGURE_ROOT = ROOT / "docs/figures/contour-tree-topology-authority/brute-force"
VERTEX_CAPACITY = 256
EDGE_CAPACITY = 2048
TRIANGLE_CAPACITY = 1024

MAST_CELLS = 400
MAST_VERTEX_CAPACITY = 8192
MAST_EDGE_CAPACITY = 65536
MAST_TRIANGLE_CAPACITY = 32768
MAST_ROWS = ((27079, 16), (22475, 50))


@dataclass(frozen=True)
class Fixture:
    """One stored terminal field and its material boundary."""

    name: str
    mesh: ContourMesh


def certificate_fixtures() -> tuple[Fixture, ...]:
    """Read every compact persisted Solov'ev certificate terminal state."""

    fixtures = []
    for path in sorted(PART_ROOT.glob("*production-route-reduced.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        render = payload.get("render_data")
        if not render:
            continue
        count = int(payload["realised_cells"])
        coordinate = np.asarray(render["coordinates_rz_m"], dtype=np.float64)[:count]
        flux = np.asarray(render["terminal_flux_wb"], dtype=np.float64)[:count]
        wall = np.asarray(render["wall_units_rz_m"][0], dtype=np.float64)
        mesh = build_contour_mesh(
            coordinate,
            flux,
            [vessel_unit(wall[:, 0], wall[:, 1], name=payload["case"])],
            vertex_capacity=VERTEX_CAPACITY,
            edge_capacity=EDGE_CAPACITY,
            triangle_capacity=TRIANGLE_CAPACITY,
        )
        if mesh.overflow:
            raise RuntimeError(f"certificate mesh capacity refused: {path.name}")
        fixtures.append(Fixture(payload["case"], mesh))
    return tuple(fixtures)


def _mast_mesh(shot: int, row: int) -> ContourMesh:
    """Build one MAST hex carrier with the stored EFIT flux on its centres.

    The field is the stored EFIT reconstruction (``efm/psirz``, converted to
    total webers at the read) sampled at the hex-cell centres with a bicubic
    spline; it is a reconstruction, not a forward state.  ``sigma`` is chosen
    so the reconstructed magnetic axis is a maximum of ``sigma * psi``.
    """
    import zarr

    group = zarr.open_group(str(Path(SHOT_STORE) / f"{shot}.zarr"), mode="r")["efm"]
    geometry = read_geometry(group)
    frame = read_frame(group, row)
    outlines, _ = hex_mesh(geometry.limiter, cells=MAST_CELLS)
    centres = np.array([[np.mean(o[:, 0]), np.mean(o[:, 1])] for o in outlines])
    spline = RectBivariateSpline(
        np.asarray(frame.height, dtype=float),
        np.asarray(frame.radius, dtype=float),
        np.asarray(frame.flux, dtype=float),
        kx=3,
        ky=3,
    )
    psi = spline.ev(centres[:, 1], centres[:, 0])
    sigma = 1 if frame.flux_axis > frame.flux_boundary else -1
    wall = (
        vessel_unit(geometry.limiter[:, 0], geometry.limiter[:, 1], name=str(shot)),
    )
    return build_contour_mesh(
        centres,
        sigma * psi,
        wall,
        vertex_capacity=MAST_VERTEX_CAPACITY,
        edge_capacity=MAST_EDGE_CAPACITY,
        triangle_capacity=MAST_TRIANGLE_CAPACITY,
    )


def tree_fits(mesh: ContourMesh) -> bool:
    """Whether the carrier fits the tree's declared fixed node capacity."""

    return int(np.sum(np.asarray(mesh.vertex_valid))) <= ContourTreeResult.node_capacity


def mast_fixtures() -> tuple[Fixture, ...]:
    """Read the MAST rows named in the plan's primary-selection section."""

    return tuple(
        Fixture(f"mast-{shot}-row-{row}", _mast_mesh(shot, row))
        for shot, row in MAST_ROWS
    )


def standalone_rows(mesh: ContourMesh) -> list[dict[str, float | int]]:
    """Count superlevel components by plain host graph search, with no tree."""

    values = np.asarray(mesh.vertex_psi)[np.asarray(mesh.vertex_valid)]
    levels = np.nextafter(np.unique(values), -np.inf)[::-1]
    return [
        {"level": float(level), "brute_force": _component_count(mesh, float(level))}
        for level in levels
    ]


def _component_count(mesh: ContourMesh, level: float) -> int:
    """Count strict superlevel components by plain host-side graph search."""

    values = np.asarray(mesh.vertex_psi)
    live = np.asarray(mesh.vertex_valid) & (values > level)
    edges = np.asarray(mesh.edges)[np.asarray(mesh.edge_valid)]
    neighbours = {index: set() for index in np.flatnonzero(live).tolist()}
    for left, right in edges:
        if live[left] and live[right]:
            neighbours[int(left)].add(int(right))
            neighbours[int(right)].add(int(left))
    remaining = set(neighbours)
    components = 0
    while remaining:
        components += 1
        frontier = [remaining.pop()]
        while frontier:
            source = frontier.pop()
            for target in neighbours[source]:
                if target in remaining:
                    remaining.remove(target)
                    frontier.append(target)
    return components


def _tree_count(mesh: ContourMesh, tree, level: float, *, corrupt: bool) -> int:
    """Count tree arcs crossing a regular level without reusing its merge code."""

    edges = np.asarray(tree.edges)[np.asarray(tree.edge_valid)]
    if corrupt:
        wall = np.asarray(mesh.vertex_is_wall)
        edges = edges[~(wall[edges[:, 0]] | wall[edges[:, 1]])]
    values = np.asarray(mesh.vertex_psi)
    crossing = (values[edges[:, 0]] > level) != (values[edges[:, 1]] > level)
    return int(np.count_nonzero(crossing))


def compare(mesh: ContourMesh, *, corrupt: bool = False) -> dict[str, object]:
    """Compare tree arcs with an independent carrier-graph traversal."""

    tree = build_contour_tree(
        mesh.vertex_psi,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        jnp.asarray(1, dtype=jnp.int32),
    )
    if bool(tree.overflow):
        raise RuntimeError("contour tree capacity refused")
    values = np.asarray(mesh.vertex_psi)[np.asarray(mesh.vertex_valid)]
    levels = np.nextafter(np.unique(values), -np.inf)[::-1]
    rows = [
        {
            "level": float(level),
            "brute_force": _component_count(mesh, float(level)),
            "tree": _tree_count(mesh, tree, float(level), corrupt=corrupt),
        }
        for level in levels
    ]
    return {
        "rows": rows,
        "node_count": int(np.count_nonzero(np.asarray(tree.node_valid))),
        "edge_count": int(np.count_nonzero(np.asarray(tree.edge_valid))),
        "tree": tree,
    }


def batched_identical(fixtures: tuple[Fixture, ...]) -> bool:
    """Confirm vmap produces the same receipt as each independent invocation."""

    meshes = tuple(item.mesh for item in fixtures)
    batched = jax.vmap(build_contour_tree)(
        jnp.stack([mesh.vertex_psi for mesh in meshes]),
        jnp.stack([mesh.vertex_valid for mesh in meshes]),
        jnp.stack([mesh.vertex_is_wall for mesh in meshes]),
        jnp.stack([mesh.edges for mesh in meshes]),
        jnp.stack([mesh.edge_valid for mesh in meshes]),
        jnp.ones(len(meshes), dtype=jnp.int32),
    )
    fields = (
        "node_vertex",
        "node_psi",
        "node_valid",
        "critical_type",
        "edges",
        "edge_valid",
        "overflow",
    )
    return all(
        all(
            np.array_equal(
                np.asarray(getattr(batched, name)[index]),
                np.asarray(getattr(compare(fixture.mesh)["tree"], name)),
            )
            for name in fields
        )
        for index, fixture in enumerate(fixtures)
    )


def render(fixtures: tuple[Fixture, ...], directory: Path = FIGURE_ROOT) -> list[Path]:
    """Render one data-ink component-count comparison per persisted fixture.

    The tree arm is drawn wherever the fixed-capacity receipt returns; on a
    carrier whose receipt refuses, the figure shows the independent brute-force
    count alone rather than a truncated tree.
    """

    import matplotlib.pyplot as plt

    try:
        plt.style.use("data-ink")
    except OSError:
        pass
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for fixture in fixtures:
        rows = None
        tree = None
        if tree_fits(fixture.mesh):
            try:
                rows = compare(fixture.mesh)["rows"]
                tree = [row["tree"] for row in rows]
            except RuntimeError:
                rows = None
        if rows is None:
            rows = standalone_rows(fixture.mesh)
        level = [row["level"] for row in rows]
        brute = [row["brute_force"] for row in rows]
        figure, axis = plt.subplots(figsize=(14, 5), dpi=100)
        axis.step(level, brute, where="post", color="#356c9b", linewidth=3.0)
        axis.set(xlabel="signed flux level [Wb]", ylabel="components")
        axis.spines[["top", "right"]].set_visible(False)
        if tree is not None:
            axis.step(
                level,
                tree,
                where="post",
                color="#222222",
                linestyle="--",
                linewidth=2.6,
            )
            axis.text(level[-1], tree[-1], "tree", color="#222222", va="top")
        axis.text(level[-1], brute[-1], "brute force", color="#356c9b", va="bottom")
        path = directory / f"{fixture.name}-components.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
