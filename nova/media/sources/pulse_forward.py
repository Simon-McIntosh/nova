"""Read a pulse of solved forward maps and trace each slice's plasma region.

``scripts/media/mast_pulse_forward.py`` writes one solved 2-D flux map per
EFIT slice, with the axis, the boundary level and the binding wall point the
solve's own topology read selected. This reader turns each map into what a
poloidal panel draws: the boundary, nested surfaces at stated normalised flux,
the current-carrying cells, and the nulls.

The boundary is the edge of the axis-connected region where the normalised
flux is below one, inside the limiter. It is not "the contour at the boundary
level that happens to be longest". Tracing the connected region is what makes
the two classes come out right without branching on topology:

- a limited plasma's region is bounded by the limiter itself where the flux
  there is below the boundary level, so its edge runs along the wall at the
  contact point rather than floating inside the vessel;
- a diverted plasma's region pinches at the X-point, and the private-flux
  pocket below it, though below the boundary level, is not connected to the
  axis and is excluded.

The map is refined with a bicubic spline before tracing, so the drawn edge
follows the field at a fraction of the solve's grid pitch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Spline refinement of the solved map before it is traced and contoured.
REFINE = 8

#: Normalised flux of the connected-region mask: the boundary level itself.
EDGE_LEVEL = 1.0 - 1.0e-6

#: Normalised flux of the drawn boundary, a hair inside the boundary level so
#: the level set closes around the axis rather than running through the
#: X-point into the legs. At this offset the curve lies within a millimetre
#: or two of the separatrix or the wall contact.
SMOOTH_LEVEL = 1.0 - 1.0e-4

#: Radius, in refined grid pitches, of the cut made at a diverted slice's
#: X-point. At the boundary level the sub-boundary region touches the saddle
#: cell and connects through it into the divertor; removing a disk this size
#: around the saddle breaks that connection, so the axis region ends at the
#: X-point rather than continuing down the legs.
SADDLE_CUT_PITCHES = 1.5

#: A cell carries plasma current when it holds at least this fraction of the
#: largest cell current; below it the cell is numerical residue.
CURRENT_FLOOR = 1.0e-3


@dataclass(frozen=True)
class ForwardSlice:
    """One solved slice as a poloidal panel draws it."""

    time: float
    row: int
    boundary: np.ndarray
    field: tuple[np.ndarray, np.ndarray, np.ndarray]
    """Refined radius, height and absolute flux ``(R, Z)`` the frame is drawn from."""
    axis_flux: float
    boundary_flux: float
    magnetic_axis: np.ndarray
    x_point: np.ndarray
    diverted: bool
    qualified: bool
    residual: float
    plasma_current: float
    target_current: float
    wall_gap: float
    current_cells: np.ndarray
    """Centres of the grid cells carrying plasma current, ``(n, 2)`` in metres."""
    x_point_gap: float = float("nan")
    """Closest approach of the boundary to the selected saddle, in metres."""

    def bounded(self, tolerance: float) -> bool:
        """Return whether the boundary is held by what its class says holds it.

        A plasma is either diverted, its boundary passing through the X-point,
        or limited, its boundary touching the wall. A closed boundary standing
        clear of both is not a plasma state, whatever the residual says, so a
        figure must not draw it as one.
        """
        gap = self.x_point_gap if self.diverted else self.wall_gap
        return bool(np.isfinite(gap) and gap <= tolerance)


@dataclass(frozen=True)
class ForwardPulse:
    """A whole pulse of solved slices on one shared grid."""

    slices: tuple[ForwardSlice, ...]
    radius: np.ndarray
    height: np.ndarray
    limiter: np.ndarray
    pitch: tuple[float, float]
    provenance: dict


def _closed_rings(generator, level: float) -> list[np.ndarray]:
    """Return the closed lines a contour generator draws at ``level``."""
    return [line for line in generator.lines(level) if np.allclose(line[0], line[-1])]


def _smooth_boundary(fine_r, fine_z, psi_norm, axis, limiter) -> np.ndarray | None:
    """Return the level set enclosing the axis just inside the boundary level.

    The curve is a contour of the bicubic flux field, so it is smooth at a
    fraction of the solve pitch rather than following refined cells. It is
    clipped to the limiter: a limited boundary then rests exactly on the wall
    at its contact instead of crossing it by the spline's few millimetres.
    ``None`` when no closed level set encloses the axis inside the wall.
    """
    import shapely
    from contourpy import contour_generator

    generator = contour_generator(fine_r, fine_z, psi_norm.T)
    point = shapely.Point(float(axis[0]), float(axis[1]))
    wall = shapely.Polygon(limiter).buffer(0)
    for ring in sorted(_closed_rings(generator, SMOOTH_LEVEL), key=len):
        polygon = shapely.Polygon(ring).buffer(0)
        if not polygon.contains(point):
            continue
        inside = polygon.intersection(wall)
        if inside.is_empty:
            continue
        if inside.geom_type == "MultiPolygon":
            inside = max(inside.geoms, key=lambda part: part.area)
        if not inside.contains(point):
            continue
        return np.asarray(inside.exterior.coords)
    return None


def trace_slice(
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
    material: np.ndarray,
    axis: np.ndarray,
    axis_flux: float,
    boundary_flux: float,
    x_point: np.ndarray | None = None,
    limiter: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], str]:
    """Return the boundary of one solved map, its refined field and the route.

    ``flux`` is radius-major on the solve lattice; ``material`` holds the
    limiter-interior flags on the refined lattice. The boundary is the smooth
    level set enclosing the axis (``_smooth_boundary``) when one closes inside
    the wall. Otherwise it is the edge of the axis-connected region below the
    boundary level, cut at a diverted slice's X-point, which is exact in
    topology but follows the refined lattice. The route taken is returned so a
    receipt can count it.
    """
    from contourpy import contour_generator
    from scipy.interpolate import RectBivariateSpline
    from scipy.ndimage import label

    fine_r = np.linspace(radius[0], radius[-1], REFINE * (radius.size - 1) + 1)
    fine_z = np.linspace(height[0], height[-1], REFINE * (height.size - 1) + 1)
    spline = RectBivariateSpline(radius, height, flux, kx=3, ky=3, s=0.0)
    psi = spline(fine_r, fine_z)
    psi_norm = (psi - axis_flux) / (boundary_flux - axis_flux)
    if material.shape != psi_norm.shape:
        raise ValueError("the material flags are not on the refined lattice")
    region = (psi_norm < EDGE_LEVEL) & material
    if x_point is not None and np.all(np.isfinite(x_point)):
        pitch = max(fine_r[1] - fine_r[0], fine_z[1] - fine_z[0])
        rr, zz = np.meshgrid(fine_r, fine_z, indexing="ij")
        cut = np.hypot(rr - x_point[0], zz - x_point[1]) <= SADDLE_CUT_PITCHES * pitch
        region &= ~cut
    labels, _count = label(region)
    ia = int(np.argmin(np.abs(fine_r - axis[0])))
    ja = int(np.argmin(np.abs(fine_z - axis[1])))
    seed = labels[ia, ja]
    if seed == 0:
        raise ValueError("the magnetic axis does not lie in a sub-boundary region")
    field = (fine_r, fine_z, psi.astype(np.float32))
    if limiter is not None:
        smooth = _smooth_boundary(fine_r, fine_z, psi_norm, axis, limiter)
        if smooth is not None:
            return smooth, field, "smooth level set"
    # Outside the connected region the field is replaced by a value above one,
    # so the contour at one is the region's own edge.
    masked = np.where(labels == seed, psi_norm, 2.0)
    edge = _closed_rings(contour_generator(fine_r, fine_z, masked.T), EDGE_LEVEL)
    if not edge:
        raise ValueError("the plasma region has no closed edge")
    return max(edge, key=len), field, "connected region edge"


def _material_on(radius, height, limiter, refine: int) -> np.ndarray:
    """Return the limiter-interior flags on the refined lattice."""
    from nova.equilibrium.wall_mask import inside_polygon

    fine_r = np.linspace(radius[0], radius[-1], refine * (radius.size - 1) + 1)
    fine_z = np.linspace(height[0], height[-1], refine * (height.size - 1) + 1)
    rr, zz = np.meshgrid(fine_r, fine_z, indexing="ij")
    flags = inside_polygon(rr.ravel(), zz.ravel(), limiter[:, 0], limiter[:, 1])
    return np.asarray(flags, dtype=bool).reshape(rr.shape)


def _merged(paths: list[Path]) -> dict[str, np.ndarray]:
    """Return the rows of several stacks on one grid, ordered by time.

    A pulse solved in shards writes one stack per shard; they must share the
    grid, wall and limiter exactly, or they are not one pulse.
    """
    shared = ("radius", "height", "material", "limiter", "wall")
    parts = []
    for path in paths:
        with np.load(path) as stack:
            parts.append({name: stack[name] for name in stack.files})
    for part in parts[1:]:
        for name in shared:
            if not np.array_equal(part[name], parts[0][name]):
                raise ValueError(f"the shard stacks disagree on {name}")
    data = {name: parts[0][name] for name in shared}
    # Only fields every shard carries are merged; a field a later writer added
    # is dropped rather than padded, so no row reports a value it never had.
    common = [name for name in parts[0] if all(name in part for part in parts)]
    for name in common:
        if name not in shared:
            data[name] = np.concatenate([part[name] for part in parts])
    # One slice per time: a row solved by two shards is kept once, from the
    # first stack named.
    _, first = np.unique(data["time_s"], return_index=True)
    order = np.sort(first)[np.argsort(data["time_s"][np.sort(first)], kind="stable")]
    for name in common:
        if name not in shared:
            data[name] = data[name][order]
    return data


def read_pulse_forward(paths: Path | str | list) -> ForwardPulse:
    """Return every solved slice of one or more forward stacks, traced to draw."""
    from nova.media.poloidal import boundary_wall_gap

    paths = [
        Path(item) for item in ([paths] if isinstance(paths, str | Path) else paths)
    ]
    data = _merged(paths)
    path = paths[0]
    radius, height, limiter = data["radius"], data["height"], data["limiter"]
    material = _material_on(radius, height, limiter, REFINE)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    centres = np.column_stack([rr.ravel(), zz.ravel()])
    slices = []
    refused = []
    routes = []
    for index in range(data["flux"].shape[0]):
        try:
            boundary, field, route = trace_slice(
                radius,
                height,
                data["flux"][index],
                material,
                data["axis"][index],
                float(data["axis_flux"][index]),
                float(data["boundary_flux"][index]),
                data["x_point"][index] if bool(data["diverted"][index]) else None,
                limiter,
            )
        except ValueError as error:
            refused.append({"row": int(data["row"][index]), "reason": str(error)})
            continue
        current = data["cell_current"][index]
        carrying = np.abs(current) > CURRENT_FLOOR * np.abs(current).max()
        routes.append(route)
        slices.append(
            ForwardSlice(
                time=float(data["time_s"][index]),
                row=int(data["row"][index]),
                boundary=boundary,
                field=field,
                axis_flux=float(data["axis_flux"][index]),
                boundary_flux=float(data["boundary_flux"][index]),
                magnetic_axis=data["axis"][index],
                x_point=data["x_point"][index],
                diverted=bool(data["diverted"][index]),
                qualified=bool(data["qualified"][index]),
                residual=float(data["residual"][index]),
                plasma_current=float(data["plasma_current"][index]),
                target_current=float(data["target_current"][index]),
                wall_gap=float(boundary_wall_gap(boundary, limiter)),
                x_point_gap=float(
                    np.min(np.hypot(*(boundary - data["x_point"][index]).T))
                ),
                current_cells=centres[carrying],
            )
        )
    receipt_path = path.with_suffix(".json")
    provenance = {
        "stacks": [str(item) for item in paths],
        "refused_slices": refused,
        "trace": (
            f"level set at normalised flux {SMOOTH_LEVEL} enclosing the axis on a "
            f"{REFINE}x bicubic refinement, clipped to the limiter; where none "
            "closes inside the wall, the edge of the axis-connected region"
        ),
        "trace_routes": {name: routes.count(name) for name in sorted(set(routes))},
    }
    if receipt_path.exists():
        import json

        provenance["solve_receipt"] = json.loads(receipt_path.read_text())
    return ForwardPulse(
        slices=tuple(slices),
        radius=radius,
        height=height,
        limiter=limiter,
        pitch=(float(np.diff(radius).mean()), float(np.diff(height).mean())),
        provenance=provenance,
    )
