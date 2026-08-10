"""Evaluate electromagnetic coupling matrices on fixed flux targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import shapely.geometry
import shapely.ops

from nova.biot.biotframe import Target
from nova.biot.null import Null1D, Null2D
from nova.jax.tree_util import Pytree


@dataclass(frozen=True)
class TargetQuadraturePolicy:
    """Immutable identity for target-material averaging and contraction."""

    rule: Literal["positive_material"] = "positive_material"
    order: int = 3
    material: Literal["actual_parent"] = "actual_parent"
    contraction: Literal["logical_cell_before_reduction"] = (
        "logical_cell_before_reduction"
    )
    backend: Literal["numpy"] = "numpy"
    precision: Literal["float64"] = "float64"
    device_eligibility: Literal["host"] = "host"

    def __post_init__(self):
        """Validate every setting that changes target-linked flux."""
        if self.rule != "positive_material":
            raise ValueError(f"unsupported target quadrature rule {self.rule!r}")
        if (
            isinstance(self.order, bool | np.bool_)
            or not isinstance(self.order, int | np.integer)
            or self.order <= 0
        ):
            raise ValueError("target quadrature order must be a positive integer")
        if self.material != "actual_parent":
            raise ValueError(f"unsupported target material rule {self.material!r}")
        if self.contraction != "logical_cell_before_reduction":
            raise ValueError(
                f"unsupported target contraction rule {self.contraction!r}"
            )
        if (self.backend, self.precision, self.device_eligibility) != (
            "numpy",
            "float64",
            "host",
        ):
            raise ValueError(
                "target quadrature requires numpy float64 evaluation on the host"
            )
        object.__setattr__(self, "order", int(self.order))

    @property
    def key(self) -> str:
        """Return the canonical cache identity."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def resolve(
        cls, value: TargetQuadraturePolicy | Mapping | str | None
    ) -> TargetQuadraturePolicy:
        """Return a validated target policy from memory or persisted metadata."""
        if value is None or value == "":
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError("invalid target quadrature policy identity") from error
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(f"cannot resolve target policy from {type(value)}")


DEFAULT_TARGET_QUADRATURE_POLICY = TargetQuadraturePolicy().key


@dataclass(frozen=True)
class TargetQuadrature:
    """Map positive material nodes to logical cells and physical target rows."""

    nodes: Target = field(repr=False)
    logical: Target = field(repr=False)
    offsets: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    physical_index: tuple[str, ...]
    physical_plasma: tuple[bool, ...]
    policy: TargetQuadraturePolicy

    def __post_init__(self):
        """Validate a contiguous, normalized, strictly positive contraction."""
        offsets = np.asarray(self.offsets, dtype=int)
        weights = np.asarray(self.weights, dtype=np.float64)
        if offsets.ndim != 1 or len(offsets) != len(self.logical):
            raise ValueError(
                "one target quadrature offset is required per logical cell"
            )
        if len(offsets) == 0 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError(
                "target quadrature offsets must be strictly increasing from zero"
            )
        if weights.shape != (len(self.nodes),):
            raise ValueError("target quadrature weights must match kernel nodes")
        if not np.all(np.isfinite(weights)) or not np.all(weights > 0):
            raise ValueError("target quadrature weights must be finite and positive")
        if len(self.physical_index) != len(self.physical_plasma):
            raise ValueError("physical target labels and plasma flags must align")
        independent = np.asarray(self.logical.biotreduce.index, dtype=object)
        independent_positions = np.asarray(
            [self.logical.index.get_loc(label) for label in independent], dtype=int
        )
        expected_index = tuple(
            np.asarray(self.logical["frame"], dtype=object)[independent_positions]
            .astype(str)
            .tolist()
        )
        expected_plasma = tuple(
            np.asarray(self.logical.plasma, dtype=bool)[independent_positions].tolist()
        )
        if (
            self.physical_index != expected_index
            or self.physical_plasma != expected_plasma
        ):
            raise ValueError(
                "physical target identity must match final logical link reduction"
            )
        cell_weight = np.add.reduceat(weights, offsets)
        if not np.allclose(cell_weight, 1.0, rtol=2e-14, atol=2e-14):
            raise ValueError(
                "each logical target cell must carry unit quadrature weight"
            )
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "weights", weights)

    def collapse(self, values: np.ndarray) -> np.ndarray:
        """Average kernel-node rows into the original logical target cells."""
        values = np.asarray(values)
        if values.shape[0] != len(self.nodes):
            raise ValueError(
                "kernel target dimension does not match target quadrature nodes"
            )
        weight_shape = (len(self.weights),) + (1,) * (values.ndim - 1)
        return np.add.reduceat(
            values * self.weights.reshape(weight_shape), self.offsets, axis=0
        )


def _shapely_material(value):
    """Return the polygonal Shapely material wrapped by Nova geometry objects."""
    geometry = value
    visited: set[int] = set()
    while not isinstance(
        geometry, shapely.geometry.Polygon | shapely.geometry.MultiPolygon
    ):
        if id(geometry) in visited or not hasattr(geometry, "poly"):
            raise NotImplementedError(
                "linked-flux material quadrature currently requires an "
                "axisymmetric polygon or multipolygon"
            )
        visited.add(id(geometry))
        geometry = geometry.poly
    if geometry.is_empty or geometry.area <= 0:
        raise ValueError("linked-flux target material must have positive area")
    return geometry


def _polygonal_intersection(cell, parent):
    """Return only positive-area polygon material from a cell-parent overlap."""
    intersection = cell.intersection(parent)
    if isinstance(
        intersection, shapely.geometry.Polygon | shapely.geometry.MultiPolygon
    ):
        material = intersection
    elif isinstance(intersection, shapely.geometry.GeometryCollection):
        polygons = [
            geometry
            for geometry in intersection.geoms
            if isinstance(geometry, shapely.geometry.Polygon) and geometry.area > 0
        ]
        material = shapely.ops.unary_union(polygons)
    else:
        material = shapely.geometry.Polygon()
    if material.is_empty or material.area <= 0:
        raise ValueError("logical target cell does not overlap its parent material")
    return material


def linked_flux_target(
    frame, subframe, policy: TargetQuadraturePolicy | Mapping | str | None = None
) -> TargetQuadrature:
    """Build positive material nodes without changing logical target identities.

    Nodes integrate each existing conducting subframe cell after clipping it to
    the exact physical parent. The normalized positive weights collapse to the
    original cell before turns, parent reduction, electrical links, or plasma
    block extraction are applied.
    """
    from nova.biot.sectionaverage import section_nodes

    policy = TargetQuadraturePolicy.resolve(policy)
    parent_names = np.asarray(frame.index, dtype=object)
    subframe_names = np.asarray(subframe.index, dtype=object)
    membership = np.asarray(subframe["frame"], dtype=object)
    segment = np.asarray(subframe["segment"], dtype=object)
    unsupported = np.setdiff1d(
        np.unique(segment), np.array(["circle", "cylinder", "polysection"])
    )
    if len(unsupported) > 0:
        raise NotImplementedError(
            "linked-flux material quadrature is axisymmetric; unsupported "
            f"segments are {unsupported.tolist()}"
        )

    frame_polygons = np.asarray(frame["poly"], dtype=object)
    cell_polygons = np.asarray(subframe["poly"], dtype=object)
    nturn = np.asarray(subframe["nturn"], dtype=float)
    plasma = np.asarray(subframe.plasma, dtype=bool)
    x = np.asarray(subframe["x"], dtype=float)
    y = np.asarray(subframe["y"], dtype=float)
    z = np.asarray(subframe["z"], dtype=float)

    logical_data = {
        "x": [],
        "y": [],
        "z": [],
        "x0": [],
        "z0": [],
        "dx": [],
        "dz": [],
        "nturn": [],
        "plasma": [],
        "frame": [],
        "link": [],
        "factor": [],
    }
    node_data = {"x": [], "y": [], "z": []}
    node_index: list[str] = []
    offsets: list[int] = []
    normalized_weights: list[float] = []
    logical_index: list[str] = []
    physical_index: list[str] = []
    physical_plasma: list[bool] = []

    frame_x = np.asarray(frame["x"], dtype=float)
    frame_z = np.asarray(frame["z"], dtype=float)
    frame_dx = np.asarray(frame["dx"], dtype=float)
    frame_dz = np.asarray(frame["dz"], dtype=float)
    frame_nturn = np.asarray(frame["nturn"], dtype=float)
    for parent_position, parent_name in enumerate(parent_names):
        positions = np.flatnonzero(membership == parent_name)
        if len(positions) == 0:
            continue
        parent = _shapely_material(frame_polygons[parent_position])
        first_label = str(subframe_names[positions[0]])
        physical_index.append(str(parent_name))
        physical_plasma.append(bool(np.any(plasma[positions])))
        for cell_number, position in enumerate(positions):
            label = str(subframe_names[position])
            cell = _shapely_material(cell_polygons[position])
            material = _polygonal_intersection(cell, parent)
            points, area_weights = section_nodes(material, order=policy.order)
            points = np.asarray(points, dtype=np.float64)
            area_weights = np.asarray(area_weights, dtype=np.float64)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or area_weights.shape != (len(points),)
                or not np.all(np.isfinite(points))
                or not np.all(np.isfinite(area_weights))
                or not np.all(area_weights > 0)
            ):
                raise ValueError("section quadrature must return finite positive nodes")
            weight_sum = float(area_weights.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError("section quadrature material weight must be positive")
            tolerance = 1e-12 * max(1.0, np.sqrt(material.area))
            covered = material.buffer(tolerance)
            if not all(
                covered.covers(shapely.geometry.Point(point)) for point in points
            ):
                raise ValueError("target quadrature node escaped conducting material")

            offsets.append(len(normalized_weights))
            normalized_weights.extend((area_weights / weight_sum).tolist())
            node_data["x"].extend(points[:, 0].tolist())
            node_data["y"].extend(np.zeros(len(points)).tolist())
            node_data["z"].extend(points[:, 1].tolist())
            node_index.extend(f"{label}:q{index}" for index in range(len(points)))

            logical_index.append(label)
            logical_data["x"].append(x[position])
            logical_data["y"].append(y[position])
            logical_data["z"].append(z[position])
            logical_data["x0"].append(frame_x[parent_position])
            logical_data["z0"].append(frame_z[parent_position])
            logical_data["dx"].append(frame_dx[parent_position])
            logical_data["dz"].append(frame_dz[parent_position])
            logical_data["nturn"].append(nturn[position])
            logical_data["plasma"].append(plasma[position])
            logical_data["frame"].append(str(parent_name))
            logical_data["link"].append("" if cell_number == 0 else first_label)
            logical_data["factor"].append(1.0)

        if not np.isclose(
            nturn[positions].sum(),
            frame_nturn[parent_position],
            rtol=2e-12,
            atol=2e-12 * max(1.0, abs(frame_nturn[parent_position])),
        ):
            raise ValueError(
                f"logical target turns do not reproduce parent {parent_name!r}"
            )

    if not logical_index:
        raise ValueError("linked-flux target contains no conducting material")
    logical = Target(logical_data, index=logical_index, available=[])
    return TargetQuadrature(
        nodes=Target(node_data, index=node_index, available=[]),
        logical=logical,
        offsets=np.asarray(offsets, dtype=int),
        weights=np.asarray(normalized_weights, dtype=np.float64),
        physical_index=tuple(physical_index),
        physical_plasma=tuple(physical_plasma),
        policy=policy,
    )


@dataclass
@jax.tree_util.register_pytree_node_class
class FluxTarget(Pytree):
    """Evaluate external-source and plasma coupling on a flux target."""

    source_target: jnp.ndarray = field(repr=False)
    plasma_target: jnp.ndarray = field(repr=False)
    null: Null1D | Null2D

    @property
    def coordinate(self):
        """Return target coordinate."""
        return self.null.coordinate

    @property
    def node_number(self):
        """Return target node number."""
        return self.null.node_number

    @jax.jit
    def external(self, external_current: jnp.ndarray):
        """Return external poloidal flux map."""
        return self.source_target @ external_current

    @jax.jit
    def internal(self, plasma_current: jnp.ndarray):
        """Return internal (plasma generated) poloidal flux map."""
        return self.plasma_target @ plasma_current

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (self.source_target, self.plasma_target, self.null)
        aux_data = {}
        return (children, aux_data)
