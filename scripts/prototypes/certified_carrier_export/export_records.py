"""Export nova's certified analytic carriers as version-2 boundary trace records.

The field-swap comparison reads imas-efit records with nova's read. This module
runs the other way: it writes nova's certified analytic carriers, one limited
and one diverted, into imas-efit's version-2 record layout, so imas-efit's own
reader and escape-level rule can be run on fields nova generated.

Each record carries the analytic psi on a fixed 65x65 grid spanning the wall's
bounding box, the read axis, the record polarity, and the per-loop wall. The
psi is nova's analytic total poloidal flux in the written COCOS-17 convention,
which the record states through its polarity field.

The carrier machine is loaded from the content-addressed certificate fixture
store through :mod:`scripts.prototypes.support.certificate_fixtures`; its
identity there is the one the certificate booking gate publishes under, so a
later run loads the object this export took its wall from.
"""

# ruff: noqa: E402  (configure_dtypes must run before jax is imported)

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.efit_field_swap.crosscheck import lattice
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField
from scripts.prototypes.efit_field_swap.trace_record import (
    TraceRecord,
    write_trace_record,
)
from scripts.prototypes.support.certificate_fixtures import cached_carrier
from tests.equilibrium.test_topology_read import _analytic_inputs

assert jax.config.jax_enable_x64

GRID = 65
CERTIFICATE_CELLS = 550
POLARITY = 1
COCOS = 17
LATTICE_RESOLUTION = 24
DIVERTED_VERDICT = 2
LIMITED_VERDICT = 1


@dataclass(frozen=True)
class CarrierExport:
    """One exported carrier: its record and the read decision behind it."""

    kind: str
    record: TraceRecord
    boundary_class: int
    axis: np.ndarray
    boundary_flux: float
    boundary_point: np.ndarray
    saddle_flux: float


def _closed_ring(vertices: np.ndarray) -> np.ndarray:
    """Close a vertex ring once, dropping consecutive duplicates."""
    keep = np.ones(len(vertices), dtype=bool)
    keep[1:] = np.any(vertices[1:] != vertices[:-1], axis=1)
    ring = vertices[keep]
    if not np.array_equal(ring[-1], ring[0]):
        ring = np.vstack((ring, ring[0]))
    return ring


def analytic_field(kind: str):
    """Return the certificate's analytic total field in the read's units.

    The limited carrier reads the static reactor case directly; the diverted
    carrier reads the single-null closed form, whose scale is flux per radian
    and so is divided by ``2 pi`` to carry whole-weber flux.
    """
    from scripts.prototypes.support.geometry import certificate_field

    _oracle, field, _wall, _axis, _saddle = _analytic_inputs(kind)
    return certificate_field(field, kind)


def wall_ring(machine) -> np.ndarray:
    """Return the carrier wall as one closed ring of RZ vertices."""
    return _closed_ring(np.asarray(machine.wall_node, dtype=np.float64))


def grid_axes(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 65-point radial and vertical axes spanning the wall bbox."""
    lower = ring.min(axis=0)
    upper = ring.max(axis=0)
    radial = np.linspace(lower[0], upper[0], GRID)
    vertical = np.linspace(lower[1], upper[1], GRID)
    return radial, vertical


def grid_flux(field, rgrid: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
    """Evaluate the analytic psi on the tensor grid, shaped ``[radial, vertical]``."""
    radial, vertical = np.meshgrid(rgrid, zgrid, indexing="ij")
    coordinates = np.column_stack((radial.ravel(), vertical.ravel()))
    values = jax.vmap(field.value)(jnp.asarray(coordinates))
    return np.asarray(values).reshape(GRID, GRID)


def record_wall(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the per-loop wall segments and their one-based loop index.

    A segment row carries the two endpoints of one edge, so the indexed range
    ``[1, nwall]`` reconstructs the whole ring through the codec's own wall
    reader.
    """
    start = ring[:-1]
    following = ring[1:]
    wall_r = np.column_stack((start[:, 0], following[:, 0]))
    wall_z = np.column_stack((start[:, 1], following[:, 1]))
    wall_index = np.asarray([[1, len(start)]], dtype=np.int32)
    return wall_r, wall_z, wall_index


def record_geometry(record: TraceRecord) -> topology.TopologyGeometry:
    """Return the read geometry a record's own wall admits."""
    ring = record.wall_ring
    cells, offsets = lattice(ring, LATTICE_RESOLUTION)
    return topology.TopologyGeometry.from_cells(cells, offsets, (ring,))


def read_record(record: TraceRecord) -> topology.TopologyRead:
    """Run nova's own read on a record's psi, wall and polarity.

    The geometry is the square lattice the record's own wall admits, so the
    read is reproducible from the record alone.
    """
    geometry = record_geometry(record)
    field = TraceSplineField.from_record(record).sampled_for(geometry)
    convention = topology.TopologyConvention.from_cocos(COCOS, POLARITY)
    return jax.jit(topology.read)(field, geometry, convention, TopologyPolicy())


def build_export(kind: str, cells: int = CERTIFICATE_CELLS) -> CarrierExport:
    """Build the version-2 record for one certificate carrier."""
    name = (
        "diverted-single-null" if kind == "diverted" else "weak-rotation-reactor-static"
    )
    carrier, _source, exact = certificate._case(name, clip_mode="exact")
    wall = certificate._diverted_wall(exact) if kind == "diverted" else None
    requested = -500 if cells == 550 else -cells
    machine = cached_carrier(carrier, requested, wall)
    field = analytic_field(kind)
    ring = wall_ring(machine)
    rgrid, zgrid = grid_axes(ring)
    psi = grid_flux(field, rgrid, zgrid)
    wall_r, wall_z, wall_index = record_wall(ring)
    provisional = TraceRecord(
        slice=1,
        iteration=1,
        verdict=DIVERTED_VERDICT if kind == "diverted" else LIMITED_VERDICT,
        saddle_source_node=0,
        wall_source_node=0,
        wall_source_segment=0,
        wall_axis_connected=True,
        handover_reverted=False,
        polarity=POLARITY,
        sweep_armed=False,
        axis_psi=0.0,
        axis_r=0.0,
        axis_z=0.0,
        saddle_contested=np.nan,
        saddle_refined=np.nan,
        wall_level=0.0,
        fed_back_psi=0.0,
        rgrid=rgrid,
        zgrid=zgrid,
        wall_r=wall_r,
        wall_z=wall_z,
        wall_index=wall_index,
        cell_r=np.empty((0, 2)),
        cell_z=np.empty((0, 2)),
        polygon_r=ring[:, 0],
        polygon_z=ring[:, 1],
        psi=psi,
    )
    reading = read_record(provisional)
    if not bool(reading.valid):
        raise ValueError(
            f"the {kind} record read refused: reason {int(reading.reason)}"
        )
    if not bool(reading.qualified):
        raise ValueError(f"the {kind} record read is unqualified")
    axis = np.asarray(reading.axis, dtype=np.float64)
    boundary = np.asarray(reading.boundary, dtype=np.float64)
    boundary_class = int(np.asarray(reading.boundary_class))
    if (boundary_class == 1) != (kind == "diverted"):
        raise ValueError(
            f"the {kind} record read class {boundary_class} "
            "does not match its certificate class"
        )
    boundary_flux = float(np.asarray(reading.boundary_flux))
    x_valid = np.asarray(reading.x_point_valid)
    if boundary_class == 1 and bool(x_valid.any()):
        index = int(np.argmax(np.asarray(reading.x_point_flux)))
        saddle_flux = float(np.asarray(reading.x_point_flux)[index])
    else:
        # A limited record admits no saddle; the codec stores only reals, so
        # the absent saddle repeats the wall-contact level rather than a NaN.
        saddle_flux = boundary_flux
    record = replace(
        provisional,
        axis_psi=float(np.asarray(reading.axis_flux)),
        axis_r=float(axis[0]),
        axis_z=float(axis[1]),
        saddle_contested=saddle_flux,
        saddle_refined=saddle_flux,
        wall_level=boundary_flux,
        fed_back_psi=boundary_flux,
    )
    return CarrierExport(
        kind=kind,
        record=record,
        boundary_class=boundary_class,
        axis=axis,
        boundary_flux=boundary_flux,
        boundary_point=boundary,
        saddle_flux=saddle_flux,
    )


def write_export(export: CarrierExport, directory: str | Path) -> Path:
    """Write one carrier export as a version-2 record and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    verdict = "diverted" if export.boundary_class == 1 else "limited"
    path = directory / f"nova-certified-{export.kind}-{verdict}.txt"
    write_trace_record(path, export.record)
    return path


def main(directory: str) -> list[Path]:
    written = []
    for kind in ("limited", "diverted"):
        export = build_export(kind)
        path = write_export(export, directory)
        written.append(path)
        print(
            "EXPORTED",
            kind,
            "class",
            export.boundary_class,
            "axis",
            np.round(export.axis, 4),
            "boundary_flux",
            export.boundary_flux,
            "boundary_point",
            np.round(export.boundary_point, 4),
            path,
            flush=True,
        )
    return written


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:2])
