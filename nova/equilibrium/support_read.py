"""The forward map's plasma support: one read and one booking per step.

A step contracts the booked source moments with the dense jet rows once, solves
the axis and X-points in the bounded warm-started row window
(``fixed_points.solve``), reads the connected membership of the resulting
nodal Bell field at those points (``topology.read_at_stationary_points``) and
books current moments on it (``current.book``). Nothing else reads support:
the legacy spline-clip support and fixed-design read are not reachable from
this route.

All numerical operands are runtime pytree leaves, so one compiled program
serves any state of the same layout.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, fixed_points, topology


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SupportOperands:
    """Geometry, nodal source rows and exterior jets of one machine.

    ``rows`` has shape (nodes, source cells, three moments, six jet entries)
    and ``exterior`` has shape (nodes, six jet entries); both follow the
    node order of ``nodes``, the unique per-cell sample points in
    ``geometry``. The total nodal jet of a state is ``exterior`` plus ``rows``
    contracted with the state's booked moments.
    """

    geometry: topology.TopologyGeometry
    mesh: fixed_points.BellMesh
    nodes: jax.Array
    sample_index: jax.Array
    rows: jax.Array
    exterior: jax.Array

    @classmethod
    def from_rows(cls, geometry, rows, exterior, nodes=None, sample_index=None):
        """Assemble operands from prebuilt rows; nodes follow the sample points."""
        if nodes is None or sample_index is None:
            samples = np.asarray(geometry.sample_points)
            nodes, inverse = np.unique(
                np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
            )
            sample_index = inverse.reshape(samples.shape[:-1])
        mesh = fixed_points.build_mesh(nodes, sample_index, geometry)
        return cls(
            geometry,
            mesh,
            jnp.asarray(nodes),
            jnp.asarray(sample_index),
            jnp.asarray(rows),
            jnp.asarray(exterior),
        )


class SupportRead(NamedTuple):
    """The read, its stationary points and one overall verdict."""

    stationary: fixed_points.FixedPointResult
    reading: topology.TopologyRead
    valid: jax.Array


def nodal_jets(operands: SupportOperands, moments) -> jax.Array:
    """Contract the booked moments once into every node's value and derivatives."""
    stacked = jnp.stack(tuple(moments), axis=-1)
    return operands.exterior + jnp.einsum("ncmq,cm->nq", operands.rows, stacked)


def conductor_jet_rows(nodes, sections):
    """Build exact nodal response jets for rectangular coils or filaments.

    Each section is ``(radius, height, radial_width, vertical_width)``; two
    zero widths select a filament. Currents are contracted after this geometry
    construction and remain runtime operands. A row is value, two gradient
    entries and three symmetric Hessian entries, per ampere.
    """
    from nova.biot.greens import traced_cylinder_greens, traced_filament_greens

    sections = np.asarray(sections, dtype=np.float64)
    if sections.ndim != 2 or sections.shape[1] != 4:
        raise ValueError("conductor sections must have shape (conductors, 4)")
    widths = sections[:, 2:]
    if np.any(widths < 0) or np.any((widths[:, 0] == 0) != (widths[:, 1] == 0)):
        raise ValueError("conductor widths must both be positive or both zero")

    def one(point, section):
        def gradient_with_value(target):
            value, radial, vertical = jax.lax.cond(
                section[2] == 0,
                lambda _: traced_filament_greens(
                    jnp, target[0], target[1], section[0], section[1]
                ),
                lambda _: traced_cylinder_greens(jnp, target[0], target[1], *section),
                None,
            )
            gradient = 2 * jnp.pi * target[0] * jnp.stack((vertical, -radial))
            return gradient, (value, gradient)

        hessian, (value, gradient) = jax.jacfwd(gradient_with_value, has_aux=True)(
            point
        )
        return fixed_points.stack_jet(topology.FieldJet(value, gradient, hessian))

    return jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None)))(
        jnp.asarray(nodes), jnp.asarray(sections)
    )


def coupling_rows(nodes, coupling, *, batch_size=3):
    """Build value and derivative rows from the existing exact source kernel."""
    return jax.jit(
        lambda nodes, coupling: jax.lax.map(
            lambda point: fixed_points._source_jets(point, coupling),
            nodes,
            batch_size=batch_size,
        )
    )(jnp.asarray(nodes), coupling)


def row_window(operands, previous, *, margin=None, capacity=512):
    """Select a bounded stationary-point window from already-built nodal rows."""
    nodes = np.asarray(operands.nodes)
    triangles = np.asarray(operands.mesh.triangles)
    centres = np.asarray(operands.mesh.generators)
    if margin is None:
        margin = 2 * float(np.median(operands.geometry.pitch))
    distance = np.min(
        np.linalg.norm(centres[:, None] - np.asarray(previous)[None], axis=-1), axis=1
    )
    radius = np.max(
        np.linalg.norm(nodes[triangles] - centres[:, None, None], axis=-1), axis=(1, 2)
    )
    chosen = np.unique(triangles[distance <= margin + radius])
    if not len(chosen) or len(chosen) > capacity:
        raise ValueError(
            f"row window needs {len(chosen)} nodes; capacity is {capacity}"
        )
    lookup = np.full(len(nodes), -1, dtype=np.int32)
    lookup[chosen] = np.arange(len(chosen))
    padding = capacity - len(chosen)
    return fixed_points.RowCache(
        operands.mesh,
        operands.nodes,
        jnp.asarray(lookup),
        jnp.pad(operands.rows[chosen], ((0, padding), (0, 0), (0, 0), (0, 0))),
        jnp.pad(operands.exterior[chosen], ((0, padding), (0, 0))),
    )


def read(
    operands: SupportOperands,
    cache: fixed_points.RowCache,
    moments,
    seeds,
    previous,
    convention,
    policy,
    axis_sign=-1.0,
    *,
    iterations: int = 12,
    jets=None,
) -> SupportRead:
    """Solve the stationary points, then read membership at them.

    ``seeds`` and ``previous`` have shape (points, 2) with the axis first;
    ``cache`` is the bounded row window kept by ``fixed_points.maintain_rows``.
    A row-window miss, a wrong root or an unconverged point makes the whole
    read invalid; the typed per-point status stays on ``stationary``.
    """
    pitch = jnp.median(operands.geometry.pitch)
    solved = fixed_points.solve(
        cache,
        jnp.stack(tuple(moments), axis=-1),
        seeds,
        previous,
        pitch,
        axis_sign,
        iterations=iterations,
    )
    jets = nodal_jets(operands, moments) if jets is None else jets
    field = fixed_points.field_from_jets(jets, operands.sample_index, operands.mesh)
    reading = topology.read_at_stationary_points(
        field, operands.geometry, convention, policy, solved
    )
    valid = jnp.all(solved.valid) & reading.valid & reading.qualified
    return SupportRead(solved, reading, valid)


def book(operands, support: SupportRead, psi, profile, quadrature):
    """Book current moments on a support read; invalid reads book non-finite."""
    booking = current.book(psi, support.reading, operands.geometry, profile, quadrature)
    valid = booking.valid & support.valid
    return booking._replace(
        **{
            name: jnp.where(valid, getattr(booking, name), jnp.nan)
            for name in ("cell_current", "radial_moment", "vertical_moment")
        },
        valid=valid,
    )


def read_and_book(
    operands,
    cache,
    moments,
    seeds,
    previous,
    convention,
    policy,
    profile,
    quadrature,
    axis_sign=-1.0,
    *,
    psi=None,
    iterations: int = 12,
):
    """One step: one contraction, one read, one booking.

    ``psi`` is the physical flux at ``geometry.sample_points``; when omitted it
    is the flux the booked moments themselves generate at those points.
    """
    jets = nodal_jets(operands, moments)
    if psi is None:
        psi = jets[operands.sample_index, 0]
    support = read(
        operands,
        cache,
        moments,
        seeds,
        previous,
        convention,
        policy,
        axis_sign,
        iterations=iterations,
        jets=jets,
    )
    return support, book(operands, support, psi, profile, quadrature)
