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

import json
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.database import content_store
from nova.equilibrium import clip_quadrature, current, fixed_points, topology
from nova.equilibrium.solve import _damped_scale

# Node and cell counts a machine's mesh is padded up to, so two geometries of
# one bucket share one compiled program and the bank program is shape-static.
SHAPE_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
BANK_MANIFEST_NAME = "bank.json"


def shape_bucket(count: int) -> int:
    """Return the smallest declared bucket that holds ``count`` nodes."""
    if count < 1:
        raise ValueError("a shape bucket needs at least one node")
    for bucket in SHAPE_BUCKETS:
        if count <= bucket:
            return bucket
    raise ValueError(f"shape bucket does not cover {count} nodes")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SupportOperands:
    """Geometry, nodal source rows and exterior jets of one machine.

    ``rows`` has shape (nodes, source cells, three moments, six jet entries)
    and ``exterior`` has shape (nodes, six jet entries); both follow the
    node order of ``nodes``, the unique per-cell sample points in
    ``geometry``. The total nodal jet of a state is ``exterior`` plus ``rows``
    contracted with the state's booked moments.

    Padded to a shape bucket, ``rows``, ``exterior`` and ``nodes`` carry
    ``bucket`` rows of which the first ``active_nodes`` are the machine's own
    and the rest are zero, so one program serves every geometry of the bucket.
    ``bucket`` is static and enters the compile key; ``active_nodes`` is a
    traced leaf, because two geometries in one bucket need not carry the same
    active count and a differing static value would recompile.
    """

    geometry: topology.TopologyGeometry
    mesh: fixed_points.BellMesh
    nodes: jax.Array
    sample_index: jax.Array
    rows: jax.Array
    exterior: jax.Array
    bucket: int | None = field(default=None, metadata=dict(static=True))
    active_nodes: jax.Array | None = None

    @classmethod
    def from_rows(
        cls,
        geometry,
        rows,
        exterior,
        nodes=None,
        sample_index=None,
        bucket=None,
    ):
        """Assemble operands from prebuilt rows; nodes follow the sample points.

        ``bucket`` pads the operator rows to a declared shape bucket so the
        bank program is shape-static; ``None`` keeps the machine's own shape.
        """
        if nodes is None or sample_index is None:
            samples = np.asarray(geometry.sample_points)
            nodes, inverse = np.unique(
                np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
            )
            sample_index = inverse.reshape(samples.shape[:-1])
        nodes = np.asarray(nodes)
        rows = np.asarray(rows)
        exterior = np.asarray(exterior)
        mesh_nodes = nodes
        active = len(nodes)
        if bucket is not None:
            bucket = shape_bucket(bucket)
            if active > bucket:
                bucket = shape_bucket(active)
            padded_rows = np.zeros((bucket, *rows.shape[1:]), dtype=rows.dtype)
            padded_rows[:active] = rows
            padded_exterior = np.zeros(
                (bucket, *exterior.shape[1:]), dtype=exterior.dtype
            )
            padded_exterior[:active] = exterior
            padded_nodes = np.zeros((bucket, nodes.shape[-1]), dtype=nodes.dtype)
            padded_nodes[:active] = nodes
            rows, exterior, nodes = padded_rows, padded_exterior, padded_nodes
        mesh = fixed_points.build_mesh(mesh_nodes, sample_index, geometry)
        return cls(
            geometry,
            mesh,
            jnp.asarray(nodes),
            jnp.asarray(sample_index),
            jnp.asarray(rows),
            jnp.asarray(exterior),
            bucket,
            jnp.asarray(active),
        )


class TrackingState(NamedTuple):
    """The previous step's tracked cells, ownership and connected decision.

    ``cells`` holds the hex cell of each tracked point, ``-1`` where the point
    is not live. ``saddle_live`` records whether an admitted saddle gave the
    state diverted ownership. ``reading`` is the connected-domain decision of
    that step, reused as long as neither changes.
    """

    cells: jax.Array
    saddle_live: jax.Array
    reading: topology.TopologyRead
    rebuilds: jax.Array
    valid: jax.Array


class SupportRead(NamedTuple):
    """The read, its stationary points and one overall verdict.

    ``cells``, ``saddle_live`` and ``rebuilds`` carry the tracking state
    forward; ``rebuilt`` says whether this call rebuilt the connected-domain
    decision. ``jets`` is the contour-free derivative source the cell-local
    fit was evaluated from, so a caller needs no second whole-mesh pass.
    """

    stationary: fixed_points.FixedPointResult
    reading: topology.TopologyRead
    valid: jax.Array
    cells: jax.Array
    saddle_live: jax.Array
    rebuilds: jax.Array
    rebuilt: jax.Array


def nodal_jets(operands: SupportOperands, moments) -> jax.Array:
    """Contract the booked moments once into every node's value and derivatives."""
    stacked = jnp.stack(tuple(moments), axis=-1)
    return operands.exterior + jnp.einsum("ncmq,cm->nq", operands.rows, stacked)


class Stencil(NamedTuple):
    """The stationary points of one state, before any connected decision.

    This is the part of a read that cannot be shared between lanes: it refines
    the state's own stationary points and resolves the cell each occupies.
    A caller carries the connected-domain decision from :class:`TrackingState`
    while refreshing its continuous geometry from these stationary points.
    """

    solved: fixed_points.FixedPointResult
    cells: jax.Array
    saddle_live: jax.Array


def stationary_stencil(
    operands: SupportOperands,
    cache: fixed_points.RowCache,
    moments,
    seeds,
    previous,
    convention,
    axis_sign=-1.0,
    *,
    iterations: int = 12,
) -> Stencil:
    """Solve the stationary points and resolve their tracked cells."""
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
    eigenvalues = jnp.linalg.eigvalsh(convention.sigma * solved.jet.hessian)
    inside = jax.vmap(
        lambda point: jnp.any(topology._inside_cells(operands.geometry, point))
    )(solved.position)
    live = solved.valid & inside
    cells = jnp.where(
        live, fixed_points.nearest_cell(operands.mesh, solved.position), -1
    )
    saddles = live & (eigenvalues[:, 0] < 0.0) & (eigenvalues[:, 1] > 0.0)
    return Stencil(solved, cells, jnp.any(saddles))


def rebuild_needed(
    stencil: Stencil, tracking: TrackingState, geometry=None
) -> jax.Array:
    """Whether tracked cells, ownership or represented saddle cells change."""
    normal_changed = jnp.asarray(False)
    if geometry is not None:
        decision = tracking.reading
        index = jnp.argmax(decision.x_point_valid)
        represented = topology.represented_saddle_cells(
            geometry,
            stencil.solved.position[index],
            decision.boundary_class == int(topology.TopologyClass.DIVERTED),
            decision.normal_form_radius,
        )
        normal_changed = jnp.any(represented != decision.normal_form_cells)
    return (
        normal_changed
        | jnp.any(stencil.cells != tracking.cells)
        | (stencil.saddle_live != tracking.saddle_live)
        | ~tracking.valid
    )


def _verdict(
    stencil: Stencil,
    reading,
    rebuilt: jax.Array,
    rebuilds: jax.Array,
) -> SupportRead:
    """One lane's read result from its stencil and the decision it carries."""
    valid = jnp.all(stencil.solved.valid) & reading.valid & reading.qualified
    return SupportRead(
        stencil.solved,
        reading,
        valid,
        stencil.cells,
        stencil.saddle_live,
        rebuilds,
        rebuilt,
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
    tracking: TrackingState | None = None,
    rebuild: bool | None = None,
) -> SupportRead:
    """Solve the stationary points, then read membership at them.

    ``seeds`` and ``previous`` have shape (points, 2) with the axis first;
    ``cache`` is the bounded row window kept by ``fixed_points.maintain_rows``.
    A row-window miss, a wrong root or an unconverged point makes the whole
    read invalid; the typed per-point status stays on ``stationary``.

    The stationary points are refined by ``fixed_points.solve`` on the
    cell-local Bell fit of whichever cells they occupy, warm-started from
    ``previous``. The connected-domain decision is rebuilt only when a tracked
    point changes cell or the limited/diverted ownership changes; otherwise the
    previous discrete decision is carried forward. The own-node coefficients,
    stationary fluxes, fragment areas, edge intervals and saddle normal form
    are refreshed from the current state on both branches.
    """
    stencil = stationary_stencil(
        operands,
        cache,
        moments,
        seeds,
        previous,
        convention,
        axis_sign,
        iterations=iterations,
    )

    def rebuild(_):
        field_jets = nodal_jets(operands, moments) if jets is None else jets
        field = fixed_points.field_from_jets(
            field_jets,
            operands.sample_index,
            operands.mesh,
            bank_rows=operands.rows,
            bank_moments=jnp.stack(tuple(moments), axis=-1),
            bank_exterior=operands.exterior,
        )
        return topology.read_at_stationary_points(
            field, operands.geometry, convention, policy, stencil.solved
        )

    if tracking is None:
        rebuilt = jnp.asarray(True)
        reading = rebuild(None)
        rebuilds = jnp.asarray(1, jnp.int32)
    else:
        rebuilt = rebuild_needed(stencil, tracking, operands.geometry)
        if rebuild is False:
            # A caller that knows no slice of its batch can rebuild takes the
            # carried decision directly. Under ``vmap`` the conditional below
            # becomes a select and evaluates the whole-mesh rebuild for every
            # lane whether or not a lane needs it; :func:`read_batch` avoids
            # that by taking one branch for the whole batch.
            reading = _rebuilt_reading(
                operands, stencil, moments, convention, policy, jets, tracking.reading
            )
        else:
            reading = jax.lax.cond(
                rebuilt,
                rebuild,
                lambda _: _rebuilt_reading(
                    operands,
                    stencil,
                    moments,
                    convention,
                    policy,
                    jets,
                    tracking.reading,
                ),
                None,
            )
        changed = topology.carried_decision_changed(
            reading, tracking.reading, operands.geometry, policy
        )
        extra = ~rebuilt & changed
        reading = jax.lax.cond(extra, rebuild, lambda _: reading, None)
        rebuilt = rebuilt | extra
        rebuilds = tracking.rebuilds + rebuilt.astype(jnp.int32)
    return _verdict(stencil, reading, rebuilt, rebuilds)


def _rebuilt_reading(
    operands, stencil, moments, convention, policy, jets, decision=None
):
    """The connected-domain decision of one stencil, fitted over the whole mesh."""
    field_jets = nodal_jets(operands, moments) if jets is None else jets
    field = fixed_points.field_from_jets(
        field_jets,
        operands.sample_index,
        operands.mesh,
        bank_rows=operands.rows,
        bank_moments=jnp.stack(tuple(moments), axis=-1),
        bank_exterior=operands.exterior,
    )
    if decision is None:
        return topology.read_at_stationary_points(
            field, operands.geometry, convention, policy, stencil.solved
        )
    return topology.refresh_at_stationary_points(
        field, operands.geometry, convention, policy, stencil.solved, decision
    )


def read_batch(
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
    tracking: TrackingState | None = None,
    certificate=None,
) -> SupportRead:
    """One read over a batch of slices, taking one decision branch for all.

    ``moments``, ``seeds``, ``previous``, ``jets`` and ``tracking`` carry one
    entry per slice on their leading axis; the operands and the row window are
    shared by every lane. The per-lane rebuild decision is formed from the
    batched stencils, and a single ``lax.cond`` on whether *any* lane needs a
    rebuild chooses between the carried decision and the whole batch's reload.
    A steady batch therefore executes the carry branch alone, where a ``vmap``
    of :func:`read`'s own conditional compiles to a select that evaluates the
    whole-mesh rebuild in every lane.

    Continuous fragment geometry is refreshed even on a carry. ``certificate``
    optionally tests that result and requests a full fresh read for the batch;
    only the full read can then supply a terminal solve certificate.

    ``rebuilt`` and the ``rebuilds`` counter are batch-uniform: when one lane
    needs the decision rebuilt the batch rebuilds all of them, so every lane
    reports the rebuild it took part in.
    """
    stack = jax.vmap(
        lambda state, seed, warm: stationary_stencil(
            operands,
            cache,
            state,
            seed,
            warm,
            convention,
            axis_sign,
            iterations=iterations,
        )
    )(moments, seeds, previous)

    def whole_mesh():
        if jets is None:
            return jax.vmap(
                lambda stencil, state: _rebuilt_reading(
                    operands, stencil, state, convention, policy, None
                )
            )(stack, moments)
        return jax.vmap(
            lambda stencil, state, jet: _rebuilt_reading(
                operands, stencil, state, convention, policy, jet
            )
        )(stack, moments, jets)

    if tracking is None:
        return jax.vmap(
            lambda stencil, reading: _verdict(
                stencil,
                reading,
                jnp.asarray(True),
                jnp.asarray(1, dtype=jnp.int32),
            )
        )(stack, whole_mesh())

    flags = jax.vmap(
        lambda stencil, tracked: rebuild_needed(stencil, tracked, operands.geometry)
    )(stack, tracking)
    # The decision is one scalar for the batch: the branch, the reported flag
    # and the counter all read it, so no lane can carry a per-lane verdict.
    needed = jnp.any(flags)

    def verdict(stencil, reading, tracked):
        return _verdict(
            stencil,
            reading,
            needed,
            tracked.rebuilds + needed.astype(tracked.rebuilds.dtype),
        )

    def full_verdict():
        return jax.vmap(
            lambda stencil, reading, tracked: _verdict(
                stencil, reading, jnp.asarray(True), tracked.rebuilds + 1
            )
        )(stack, whole_mesh(), tracking)

    def carry(_):
        if jets is None:
            reading = jax.vmap(
                lambda stencil, state, decision: _rebuilt_reading(
                    operands, stencil, state, convention, policy, None, decision
                )
            )(stack, moments, tracking.reading)
        else:
            reading = jax.vmap(
                lambda stencil, state, jet, decision: _rebuilt_reading(
                    operands, stencil, state, convention, policy, jet, decision
                )
            )(stack, moments, jets, tracking.reading)
        carried = jax.vmap(verdict)(stack, reading, tracking)
        changed = jax.vmap(
            lambda reading, decision: topology.carried_decision_changed(
                reading, decision, operands.geometry, policy
            )
        )(reading, tracking.reading)
        if certificate is not None:
            changed = changed | certificate(carried)
        return jax.lax.cond(
            jnp.any(changed), lambda _: full_verdict(), lambda _: carried, None
        )

    def rebuild(_):
        return jax.vmap(verdict)(stack, whole_mesh(), tracking)

    return jax.lax.cond(needed, rebuild, carry, None)


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
    tracking: TrackingState | None = None,
):
    """One step: one contraction, one read, one booking.

    ``psi`` is the physical flux at ``geometry.sample_points``; when omitted it
    is the flux the booked moments themselves generate at those points. That
    contraction is over every node, which booking needs, so it is not part of
    what tracking saves; the read's own whole-mesh fit and Bell rebuild are.
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
        tracking=tracking,
    )
    return support, book(operands, support, psi, profile, quadrature)


def tracking_of(support: SupportRead) -> TrackingState:
    """Carry one step's tracked cells, ownership and decision to the next."""
    return TrackingState(
        support.cells,
        support.saddle_live,
        support.reading,
        support.rebuilds,
        support.valid,
    )


def bank_program(operands: SupportOperands, moments) -> jax.Array:
    """Apply the bank to booked moments: the minimal program it is measured through.

    Every operand is a traced argument, so the program is reused for any state
    and any geometry of the operand's shape bucket; only ``bucket`` is static.
    """
    jets = nodal_jets(operands, moments)
    active = jnp.asarray(
        jets.shape[0] if operands.active_nodes is None else operands.active_nodes
    )
    mask = (jnp.arange(jets.shape[0]) < active)[:, None]
    return jnp.where(mask, jets, 0.0)


class StepResidual(NamedTuple):
    """Moment mismatch under one fixed connected-support decision."""

    residual: tuple[jax.Array, jax.Array, jax.Array]
    booked: current.Booking
    reading: topology.TopologyRead
    valid: jax.Array


def _directional_field(operands: SupportOperands, direction):
    """Apply only the bank's linear rows, without its exterior constant."""
    stacked = jnp.stack(tuple(direction), axis=-1)
    jets = jnp.einsum("ncmq,cm->nq", operands.rows, stacked)
    return jets, fixed_points.field_from_jets(
        jets, operands.sample_index, operands.mesh
    )


def step_residual(
    operands,
    base_moments,
    moments,
    support: SupportRead,
    profile,
    quadrature,
) -> StepResidual:
    """Book a state while preserving one read's ownership and clip set.

    Continuous coefficients and stationary flux levels follow the bank. The
    point coordinates stay fixed by the stationary-point envelope argument;
    changes of ownership or clipped-cell set require a new support read.
    """
    direction = tuple(
        value - base for value, base in zip(moments, base_moments, strict=True)
    )
    _, field = _directional_field(operands, direction)
    reading = support.reading._replace(
        field_coefficients=support.reading.field_coefficients
        + field.sampled_coefficient(operands.geometry),
        axis_flux=support.reading.axis_flux + field.value(support.reading.axis),
        boundary_flux=support.reading.boundary_flux
        + field.value(support.reading.boundary),
    )
    psi = nodal_jets(operands, moments)[operands.sample_index, 0]
    booked = current.book(psi, reading, operands.geometry, profile, quadrature)
    valid = support.valid & booked.valid
    residual = tuple(
        jnp.where(valid, value - predicted, jnp.nan)
        for value, predicted in zip(moments, booked[:3], strict=True)
    )
    return StepResidual(residual, booked, reading, valid)


def step_jacobian_vector(
    operands,
    moments,
    direction,
    support: SupportRead,
    profile,
    density_gradient,
    quadrature,
    booked: current.Booking | None = None,
    *,
    fresh_geometry: bool = False,
):
    """Apply the bank, booking tangent and optional moving saddle geometry."""
    psi = nodal_jets(operands, moments)[operands.sample_index, 0]
    if booked is None:
        booked = current.book(
            psi, support.reading, operands.geometry, profile, quadrature
        )
    jets, field = _directional_field(operands, direction)
    tangent = current.book_tangent(
        psi,
        jets[operands.sample_index, 0],
        support.reading,
        field.sampled_coefficient(operands.geometry),
        field.value(support.reading.axis),
        field.value(support.reading.boundary),
        operands.geometry,
        profile,
        density_gradient,
        quadrature,
        booked,
    )
    if fresh_geometry:
        geometry_tangent = _normal_geometry_booking_tangent(
            operands, moments, jets, support, profile, quadrature, booked
        )
        tangent = jax.tree.map(lambda a, b: a + b, tangent, geometry_tangent)
    return tuple(
        value - change for value, change in zip(direction, tangent, strict=True)
    )


def _bell_derivatives(field, point, orders):
    return fixed_points.polynomial_derivatives(field, point, orders)


def _tensor_orders(rank):
    return tuple(
        (sum(index == 0 for index in axes), sum(index == 1 for index in axes))
        for axes in np.ndindex((2,) * rank)
    )


_HESSIAN_ORDERS = _tensor_orders(2)
_GRADIENT_ORDERS = _tensor_orders(1)
_THIRD_ORDERS = _tensor_orders(3)
_FOURTH_ORDERS = _tensor_orders(4)
_FIFTH_ORDERS = _tensor_orders(5)


def _curvature_rates(field, direction_field, point, point_rate, pitch):
    """Use unrounded bank curvature and its matching linear polynomial rates."""
    del pitch
    orders = _HESSIAN_ORDERS + _THIRD_ORDERS + _FOURTH_ORDERS + _FIFTH_ORDERS
    primal = _bell_derivatives(field, point, orders)
    varied = _bell_derivatives(direction_field, point, orders)
    hessian = primal[:4].reshape(2, 2)
    third = primal[4:12].reshape(2, 2, 2)
    fourth = primal[12:28].reshape(2, 2, 2, 2)
    fifth = primal[28:].reshape(2, 2, 2, 2, 2)
    hessian_rate = varied[:4].reshape(2, 2) + jnp.einsum("ijk,k->ij", third, point_rate)
    third_rate = varied[4:12].reshape(2, 2, 2) + jnp.einsum(
        "ijkl,l->ijk", fourth, point_rate
    )
    fourth_rate = varied[12:28].reshape(2, 2, 2, 2) + jnp.einsum(
        "ijklm,m->ijkl", fifth, point_rate
    )
    return hessian, hessian_rate, third, third_rate, fourth, fourth_rate


def _normal_form_rate(
    form, hessian, hessian_rate, third, third_rate, fourth, fourth_rate
):
    """Differentiate the two Hessian eigendirections and cubic ray correction."""
    eigenvalue, eigenvector = jnp.linalg.eigh(hessian)
    eigen_rate = jnp.einsum("ia,ij,ja->a", eigenvector, hessian_rate, eigenvector)
    coupling = eigenvector[:, 1] @ hessian_rate @ eigenvector[:, 0]
    gap = eigenvalue[0] - eigenvalue[1]
    vector_rate = jnp.stack(
        (eigenvector[:, 1] * coupling / gap, -eigenvector[:, 0] * coupling / gap),
        axis=1,
    )
    first = jnp.sqrt(eigenvalue[1]) * eigenvector[:, 0]
    second = jnp.sqrt(-eigenvalue[0]) * eigenvector[:, 1]
    first_rate = jnp.sqrt(eigenvalue[1]) * vector_rate[:, 0] + first * eigen_rate[1] / (
        2 * eigenvalue[1]
    )
    second_rate = jnp.sqrt(-eigenvalue[0]) * vector_rate[:, 1] + second * eigen_rate[
        0
    ] / (2 * eigenvalue[0])
    raw = jnp.stack((first + second, first - second, -first - second, -first + second))
    raw_rate = jnp.stack(
        (
            first_rate + second_rate,
            first_rate - second_rate,
            -first_rate - second_rate,
            -first_rate + second_rate,
        )
    )
    length = jnp.linalg.norm(raw, axis=1, keepdims=True)
    direction = raw / length
    direction_rate = (
        raw_rate - direction * jnp.sum(direction * raw_rate, axis=1, keepdims=True)
    ) / length
    order = jnp.argsort(jnp.arctan2(direction[:, 1], direction[:, 0]))
    direction_rate = direction_rate[order]
    direction = form.direction
    normal = direction @ hessian
    normal_rate = direction_rate @ hessian + direction @ hessian_rate
    third_ray = jnp.einsum("ijk,ai,aj,ak->a", third, direction, direction, direction)
    third_ray_rate = jnp.einsum(
        "ijk,ai,aj,ak->a", third_rate, direction, direction, direction
    ) + 3 * jnp.einsum("ijk,ai,aj,ak->a", third, direction_rate, direction, direction)
    norm = jnp.sum(normal * normal, axis=1)
    norm_rate = 2 * jnp.sum(normal * normal_rate, axis=1)
    curvature = form.curvature
    curvature_rate = -(
        third_ray_rate[:, None] * normal + third_ray[:, None] * normal_rate
    ) / (6 * norm[:, None]) + third_ray[:, None] * normal * norm_rate[:, None] / (
        6 * norm[:, None] ** 2
    )
    remainder = (
        0.5 * jnp.einsum("ai,ij,aj->a", curvature, hessian, curvature)
        + 0.5 * jnp.einsum("ijk,ai,aj,ak->a", third, direction, direction, curvature)
        + jnp.einsum(
            "ijkl,ai,aj,ak,al->a", fourth, direction, direction, direction, direction
        )
        / 24
    )
    remainder_rate = (
        jnp.einsum("ai,ij,aj->a", curvature_rate, hessian, curvature)
        + 0.5 * jnp.einsum("ai,ij,aj->a", curvature, hessian_rate, curvature)
        + 0.5
        * jnp.einsum("ijk,ai,aj,ak->a", third_rate, direction, direction, curvature)
        + jnp.einsum("ijk,ai,aj,ak->a", third, direction_rate, direction, curvature)
        + 0.5
        * jnp.einsum("ijk,ai,aj,ak->a", third, direction, direction, curvature_rate)
        + jnp.einsum(
            "ijkl,ai,aj,ak,al->a",
            fourth_rate,
            direction,
            direction,
            direction,
            direction,
        )
        / 24
        + jnp.einsum(
            "ijkl,ai,aj,ak,al->a",
            fourth,
            direction_rate,
            direction,
            direction,
            direction,
        )
        / 6
    )
    cubic_rate = (
        -(remainder_rate[:, None] * normal + remainder[:, None] * normal_rate)
        / norm[:, None]
        + remainder[:, None] * normal * norm_rate[:, None] / norm[:, None] ** 2
    )
    return direction_rate, curvature_rate, cubic_rate


def _ray_cut_rates(vertices, count, form, changes, tolerance):
    """Keep each cubic root's derivative beside it through subsequent sorts."""
    position_rate, direction_rate, curvature_rate, cubic_rate = changes
    width = vertices.shape[0]
    slot = jnp.arange(width)
    following = jnp.where(slot + 1 < count, slot + 1, 0)
    edge = vertices[following] - vertices
    relative = vertices - form.position
    cross = topology._cross_plane
    roots = topology._cubic_parameters(
        cross(form.cubic[:, None], edge[None]),
        cross(form.curvature[:, None], edge[None]),
        cross(form.direction[:, None], edge[None]),
        -cross(relative[None], edge[None]),
    )
    safe = jnp.where(jnp.isfinite(roots), roots, 0.0)
    ray = (
        form.direction[:, None, None]
        + 2 * safe[..., None] * form.curvature[:, None, None]
        + 3 * safe[..., None] ** 2 * form.cubic[:, None, None]
    )
    fixed_rate = (
        position_rate
        + safe[..., None] * direction_rate[:, None, None]
        + safe[..., None] ** 2 * curvature_rate[:, None, None]
        + safe[..., None] ** 3 * cubic_rate[:, None, None]
    )
    denominator = cross(ray, edge[None, :, None])
    root_rate = -cross(fixed_rate, edge[None, :, None]) / jnp.where(
        denominator != 0, denominator, 1.0
    )
    points = (
        safe[..., None] * form.direction[:, None, None]
        + safe[..., None] ** 2 * form.curvature[:, None, None]
        + safe[..., None] ** 3 * form.cubic[:, None, None]
    )
    square = jnp.sum(edge * edge, axis=-1)
    fraction = (
        jnp.sum((points - relative[None, :, None]) * edge[None, :, None], axis=-1)
        / jnp.where(square > 0, square, 1)[None, :, None]
    )
    fraction_rate = (
        jnp.sum(
            (fixed_rate + root_rate[..., None] * ray) * edge[None, :, None], axis=-1
        )
        / jnp.where(square > 0, square, 1)[None, :, None]
    )
    hit = (
        (slot < count)[None, :, None]
        & jnp.isfinite(roots)
        & (roots >= -tolerance)
        & (fraction >= -tolerance)
        & (fraction <= 1 + tolerance)
    )
    return roots, root_rate, fraction, fraction_rate, hit, edge, relative


def _sorted_rates(values, rates, lower, upper):
    clipped = jnp.clip(values, lower, upper)
    rate = jnp.where((values > lower) & (values < upper), rates, 0.0)
    return jax.lax.sort_key_val(clipped, rate, dimension=-1, is_stable=True)


def _edge_interval_rate(vertices, count, form, changes, tolerance):
    roots, _, fraction, fraction_rate, hit, edge, _ = _ray_cut_rates(
        vertices, count, form, changes, tolerance
    )
    width = vertices.shape[0]
    values = jnp.transpose(jnp.where(hit, fraction, jnp.inf), (1, 0, 2)).reshape(
        width, 12
    )
    rates = jnp.transpose(jnp.where(hit, fraction_rate, 0.0), (1, 0, 2)).reshape(
        width, 12
    )
    cuts, cut_rate = _sorted_rates(values, rates, 0.0, 1.0)
    cuts = jnp.concatenate((jnp.zeros((width, 1)), cuts, jnp.ones((width, 1))), -1)
    cut_rate = jnp.pad(cut_rate, ((0, 0), (1, 1)))
    lower, upper = cuts[:, :-1], cuts[:, 1:]
    middle = vertices[:, None] + ((lower + upper) / 2)[..., None] * edge[:, None]
    centre = (
        jnp.sum(jnp.where((jnp.arange(width) < count)[:, None], vertices, 0.0), axis=0)
        / count
    )
    sectors = topology._normal_form_sectors(middle + 1e-10 * (centre - middle), form)
    present = (jnp.arange(width) < count)[:, None] & (upper > lower)
    mask = jnp.transpose(sectors, (2, 0, 1))[..., None] & present[None, :, :, None]
    return jnp.where(
        mask, jnp.stack((cut_rate[:, :-1], cut_rate[:, 1:]), axis=-1)[None], 0.0
    )


def _normal_ring_rates(
    vertices, count, centre, form, selected, intervals, interval_rate, changes, segments
):
    position_rate, direction_rate, curvature_rate, cubic_rate = changes
    width = vertices.shape[0]
    slot = jnp.arange(width)
    following = jnp.where(slot + 1 < count, slot + 1, 0)
    edges = vertices[following] - vertices
    live_edge = slot < count
    first_rate = interval_rate[..., 0, None] * edges[None, :, None]
    last_rate = interval_rate[..., 1, None] * edges[None, :, None]
    straight_rate = jnp.stack(
        (jnp.zeros_like(first_rate), first_rate, last_rate), axis=-2
    ).reshape(-1, 3, 2)
    roots, root_rate, fraction, _, hit, _, relative = _ray_cut_rates(
        vertices, count, form, changes, 1e-12
    )
    bound_norm = jnp.linalg.norm(relative, axis=-1)
    index = jnp.argmax(bound_norm)
    bound = 2 * bound_norm[index]
    bound_rate = -2 * jnp.dot(relative[index], position_rate) / bound_norm[index]
    values = jnp.where(hit, roots, jnp.inf).reshape(4, -1)
    rates = jnp.where(hit, root_rate, 0.0).reshape(4, -1)
    parameters, parameter_rate = _sorted_rates(values, rates, 0.0, bound)
    parameter_rate = jnp.where(parameters >= bound, bound_rate, parameter_rate)
    parameters = jnp.concatenate(
        (jnp.zeros((4, 1)), parameters, jnp.full((4, 1), bound)), -1
    )
    parameter_rate = jnp.concatenate(
        (jnp.zeros((4, 1)), parameter_rate, jnp.full((4, 1), bound_rate)), -1
    )
    begin, end = parameters[:, :-1], parameters[:, 1:]
    begin_rate, end_rate = parameter_rate[:, :-1], parameter_rate[:, 1:]
    middle = (begin + end) / 2
    curve = (
        middle[..., None] * form.direction[:, None]
        + middle[..., None] ** 2 * form.curvature[:, None]
        + middle[..., None] ** 3 * form.cubic[:, None]
    )
    cross = topology._cross_plane
    inside = jnp.all(
        jnp.where(
            live_edge[None, None],
            cross(edges[None, None], curve[:, :, None] - relative[None, None])
            >= -1e-12,
            True,
        ),
        axis=-1,
    )
    chosen = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
    sector_live = jnp.zeros(4, dtype=bool).at[chosen].set(selected[:2])
    sign = sector_live.astype(vertices.dtype) - jnp.roll(sector_live, 1).astype(
        vertices.dtype
    )
    active = inside & (end > begin) & (sign[:, None] != 0)
    parameter = jnp.linspace(0, 1, segments + 1)
    t = begin[..., None] + (end - begin)[..., None] * parameter
    t_rate = begin_rate[..., None] + (end_rate - begin_rate)[..., None] * parameter
    arc_rate = (
        position_rate
        + t_rate[..., None]
        * (
            form.direction[:, None, None]
            + 2 * t[..., None] * form.curvature[:, None, None]
            + 3 * t[..., None] ** 2 * form.cubic[:, None, None]
        )
        + t[..., None] * direction_rate[:, None, None]
        + t[..., None] ** 2 * curvature_rate[:, None, None]
        + t[..., None] ** 3 * cubic_rate[:, None, None]
    )
    arc_rate = jnp.where(
        (sign < 0)[:, None, None, None], arc_rate[:, :, ::-1], arc_rate
    )
    arc_rate = arc_rate.reshape(-1, segments + 1, 2)
    arc_rate = jnp.pad(arc_rate, ((0, 0), (1, 0), (0, 0)))
    arc_rate = jnp.where(active.reshape(-1, 1, 1), arc_rate, 0.0)
    straight_rate = jnp.pad(straight_rate, ((0, 0), (0, segments - 1), (0, 0)))
    return jnp.concatenate((straight_rate, arc_rate))


def _normal_geometry_booking_tangent(
    operands, moments, jet_tangent, support, profile, quadrature, booked
):
    """Move admitted saddle rays and their edge intervals with the Bell field.

    The only differentiated inputs are the jets at the saddle's Bell triangle.
    Its stationary position follows the implicit Hessian equation. The small
    normal-form and edge-intersection expressions then supply ring velocities;
    moving-polygon moments integrate those velocities on the selected rings.
    """
    geometry = operands.geometry
    reading = support.reading
    cells = len(geometry.centre)
    zero = current.ClippedCurrentMoments(
        *(jnp.zeros(cells, dtype=jet_tangent.dtype) for _ in range(3))
    )
    active = (
        (reading.membership > 0)
        & (reading.membership < 1 - 1e-14)
        & reading.normal_form_cells
    )

    def evaluate(_):
        total_jets = nodal_jets(operands, moments)
        field = fixed_points.field_from_jets(
            total_jets,
            operands.sample_index,
            operands.mesh,
            bank_rows=operands.rows,
            bank_moments=jnp.stack(tuple(moments), axis=-1),
            bank_exterior=operands.exterior,
        )
        direction_field = fixed_points.field_from_jets(
            jet_tangent, operands.sample_index, operands.mesh
        )
        point = reading.saddle_form.position
        pitch = jnp.median(geometry.pitch)
        point_tangent = -jnp.linalg.solve(
            _bell_derivatives(field, point, _HESSIAN_ORDERS).reshape(2, 2),
            _bell_derivatives(direction_field, point, _GRADIENT_ORDERS),
        )

        hessian, hessian_tangent, third, third_tangent, fourth, fourth_tangent = (
            _curvature_rates(field, direction_field, point, point_tangent, pitch)
        )
        sigma = quadrature.sigma
        form = reading.saddle_form
        form_rates = _normal_form_rate(
            form,
            sigma * hessian,
            sigma * hessian_tangent,
            sigma * third,
            sigma * third_tangent,
            sigma * fourth,
            sigma * fourth_tangent,
        )
        psi = total_jets[operands.sample_index, 0]
        coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
        span = reading.boundary_flux - reading.axis_flux
        normalized = coefficients.at[:, 0].add(-reading.axis_flux) / span
        chosen = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
        indices = jnp.nonzero(active, size=cells, fill_value=0)[0]
        count = jnp.sum(active, dtype=jnp.int32)

        def add(index, carried):
            cell = indices[index]
            vertices = geometry.vertices[cell]
            vertex_count = geometry.vertex_count[cell]
            centre = geometry.centre[cell]
            intervals = reading.edge_interval[cell]
            selected = reading.fragment_selected[cell]

            changes = (point_tangent, *form_rates)
            interval_tangent = _edge_interval_rate(
                vertices,
                vertex_count,
                form,
                changes,
                topology.TopologyPolicy().edge_tolerance,
            )[chosen]
            interval_tangent = jnp.pad(
                interval_tangent,
                ((0, intervals.shape[0] - 2), (0, 0), (0, 0), (0, 0)),
            )

            rings_now, ring_count = current._normal_rings(
                vertices,
                vertex_count,
                centre,
                form,
                selected,
                intervals,
                quadrature.segments,
            )
            velocity = _normal_ring_rates(
                vertices,
                vertex_count,
                centre,
                form,
                selected,
                intervals,
                interval_tangent,
                changes,
                quadrature.segments,
            )

            moved = clip_quadrature.moving_polygon_current_moment_tangents(
                rings_now,
                velocity,
                ring_count,
                centre,
                geometry.pitch[cell],
                normalized[cell],
                profile,
            )
            return jax.tree.map(
                lambda prior, change: prior.at[cell].add(jnp.sum(change)),
                carried,
                moved,
            )

        raw = jax.lax.fori_loop(0, count, add, zero)
        total = jnp.sum(raw.cell_current)
        return current.ClippedCurrentMoments(
            *(
                booked.normalization * change
                - (
                    0.0
                    if quadrature.target_current is None
                    else value * total / booked.raw_current
                )
                for change, value in zip(raw, booked[:3], strict=True)
            )
        )

    return jax.lax.cond(jnp.any(active), evaluate, lambda _: zero, None)


class NewtonFactor(NamedTuple):
    """Balanced matrix and factor carried between contracting Newton steps."""

    matrix: jax.Array
    row: jax.Array
    column: jax.Array
    block_inverse: jax.Array
    indices: jax.Array


class NewtonStep(NamedTuple):
    """One Newton update; the linear residual belongs to the factor's matrix."""

    update: tuple[jax.Array, jax.Array, jax.Array]
    condition_before: jax.Array
    condition_after: jax.Array
    linear_residual: jax.Array
    converged: jax.Array
    factor: NewtonFactor
    refactored: jax.Array
    krylov_condition: jax.Array = 1.0
    krylov_residual: jax.Array = 0.0
    krylov_iterations: jax.Array = 0


_LOCAL_BOOKING_CUT_BATCH_WIDTH = 128


def _local_booking_sensitivities(
    operands, moments, support, profile, density_gradient, quadrature, booked
):
    """Differentiate one booking against its local flux and level inputs."""
    cells, samples = operands.sample_index.shape
    psi = nodal_jets(operands, moments)[operands.sample_index, 0]
    channels = samples + 8
    indices = jnp.arange(channels)
    sample_basis = jnp.eye(samples, dtype=psi.dtype)
    psi_direction = jnp.broadcast_to(
        sample_basis[jnp.minimum(indices, samples - 1), None]
        * (indices < samples)[:, None, None],
        (channels, cells, samples),
    )
    field_basis = jnp.eye(6, dtype=psi.dtype)
    field_direction = jnp.broadcast_to(
        field_basis[jnp.clip(indices - samples, 0, 5), None]
        * ((indices >= samples) & (indices < samples + 6))[:, None, None],
        (channels, cells, 6),
    )
    # Moving the boundary level is the negative constant-field direction.
    # Add that constant to cancel its ring velocity, retaining the interior
    # normalization derivative; recover the level channel by linearity below.
    field_direction = field_direction.at[samples + 7, :, 0].set(1.0)
    if quadrature.cut_batch_width == current.Quadrature().cut_batch_width:
        quadrature = replace(
            quadrature, cut_batch_width=_LOCAL_BOOKING_CUT_BATCH_WIDTH
        )
    response = current.book_tangent_channels(
        psi,
        psi_direction,
        support.reading,
        field_direction,
        (indices == samples + 6).astype(psi.dtype),
        (indices == samples + 7).astype(psi.dtype),
        operands.geometry,
        profile,
        density_gradient,
        quadrature,
        booked,
        jnp.arange(samples, samples + 6),
    )
    return response.at[:, :, samples + 7].add(-response[:, :, samples])


_FRESH_GEOMETRY_NODE_CAPACITY = 3


def step_jacobian(
    operands,
    moments,
    support,
    profile,
    density_gradient,
    quadrature,
    booked=None,
    *,
    fresh_geometry: bool = False,
):
    """Compose the bank with closed-form cell-local booking sensitivities.

    Fresh geometry admits one Bell triangle across the curvature stencil.
    A stencil crossing into further nodes returns a nonfinite matrix, so its
    truncated geometry cannot supply an accepted Newton correction.
    """
    if booked is None:
        psi = nodal_jets(operands, moments)[operands.sample_index, 0]
        booked = current.book(
            psi, support.reading, operands.geometry, profile, quadrature
        )
    if quadrature.target_current is None:
        local_quadrature, local_booked = quadrature, booked
    else:
        local_quadrature = replace(quadrature, target_current=None)
        local_booked = current.Booking(
            *(jnp.zeros_like(value) for value in booked[:3]),
            booked.raw_current,
            jnp.ones_like(booked.normalization),
            booked.valid,
        )
    local = _local_booking_sensitivities(
        operands,
        moments,
        support,
        profile,
        density_gradient,
        local_quadrature,
        local_booked,
    )
    cells = moments[0].size

    def bank_channels(index):
        direction = tuple(
            jnp.where(
                index // cells == block, jax.nn.one_hot(index % cells, cells), 0.0
            )
            for block in range(3)
        )
        jets, field = _directional_field(operands, direction)
        return jnp.concatenate(
            (
                jets[operands.sample_index, 0],
                field.sampled_coefficient(operands.geometry),
                jnp.broadcast_to(field.value(support.reading.axis), (cells, 1)),
                jnp.broadcast_to(field.value(support.reading.boundary), (cells, 1)),
            ),
            axis=-1,
        )

    channels = jax.vmap(bank_channels)(jnp.arange(3 * cells))
    response = jnp.einsum("cok,ick->coi", local, channels)
    response = jnp.transpose(response, (1, 0, 2)).reshape(3 * cells, 3 * cells)
    if quadrature.target_current is not None:
        current_tangent = jnp.sum(response[:cells], axis=0)
        booked_vector = jnp.concatenate(booked[:3])
        response = (
            booked.normalization * response
            - (booked_vector[:, None] / booked.raw_current) * current_tangent[None, :]
        )
    matrix = jnp.eye(3 * cells, dtype=response.dtype) - response
    if fresh_geometry:
        point = support.reading.saddle_form.position
        cached = fixed_points._CachedField(
            operands.mesh,
            jnp.arange(len(operands.nodes)),
            operands.rows,
            jnp.stack(tuple(moments), axis=-1),
            operands.exterior,
        )
        pitch = jnp.median(operands.geometry.pitch)
        offsets = jnp.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            )
        )
        stencil = jax.vmap(lambda p: cached._location(p)[2])(
            point + 0.01 * pitch * offsets
        ).reshape(-1)
        slot = jnp.arange(stencil.size)
        duplicate = (stencil[:, None] == stencil[None, :]) & (
            slot[:, None] > slot[None, :]
        )
        first = ~jnp.any(duplicate, axis=1)
        count = jnp.sum(first, dtype=jnp.int32)
        nodes = stencil[
            jnp.nonzero(first, size=_FRESH_GEOMETRY_NODE_CAPACITY, fill_value=0)[0]
        ]
        jet_rows = operands.rows[nodes]
        channels = jnp.transpose(jet_rows, (0, 3, 2, 1)).reshape(-1, 3 * cells)

        def local_channel(index):
            direction = (
                jnp.zeros_like(operands.exterior)
                .at[nodes[index // 6], index % 6]
                .set(1.0)
            )
            response = _normal_geometry_booking_tangent(
                operands, moments, direction, support, profile, quadrature, booked
            )
            return jnp.stack(tuple(response), axis=-1)

        channel_index = jnp.arange(6 * _FRESH_GEOMETRY_NODE_CAPACITY)
        local = jax.vmap(local_channel)(channel_index)
        local = jnp.where((channel_index < 6 * count)[:, None, None], local, 0.0)
        response = jnp.einsum("kco,ki->coi", local, channels)
        response = jnp.transpose(response, (1, 0, 2)).reshape(3 * cells, 3 * cells)
        matrix = jnp.where(
            count <= _FRESH_GEOMETRY_NODE_CAPACITY, matrix - response, jnp.nan
        )
    return matrix


def _equilibrate(matrix, moments, booked):
    """Scale moment blocks physically, then balance row and column norms."""
    scales = jnp.stack(
        tuple(
            jnp.maximum(jnp.maximum(jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))), 1e-12)
            for a, b in zip(moments, booked[:3], strict=True)
        )
    )
    column = jnp.repeat(scales, moments[0].size)
    row = column
    balanced = matrix * column[None, :] / row[:, None]
    for _ in range(3):
        row_norm = jnp.sqrt(jnp.max(jnp.abs(balanced), axis=1))
        row_norm = jnp.maximum(row_norm, 1e-12)
        balanced = balanced / row_norm[:, None]
        row = row * row_norm
        column_norm = jnp.sqrt(jnp.max(jnp.abs(balanced), axis=0))
        column_norm = jnp.maximum(column_norm, 1e-12)
        balanced = balanced / column_norm[None, :]
        column = column / column_norm
    return balanced, row, column


def _krylov_condition_estimate(matvec, size: int, width: int = 8):
    """Estimate conditioning from the singular values of an Arnoldi block."""
    width = min(width, size)
    seed = jnp.linspace(-1.0, 1.0, size)
    seed = seed / jnp.linalg.norm(seed)
    basis = jnp.zeros((width + 1, size), seed.dtype).at[0].set(seed)
    hessenberg = jnp.zeros((width + 1, width), seed.dtype)

    def advance(column, carried):
        vectors, h = carried
        candidate = matvec(vectors[column])

        def orthogonalize(row, state):
            remaining, matrix = state
            coefficient = jnp.vdot(vectors[row], remaining)
            remaining = remaining - coefficient * vectors[row]
            return remaining, matrix.at[row, column].set(coefficient)

        candidate, h = jax.lax.fori_loop(0, column + 1, orthogonalize, (candidate, h))
        length = jnp.linalg.norm(candidate)
        vectors = vectors.at[column + 1].set(
            candidate / jnp.where(length > 0, length, 1.0)
        )
        return vectors, h.at[column + 1, column].set(length)

    _, hessenberg = jax.lax.fori_loop(0, width, advance, (basis, hessenberg))
    singular = jnp.linalg.svd(hessenberg[:width, :width], compute_uv=False)
    largest = jnp.max(singular)
    cutoff = 64.0 * jnp.finfo(singular.dtype).eps * largest
    smallest = jnp.min(jnp.where(singular > cutoff, singular, jnp.inf))
    return jnp.maximum(largest / smallest, 1.0)


def _block_preconditioner(matrix):
    """Invert each cell's three moment components, stored in component blocks."""
    cells = matrix.shape[0] // 3
    indices = (
        jnp.arange(cells, dtype=jnp.int32)[:, None]
        + cells * jnp.arange(3, dtype=jnp.int32)[None, :]
    )
    blocks = matrix[indices[:, :, None], indices[:, None, :]]
    cofactors = jnp.stack(
        (
            jnp.cross(blocks[:, 1], blocks[:, 2]),
            jnp.cross(blocks[:, 2], blocks[:, 0]),
            jnp.cross(blocks[:, 0], blocks[:, 1]),
        ),
        axis=1,
    )
    determinant = jnp.sum(blocks[:, 0] * cofactors[:, 0], axis=-1)
    inverse = jnp.swapaxes(cofactors, -1, -2) / determinant[:, None, None]
    return inverse.reshape(-1, 3), indices.reshape(-1)


def _apply_block_inverse(inverse, indices, vector):
    values = jnp.einsum(
        "cij,cj->ci", inverse.reshape(-1, 3, 3), vector[indices].reshape(-1, 3)
    )
    return values.T.reshape(-1)


def _block_gmres(matrix, right, inverse, indices, *, max_iterations=400):
    """Restarted Arnoldi solve with a measured preconditioned residual bound.

    Twice-applied classical Gram-Schmidt keeps the basis orthogonal without
    a kernel per earlier vector. QR solves the small least-squares problem;
    the true residual, rather than its Hessenberg estimate, decides each restart.
    """
    from jax.scipy.linalg import solve_triangular

    def precondition(value):
        return _apply_block_inverse(inverse, indices, value)

    def matvec(value):
        return precondition(matrix @ value)

    condition = _krylov_condition_estimate(matvec, right.size)
    target = precondition(right)
    target_norm = jnp.linalg.norm(target)
    denominator = jnp.where(target_norm > 0, target_norm, 1.0)
    width = min(20, right.size)

    def relative(solution):
        return jnp.linalg.norm(precondition(right - matrix @ solution)) / denominator

    def cycle(carried):
        solution, steps, _ = carried
        residual = precondition(right - matrix @ solution)
        norm = jnp.linalg.norm(residual)
        vectors = jnp.zeros((right.size, width + 1), right.dtype)
        vectors = vectors.at[:, 0].set(residual / jnp.where(norm > 0, norm, 1.0))
        hessenberg = jnp.zeros((width + 1, width), right.dtype)

        def advance(state):
            basis, h, count, _ = state
            candidate = matvec(basis[:, count])
            scale = jnp.linalg.norm(candidate)
            projection = basis.T @ candidate
            candidate = candidate - basis @ projection
            correction = basis.T @ candidate
            candidate = candidate - basis @ correction
            length = jnp.linalg.norm(candidate)
            breakdown = length <= jnp.finfo(right.dtype).eps * scale
            basis = basis.at[:, count + 1].set(
                jnp.where(
                    breakdown, 0.0, candidate / jnp.where(length > 0, length, 1.0)
                )
            )
            h = h.at[:, count].set(projection + correction)
            h = h.at[count + 1, count].set(length)
            return basis, h, count + 1, breakdown

        vectors, hessenberg, count, _ = jax.lax.while_loop(
            lambda state: (
                (state[2] < width) & (steps + state[2] < max_iterations) & ~state[3]
            ),
            advance,
            (vectors, hessenberg, jnp.asarray(0), jnp.asarray(False)),
        )
        written = jnp.arange(width) < count
        hessenberg = jnp.where(written[None, :], hessenberg, 0.0)
        orthogonal, triangular = jnp.linalg.qr(hessenberg, mode="reduced")
        triangular = triangular + jnp.diag((~written).astype(right.dtype))
        coefficients = solve_triangular(triangular, norm * orthogonal[0])
        update = jnp.where(written[None, :], vectors[:, :width], 0.0) @ coefficients
        solution = solution + update
        return solution, steps + count, relative(solution)

    solution, steps, residual = jax.lax.while_loop(
        lambda state: (state[1] < max_iterations) & (condition * state[2] > 1e-8),
        cycle,
        (jnp.zeros_like(right), jnp.asarray(0), relative(jnp.zeros_like(right))),
    )
    return solution, condition, residual, steps


def newton_step(
    operands,
    moments,
    support: SupportRead,
    profile,
    density_gradient,
    quadrature,
    *,
    fresh_geometry: bool = False,
    condition_diagnostics: bool | None = None,
    factor_cache: NewtonFactor | None = None,
    contraction_ratio: jax.Array | float | None = None,
) -> NewtonStep:
    """Solve the bank-backed linear system at one fixed support decision.

    Local analytic booking sensitivities compose with the bank into a dense
    matrix. Physical moment scales and diagonal equilibration precede block
    GMRES. Its condition estimate and residual both describe the equilibrated,
    left-preconditioned operator; their product must be at most 1e-8.
    A carried block inverse skips assembly while the fresh residual contracts by at
    least half. Its linear residual certifies the carried matrix solve; the
    caller must check convergence against a fresh read of the new state.
    Exact condition estimates are automatic only for a single-cell diagnostic
    system; larger systems omit the two dense SVDs unless requested.
    """
    residual = step_residual(operands, moments, moments, support, profile, quadrature)
    cells = moments[0].size
    if condition_diagnostics is None:
        condition_diagnostics = cells == 1
    boundaries = (cells, 2 * cells)
    right = -jnp.concatenate(tuple(jnp.ravel(value) for value in residual.residual))

    def rebuild(_):
        matrix = step_jacobian(
            operands,
            moments,
            support,
            profile,
            density_gradient,
            quadrature,
            residual.booked,
            fresh_geometry=fresh_geometry,
        )
        balanced, row, column = _equilibrate(matrix, moments, residual.booked)
        inverse, indices = _block_preconditioner(balanced)
        return NewtonFactor(matrix, row, column, inverse, indices)

    if factor_cache is None:
        factor = rebuild(None)
        refactored = jnp.asarray(True)
    else:
        if contraction_ratio is None:
            raise ValueError(
                "factor reuse requires a measured residual contraction ratio"
            )
        reuse = jnp.isfinite(contraction_ratio) & (contraction_ratio <= 0.5)
        factor = jax.lax.cond(reuse, lambda _: factor_cache, rebuild, None)
        refactored = ~reuse
    balanced = factor.matrix * factor.column[None, :] / factor.row[:, None]
    solution, condition, krylov_residual, iterations = _block_gmres(
        balanced, right / factor.row, factor.block_inverse, factor.indices
    )
    solution = factor.column * solution
    right_norm = jnp.linalg.norm(right)
    linear_residual = jnp.linalg.norm(factor.matrix @ solution - right) / jnp.where(
        right_norm > 0, right_norm, 1.0
    )
    if condition_diagnostics:
        condition_before = jnp.linalg.cond(factor.matrix)
        condition_after = jnp.linalg.cond(balanced)
    else:
        condition_before = jnp.asarray(jnp.nan, dtype=factor.matrix.dtype)
        condition_after = condition_before
    return NewtonStep(
        tuple(jnp.split(solution, boundaries)),
        condition_before,
        condition_after,
        linear_residual,
        jnp.isfinite(krylov_residual) & (condition * krylov_residual <= 1e-8),
        factor,
        refactored,
        condition,
        krylov_residual,
        iterations,
    )


class BatchStep(NamedTuple):
    """A batched warm step and the one read decision the whole batch took."""

    step: NewtonStep
    support: SupportRead


def step_batch(
    operands,
    cache,
    moments,
    seeds,
    previous,
    convention,
    policy,
    profile,
    density_gradient,
    quadrature,
    axis_sign=-1.0,
    *,
    iterations: int = 12,
    tracking: TrackingState | None = None,
    fresh_geometry: bool = False,
    condition_diagnostics: bool = False,
    converged: jax.Array | None = None,
) -> BatchStep:
    """One warm Newton step over independent slices in one compiled program.

    ``moments``, ``seeds`` and ``previous`` carry one entry per slice on their
    leading axis, exactly as in :func:`read_batch`, and the operands and row
    window are shared by every lane. The read is :func:`read_batch`, so the
    batch takes one ``lax.cond`` on whether any lane needs a rebuild, outside
    the map; the step that consumes that read is a ``vmap`` over the lanes.
    A steady batch therefore shares the read's and the step's launches across
    the slices instead of charging each of them the whole per-call cost.

    This is the minimal-factor route: every lane assembles and factors its own
    matrix. A carried :class:`NewtonFactor` cannot be shared here, because the
    lanes hold different states and each factor belongs to one lane.

    ``converged`` is an optional per-lane flag: a lane flagged converged has
    its update zeroed, so its state is carried unchanged while the remaining
    lanes iterate. A lane that has met a residual bound would otherwise keep
    moving under a full Newton step and step back off the solution, which is
    what makes a whole-batch bound unreachable even where every lane passes it
    at some iteration.
    """
    support = read_batch(
        operands,
        cache,
        moments,
        seeds,
        previous,
        convention,
        policy,
        axis_sign,
        iterations=iterations,
        tracking=tracking,
    )

    def one(state, lane_support):
        return newton_step(
            operands,
            state,
            lane_support,
            profile,
            density_gradient,
            quadrature,
            fresh_geometry=fresh_geometry,
            condition_diagnostics=condition_diagnostics,
        )

    stepped = jax.vmap(one)(moments, support)
    if converged is None:
        return BatchStep(stepped, support)

    hold = jnp.asarray(converged, dtype=bool)
    update = tuple(
        jnp.where(
            hold.reshape(hold.shape + (1,) * (change.ndim - 1)),
            jnp.zeros_like(change),
            change,
        )
        for change in stepped.update
    )
    return BatchStep(stepped._replace(update=update), support)


class SolveResult(NamedTuple):
    """A capped bank solve certified by its terminal fresh read.

    ``reads`` counts every fresh evaluation per lane, including companion reads
    after a lane stops moving while another lane continues its search or solve.
    """

    moments: tuple[jax.Array, jax.Array, jax.Array]
    support: SupportRead
    residual: jax.Array
    converged: jax.Array
    steps: jax.Array
    refactors: jax.Array
    reads: jax.Array
    freezes: jax.Array
    unfreezes: jax.Array
    residual_path: jax.Array
    update_path: jax.Array
    factor: NewtonFactor
    factor_valid: jax.Array


def solve_batch(
    operands,
    cache,
    moments,
    seeds,
    previous,
    convention,
    policy,
    profile,
    density_gradient,
    quadrature,
    axis_sign=-1.0,
    *,
    max_steps: int = 60,
    tolerance: float = 1e-10,
    freeze_threshold: float = 1e-6,
    reuse_factor: bool = False,
    factor_cache: NewtonFactor | None = None,
    factor_valid: jax.Array | bool = True,
    contraction_limit: float = 0.5,
    freeze_active_set: bool = True,
    iterations: int = 12,
) -> SolveResult:
    """Converge independent lanes in one staged loop with a batch refactor branch.

    Intermediate reads carry the discrete connected decision while tracked
    cells and saddle liveness hold, refreshing all continuous geometry. A trial
    that appears converged is re-read in full before acceptance; the last
    allowed update also takes a full read. Once close to the root, the
    booking keeps the cut-cell, normal-form and fragment decisions of the first
    close read while every continuous reading remains current. A changed fresh
    decision or failed contraction releases that lock. Candidate updates halve
    until their fresh residual contracts. If no halving improves the incumbent,
    retain the smallest-residual valid trial: retaining the incumbent can stall
    at a non-smooth residual plateau above tolerance. Factor reuse is opt-in
    because extra iterations can cost more than skipped refactors. An incoming
    factor proposes the first update; a fresh-residual contraction check rejects
    that trial and rebuilds at the incumbent if it fails the contraction limit.
    The returned factor is usable only where ``factor_valid`` is true.

    ``profile`` is either one shared profile or a callable taking a traced
    lane index and returning that lane's profile. The callable is evaluated
    inside the lane map for both residuals and Newton booking derivatives.
    """
    lanes, cells = moments[0].shape
    lane_profile = profile if callable(profile) else lambda _: profile
    lane_indices = jnp.arange(lanes)
    support = read_batch(
        operands,
        cache,
        moments,
        seeds,
        previous,
        convention,
        policy,
        axis_sign,
        iterations=iterations,
    )

    def magnitude(state, reading):
        result = jax.vmap(
            lambda one, one_read, lane: step_residual(
                operands, one, one, one_read, lane_profile(lane), quadrature
            )
        )(state, reading, lane_indices)
        return jnp.max(jnp.abs(jnp.concatenate(result.residual, axis=-1)), axis=-1)

    def selection_changed(reading, cut_set, normal_set, fragments):
        cut = (reading.membership > 0) & (reading.membership < 1 - 1e-14)
        return (
            jnp.any(cut != cut_set, axis=-1)
            | jnp.any(reading.normal_form_cells != normal_set, axis=-1)
            | jnp.any(
                (reading.fragment_selected != fragments).reshape(lanes, -1), axis=-1
            )
        )

    norm = magnitude(moments, support)
    accepted = support.valid & jnp.isfinite(norm) & (norm <= tolerance)
    size = 3 * cells
    dtype = moments[0].dtype
    empty_factor = NewtonFactor(
        jnp.zeros((lanes, size, size), dtype),
        jnp.ones((lanes, size), dtype),
        jnp.ones((lanes, size), dtype),
        jnp.zeros((lanes, size, 3), dtype),
        jnp.zeros((lanes, size), jnp.int32),
    )
    empty_set = jnp.zeros_like(support.reading.normal_form_cells)
    locked_fragments = support.reading.fragment_selected
    initial = (
        moments,
        support,
        norm,
        accepted,
        jnp.zeros(lanes, bool),
        empty_set,
        empty_set,
        locked_fragments,
        empty_factor if factor_cache is None else factor_cache,
        jnp.asarray(False)
        if factor_cache is None
        else jnp.all(jnp.asarray(factor_valid)),
        jnp.full((lanes,), jnp.inf, dtype),
        jnp.zeros(lanes, jnp.int32),
        jnp.zeros(lanes, jnp.int32),
        jnp.ones(lanes, jnp.int32),
        jnp.zeros(lanes, jnp.int32),
        jnp.zeros(lanes, jnp.int32),
        jnp.full((lanes, max_steps + 1), jnp.nan, dtype).at[:, 0].set(norm),
        jnp.full((lanes, max_steps), jnp.nan, dtype),
    )

    def keep_going(carried):
        return jnp.any(~carried[3]) & (jnp.max(carried[11]) < max_steps)

    def advance(carried):
        (
            state,
            fresh,
            current_norm,
            done,
            frozen,
            cut_set,
            normal_set,
            locked_fragments,
            factor,
            have_factor,
            previous_norm,
            steps,
            refactors,
            reads,
            freezes,
            unfreezes,
            residual_path,
            update_path,
        ) = carried
        cut_now = (fresh.reading.membership > 0) & (
            fresh.reading.membership < 1 - 1e-14
        )
        normal_now = fresh.reading.normal_form_cells
        ratio = current_norm / previous_norm
        changed = (
            (~done)
            & frozen
            & (
                selection_changed(fresh.reading, cut_set, normal_set, locked_fragments)
                | ~jnp.isfinite(ratio)
                | (ratio > contraction_limit)
            )
        )
        start_freeze = (
            (~done)
            & (~frozen)
            & (~changed)
            & fresh.valid
            & (current_norm < freeze_threshold)
        )
        if not freeze_active_set:
            start_freeze = jnp.zeros_like(start_freeze)
        lock = (frozen & ~changed) | start_freeze
        cut_set = jnp.where(start_freeze[:, None], cut_now, cut_set)
        normal_set = jnp.where(start_freeze[:, None], normal_now, normal_set)
        locked_fragments = jnp.where(
            start_freeze.reshape(
                (lanes,) + (1,) * (fresh.reading.fragment_selected.ndim - 1)
            ),
            fresh.reading.fragment_selected,
            locked_fragments,
        )
        step_reading = fresh.reading._replace(
            normal_form_cells=jnp.where(lock[:, None], normal_set, normal_now),
            fragment_selected=jnp.where(
                lock.reshape(
                    (lanes,) + (1,) * (fresh.reading.fragment_selected.ndim - 1)
                ),
                locked_fragments,
                fresh.reading.fragment_selected,
            ),
        )
        step_support = fresh._replace(reading=step_reading)
        active = ~done
        rebuild = ~have_factor | jnp.any(
            active & (~jnp.isfinite(ratio) | (ratio > contraction_limit) | changed)
        )
        if not reuse_factor:
            rebuild = jnp.asarray(True)

        def one_with_factor(items):
            one, one_read, prior, lane = items
            return newton_step(
                operands,
                one,
                one_read,
                lane_profile(lane),
                density_gradient,
                quadrature,
                fresh_geometry=True,
                condition_diagnostics=False,
                factor_cache=prior,
                contraction_ratio=jnp.asarray(0.0),
            )

        def one_fresh(items):
            one, one_read, lane = items
            return newton_step(
                operands,
                one,
                one_read,
                lane_profile(lane),
                density_gradient,
                quadrature,
                fresh_geometry=True,
                condition_diagnostics=False,
            )

        correction = jax.lax.cond(
            rebuild,
            lambda _: jax.vmap(one_fresh)((state, step_support, lane_indices)),
            lambda _: jax.lax.map(
                one_with_factor, (state, step_support, factor, lane_indices)
            ),
            None,
        )

        def evaluate(trial):
            trial_read = read_batch(
                operands,
                cache,
                trial,
                seeds,
                previous,
                convention,
                policy,
                axis_sign,
                iterations=iterations,
                tracking=tracking_of(fresh),
                certificate=lambda reading: (
                    (magnitude(trial, reading) <= tolerance)
                    | (jnp.max(steps) + 1 >= max_steps)
                ),
            )
            return magnitude(trial, trial_read), trial_read.valid, trial_read

        search = _damped_scale(
            evaluate,
            state,
            correction.update,
            current_norm,
            fresh,
            active=active,
            max_steps=8,
            tolerance=tolerance,
            keep_incumbent=False,
        )
        candidate, next_read, next_norm = (
            search.state,
            search.auxiliary,
            search.residual,
        )
        if factor_cache is not None and reuse_factor:
            trial_ratio = next_norm / jnp.where(current_norm > 0, current_norm, 1.0)
            retry = (
                (~rebuild)
                & (jnp.max(steps) == 0)
                & jnp.any(
                    active
                    & (~jnp.isfinite(trial_ratio) | (trial_ratio > contraction_limit))
                )
            )

            def rebuild_trial(_):
                rebuilt = jax.vmap(one_fresh)((state, step_support, lane_indices))
                retried = _damped_scale(
                    evaluate,
                    state,
                    rebuilt.update,
                    current_norm,
                    fresh,
                    active=active,
                    max_steps=8,
                    tolerance=tolerance,
                    keep_incumbent=False,
                )
                return rebuilt, retried

            correction, checked = jax.lax.cond(
                retry, rebuild_trial, lambda _: (correction, search), None
            )
            candidate, next_read, next_norm = (
                checked.state,
                checked.auxiliary,
                checked.residual,
            )
            search = checked._replace(
                evaluations=search.evaluations
                + jnp.where(retry, checked.evaluations, 0)
            )
            rebuild = rebuild | retry
        drift = lock & selection_changed(
            next_read.reading, cut_set, normal_set, locked_fragments
        )
        next_done = done | (
            next_read.valid
            & jnp.isfinite(next_norm)
            & (next_norm <= tolerance)
            & ~drift
        )
        slot = jnp.max(steps)
        update_norm = jnp.max(
            jnp.abs(
                jnp.concatenate(
                    tuple(
                        after - before
                        for after, before in zip(candidate, state, strict=True)
                    ),
                    axis=-1,
                )
            ),
            axis=-1,
        )
        return (
            candidate,
            next_read,
            next_norm,
            next_done,
            lock & ~drift,
            cut_set,
            normal_set,
            locked_fragments,
            correction.factor,
            jnp.asarray(True),
            current_norm,
            steps + active.astype(jnp.int32),
            refactors + (active & rebuild).astype(jnp.int32),
            reads + search.evaluations,
            freezes + start_freeze.astype(jnp.int32),
            unfreezes + (changed | drift).astype(jnp.int32),
            residual_path.at[:, slot + 1].set(next_norm),
            update_path.at[:, slot].set(jnp.where(active, update_norm, 0.0)),
        )

    final = jax.lax.while_loop(keep_going, advance, initial)
    # A search with no valid trial can retain a carried incumbent at the cap.
    # Return its full fresh read too, even though the lane did not converge.
    terminal_needed = jnp.any(~final[3] & ~final[1].rebuilt)

    def terminal_read(_):
        reading = read_batch(
            operands,
            cache,
            final[0],
            seeds,
            previous,
            convention,
            policy,
            axis_sign,
            iterations=iterations,
        )
        return reading, magnitude(final[0], reading)

    terminal, terminal_norm = jax.lax.cond(
        terminal_needed, terminal_read, lambda _: (final[1], final[2]), None
    )
    return SolveResult(
        final[0],
        terminal,
        terminal_norm,
        final[3]
        & terminal.valid
        & jnp.isfinite(terminal_norm)
        & (terminal_norm <= tolerance),
        final[11],
        final[12],
        final[13] + terminal_needed.astype(jnp.int32),
        final[14],
        final[15],
        final[16],
        final[17],
        final[8],
        jnp.full((lanes,), final[9], bool),
    )


def solve(
    operands,
    cache,
    moments,
    seeds,
    previous,
    convention,
    policy,
    profile,
    density_gradient,
    quadrature,
    axis_sign=-1.0,
    **kwargs,
) -> SolveResult:
    """Converge one seed using the same staged driver as independent lanes."""
    batched = solve_batch(
        operands,
        cache,
        jax.tree.map(lambda value: value[None], moments),
        seeds[None],
        previous[None],
        convention,
        policy,
        profile,
        density_gradient,
        quadrature,
        axis_sign,
        **kwargs,
    )
    return jax.tree.map(lambda value: value[0], batched)


def _bank_payload(operands: SupportOperands) -> bytes:
    """Serialize the bank's dense operands and their shape identity."""
    import io

    buffer = io.BytesIO()
    np.savez(
        buffer,
        rows=np.asarray(operands.rows),
        exterior=np.asarray(operands.exterior),
        nodes=np.asarray(operands.nodes),
        sample_index=np.asarray(operands.sample_index),
        bucket=np.asarray(-1 if operands.bucket is None else operands.bucket),
        active_nodes=np.asarray(
            np.asarray(operands.rows).shape[0]
            if operands.active_nodes is None
            else operands.active_nodes
        ),
    )
    return buffer.getvalue()


class _BankFile:
    """A published bank payload with its content identity."""

    def __init__(self, name: str, payload: bytes):
        self.name = name
        self.sha256 = content_store.sha256_bytes(payload)
        self.size = len(payload)


def write_bank(operands: SupportOperands, cache_directory) -> str:
    """Publish the bank into the content store and return its digest."""
    payload = _bank_payload(operands)
    digest = "sha256:" + content_store.sha256_bytes(payload)
    manifest = json.dumps(
        {
            "kind": "nova-support-operator-bank",
            "bucket": operands.bucket,
            "active_nodes": None
            if operands.active_nodes is None
            else int(operands.active_nodes),
        },
        sort_keys=True,
    ).encode()
    content_store.publish_files(
        cache_directory,
        digest,
        files=(_BankFile("bank.npz", payload),),
        sources={"bank.npz": payload},
        manifest_bytes=manifest,
    )
    return digest


def read_bank(cache_directory, digest: str, geometry, mesh) -> SupportOperands:
    """Load a published bank and bind it to the machine's geometry and mesh."""
    object_root = content_store.verified_object_root(cache_directory, create=False)
    destination = content_store.verified_destination(
        object_root, content_store.digest_hex(digest)
    )
    if destination is None:
        raise content_store.ContentStoreError(f"bank {digest} is absent from the store")
    with np.load(destination / "bank.npz", allow_pickle=False) as archive:
        rows = archive["rows"]
        exterior = archive["exterior"]
        nodes = archive["nodes"]
        sample_index = archive["sample_index"]
        bucket = int(archive["bucket"])
        active_nodes = int(archive["active_nodes"])
    return SupportOperands(
        geometry,
        mesh,
        jnp.asarray(nodes),
        jnp.asarray(sample_index),
        jnp.asarray(rows),
        jnp.asarray(exterior),
        None if bucket < 0 else bucket,
        jnp.asarray(active_nodes),
    )
