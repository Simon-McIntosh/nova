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
from nova.equilibrium import current, fixed_points, topology

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
    previous decision is carried forward, so the own-node fit over every node
    (``sampled_coefficient``) and the whole-mesh Bell rebuild
    (``field_from_jets``) are paid by a rebuild step only.
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
    eigenvalues = jnp.linalg.eigvalsh(convention.sigma * solved.jet.hessian)
    inside = jax.vmap(
        lambda point: jnp.any(topology._inside_cells(operands.geometry, point))
    )(solved.position)
    live = solved.valid & inside
    cells = jnp.where(
        live, fixed_points.nearest_cell(operands.mesh, solved.position), -1
    )
    saddles = live & (eigenvalues[:, 0] < 0.0) & (eigenvalues[:, 1] > 0.0)
    saddle_live = jnp.any(saddles)

    def rebuild(_):
        field_jets = nodal_jets(operands, moments) if jets is None else jets
        field = fixed_points.field_from_jets(
            field_jets, operands.sample_index, operands.mesh
        )
        return topology.read_at_stationary_points(
            field, operands.geometry, convention, policy, solved
        )

    if tracking is None:
        rebuilt = jnp.asarray(True)
        reading = rebuild(None)
        rebuilds = jnp.asarray(1, jnp.int32)
    else:
        rebuilt = (
            jnp.any(cells != tracking.cells)
            | (saddle_live != tracking.saddle_live)
            | ~tracking.valid
        )
        reading = jax.lax.cond(
            rebuilt,
            rebuild,
            lambda _: tracking.reading,
            None,
        )
        rebuilds = tracking.rebuilds + rebuilt.astype(jnp.int32)
    valid = jnp.all(solved.valid) & reading.valid & reading.qualified
    return SupportRead(solved, reading, valid, cells, saddle_live, rebuilds, rebuilt)


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
            total_jets, operands.sample_index, operands.mesh
        )
        direction_field = fixed_points.field_from_jets(
            jet_tangent, operands.sample_index, operands.mesh
        )
        point = reading.saddle_form.position
        pitch = jnp.median(geometry.pitch)
        jet = field.evaluate(point)
        point_tangent = -jnp.linalg.solve(
            jet.hessian, direction_field.evaluate(point).gradient
        )

        def varied_field(amount):
            return replace(
                field,
                coefficients=field.coefficients + amount * direction_field.coefficients,
                jets=field.jets + amount * direction_field.jets,
            )

        def form_inputs(amount, position):
            varied = varied_field(amount)
            derivatives = topology._curvature_derivatives(varied, position, pitch)
            return varied.evaluate(position).hessian, *derivatives

        (hessian, third, fourth), (hessian_tangent, third_tangent, fourth_tangent) = (
            jax.jvp(
                form_inputs,
                (jnp.zeros((), point.dtype), point),
                (jnp.ones((), point.dtype), point_tangent),
            )
        )
        sigma = quadrature.sigma
        form = reading.saddle_form
        _, form_tangent = jax.jvp(
            lambda position, h, d3, d4: topology.saddle_normal_form(
                position,
                sigma * h,
                sigma * d3,
                topology.TopologyPolicy().hessian_tolerance,
                sigma * d4,
            ),
            (point, hessian, third, fourth),
            (point_tangent, hessian_tangent, third_tangent, fourth_tangent),
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

            def edges(position, direction, curvature, cubic):
                changed = form._replace(
                    position=position,
                    direction=direction,
                    curvature=curvature,
                    cubic=cubic,
                )
                return topology._normal_form_cell_fragments(
                    vertices,
                    vertex_count,
                    changed,
                    topology.TopologyPolicy().edge_tolerance,
                ).edge_interval[chosen]

            inputs = (form.position, form.direction, form.curvature, form.cubic)
            changes = (
                form_tangent.position,
                form_tangent.direction,
                form_tangent.curvature,
                form_tangent.cubic,
            )
            _, interval_tangent = jax.jvp(edges, inputs, changes)
            interval_tangent = jnp.pad(
                interval_tangent,
                ((0, intervals.shape[0] - 2), (0, 0), (0, 0), (0, 0)),
            )

            def rings(position, direction, curvature, cubic, interval):
                changed = form._replace(
                    position=position,
                    direction=direction,
                    curvature=curvature,
                    cubic=cubic,
                )
                return current._normal_rings(
                    vertices,
                    vertex_count,
                    centre,
                    changed,
                    selected,
                    interval,
                    quadrature.segments,
                )[0]

            rings_now, ring_count = current._normal_rings(
                vertices,
                vertex_count,
                centre,
                form,
                selected,
                intervals,
                quadrature.segments,
            )
            _, velocity = jax.jvp(
                rings,
                (*inputs, intervals),
                (*changes, interval_tangent),
            )

            def moving(ring, speed, number):
                return current.moving_polygon_current_moment_tangent(
                    ring,
                    speed,
                    number,
                    centre,
                    geometry.pitch[cell],
                    normalized[cell],
                    profile,
                )

            moved = jax.vmap(moving)(rings_now, velocity, ring_count)
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


class NewtonStep(NamedTuple):
    """One dense Newton update and its conditioning ledger."""

    update: tuple[jax.Array, jax.Array, jax.Array]
    condition_before: jax.Array
    condition_after: jax.Array
    linear_residual: jax.Array
    converged: jax.Array


def _local_booking_sensitivities(
    operands, moments, support, profile, density_gradient, quadrature, booked
):
    """Differentiate one booking against its local flux and level inputs."""
    cells, samples = operands.sample_index.shape
    psi = nodal_jets(operands, moments)[operands.sample_index, 0]
    channels = samples + 8
    sample_basis = jnp.eye(samples, dtype=psi.dtype)
    field_basis = jnp.eye(6, dtype=psi.dtype)

    def evaluate(index):
        psi_direction = jnp.broadcast_to(
            sample_basis[jnp.minimum(index, samples - 1)] * (index < samples),
            psi.shape,
        )
        field_index = jnp.clip(index - samples, 0, 5)
        field_direction = jnp.broadcast_to(
            field_basis[field_index] * ((index >= samples) & (index < samples + 6)),
            (cells, 6),
        )
        tangent = current.book_tangent(
            psi,
            psi_direction,
            support.reading,
            field_direction,
            (index == samples + 6).astype(psi.dtype),
            (index == samples + 7).astype(psi.dtype),
            operands.geometry,
            profile,
            density_gradient,
            quadrature,
            booked,
        )
        return jnp.stack(tuple(tangent), axis=-1)

    return jnp.moveaxis(jax.lax.map(evaluate, jnp.arange(channels)), 0, -1)


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
    """Compose the bank with closed-form cell-local booking sensitivities."""
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
        nodes = stencil[jnp.nonzero(first, size=stencil.size, fill_value=0)[0]]
        jet_rows = operands.rows[nodes]
        channels = jnp.transpose(jet_rows, (0, 3, 2, 1)).reshape(-1, 3 * cells)

        def local_channel(index, carried):
            direction = (
                jnp.zeros_like(operands.exterior)
                .at[nodes[index // 6], index % 6]
                .set(1.0)
            )
            response = _normal_geometry_booking_tangent(
                operands, moments, direction, support, profile, quadrature, booked
            )
            return carried.at[index].set(jnp.stack(tuple(response), axis=-1))

        local = jax.lax.fori_loop(
            0,
            6 * count,
            local_channel,
            jnp.zeros((6 * stencil.size, cells, 3), dtype=matrix.dtype),
        )
        response = jnp.einsum("kco,ki->coi", local, channels)
        response = jnp.transpose(response, (1, 0, 2)).reshape(3 * cells, 3 * cells)
        matrix = matrix - response
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


def newton_step(
    operands,
    moments,
    support: SupportRead,
    profile,
    density_gradient,
    quadrature,
    *,
    fresh_geometry: bool = False,
) -> NewtonStep:
    """Solve the bank-backed linear system at one fixed support decision.

    Local analytic booking sensitivities compose with the bank into a dense
    matrix. Physical moment scales and diagonal equilibration precede LU.
    """
    residual = step_residual(operands, moments, moments, support, profile, quadrature)
    cells = moments[0].size
    boundaries = (cells, 2 * cells)
    right = -jnp.concatenate(tuple(jnp.ravel(value) for value in residual.residual))
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
    from jax.scipy import linalg

    factor = linalg.lu_factor(balanced)
    solution = column * linalg.lu_solve(factor, right / row)
    linear_residual = jnp.linalg.norm(matrix @ solution - right) / jnp.linalg.norm(
        right
    )
    return NewtonStep(
        tuple(jnp.split(solution, boundaries)),
        jnp.linalg.cond(matrix),
        jnp.linalg.cond(balanced),
        linear_residual,
        linear_residual <= 1e-8,
    )


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
