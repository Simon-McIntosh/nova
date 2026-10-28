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


class SupportRead(NamedTuple):
    """The read, its stationary points and one overall verdict."""

    stationary: fixed_points.FixedPointResult
    reading: topology.TopologyRead
    valid: jax.Array


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
):
    """Apply the bank and the three closed-form booking tangent channels."""
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
    return tuple(
        value - change for value, change in zip(direction, tangent, strict=True)
    )


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
    operands, moments, support, profile, density_gradient, quadrature, booked=None
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
    return jnp.eye(3 * cells, dtype=response.dtype) - response


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


def _krylov_solve(matvec, right, tolerance, restart: int = 20, maxiter: int = 20):
    """Solve a linear operator without tracing a transpose of its program."""
    width = min(restart, right.size)
    basis_dtype = right.dtype
    right_norm = jnp.linalg.norm(right)
    threshold = tolerance * right_norm

    def one_restart(state):
        solution, iteration, _ = state
        residual = right - matvec(solution)
        length = jnp.linalg.norm(residual)
        basis = (
            jnp.zeros((width + 1, right.size), basis_dtype)
            .at[0]
            .set(residual / jnp.where(length > 0, length, 1.0))
        )
        hessenberg = jnp.zeros((width + 1, width), basis_dtype)

        def advance(column, carried):
            vectors, h = carried
            candidate = matvec(vectors[column])

            def orthogonalize(row, value):
                remaining, matrix = value
                coefficient = jnp.vdot(vectors[row], remaining)
                remaining = remaining - coefficient * vectors[row]
                return remaining, matrix.at[row, column].set(coefficient)

            candidate, h = jax.lax.fori_loop(
                0, column + 1, orthogonalize, (candidate, h)
            )
            norm = jnp.linalg.norm(candidate)
            vectors = vectors.at[column + 1].set(
                candidate / jnp.where(norm > 0, norm, 1.0)
            )
            return vectors, h.at[column + 1, column].set(norm)

        basis, hessenberg = jax.lax.fori_loop(0, width, advance, (basis, hessenberg))
        target = jnp.zeros(width + 1, basis_dtype).at[0].set(length)
        coefficients = jnp.linalg.pinv(hessenberg, rtol=1e-12) @ target
        solution = solution + basis[:width].T @ coefficients
        error = jnp.linalg.norm(right - matvec(solution))
        return solution, iteration + 1, error

    def continue_solve(state):
        _, iteration, error = state
        return (iteration < maxiter) & (error > threshold)

    initial = (jnp.zeros_like(right), jnp.asarray(0), right_norm)
    solution, _, error = jax.lax.while_loop(continue_solve, one_restart, initial)
    return solution, error / jnp.maximum(right_norm, jnp.finfo(right.dtype).tiny)


def newton_step(
    operands,
    moments,
    support: SupportRead,
    profile,
    density_gradient,
    quadrature,
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
