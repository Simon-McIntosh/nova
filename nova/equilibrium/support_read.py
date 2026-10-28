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
from dataclasses import dataclass, field
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


def stationary_cells(operands: SupportOperands, position, live) -> jax.Array:
    """Index the hex cell that holds each tracked point, ``-1`` when not live.

    The cell is the one whose generator is nearest, which is the same
    selection ``fixed_points._CachedField`` evaluates the cell-local Bell fit
    in, so a reused decision and a rebuilt one agree on where the point sits.
    """
    distance = jnp.sum(
        (operands.mesh.generators[None, :, :] - position[:, None, :]) ** 2, axis=-1
    )
    return jnp.where(live, jnp.argmin(distance, axis=-1), -1)


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
    cells = stationary_cells(operands, solved.position, live)
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
    return SupportRead(
        solved, reading, valid, cells, saddle_live, rebuilds, rebuilt
    )


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
