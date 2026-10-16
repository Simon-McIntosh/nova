"""Precompute polygon point jets and read a cellwise Hermite field."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter
from types import FunctionType
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.polygonanalytic import packed_analytic_moments
from nova.database.content_store import (
    ContentStoreError,
    MANIFEST_FILENAME,
    entry_metadata,
    file_content_identity,
    publish_files,
    read_regular_bytes,
    sha256_bytes,
    verified_destination,
    verified_object_root,
)
from nova.database.filepath import operand_cache_root
from nova.equilibrium import topology

JETS_KIND = "jets"
VALUES_KIND = "values"
MATRIX_NAME = "matrix.npy"


@dataclass(frozen=True)
class OperandFile:
    """One declared operand file, in the form the content store verifies."""

    name: str
    sha256: str
    size: int


def source_bytes() -> tuple[bytes, ...]:
    """Return the bytes of every source file a jet key is computed over."""
    from nova.biot import polygonanalytic

    return (Path(__file__).read_bytes(), Path(polygonanalytic.__file__).read_bytes())


def operand_key(*chunks: bytes) -> str:
    """Return the content-store digest naming one operand from its key chunks."""
    return "sha256:" + sha256_bytes(b"".join(chunks))


def kind_root(kind: str) -> Path:
    """Return the cache directory one operand kind is published under."""
    return operand_cache_root() / kind


def operand_entry(kind: str, digest: str) -> Path | None:
    """Return the store's verified directory for one operand key, else ``None``.

    A cold cache is a miss, not an error: the object root does not exist until
    the first operand of that kind is published. Everything else the store
    reports about the root and the destination, such as an object root that is a
    symlink or a destination that escaped it, is raised rather than read.
    """
    try:
        object_root = verified_object_root(kind_root(kind), create=False)
    except ContentStoreError:
        return None
    return verified_destination(object_root, digest.removeprefix("sha256:"))


@dataclass(frozen=True)
class OperandEntry:
    """A published operand: its mapped matrix, recorded build wall and directory."""

    matrix: np.ndarray
    build_seconds: float
    directory: Path


def read_operand(kind: str, digest: str, shape: tuple[int, ...]) -> OperandEntry | None:
    """Return the published operand for one key, else ``None``.

    The matrix is memory-mapped rather than read, so a 1.9 GB entry costs a few
    milliseconds to open and streams from GPFS as it is consumed. For the same
    reason the read does not re-hash the matrix: a full verification would read
    more from disk than the assembly the entry exists to save. What it checks is
    that the store's own root and destination are canonical, that the manifest
    names this kind and shape, and that the matrix is the declared size. The
    content digest of every published file is verified by the store before the
    entry becomes visible, which is where that guarantee belongs.
    """
    directory = operand_entry(kind, digest)
    if directory is None:
        return None
    manifest_path = directory / MANIFEST_FILENAME
    if entry_metadata(manifest_path) is None:
        return None
    payload = json.loads(read_regular_bytes(manifest_path))
    if payload.get("kind") != kind or tuple(payload.get("shape", ())) != tuple(shape):
        return None
    declared = {item["name"]: item for item in payload.get("files", [])}
    if MATRIX_NAME not in declared:
        return None
    matrix_path = directory / MATRIX_NAME
    metadata = entry_metadata(matrix_path)
    if metadata is None:
        return None
    if metadata.st_size != declared[MATRIX_NAME]["size"]:
        raise ContentStoreError(f"operand is not the declared size: {matrix_path}")
    matrix = np.load(matrix_path, mmap_mode="r")
    return OperandEntry(matrix, float(payload["build_seconds"]), directory)


def publish_operand(
    kind: str, digest: str, matrix: np.ndarray, build_seconds: float
) -> Path:
    """Publish one operand through the content store and return its directory.

    The matrix lands in a private directory that is renamed into place without
    replacement, so a reader sees the whole entry or none of it and two
    concurrent builders of one key cannot interleave writes into one file.
    """
    root = kind_root(kind)
    root.mkdir(parents=True, exist_ok=True)
    pending = root / f".pending-{os.getpid()}.npy"
    try:
        np.save(pending, np.asarray(matrix))
        identity, size = file_content_identity(pending)
        manifest = json.dumps(
            {
                "kind": kind,
                "shape": [int(length) for length in np.shape(matrix)],
                "build_seconds": float(build_seconds),
                "files": [{"name": MATRIX_NAME, "sha256": identity, "size": size}],
            },
            sort_keys=True,
        ).encode()
        return publish_files(
            root,
            digest,
            [OperandFile(MATRIX_NAME, identity, size)],
            {MATRIX_NAME: pending},
            manifest,
        )
    finally:
        pending.unlink(missing_ok=True)


def operand_matrix(
    kind: str,
    digest: str,
    shape: tuple[int, ...],
    assemble: Callable[[], np.ndarray],
    transfer: Callable[[np.ndarray], Any],
    persist: bool = True,
    receipt: dict[str, Any] | None = None,
):
    """Return ``(device matrix, entry directory)`` for one operand of ``kind``.

    On a hit the mapped matrix is transferred to the device; on a miss
    ``assemble`` builds it, ``transfer`` moves it, and the store publishes it
    under ``digest``. The receipt prices only what this process paid:
    ``load_seconds`` is the hit wall from opening the entry to a device-resident
    matrix, ``build_seconds`` the miss assembly, and ``matrix_ready_seconds``
    the wall that got the matrix ready either way. ``stored_build_seconds`` is
    the wall the entry's own builder recorded, which is not this process's cost.
    """
    entry = read_operand(kind, digest, shape) if persist else None
    if entry is not None:
        load_started = perf_counter()
        matrix = transfer(entry.matrix)
        load_seconds = perf_counter() - load_started
        if receipt is not None:
            receipt.update(
                kind=kind,
                digest=digest,
                cache_hit=True,
                cache_path=str(entry.directory / MATRIX_NAME),
                build_seconds=None,
                load_seconds=load_seconds,
                publish_seconds=None,
                stored_build_seconds=entry.build_seconds,
                matrix_ready_seconds=load_seconds,
                matrix_bytes=int(entry.matrix.nbytes),
            )
        return matrix, entry.directory

    build_started = perf_counter()
    host = assemble()
    build_seconds = perf_counter() - build_started
    transfer_started = perf_counter()
    matrix = transfer(host)
    ready_seconds = build_seconds + perf_counter() - transfer_started
    directory = None
    publish_seconds = None
    stored_build_seconds = None
    if persist:
        publish_started = perf_counter()
        try:
            directory = publish_operand(kind, digest, host, build_seconds)
        except ContentStoreError:
            published = read_operand(kind, digest, shape)
            if published is None:
                raise
            directory = published.directory
            stored_build_seconds = published.build_seconds
        publish_seconds = perf_counter() - publish_started
    if receipt is not None:
        receipt.update(
            kind=kind,
            digest=digest,
            cache_hit=False,
            cache_path=None if directory is None else str(directory / MATRIX_NAME),
            build_seconds=build_seconds,
            load_seconds=None,
            stored_build_seconds=stored_build_seconds,
            publish_seconds=publish_seconds,
            matrix_ready_seconds=ready_seconds,
            matrix_bytes=int(host.nbytes),
        )
    return matrix, directory


def _source_jets(point, coupling):
    """Value, gradient and symmetric Hessian for each booked moment basis."""

    def value_gradient(target):
        rows = packed_analytic_moments(
            jnp,
            jnp.broadcast_to(target[0], coupling.norm.shape),
            jnp.broadcast_to(target[1], coupling.norm.shape),
            coupling.edge,
            coupling.weight,
            coupling.norm,
            coupling.centre.T,
            coupling.centre.T,
            coupling.reflection_axis,
            coupling.reflection_partner,
        )
        value = jnp.stack(rows[:3], axis=-1)
        gradient = (
            2
            * jnp.pi
            * target[0]
            * jnp.stack(
                (jnp.stack(rows[6:9], axis=-1), -jnp.stack(rows[3:6], axis=-1)),
                axis=-1,
            )
        )
        return value, gradient

    value, gradient = value_gradient(point)
    hessian = jax.jacfwd(lambda target: value_gradient(target)[1])(point)
    coefficients = jnp.concatenate(
        (
            value[..., None],
            gradient,
            hessian[..., 0, 0, None],
            hessian[..., 0, 1, None],
            hessian[..., 1, 1, None],
        ),
        axis=-1,
    )
    inverse = coupling.inverse_second_moment
    transform = jnp.stack(
        (
            jnp.stack(
                (jnp.ones(len(value)), jnp.zeros(len(value)), jnp.zeros(len(value))),
                axis=-1,
            ),
            jnp.stack(
                (jnp.zeros(len(value)), inverse[:, 0, 0], inverse[:, 0, 1]), axis=-1
            ),
            jnp.stack(
                (jnp.zeros(len(value)), inverse[:, 1, 0], inverse[:, 1, 1]), axis=-1
            ),
        ),
        axis=1,
    )
    return jnp.einsum("cbq,cbm->cmq", coefficients, transform)


def build_jet_matrix(
    coupling,
    geometry,
    total,
    reference_moments,
    batch=4,
    persist=True,
    receipt: dict[str, Any] | None = None,
):
    """Build the three (nodes, cells, six-jet) source matrices in batches.

    With ``persist`` set the matrix is loaded from the shared operand cache
    when an entry for the same key is published, and otherwise built and
    published there for the next process.  The key covers the unique node
    coordinates, the coupling operands and the hashed source files.
    """
    started = perf_counter()
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    sample_index = inverse.reshape(samples.shape[:-1])
    shape = (len(nodes), len(coupling.centre), 3, 6)
    operands = (
        np.asarray(leaf).tobytes() for leaf in jax.tree_util.tree_leaves(coupling)
    )
    digest = operand_key(nodes.tobytes(), *operands, *source_bytes())
    if receipt is not None:
        receipt["digest"] = digest
        receipt["cache_root"] = str(operand_cache_root())

    def assemble():
        rows = np.empty(shape, dtype=np.float64)
        point_batch = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))
        for first in range(0, len(nodes), batch):
            last = min(first + batch, len(nodes))
            points = nodes[first:last]
            if len(points) < batch:
                points = np.concatenate(
                    (points, np.repeat(points[-1:], batch - len(points), axis=0))
                )
            rows[first:last] = np.asarray(point_batch(jnp.asarray(points), coupling))[
                : last - first
            ]
            if first % (batch * 500) == 0:
                elapsed = perf_counter() - started
                print(
                    f"JET_BUILD nodes={last}/{len(nodes)} seconds={elapsed:.3f}",
                    flush=True,
                )
        return rows

    def transfer(rows):
        matrix = jnp.asarray(rows)
        jax.block_until_ready(matrix)
        return matrix

    matrix, directory = operand_matrix(
        JETS_KIND, digest, shape, assemble, transfer, persist=persist, receipt=receipt
    )
    if receipt is not None and receipt["cache_hit"]:
        print(
            f"JET_CACHE path={directory / MATRIX_NAME} "
            f"bytes={receipt['matrix_bytes']} "
            f"stored_build_seconds={receipt['stored_build_seconds']:.6f} "
            f"load_seconds={receipt['load_seconds']:.6f}",
            flush=True,
        )
    reference = np.stack(
        tuple(np.asarray(value) for value in reference_moments), axis=-1
    )
    total_jet = jax.jit(jax.vmap(total.evaluate))(jnp.asarray(nodes))
    exterior = jnp.stack(
        (
            total_jet.value,
            total_jet.gradient[:, 0],
            total_jet.gradient[:, 1],
            total_jet.hessian[:, 0, 0],
            total_jet.hessian[:, 0, 1],
            total_jet.hessian[:, 1, 1],
        ),
        axis=-1,
    ) - jnp.einsum("ncmq,cm->nq", matrix, jnp.asarray(reference))
    jax.block_until_ready(exterior)
    elapsed = perf_counter() - started
    print(
        f"JET_READY nodes={len(nodes)} cells={len(coupling.centre)} "
        f"bytes={int(matrix.size) * matrix.dtype.itemsize} "
        f"seconds={elapsed:.6f}",
        flush=True,
    )
    return matrix, exterior, jnp.asarray(nodes), jnp.asarray(sample_index), elapsed


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HermiteField:
    """Seven-node quadratic Taylor jets blended with sixth-order Shepard weights.

    The element reproduces each node's value, gradient and Hessian exactly.
    For a smooth field its local value, gradient and Hessian errors are third,
    second and first order in cell pitch, respectively.
    """

    jets: jax.Array
    nodes: jax.Array
    sample_index: jax.Array
    centre: jax.Array
    pitch: jax.Array

    def value(self, point):
        cell = jnp.argmin(jnp.sum((self.centre - point) ** 2, axis=-1))
        return self.local_value(cell, point)

    def local_value(self, cell, point):
        """Evaluate this cell explicitly, including on a shared edge."""
        node = self.nodes[self.sample_index[cell]]
        jet = self.jets[self.sample_index[cell]]
        delta = point[None, :] - node
        squared = jnp.sum((delta / self.pitch[cell]) ** 2, axis=-1)
        factors = jnp.where(
            jnp.eye(len(squared), dtype=bool), 1.0, squared[None, :] ** 3
        )
        weight = jnp.prod(factors, axis=-1)
        weight = weight / jnp.sum(weight)
        taylor = (
            jet[:, 0]
            + jnp.sum(jet[:, 1:3] * delta, axis=-1)
            + 0.5
            * (
                jet[:, 3] * delta[:, 0] ** 2
                + 2 * jet[:, 4] * delta[:, 0] * delta[:, 1]
                + jet[:, 5] * delta[:, 1] ** 2
            )
        )
        return jnp.sum(weight * taylor)

    def evaluate(self, point):
        value = self.value(point)
        gradient = jax.grad(self.value)(point)
        hessian = jax.hessian(self.value)(point)
        return topology.FieldJet(value, gradient, hessian)


def read_with_jets(
    matrix, moments, exterior, nodes, sample_index, geometry, convention, policy
):
    """Contract the runtime jet matrix once, then call the unchanged read body."""
    stacked = jnp.stack(tuple(moments), axis=-1)
    jets = exterior + jnp.einsum("ncmq,cm->nq", matrix, stacked)
    field = HermiteField(jets, nodes, sample_index, geometry.centre, geometry.pitch)

    return read_field(field, geometry, convention, policy, finite_curvature=True)


def read_field(field, geometry, convention, policy, *, finite_curvature=False):
    """Adapt nodal samples once for both polynomial field representations."""

    def sampled_flux(field, geometry):
        values = field.jets[field.sample_index, 0]
        if hasattr(field, "cell_value"):
            centres = jax.vmap(field.cell_value)(
                jnp.arange(len(values)), geometry.centre
            )
            values = values.at[:, 0].set(centres)
        return jnp.einsum("nij,nj->ni", geometry.fit_inverse, values), values[:, 0]

    curvature = topology._curvature_derivatives
    if finite_curvature:
        curvature = FunctionType(
            curvature.__code__, dict(curvature.__globals__, TotalField=type(field))
        )
    adapted = FunctionType(
        topology.read.__code__,
        dict(
            topology.read.__globals__,
            _sampled_flux=sampled_flux,
            _curvature_derivatives=curvature,
        ),
    )
    return adapted(field, geometry, convention, policy)
