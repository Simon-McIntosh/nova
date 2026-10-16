"""Precompute polygon point jets and read a cellwise Hermite field."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from types import FunctionType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.polygonanalytic import packed_analytic_moments
from nova.equilibrium import topology
from nova.database.filepath import operand_cache_root

JETS_KIND = "jets"
VALUES_KIND = "values"
COMPLETE_NAME = "complete.json"
MATRIX_NAME = "matrix.npy"


def source_bytes() -> tuple[bytes, ...]:
    """Return the bytes of every source file a jet key is computed over."""
    from nova.biot import polygonanalytic

    return (Path(__file__).read_bytes(), Path(polygonanalytic.__file__).read_bytes())


def operand_digest(*chunks: bytes) -> str:
    """Return the hex digest of every chunk, in order."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def operand_cache_directory(kind: str, digest: str) -> Path:
    """Return (creating if absent) the cache directory for one operand key."""
    directory = operand_cache_root() / kind / digest
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_operand(directory: Path, shape: tuple[int, ...]):
    """Return ``(matrix, build_seconds)`` for a completed entry, else ``None``.

    A directory without its completion marker is a build in flight, not a
    cache entry: the reader returns ``None`` and lets the caller rebuild
    rather than serve a partial matrix.
    """
    complete = directory / COMPLETE_NAME
    if not complete.is_file():
        return None
    matrix = np.load(directory / MATRIX_NAME, mmap_mode="r")
    if tuple(matrix.shape) != tuple(shape):
        return None
    return matrix, json.loads(complete.read_text())["build_seconds"]


def store_operand(directory: Path, matrix: np.ndarray, build_seconds: float):
    """Write a completed matrix and marker, each through an atomic rename.

    The matrix and its marker land under process-unique names and are moved
    into place with :func:`os.replace`, so a concurrent builder of the same
    key replaces whole files rather than interleaving writes into one, and a
    reader never observes a partial matrix.
    """
    pid = os.getpid()
    pending_matrix = directory / f"{MATRIX_NAME}.{pid}.pending.npy"
    np.save(pending_matrix, np.asarray(matrix))
    os.replace(pending_matrix, directory / MATRIX_NAME)
    pending_marker = directory / f"{COMPLETE_NAME}.{pid}.pending"
    pending_marker.write_text(
        json.dumps({"build_seconds": build_seconds, "bytes": int(matrix.nbytes)})
    )
    os.replace(pending_marker, directory / COMPLETE_NAME)


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
    when a completed entry for the same key exists, and otherwise built and
    stored there for the next process.  The key covers the unique node
    coordinates, the coupling operands and the hashed source files.
    """
    started = perf_counter()
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    sample_index = inverse.reshape(samples.shape[:-1])
    shape = (len(nodes), len(coupling.centre), 3, 6)
    directory = None
    load_seconds = None
    build_seconds = None
    cached = None
    if persist:
        operands = (
            np.asarray(leaf).tobytes() for leaf in jax.tree_util.tree_leaves(coupling)
        )
        digest = operand_digest(nodes.tobytes(), *operands, *source_bytes())
        directory = operand_cache_directory(JETS_KIND, digest)
        if receipt is not None:
            receipt["digest"] = digest
            receipt["cache_path"] = str(directory / MATRIX_NAME)
            receipt["cache_root"] = str(operand_cache_root())
        load_started = perf_counter()
        cached = load_operand(directory, shape)
        if cached is not None:
            load_seconds = perf_counter() - load_started
    if cached is not None:
        rows, build_seconds = cached
        rows = np.asarray(rows)
        print(
            f"JET_CACHE path={directory / MATRIX_NAME} bytes={rows.nbytes} "
            f"build_seconds={build_seconds:.6f} load_seconds={load_seconds:.6f}",
            flush=True,
        )
    else:
        rows = np.empty(shape, dtype=np.float64)
        build_started = perf_counter()
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
        build_seconds = perf_counter() - build_started
        if directory is not None:
            store_operand(directory, rows, build_seconds)
    matrix = jnp.asarray(rows)
    jax.block_until_ready(matrix)
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
    if receipt is not None:
        receipt.update(
            cache_hit=cached is not None,
            build_seconds=build_seconds,
            load_seconds=load_seconds,
        )
    print(
        f"JET_READY nodes={len(nodes)} cells={len(coupling.centre)} "
        f"bytes={rows.nbytes} seconds={elapsed:.6f}",
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
