"""Exact analytic exterior jets supplied to the production moment operator."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from nova.equilibrium import fixed_points, support_read, topology


def bind_analytic_exterior(
    operator, field, exact_moments, roots, *, value_tolerance=1e-11, row_cache=None
):
    """Bind closed-form jets minus the exact-support current's kernel image.

    The fixture's existing sampled exterior is an independent value check at
    every shared node. No legacy support read is used to construct either the
    jet field or its current moments.
    """
    moment_geometry = operator.moment_geometry
    wall = np.asarray(operator.wall.coordinate)
    units = tuple(
        wall[start:stop]
        for start, stop in zip(
            operator.wall_unit_offsets[:-1], operator.wall_unit_offsets[1:], strict=True
        )
    )
    geometry = topology.TopologyGeometry.from_cells(
        moment_geometry.polygons, moment_geometry.sampling_vertices, units
    )
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    coupling = topology.BiotMomentCoupling.from_polygons(moment_geometry.polygons)
    digest = hashlib.sha256()
    for value in (nodes, *jax.tree.leaves(coupling)):
        array = np.asarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    source_root = Path(__file__).resolve().parents[3]
    sources = sorted((source_root / "nova/biot").rglob("*.py"))
    sources += [
        Path(module.__file__) for module in (support_read, fixed_points, topology)
    ]
    for source in sources:
        digest.update(source.relative_to(source_root).as_posix().encode())
        digest.update(source.read_bytes())
    identity = digest.hexdigest()
    cached = None if row_cache is None else Path(row_cache) / (identity + ".npy")
    hit = cached is not None and cached.is_file()
    if hit:
        rows = jnp.asarray(np.load(cached, allow_pickle=False))
        if rows.shape != (len(nodes), len(coupling.norm), 3, 6):
            raise ValueError("cached jet-row shape differs from its source geometry")
    else:
        rows = support_read.coupling_rows(nodes, coupling)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            partial = cached.with_suffix(".partial.npy")
            np.save(partial, np.asarray(rows), allow_pickle=False)
            partial.replace(cached)
            cached.with_suffix(".json").write_text(
                json.dumps(
                    dict(
                        identity=identity,
                        shape=rows.shape,
                        dtype=str(rows.dtype),
                        sources=[
                            str(source.relative_to(source_root)) for source in sources
                        ],
                    )
                )
                + "\n"
            )
    closed = jax.jit(
        lambda field, nodes: fixed_points.stack_jet(jax.vmap(field.evaluate)(nodes))
    )(field, jnp.asarray(nodes))
    moments = jnp.stack(tuple(exact_moments), axis=-1)
    exterior = closed - jnp.einsum("ncmq,cm->nq", rows, moments)
    if not np.isfinite(np.asarray(exterior)).all():
        raise ValueError(
            "the exact exterior jet contains a nonfinite kernel derivative"
        )
    coordinates = np.vstack(
        (operator.grid.coordinate, operator.wall.coordinate, operator.sample.coordinate)
    )
    distance, indices = cKDTree(coordinates).query(nodes)
    shared = distance < 2e-11
    if not np.any(shared):
        raise ValueError("the exterior comparison found no shared fixture nodes")
    actual = np.asarray(exterior)[shared, 0]
    expected = np.asarray(operator.external())[indices[shared]]
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    error = float(np.max(np.abs(actual - expected)))
    if error > value_tolerance * scale:
        raise ValueError(
            f"exterior jet value mismatch: {error / scale:.9g} "
            f"relative on {shared.sum()} nodes"
        )
    operands = support_read.SupportOperands.from_rows(
        geometry, rows, exterior, nodes, inverse.reshape(samples.shape[:-1])
    )
    operator.bind_moment_support(operands, roots)
    reconstructed = jax.jit(
        lambda rows, exterior, moments: (
            exterior + jnp.einsum("ncmq,cm->nq", rows, moments)
        )
    )(rows, exterior, moments)
    closure = np.asarray(jnp.max(jnp.abs(reconstructed - closed), axis=0))
    reference_scale = np.asarray(jnp.max(jnp.abs(closed), axis=0))
    assert reference_scale[0] > 0 and np.max(reference_scale[1:3]) > 0
    assert np.max(reference_scale[3:]) > 0
    return dict(
        row_cache_hit=hit,
        row_cache_identity=identity,
        row_cache_path=None if cached is None else str(cached),
        shared_nodes=int(shared.sum()),
        nodes=len(nodes),
        value_error=error,
        relative_value_error=error / scale,
        coupling_bytes=rows.nbytes,
        closed_jet_scale=reference_scale.tolist(),
        closed_jet_reconstruction_error=closure.tolist(),
        source_row_component_max=np.asarray(
            jnp.max(jnp.abs(rows), axis=(0, 1, 2))
        ).tolist(),
        exterior_jet_component_max=np.asarray(
            jnp.max(jnp.abs(exterior), axis=0)
        ).tolist(),
    )
