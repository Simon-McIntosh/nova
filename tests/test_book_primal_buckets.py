"""Primal bucket quadrature preserves the sequential ring formulation."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, topology
from nova.equilibrium.current import (
    Booking,
    FragmentSupport,
    Quadrature,
    _conic_rings,
    _normal_rings,
)
from nova.equilibrium.clip_quadrature import (
    ClippedCurrentMoments,
    clipped_support_current_moments,
)
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial

assert jax.config.jax_enable_x64
pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


def test_scalar_cut_tile_keeps_wide_batches():
    # Wide conic tiles reduce strip-propagation batches in diverted banks.
    # Normal-form rings use their own smaller tile width.
    assert current._SCALAR_CUT_BATCH_WIDTH == 128


def reference_book(psi, read, geometry, profile, quadrature=Quadrature()):
    """Integrate every selected fragment and normalize its current once.

    ``psi`` contains physical flux at the geometry's per-cell sample points.
    The supplied read must be its topology read; using a stale read is a caller
    error. Invalid reads and zero/nonfinite normalization denominators return
    an invalid receipt with nonfinite moments, including under jit and vmap.
    """
    if quadrature.segments < 2:
        raise ValueError("quadrature segments must be at least two")
    psi = jnp.asarray(psi)
    if psi.shape != geometry.sample_points.shape[:-1]:
        raise ValueError("psi must carry flux at every per-cell sample point")
    coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
    span = read.boundary_flux - read.axis_flux
    safe_span = jnp.where(span != 0, span, 1.0)
    normalized = coefficients.at[:, 0].add(-read.axis_flux) / safe_span
    signed = quadrature.sigma * read.field_coefficients
    signed = signed.at[:, 0].add(-quadrature.sigma * read.boundary_flux)

    def cell_field(index):
        centre = geometry.centre[index]
        field = FluxFieldPolynomial(
            normalized[index][None],
            centre[None],
            jnp.full((1, 2), geometry.pitch[index]),
            jnp.ones(1, dtype=bool),
        )
        return centre, field

    def integrate(index, rings, counts):
        centre, field = cell_field(index)
        support = FragmentSupport(
            rings[None],
            counts[None],
            centre[None],
            jnp.ones(1, dtype=bool),
            jnp.ones(1, dtype=bool),
        )
        moments = clipped_support_current_moments(
            support,
            support.included,
            field,
            profile,
            cut_cell_capacity=1,
            boundary_reduction=False,
        )
        return jax.tree.map(lambda value: value[0], moments)

    def whole_cell(index):
        rings = geometry.vertices[index][None]
        counts = geometry.vertex_count[index][None]
        return integrate(index, rings, counts)

    def integrate_live_rings(index, rings, counts):
        live = counts >= 3
        live_count = jnp.sum(live, dtype=jnp.int32)
        ring_index = jnp.nonzero(live, size=len(counts), fill_value=0)[0]
        zero = ClippedCurrentMoments(*(jnp.zeros((), rings.dtype) for _ in range(3)))

        def add_ring(slot, total):
            ring = ring_index[slot]
            value = integrate(index, rings[ring][None], counts[ring][None])
            return jax.tree.map(lambda prior, addition: prior + addition, total, value)

        return jax.lax.fori_loop(0, live_count, add_ring, zero)

    def cut_cell(index):
        centre = geometry.centre[index]
        vertices, count = geometry.vertices[index], geometry.vertex_count[index]

        def conic(_):
            rings, counts, valid = _conic_rings(
                vertices,
                count,
                centre,
                geometry.pitch[index],
                signed[index],
                read.fragment_selected[index],
                quadrature.segments,
            )
            moments = integrate_live_rings(index, rings, counts)
            return jax.tree.map(lambda value: jnp.where(valid, value, jnp.nan), moments)

        def normal(_):
            rings, counts = _normal_rings(
                vertices,
                count,
                centre,
                read.saddle_form,
                read.fragment_selected[index],
                read.edge_interval[index],
                quadrature.segments,
            )
            return integrate_live_rings(index, rings, counts)

        return jax.lax.cond(read.normal_form_cells[index], normal, conic, None)

    cells = len(geometry.centre)
    full = read.membership >= 1 - 1e-14
    cut = (read.membership > 0) & ~full
    whole = jax.vmap(whole_cell)(jnp.arange(cells))
    whole = jax.tree.map(lambda value: jnp.where(full, value, 0.0), whole)

    # Only the compact cut batch carries sampled rings. The number of batches
    # varies with the read, while the compiled batch width bounds device memory.
    batch_width = 32
    cut_count = jnp.sum(cut, dtype=jnp.int32)
    cut_index = jnp.nonzero(cut, size=cells, fill_value=0)[0]
    cut_index = jnp.pad(cut_index, (0, (-cells) % batch_width))

    def add_batch(batch, carried):
        start = batch * batch_width
        indices = jax.lax.dynamic_slice(cut_index, (start,), (batch_width,))
        active = start + jnp.arange(batch_width) < cut_count
        values = jax.vmap(cut_cell)(indices)

        def scatter(total, value):
            return total.at[indices].add(jnp.where(active, value, 0.0))

        return jax.tree.map(scatter, carried, values)

    cuts = jax.tree.map(jnp.zeros_like, whole)
    cuts = jax.lax.fori_loop(
        0, (cut_count + batch_width - 1) // batch_width, add_batch, cuts
    )
    moments = jax.tree.map(
        lambda full_value, cut_value: full_value + cut_value, whole, cuts
    )
    raw = jnp.sum(moments.cell_current)
    valid = read.valid & read.qualified & (span != 0) & jnp.isfinite(raw)
    if quadrature.target_current is None:
        normalization = jnp.ones_like(raw)
    else:
        valid &= (raw != 0) & jnp.isfinite(quadrature.target_current)
        normalization = quadrature.target_current / jnp.where(raw != 0, raw, 1.0)
    return Booking(
        *(jnp.where(valid, value * normalization, jnp.nan) for value in moments),
        raw,
        normalization,
        valid,
    )


def assert_moments_match(psi, reading, geometry, profile, quadrature, label):
    expected = reference_book(psi, reading, geometry, profile, quadrature)
    actual = current.book(psi, reading, geometry, profile, quadrature)
    assert bool(expected.valid) and bool(actual.valid)
    full = reading.membership >= 1 - 1e-14
    normal = (reading.membership > 0) & ~full & reading.normal_form_cells
    rows = {}
    for name, before, after in zip(Booking._fields, expected, actual, strict=True):
        before, after = np.asarray(before), np.asarray(after)
        delta = np.max(np.abs(after.astype(float) - before.astype(float)))
        scale = np.max(np.abs(before.astype(float)))
        relative = float(delta / scale) if scale else float(delta)
        rows[name] = dict(
            relative=relative,
            absolute=float(delta),
            identical=before.tobytes() == after.tobytes(),
            sha256=hashlib.sha256(after.tobytes()).hexdigest(),
        )
    print(
        "MOMENT_PARITY",
        json.dumps(dict(case=label, normal_cells=int(normal.sum()), fields=rows)),
        flush=True,
    )
    assert all(row["relative"] <= 1e-14 for row in rows.values()), rows


def test_diverted_fixture_matches_sequential_booking():
    from tests.test_support_read import _case, UnitDensity
    from nova.equilibrium import support_read

    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    support, _ = step(operands, cache, roots + 0.03 * pitch, roots)
    moments = tuple(jnp.zeros(len(operands.geometry.centre)) for _ in range(3))
    psi = support_read.nodal_jets(operands, moments)[operands.sample_index, 0]
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    assert_moments_match(
        psi,
        support.reading,
        operands.geometry,
        UnitDensity(),
        quadrature,
        "diverted-50",
    )


@pytest.fixture(scope="module", params=["limited", "diverted"])
def certificate(request):
    from benchmarks import solovev_certificate
    from scripts.analytic_oracle_fixtures.measure import analytic_profile
    from scripts.prototypes.read_route.measure import sample_field
    from tests.equilibrium.test_topology_read import (
        _analytic_inputs,
        _realised_hex_geometry,
    )
    from nova.equilibrium.solve_request import TopologyPolicy

    kind = request.param
    _, analytic, wall, _, saddle = _analytic_inputs(kind)
    geometry = _realised_hex_geometry(wall, 550)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0
    )
    fields = {"converged": analytic, "trial": sample_field(analytic, geometry)}
    readings = {
        name: jax.jit(topology.read)(field, geometry, convention, policy)
        for name, field in fields.items()
    }
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    _, source, _ = solovev_certificate._case(name, clip_mode="exact")
    profile = analytic_profile(source)
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    return kind, psi, readings, geometry, profile, quadrature


@pytest.mark.parametrize("state", ["trial", "converged"])
def test_certificate_moments_match_sequential_booking(certificate, state):
    kind, psi, readings, geometry, profile, quadrature = certificate
    assert_moments_match(
        psi, readings[state], geometry, profile, quadrature, f"{kind}-550-{state}"
    )
