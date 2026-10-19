"""Connected-fragment booking, signed rings and moving-support tangents."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray

from nova.equilibrium.clip_quadrature import clipped_support_current_moments
from nova.equilibrium.current import FragmentSupport, Quadrature, book
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial
from nova.equilibrium import topology

assert jax.config.jax_enable_x64


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Density:
    slope: object = 0.0

    def current_density(self, radius, normalized):
        return jnp.ones_like(radius) + self.slope * normalized


def _rings():
    outer = [[1, -1], [3, -1], [3, 1], [1, 1]]
    hole = [[1.5, -0.5], [1.5, 0.5], [2.5, 0.5], [2.5, -0.5]]
    island = [[4, -0.5], [5, -0.5], [5, 0.5], [4, 0.5]]
    return FragmentSupport(
        jnp.asarray([[outer, hole, island]], dtype=jnp.float64),
        jnp.asarray([[4, 4, 4]]),
        jnp.asarray([[2.0, 0.0]]),
        jnp.asarray([True]),
        jnp.asarray([True]),
    )


@pytest.mark.parametrize("polynomial", [False, True])
def test_disconnected_polygons_and_holes(polynomial):
    support = _rings()
    field = FluxFieldPolynomial(
        jnp.zeros((1, 6)), support.centroids, jnp.ones((1, 2)), jnp.ones(1, dtype=bool)
    )
    result = jax.jit(
        lambda value: clipped_support_current_moments(
            value,
            value.included,
            field,
            Density(),
            cut_cell_capacity=1,
            boundary_reduction=polynomial,
        )
    )(support)
    np.testing.assert_allclose(result.cell_current, [4], atol=2e-13)
    np.testing.assert_allclose(result.radial_moment, [2.5], atol=2e-13)
    np.testing.assert_allclose(result.vertical_moment, [0], atol=2e-13)


def _split_cell(level=0.0, axis=1.0, shift=0.0, slope=0.2):
    vertices = jnp.asarray([[1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [1.0, 1.0]])
    centre = jnp.asarray([2.0, 0.0])
    points = jnp.asarray(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]], dtype=jnp.float64
    )
    x, y = points.T
    design = jnp.stack([jnp.ones(6), x, y, x * x, x * y, y * y], axis=1)
    coefficients = jnp.asarray([shift * shift - 0.04, 0.0, -2 * shift, 0.0, 0.0, 1.0])
    fragment = topology.quadratic_cell_fragments(
        vertices - centre, jnp.asarray(4), coefficients.at[0].add(-level)
    )
    selected = jnp.asarray([[False, True]])
    if os.environ.get("NOVA_BOOK_LEVEL_MEMBERSHIP") == "1":
        selected = fragment.area[None] > 0
    read = SimpleNamespace(
        field_coefficients=coefficients[None],
        axis_flux=jnp.asarray(axis),
        boundary_flux=jnp.asarray(level),
        fragment_selected=selected,
        membership=jnp.sum(jnp.where(selected, fragment.area[None], 0), axis=1) / 4,
        normal_form_cells=jnp.asarray([False]),
        saddle_form=topology.saddle_normal_form(
            jnp.asarray([2.0, 0.0]),
            jnp.asarray([[1.0, 0.0], [0.0, -1.0]]),
            jnp.zeros((2, 2, 2)),
            1e-12,
            jnp.zeros((2, 2, 2, 2)),
        ),
        edge_interval=jnp.pad(
            fragment.edge_interval[None], ((0, 0), (0, 0), (0, 0), (0, 10), (0, 0))
        ),
        valid=jnp.asarray(True),
        qualified=jnp.asarray(True),
    )
    geometry = SimpleNamespace(
        vertices=vertices[None],
        vertex_count=jnp.asarray([4]),
        centre=centre[None],
        pitch=jnp.ones(1),
        sample_points=(points + centre)[None],
        fit_inverse=jnp.linalg.inv(design)[None],
    )
    return book(
        (design @ coefficients)[None],
        read,
        geometry,
        Density(slope),
        Quadrature(segments=64),
    )


def test_private_fragment_current_is_zero():
    result = jax.jit(lambda: _split_cell(slope=0.0))()
    # The upper strip's exact moment identifies its membership, even when the
    # lower private strip shares the same cell and signed flux level.
    private_current = (result.cell_current[0] - result.vertical_moment[0] / 0.6) / 2
    print(f"PRIVATE_FLUX_CURRENT={float(private_current):.17g}", flush=True)
    assert abs(float(private_current)) < 1e-12, f"private current {private_current}"
    np.testing.assert_allclose(result.cell_current, [1.6], atol=1e-12)
    np.testing.assert_allclose(result.vertical_moment, [0.96], atol=1e-12)


@pytest.mark.parametrize("stage", ["smooth", "axis", "boundary", "support"])
def test_booking_jvp_matches_central_difference(stage):
    def evaluate(parameter):
        values = dict(slope=0.2, axis=1.0, level=0.0, shift=0.0)
        values[
            {
                "smooth": "slope",
                "axis": "axis",
                "boundary": "level",
                "support": "shift",
            }[stage]
        ] += parameter
        result = _split_cell(**values)
        return jnp.stack(result[:3])

    primal = jnp.asarray(0.0)
    derivative = jax.jit(
        lambda value: jax.jvp(evaluate, (value,), (jnp.ones_like(value),))[1]
    )(primal)
    compiled = jax.jit(evaluate)
    errors = []
    for step in [1e-3, 3e-5, 1e-5]:
        central = (compiled(primal + step) - compiled(primal - step)) / (2 * step)
        error = float(
            jnp.max(jnp.abs(central - derivative))
            / jnp.maximum(1.0, jnp.max(jnp.abs(derivative)))
        )
        errors.append(error)
    print(f"JVP_STAGE={stage} ERRORS={errors}", flush=True)
    assert np.isfinite(errors).all()
    assert errors[-1] < (1e-10 if stage == "smooth" else 1e-8)


def test_fragment_capacity_is_refused():
    support = _rings()
    field = FluxFieldPolynomial(
        jnp.zeros((1, 6)), support.centroids, jnp.ones((1, 2)), jnp.ones(1, dtype=bool)
    )
    result = clipped_support_current_moments(
        support, support.included, field, Density(), cut_cell_capacity=1
    )
    assert float(result.cell_current.sum()) > 0
    # Capacity refusal must not silently discard a whole occupied cell.
    doubled = jax.tree.map(lambda value: jnp.concatenate((value, value)), support)
    doubled_field = jax.tree.map(lambda value: jnp.concatenate((value, value)), field)
    refused = clipped_support_current_moments(
        doubled, doubled.included, doubled_field, Density(), cut_cell_capacity=1
    )
    assert np.isnan(refused.cell_current).all()


@pytest.mark.parametrize("kind", ["limited", "diverted"])
@pytest.mark.parametrize("cells", [550, 2000, 5000])
def test_analytic_certificate_moments(kind, cells, tmp_path):
    backend = jax.default_backend()
    if backend != "gpu":
        pytest.skip(f"certificate rows require the GPU measurement lane, not {backend}")

    from scripts.prototypes.support.book import measure

    row = measure(kind, cells, os.environ.get("NOVA_BOOK_RESULTS") or tmp_path)
    assert row["net_current_relative_error"] <= 1e-12, row
    assert row["private_cell_count"] > 0, row
    assert row["private_current"] == 0, row
    assert row["smooth_current_budget_ratio"] <= 1, row
    assert row["smooth_moment_budget_ratio"] <= 1, row
    assert row["saddle_current_budget_ratio"] <= 1, row
    assert row["saddle_moment_budget_ratio"] <= 1, row
    assert row["clip_current_error"] <= 2e-6, row
    assert row["clip_first_moment_error"] <= 2e-6, row
    factor = 2 if kind == "limited" else 1
    assert (
        row["read_image_error"]
        <= factor * row["exact_image_error"] + row["oracle_area_fraction_uncertainty"]
    ), row
    if kind == "limited":
        assert row["read_image_error"] <= 1e-4, row


def test_certificate_fixture_is_addressed_by_the_carrier_identity(
    tmp_path, monkeypatch
):
    """A moved identity addresses a different fixture rather than a stale one.

    The shared fixture key is the carrier identity the fixture sources define,
    so perturbing one identity input must miss the object published under the
    original identity. A key that ignores an input would serve the stale
    fixture, which is what this refuses.
    """
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures import measure as fixture
    from scripts.prototypes.support import certificate_fixtures

    monkeypatch.setenv("NOVA_BOOK_FIXTURE_ROOT", str(tmp_path))
    carrier, _source, _exact = certificate._case(
        "weak-rotation-reactor-static", clip_mode="exact"
    )
    identity = fixture.cache_identity(
        carrier, requested_cells=-550, wall_nodes=fixture.WALL_POINT_COUNT
    )
    moved = json.loads(json.dumps(identity, sort_keys=True))
    moved["discretisation"]["requested_cells"] = -552
    kind = certificate_fixtures.CARRIER_KIND
    payload = xarray.Dataset({"cell_current": (("cell",), np.arange(3.0))})
    digest = certificate_fixtures.fixture_digest(identity)
    # Recompute the key here from the canonical JSON rather than reading it back
    # out of the store: a store that ignored an identity input, or named objects
    # by anything coarser than the defining identity, would agree with itself and
    # still fail this.
    canonical = json.dumps(
        certificate_fixtures._defining_identity(identity),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    independently_computed = hashlib.sha256(canonical).hexdigest()
    assert digest == "sha256:" + independently_computed, digest
    published = certificate_fixtures.publish_fixture(kind, digest, payload)
    assert published.name == independently_computed, published
    entry = certificate_fixtures.fixture_entry(kind, digest)
    assert entry is not None and entry.name == independently_computed, entry
    assert (
        certificate_fixtures.fixture_entry(
            kind, certificate_fixtures.fixture_digest(moved)
        )
        is None
    ), "a moved identity loaded the fixture published under the original"


class _RefusingSecondBuild:
    """A store that refuses to publish one identity a second time.

    A fixture owner publishes only on a miss, so a second call with the same
    identity must load the object the first call published. If the load path is
    bypassed the second call builds again and reaches this refusal, which is how
    a hit-path test can fail rather than merely observing a receipt field.
    """

    def __init__(self, inner):
        self.inner = inner
        self.published: list[str] = []

    @contextmanager
    def session(self, identity):
        with self.inner.session(identity) as inner_session:
            yield _RefusingSecondBuildSession(inner_session, self)


class _RefusingSecondBuildSession:
    def __init__(self, inner, owner):
        self.inner = inner
        self.owner = owner

    @property
    def key(self):
        return self.inner.key

    @property
    def location(self):
        return self.inner.location

    @property
    def lock_wait(self):
        return self.inner.lock_wait

    def load(self):
        return self.inner.load()

    def publish(self, data):
        if self.inner.key in self.owner.published:
            raise RuntimeError(
                f"a builder reached the store twice for {self.inner.key}"
            )
        self.owner.published.append(self.inner.key)
        self.inner.publish(data)


def test_second_call_loads_the_published_carrier(tmp_path, monkeypatch):
    """A second call on one identity loads the fixture; it does not build it."""
    from scripts.analytic_oracle_fixtures import measure as fixture
    from scripts.prototypes.support import certificate_fixtures

    monkeypatch.setenv("NOVA_BOOK_FIXTURE_ROOT", str(tmp_path))
    store = _RefusingSecondBuild(certificate_fixtures.carrier_store())
    case = fixture.analytic_case()
    request = dict(wall_nodes=fixture.WALL_POINT_COUNT)
    cold = fixture.cached_machine(case, -110, store=store, **request)
    warm = fixture.cached_machine(case, -110, store=store, **request)
    assert cold.cache["hit"] is False, cold.cache
    assert cold.cache["build_seconds"] > 0.0, cold.cache
    assert warm.cache["hit"] is True, warm.cache
    assert warm.cache["build_seconds"] == 0.0, warm.cache
    assert store.published == [cold.cache["semantic_key"]], store.published
    assert warm.cache["semantic_key"] == cold.cache["semantic_key"]
    np.testing.assert_array_equal(warm.node, cold.node)
