"""Contract for evaluating a tile on an accelerator backend.

The tiled build already guarantees that every kernel call in a build has the
same array shape. That property only earns its keep if a second implementation
can exploit it, so the checks here are about the two things that could make an
accelerator path worthless: it could compute something else, or it could
recompile.

Parity is pinned in float64 against the numpy kernel that the operator is
already validated against -- an accelerator path is a cost change, never a
physics change. Compilation is counted directly: the traced kernel must be
compiled ONCE for a whole build, no matter how many tiles it evaluates, which
is only true if the padded shapes really are constant. Once per build is not
once, though, and the closed-form kernel costs a hundred seconds to compile
against under two to run a tile with -- so the reuse is pinned as well: a
caller building the operator at several geometries must compile at the first
one only, and a second process must find the executable on disk.

The tolerance is stated against each component's own peak rather than pointwise:
a device reassociates the quadrature sum, and the smallest entries of Br and Bz
are differences of much larger terms, so a pointwise ratio there measures the
reference's cancellation instead of the backends' agreement.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from nova.biot import polygon
from nova.biot.tiledassembly import (
    COMPONENTS,
    TilePlan,
    assemble,
    compilation_cache,
    forget_evaluators,
    tile_coupling,
    tile_evaluator,
)

jax = pytest.importorskip("jax")
zarr = pytest.importorskip("zarr")

pytestmark = pytest.mark.slow


def hexagon(r0, z0, radius=0.06):
    """Return regular hexagon vertices, counter-clockwise."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def clipped(r0, z0, radius=0.06):
    """Return a wall-clipped cell: the same hexagon with a corner cut off."""
    return np.delete(hexagon(r0, z0, radius), 2, axis=0)


def mesh(count=9):
    """Return a small mixed-shape cell set and the matching target points."""
    offset = np.linspace(-0.3, 0.3, count)
    sections = [
        clipped(6.2 + d, 0.15 * s) if i % 3 == 0 else hexagon(6.2 + d, 0.15 * s)
        for i, (d, s) in enumerate(zip(offset, np.cos(3.0 * offset)))
    ]
    target_r = 6.2 + np.linspace(-0.35, 0.35, count + 2)
    target_z = 0.1 * np.sin(5.0 * target_r)
    return sections, target_r, target_z


@pytest.mark.parametrize("batched", [False, True])
def test_the_traced_tile_matches_the_numpy_tile(batched):
    """Same operator to float64 tolerance, mapped over blocks either way."""
    sections, target_r, target_z = mesh()
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(
        target_tile=target_r.size,
        source_tile=len(sections),
        block=8,
        n_panels=16,
        n_nodes=48,
    )
    computed = tile_evaluator(plan, batched=batched)(
        target_r, target_z, edge, weight, norm
    )
    want = tile_coupling(target_r, target_z, edge, weight, norm, block=8)
    for got, reference in zip(computed, want):
        np.testing.assert_allclose(
            got, reference, rtol=1e-11, atol=1e-11 * np.max(np.abs(reference))
        )


def test_the_traced_kernel_carries_float64():
    """A float32 operator would be a silent accuracy regression, not a speedup."""
    sections, target_r, target_z = mesh(5)
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(
        target_tile=target_r.size, source_tile=5, block=4, n_panels=4, n_nodes=8
    )
    evaluate = tile_evaluator(plan)
    assert all(
        block.dtype == np.float64
        for block in evaluate(target_r, target_z, edge, weight, norm)
    )


def test_tile_precision_is_selected_per_evaluator():
    """Automatic retains fp64 while explicit fp32 owns a separate executable."""
    from nova.jax.config import Precision

    sections, target_r, target_z = mesh(5)
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(
        target_tile=target_r.size, source_tile=5, block=4, n_panels=4, n_nodes=8
    )
    automatic = tile_evaluator(plan)
    single = tile_evaluator(plan, precision=Precision.SINGLE)

    assert automatic is not single
    assert all(
        block.dtype == np.float64
        for block in automatic(target_r, target_z, edge, weight, norm)
    )
    assert all(
        block.dtype == np.float32
        for block in single(target_r, target_z, edge, weight, norm)
    )


@pytest.mark.parametrize("batched", [False, True])
def test_a_whole_build_compiles_the_kernel_once(batched):
    """Fixed padded shapes across tiles: the compile is amortised over the build."""
    sections, target_r, target_z = mesh(12)
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(target_tile=4, source_tile=4, block=8, n_panels=4, n_nodes=8)
    evaluate = tile_evaluator(plan, batched=batched)
    tiles = list(plan.tiles(target_r.size, len(sections)))
    assert len(tiles) > 1
    for rows, columns in tiles:
        evaluate(
            target_r[rows],
            target_z[rows],
            edge[:, :, columns],
            weight[:, columns],
            norm[columns],
        )
    assert evaluate.compile_count == 1


def test_a_ragged_tail_tile_does_not_force_a_recompile():
    """A tile at the edge of the matrix holds fewer pairs, not a different shape."""
    sections, target_r, target_z = mesh(7)
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(target_tile=4, source_tile=4, block=8, n_panels=4, n_nodes=8)
    evaluate = tile_evaluator(plan)
    full = evaluate(target_r[:4], target_z[:4], edge[:, :, :4], weight[:, :4], norm[:4])
    tail = evaluate(target_r[:4], target_z[:4], edge[:, :, 5:], weight[:, 5:], norm[5:])
    assert full[0].shape == (4, 4)
    assert tail[0].shape == (4, 2)
    assert evaluate.compile_count == 1
    want = tile_coupling(
        target_r[:4],
        target_z[:4],
        edge[:, :, 5:],
        weight[:, 5:],
        norm[5:],
        n_panels=plan.n_panels,
        n_nodes=plan.n_nodes,
        block=plan.block,
    )[0]
    np.testing.assert_allclose(
        tail[0], want, rtol=1e-11, atol=1e-11 * np.max(np.abs(want))
    )


def test_a_tile_larger_than_the_plan_is_refused():
    """Silently truncating a tile would corrupt the operator; it must raise."""
    sections, target_r, target_z = mesh(6)
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(target_tile=3, source_tile=3, block=8, n_panels=4, n_nodes=8)
    with pytest.raises(ValueError, match="exceeds the configured shape"):
        tile_evaluator(plan)(target_r, target_z, edge, weight, norm)


def test_a_traced_build_streams_the_same_store(tmp_path):
    """The backend is a tile evaluator swap; the assembled store is unchanged."""
    sections, target_r, target_z = mesh(11)
    plan = TilePlan(target_tile=5, source_tile=4, block=8, n_panels=8, n_nodes=16)
    reference, traced = tmp_path / "numpy.zarr", tmp_path / "traced.zarr"
    assemble(reference, target_r, target_z, sections, plan=plan)
    assemble(traced, target_r, target_z, sections, plan=plan, backend="jax")
    one, other = (zarr.open_group(str(path), mode="r") for path in (reference, traced))
    for name in COMPONENTS:
        want = np.asarray(one[name][:])
        np.testing.assert_allclose(
            np.asarray(other[name][:]),
            want,
            rtol=1e-11,
            atol=1e-11 * np.max(np.abs(want)),
        )
        assert other[name].chunks == one[name].chunks


# The closed-form kernel is the reason the elliptic primitives were made
# complement-native and trip-bounded: scipy's routines cannot enter a trace at all,
# so before this the accurate kernel was host-only. What has to be pinned is that
# the traced reduction computes the SAME thing as the host one and still compiles
# once -- its cost is a separate question, measured in benchmarks/tiled_backend.py.


def triangle(r0, z0, radius=0.05):
    """Return a three-cornered section, the cheapest shape that closes a chain."""
    angle = np.pi / 2 + np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def pair_geometry(target_r, target_z, edge, weight, norm):
    """Return a tile's geometry flattened onto its pair list, target-major."""
    rows, columns = np.divmod(np.arange(target_r.size * norm.size), norm.size)
    return (
        target_r[rows],
        target_z[rows],
        edge[:, :, columns],
        weight[:, columns],
        norm[columns],
    )


def test_the_traced_closed_form_matches_numpy_and_the_kernel_it_replaces():
    """One implementation, two namespaces, over a batch of unlike sections.

    Both references off ONE compilation, because that compilation is expensive: the
    reduction unrolls its moment recursions, so the trace is large and grows with the
    corner count. Three-cornered sections keep it as small as a closed chain can be
    and the sections still differ from each other, which is what the padded batch
    exists for.

    Against the same driver on numpy, to a few parts in 1e9 -- a compiler is free to
    reassociate and to contract a multiply-add, which moves the last bits of a
    reduction whose section sum differences an antiderivative of order the squared
    major radius. And against the quadrature the closed form replaces, at targets far
    enough from the contour for the rule to be converged: NOT a tolerance the closed
    form should be judged by -- it is one to two orders more accurate, and the
    acceptance gate in ``tests/test_biotpolygonanalytic.py`` is where that is measured
    -- but the only check that would catch a transposed section or a mis-taken pair
    column, which a self-comparison cannot see.
    """
    from nova.biot.polygonanalytic import packed_analytic_greens

    sections = [triangle(6.2, 0.0), triangle(6.3, 0.08, 0.03), triangle(6.1, -0.05)]
    target_r = np.array([6.6, 5.8, 6.2])
    target_z = np.array([0.35, -0.3, 0.4])
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(
        target_tile=target_r.size,
        source_tile=len(sections),
        block=target_r.size * len(sections),
        n_panels=16,
        n_nodes=48,
    )
    evaluate = tile_evaluator(plan, batched=True, kernel="closed")
    got = evaluate(target_r, target_z, edge, weight, norm)
    assert evaluate.compile_count == 1
    host = packed_analytic_greens(
        np, *pair_geometry(target_r, target_z, edge, weight, norm)
    )
    quadrature = tile_coupling(target_r, target_z, edge, weight, norm, block=plan.block)
    for component, reference, rule in zip(got, host, quadrature):
        reference = np.asarray(reference).reshape(target_r.size, len(sections))
        assert component.dtype == np.float64
        np.testing.assert_allclose(
            component, reference, rtol=1e-9, atol=1e-9 * np.max(np.abs(reference))
        )
        np.testing.assert_allclose(component, rule, rtol=2e-08, atol=0.0)


def test_an_unknown_kernel_is_refused_rather_than_silently_ignored():
    """A misspelled kernel must not fall back to the one it is not asking for."""
    plan = TilePlan(target_tile=2, source_tile=2, block=4, n_panels=4, n_nodes=8)
    with pytest.raises(ValueError, match="kernel"):
        tile_evaluator(plan, kernel="analytic")


# What a compile is paid PER.  Compiling once per build is only a bounded cost
# if a process performs one build; a caller sweeping a winding pack through
# positions performs one per position, and the closed-form executable costs
# more to produce than every tile of a small operator costs to evaluate.  The
# geometry is an argument to the kernel rather than a constant of it, so the
# same executable serves every position -- these checks pin that it is actually
# reused, in the process and across a process boundary.


def moved(sections, shift):
    """Return the same sections translated in R -- a geometry scan's next step."""
    return [section + np.array([shift, 0.0]) for section in sections]


def test_the_same_tile_shape_hands_back_the_same_compiled_kernel():
    """Memoised on the plan: two builds of one shape cannot compile twice."""
    plan = TilePlan(target_tile=4, source_tile=4, block=8, n_panels=4, n_nodes=8)
    other = TilePlan(target_tile=4, source_tile=4, block=8, n_panels=4, n_nodes=16)
    evaluate = tile_evaluator(plan)
    assert tile_evaluator(plan) is evaluate
    assert tile_evaluator(plan, batched=True) is not evaluate
    assert tile_evaluator(plan, kernel="closed") is not evaluate
    assert tile_evaluator(other) is not evaluate


def test_a_scan_over_positions_compiles_at_the_first_position_only(tmp_path):
    """Moving a section changes argument VALUES, which cannot force a retrace."""
    sections, target_r, target_z = mesh(8)
    plan = TilePlan(target_tile=5, source_tile=4, block=8, n_panels=4, n_nodes=8)
    evaluate = tile_evaluator(plan, batched=True)
    stores = []
    for index, shift in enumerate((0.0, 0.05, 0.11)):
        path = tmp_path / f"position-{index}.zarr"
        assemble(
            path,
            target_r + shift,
            target_z,
            moved(sections, shift),
            plan=plan,
            backend="jax",
            batched=True,
        )
        stores.append(np.asarray(zarr.open_group(str(path), mode="r")["Psi"][:]))
    assert evaluate.compile_count == 1
    # the positions really are different operators, so the single compilation is
    # reuse rather than three builds of the same thing
    assert not np.allclose(stores[0], stores[1])
    assert not np.allclose(stores[1], stores[2])


def test_a_caller_can_hand_in_the_kernel_it_already_holds(tmp_path):
    """An evaluator passed in builds the same store the default path builds."""
    sections, target_r, target_z = mesh(7)
    plan = TilePlan(target_tile=4, source_tile=4, block=8, n_panels=4, n_nodes=8)
    evaluate = tile_evaluator(plan, batched=True)
    reference, supplied = tmp_path / "default.zarr", tmp_path / "supplied.zarr"
    for path, given in ((reference, None), (supplied, evaluate)):
        assemble(
            path,
            target_r,
            target_z,
            sections,
            plan=plan,
            backend="jax",
            batched=True,
            evaluator=given,
        )
    paths = (reference, supplied)
    one, other = (zarr.open_group(str(path), mode="r") for path in paths)
    for name in COMPONENTS:
        np.testing.assert_array_equal(
            np.asarray(other[name][:]), np.asarray(one[name][:])
        )
    assert evaluate.compile_count == 1


def test_a_kernel_built_for_another_build_is_refused(tmp_path):
    """Silently ignoring the mismatch would build something else than was asked."""
    sections, target_r, target_z = mesh(6)
    plan = TilePlan(target_tile=4, source_tile=4, block=8, n_panels=4, n_nodes=8)
    other = TilePlan(target_tile=3, source_tile=3, block=8, n_panels=4, n_nodes=8)
    evaluate = tile_evaluator(plan, batched=True)
    build = dict(plan=plan, backend="jax", batched=True)
    for wrong in (dict(build, plan=other), dict(build, batched=False)):
        with pytest.raises(ValueError, match="evaluator built for"):
            assemble(
                tmp_path / "refused.zarr",
                target_r,
                target_z,
                sections,
                evaluator=evaluate,
                **wrong,
            )
    with pytest.raises(ValueError, match="no compiled evaluator"):
        assemble(
            tmp_path / "refused.zarr",
            target_r,
            target_z,
            sections,
            plan=plan,
            evaluator=evaluate,
        )


def test_forgetting_the_warm_kernels_compiles_the_next_one_again():
    """The escape hatch: a measurement of the compile needs a cold evaluator."""
    plan = TilePlan(target_tile=4, source_tile=4, block=8, n_panels=8, n_nodes=8)
    evaluate = tile_evaluator(plan)
    forget_evaluators()
    assert tile_evaluator(plan) is not evaluate


# The persistent cache is JAX's, so what is checked is the wiring: the directory
# this package chooses, the switch that turns it off, and -- the property that
# matters -- that a SECOND process finds the executable the first one compiled.
# A tile kernel is not needed to check that and would cost a compile to check it
# with, so the child compiles the cheapest graph that reaches XLA at all.

_CHILD_PROGRAM = """
import json
import jax
import jax.monitoring
import jax.numpy as jnp

from nova.biot.tiledassembly import compilation_cache

events = []
jax.monitoring.register_event_listener(lambda event, **kwargs: events.append(event))
directory = compilation_cache(min_compile_seconds=0.0)
jax.jit(lambda x: jnp.sin(x) @ jnp.cos(x).T)(jnp.arange(64.0).reshape(8, 8))
print(json.dumps({
    "directory": str(directory),
    "hits": events.count("/jax/compilation_cache/cache_hits"),
}))
"""


def run_child(cache_directory):
    """Return one fresh process's cache report, sharing an on-disk cache."""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_CHILD_PROGRAM)],
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            NOVA_COMPILATION_CACHE=str(cache_directory),
            JAX_PLATFORMS="cpu",
        ),
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_a_compiled_kernel_outlives_the_process_that_produced_it(tmp_path):
    """Cold process compiles and stores; the next one finds it and does not."""
    cache = tmp_path / "kernels"
    cold = run_child(cache)
    assert cold["hits"] == 0
    assert list(cache.iterdir()), "nothing was written to the cache"
    assert run_child(cache)["hits"] >= 1


@pytest.fixture
def unconfigured_cache():
    """Run with no cache directory on JAX, and put back whatever there was."""
    previous = jax.config.jax_compilation_cache_dir
    jax.config.update("jax_compilation_cache_dir", None)
    yield
    jax.config.update("jax_compilation_cache_dir", previous)


def test_the_persistent_cache_can_be_switched_off(monkeypatch, unconfigured_cache):
    """A measurement of the compile itself must be able to refuse the cache."""
    monkeypatch.setenv("NOVA_COMPILATION_CACHE", "off")
    assert compilation_cache() is None
    assert jax.config.jax_compilation_cache_dir is None


def test_the_cache_directory_comes_from_the_environment(
    monkeypatch, tmp_path, unconfigured_cache
):
    """Named in the environment, so a build on a scratch filesystem can say so."""
    monkeypatch.setenv("NOVA_COMPILATION_CACHE", str(tmp_path / "named"))
    assert compilation_cache(tmp_path / "explicit") == tmp_path / "explicit"
    assert jax.config.jax_compilation_cache_dir == str(tmp_path / "explicit")


def test_a_directory_already_chosen_wins_over_the_default(
    monkeypatch, tmp_path, unconfigured_cache
):
    """JAX_COMPILATION_CACHE_DIR is the caller's own choice; do not overrule it."""
    jax.config.update("jax_compilation_cache_dir", str(tmp_path / "theirs"))
    monkeypatch.setenv("NOVA_COMPILATION_CACHE", str(tmp_path / "ours"))
    assert compilation_cache() == tmp_path / "theirs"
