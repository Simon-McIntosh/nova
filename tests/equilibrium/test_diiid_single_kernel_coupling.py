"""The DIII-D builder attaches all plasma moments from one polygon evaluation."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from benchmarks import diiid_forward_gs_match as bench
from nova.jax.config import configure_dtypes


@pytest.fixture
def built(monkeypatch):
    configure_dtypes()
    assert jax.config.jax_enable_x64
    radius = np.linspace(1.0, 2.0, 13)
    height = np.linspace(-0.8, 0.8, 13)
    label = (radius[:, None] - 1.5) ** 2 + height[None, :] ** 2
    wall = np.array([[0.9, -0.9], [2.1, -0.9], [2.1, 0.9], [0.9, 0.9]])
    surfaces = np.linspace(0.0, 1.0, 5)
    monkeypatch.setattr(bench, "canonical_axes", lambda row: (radius, height))
    monkeypatch.setattr(
        bench, "_label_state", lambda row, frame: (label, surfaces, surfaces, surfaces)
    )
    physical = SimpleNamespace(physical_digest="synthetic-conductor")
    monkeypatch.setattr(
        bench,
        "dataset_machine_description",
        lambda *a, **k: SimpleNamespace(physical=physical),
    )
    monkeypatch.setattr(
        bench,
        "_topology_surface",
        lambda row, lattice, *a: (
            wall,
            np.ones(lattice.node_count, dtype=bool),
            "synthetic-wall",
            "wall",
            {},
        ),
    )
    monkeypatch.setattr(
        bench,
        "vacuum_response",
        lambda machine, r, z: (("coil",), np.full((1, len(z), len(r)), 0.25)),
    )
    monkeypatch.setattr(
        bench,
        "_wall_source_response",
        lambda machine, names, target: np.full((len(target), 1), 0.5),
    )
    monkeypatch.setattr(bench, "_source_current", lambda *a: np.array([2.0]))
    monkeypatch.setattr(bench, "_boundary_field_function", lambda *a: 1.0)
    monkeypatch.setattr(bench, "_COUPLING_CACHE", {})
    calls = []
    original = bench.build_flux_blocks

    def observed(target, cells, centres):
        blocks = original(target, cells, centres)
        calls.append((np.asarray(target), blocks))
        return blocks

    monkeypatch.setattr(bench, "build_flux_blocks", observed)
    profile, *_ = bench.build_profile({"efit_times": [0.0]}, 0)
    return profile.operator, calls


@pytest.mark.parametrize(
    "target_name,call_index", [("grid", 0), ("wall", 1), ("sample", 2)]
)
def test_builder_attaches_same_polygon_evaluation(built, target_name, call_index):
    operator, calls = built
    assert len(calls) == 3, "each target set must be evaluated exactly once"
    target = getattr(operator, target_name)
    coordinate, blocks = calls[call_index]
    np.testing.assert_array_equal(target.coordinate, coordinate)
    for attribute, expected in zip(
        ("plasma_target", "plasma_target_r", "plasma_target_z"), blocks, strict=True
    ):
        np.testing.assert_array_equal(
            getattr(target, attribute),
            expected,
            err_msg=f"{target_name}.{attribute} must use the same polygon evaluation",
        )
    np.testing.assert_array_equal(
        target.source_target, 0.25 if target_name == "grid" else 0.5
    )


def test_polygon_uniform_control_distinguishes_hybrid(built):
    operator, calls = built
    grid = np.asarray(operator.grid.coordinate)
    width = np.diff(np.unique(grid[:, 0])).mean()
    height = np.diff(np.unique(grid[:, 1])).mean()
    for coordinate, blocks in calls:
        hybrid = bench._green_block(coordinate, grid, width, height)
        relative = np.linalg.norm(hybrid - blocks[0]) / np.linalg.norm(blocks[0])
        assert relative > 1e-5, "the fixture must detect a hybrid uniform substitution"
