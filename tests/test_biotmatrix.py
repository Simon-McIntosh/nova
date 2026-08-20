import pytest

import numpy as np

from nova.biot.biotframe import Source, Target
from nova.biot.matrix import Matrix
from nova.biot.polysection import PolySectionPolicy
from nova.biot.solve import Solve
from nova.biot.target import TargetQuadrature, TargetQuadraturePolicy
from nova.geometry.polyline import PolyLine


@pytest.fixture
def matrix():
    points = np.array([[-2, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0], [3, 0, 0]], float)
    polyline = PolyLine(points, minimum_arc_nodes=4)
    source = Source(polyline.path_geometry)
    target = Target({"x": np.linspace(5, 7.5, 10), "z": 0.5})
    return Matrix(source, target)


def test_coordinate_axes_shape(matrix):
    assert matrix.coordinate_axes.shape == (10, 4, 3, 3)


def test_stack_shape(matrix):
    points = matrix.target.stack(*list("xyz"))
    assert points.shape == (10, 4, 3)


def test_coordinate_axes_einsum_shape(matrix):
    points = matrix.target.stack(*list("xyz"))
    _points = np.einsum("ijk,ijkm->ijm", points, matrix.coordinate_axes)
    assert points.shape == _points.shape


def test_coord_loc(matrix):
    assert len(matrix.loc.data["source"]) == 0
    assert len(matrix.loc.data["target"]) == 0
    assert matrix.loc["source", "x"].shape == matrix.shape
    assert list(matrix.loc.data["source"].keys()) == ["x", "y", "z"]


def test_source_coordinates_roundtrip(matrix):
    points = matrix.source.stack("x1", "y1", "z1")
    local_points = matrix.loc.to_local(points)
    global_points = matrix.loc.to_global(local_points)
    assert np.allclose(local_points[..., :2], 0)
    assert np.allclose(points, global_points)


@pytest.mark.parametrize("frame", ["source", "target"])
def test_local_frame_roundtrip(matrix, frame):
    points = getattr(matrix, frame).stack(*list("xyz"))
    local_points = np.stack([matrix.loc[frame, attr] for attr in "xyz"], axis=-1)
    global_points = matrix.loc.to_global(local_points)
    assert np.allclose(points, global_points)


class ScalarGenerator:
    """Small deterministic generator exposing Solve batching without a field kernel."""

    def __init__(
        self,
        source,
        target,
        turns,
        reduce,
        target_quadrature=None,
        policy=None,
    ):
        self.source = source
        self.target = target
        self.source.turns = turns[0]
        self.source.reduce = reduce[0]
        self.target.turns = turns[1]
        self.target.reduce = reduce[1]
        self.policy = policy

    def compute(self, attr):
        """Return source x values after the same column operations as Matrix."""
        assert attr == "Psi"
        matrix = np.broadcast_to(
            np.asarray(self.source["x"], dtype=float),
            (len(self.target), len(self.source)),
        ).copy()
        plasma = np.asarray(self.source.plasma, dtype=bool)
        target_plasma = matrix[:, plasma]
        plasma_source = np.empty((0, len(self.source)))
        plasma_plasma = np.empty((0, plasma.sum()))
        if self.source.turns:
            matrix *= np.asarray(self.source["nturn"], dtype=float)[None, :]
        if self.source.reduce and self.source.biotreduce.reduce:
            matrix = np.add.reduceat(matrix, self.source.biotreduce.indices, axis=1)
        links = self.source.biotreduce.link
        if self.source.reduce and links:
            for link, (reference, factor) in links.items():
                matrix[:, reference] += factor * matrix[:, link]
            matrix = np.delete(matrix, list(links), axis=1)
        return matrix, target_plasma, plasma_source, plasma_plasma


class ReverseBatchSolve(Solve):
    """Evaluate the same immutable batches in reverse dispatch order."""

    def compose(self):
        """Compute batches from the last group to the first."""
        for batch in reversed(self.source_batches):
            self.compute(batch)


def scalar_source(number, *, policy=None, plasma=False):
    """Return independent scalar sources with stable labels and route metadata."""
    labels = [f"source_{index}" for index in range(number)]
    data = {
        "x": np.arange(1, number + 1, dtype=float),
        "z": np.zeros(number),
        "segment": np.full(number, "circle", dtype=object),
        "frame": labels,
        "nturn": np.ones(number),
        "plasma": np.full(number, plasma),
        "link": np.full(number, "", dtype=object),
        "factor": np.ones(number),
    }
    if policy is not None:
        data["polysection_policy"] = np.full(number, policy.key, dtype=object)
    return Source(data, index=labels)


@pytest.mark.parametrize(
    ("number", "batch_sizes"),
    [(499, [499]), (500, [500]), (501, [500, 1]), (1001, [500, 500, 1])],
)
def test_solve_batches_have_a_hard_source_limit(monkeypatch, number, batch_sizes):
    """Boundary sizes neither drop nor duplicate a source column."""
    monkeypatch.setitem(Solve.generator, "circle", ScalarGenerator)
    source = scalar_source(number)
    solve = Solve(
        source,
        Target({"x": [2.0], "z": [0.0]}),
        attrs=["Psi"],
        turns=False,
        reduce=False,
    )
    assert [len(batch.positions) for batch in solve.source_batches] == batch_sizes
    np.testing.assert_array_equal(solve.data.Psi.data[0], source.x)


def test_solve_policy_and_plasma_lanes_are_batch_distinct(monkeypatch):
    """Equal segment names cannot merge different policies or electrical lanes."""
    monkeypatch.setitem(Solve.generator, "polysection", ScalarGenerator)
    exact = PolySectionPolicy()
    quadrature = PolySectionPolicy(exact_kernel="quadrature")
    source = scalar_source(4, policy=exact)
    source["segment"] = np.full(4, "polysection", dtype=object)
    source["polysection_policy"] = np.array(
        [exact.key, quadrature.key, exact.key, quadrature.key], dtype=object
    )
    source["plasma"] = np.array([False, False, True, True])
    solve = Solve(
        source,
        Target({"x": [2.0], "z": [0.0]}),
        attrs=["Psi"],
        turns=False,
        reduce=False,
    )
    identities = {(batch.lane, batch.policy.key) for batch in solve.source_batches}
    assert identities == {
        ("conductor", exact.key),
        ("conductor", quadrature.key),
        ("plasma", exact.key),
        ("plasma", quadrature.key),
    }
    np.testing.assert_array_equal(solve.data.Psi.data[0], source.x)


def test_multiple_physical_plasma_targets_fail_before_turn_updates():
    """A scalar update index cannot silently stand for two physical parents."""
    logical = Target(
        {
            "x": [2.0, 3.0],
            "z": [0.0, 0.0],
            "nturn": [1.0, 1.0],
            "plasma": [True, True],
            "frame": ["plasma_a", "plasma_b"],
            "link": ["", ""],
            "factor": [1.0, 1.0],
        },
        index=["plasma_a_cell", "plasma_b_cell"],
        available=[],
    )
    quadrature = TargetQuadrature(
        nodes=Target(
            {"x": [2.0, 3.0], "z": [0.0, 0.0]},
            index=["plasma_a_node", "plasma_b_node"],
            available=[],
        ),
        logical=logical,
        offsets=np.array([0, 1]),
        weights=np.ones(2),
        physical_index=("plasma_a", "plasma_b"),
        physical_plasma=(True, True),
        policy=TargetQuadraturePolicy(),
    )
    with pytest.raises(ValueError, match="require one physical target parent"):
        Solve(
            scalar_source(1),
            quadrature.nodes,
            attrs=["Psi"],
            turns=False,
            reduce=True,
            target_quadrature=quadrature,
        )


def test_finite_arcs_never_dispatch_through_the_ring_backend(monkeypatch):
    """Polygon route metadata cannot replace the geometry-specific arc element."""
    monkeypatch.setitem(Solve.generator, "arc", ScalarGenerator)
    policy = PolySectionPolicy(
        exact_kernel="quadrature",
        quadrature=(2, 4),
        backend="jax",
        device_eligibility="axisymmetric_ring",
    )
    source = scalar_source(1, policy=policy)
    source["segment"] = np.array(["arc"], dtype=object)
    solve = Solve(
        source,
        Target({"x": [2.0], "z": [0.0]}),
        attrs=["Psi"],
        turns=False,
        reduce=False,
    )
    assert solve.source_batches[0].policy is None
    assert solve.data.Psi.item() == 1.0


@pytest.mark.parametrize("reverse", [False, True])
def test_solve_applies_nonunit_links_after_route_batching(monkeypatch, reverse):
    """A dependent keeps its circuit factor when the reference is in another batch."""
    monkeypatch.setitem(Solve.generator, "polysection", ScalarGenerator)
    exact = PolySectionPolicy()
    quadrature = PolySectionPolicy(exact_kernel="quadrature")
    source = Source(
        {
            "x": [10.0, 2.0, 4.0],
            "z": np.zeros(3),
            "segment": np.full(3, "polysection", dtype=object),
            "polysection_policy": [exact.key, quadrature.key, quadrature.key],
            "frame": np.full(3, "coil", dtype=object),
            "nturn": np.ones(3),
            "plasma": np.zeros(3, dtype=bool),
            "link": ["head", "head", "head"],
            "factor": [1.0, -0.5, -0.5],
        },
        index=["head", "dependent_a", "dependent_b"],
    )
    source.at["head", "link"] = ""
    solve_class = ReverseBatchSolve if reverse else Solve
    solve = solve_class(
        source,
        Target({"x": [2.0], "z": [0.0]}),
        attrs=["Psi"],
        turns=False,
        reduce=[True, False],
    )
    assert len(solve.source_batches) == 2
    assert solve.data.source.values.tolist() == ["head"]
    assert solve.data.Psi.item() == pytest.approx(7.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_solve_applies_nonunit_links_across_the_chunk_boundary(monkeypatch, reverse):
    """The first dependent after column 500 keeps its absolute electrical factor."""
    monkeypatch.setitem(Solve.generator, "circle", ScalarGenerator)
    number = 501
    labels = ["head", *[f"dependent_{index}" for index in range(1, number)]]
    source = Source(
        {
            "x": np.r_[10.0, np.full(number - 1, 2.0)],
            "z": np.zeros(number),
            "segment": np.full(number, "circle", dtype=object),
            "frame": np.full(number, "coil", dtype=object),
            "nturn": np.ones(number),
            "plasma": np.zeros(number, dtype=bool),
            "link": ["", *np.full(number - 1, "head", dtype=object)],
            "factor": np.r_[1.0, np.full(number - 1, -0.5)],
        },
        index=labels,
    )
    solve_class = ReverseBatchSolve if reverse else Solve
    solve = solve_class(
        source,
        Target({"x": [2.0], "z": [0.0]}),
        attrs=["Psi"],
        turns=False,
        reduce=[True, False],
    )
    assert [len(batch.positions) for batch in solve.source_batches] == [500, 1]
    assert solve.data.Psi.item() == pytest.approx(-490.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_solve_linked_cancellation_survives_batch_order(monkeypatch, reverse):
    """Linked contributions straddling batch boundaries cancel without misrouting."""
    monkeypatch.setitem(Solve.generator, "circle", ScalarGenerator)
    number = 1001
    labels = ["coil", *[f"coil_{index}" for index in range(1, number)]]
    sign = np.where(np.arange(number) % 2, -1.0, 1.0)
    order = np.arange(number)
    source = Source(
        {
            "x": np.ones(number)[order],
            "z": np.zeros(number),
            "segment": np.full(number, "circle", dtype=object),
            "frame": np.full(number, "coil", dtype=object),
            "nturn": sign[order],
            "plasma": np.zeros(number, dtype=bool),
            "link": np.array(["" if label == "coil" else "coil" for label in labels])[
                order
            ],
            "factor": np.ones(number),
        },
        index=np.asarray(labels)[order].tolist(),
    )
    solve_class = ReverseBatchSolve if reverse else Solve
    solve = solve_class(
        source,
        Target({"x": [2.0], "z": [0.0]}),
        attrs=["Psi"],
        turns=[True, False],
        reduce=[True, False],
    )
    assert solve.data.source.values.tolist() == ["coil"]
    assert solve.data.Psi.item() == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__])
