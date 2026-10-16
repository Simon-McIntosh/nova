r"""Contract of the shared operand cache for Biot-Savart value and jet matrices.

The value and jet matrices are gigabytes at production resolution and identical
across processes, so they are written once to a shared root and loaded by every
later measurement.  This module pins the four properties that make that safe:
persistence is on by default, a completed entry is served as a hit, a changed
geometry or source byte moves the key and forces a rebuild, and a loaded matrix
is bitwise the matrix that was built.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from nova.database.filepath import operand_cache_root
from nova.equilibrium import topology
from nova.jax.config import configure_dtypes
from scripts.prototypes.lean_map import values as value_module
from scripts.prototypes.lean_map.quadrature import triangle_rule
from scripts.prototypes.lean_map.values import build_values, values_digest

PITCH = 0.4


@dataclass(frozen=True)
class QuadraticField:
    """A manufactured flux field carrying every derivative a jet needs."""

    def potential(self, point):
        radius, height = point[0], point[1]
        return (
            1.3 + 0.2 * radius - 0.4 * height + 0.7 * radius**2 + 0.3 * radius * height
        )

    def value(self, point):
        return self.potential(point)

    def evaluate(self, point):
        import jax

        return topology.FieldJet(
            self.potential(point),
            jax.grad(self.potential)(point),
            jax.hessian(self.potential)(point),
        )


@dataclass(frozen=True)
class HexCarrier:
    """One small hex carrier with the geometry, field and moments a build needs."""

    centres: np.ndarray
    polygons: tuple[np.ndarray, ...]
    geometry: object
    field: QuadraticField
    reference: object

    @property
    def machine(self):
        polygons = self.polygons

        class _Machine:
            cell_polygons = polygons

        return _Machine()


def hexagon(centre: np.ndarray, pitch: float) -> np.ndarray:
    """Return a counterclockwise regular hexagon of the given pitch."""
    angle = np.arange(6) * np.pi / 3.0 + np.pi / 6.0
    radius = pitch / np.sqrt(3.0)
    return centre + radius * np.c_[np.cos(angle), np.sin(angle)]


def carrier() -> HexCarrier:
    """Return one four-cell hex carrier, its geometry and a manufactured field."""
    import jax.numpy as jnp

    configure_dtypes()
    centres = np.array(
        [
            [0.0, 0.0],
            [PITCH, 0.0],
            [PITCH * np.sqrt(3) / 2.0, PITCH * 0.5],
            [0.0, PITCH * np.sqrt(3) / 2.0],
        ]
    )
    polygons = tuple(hexagon(centre, PITCH) for centre in centres)
    angle = np.arange(6) * np.pi / 3.0 + np.pi / 6.0
    offsets = (PITCH / np.sqrt(3.0)) * np.c_[np.cos(angle), np.sin(angle)]
    sampling = centres[:, None, :] + offsets[None, :, :]
    wall = np.array(
        [[0.0, -2.0], [3.0, -2.0], [3.0, 3.0], [-2.0, 3.0], [-2.0, -2.0], [0.0, -2.0]]
    )
    geometry = topology.TopologyGeometry.from_cells(polygons, sampling, (wall,))
    reference = jnp.linspace(0.5, 1.5, len(centres) * 3).reshape(len(centres), 3)
    return HexCarrier(centres, polygons, geometry, QuadraticField(), reference)


@pytest.fixture(autouse=True)
def local_cache(tmp_path, monkeypatch):
    """Point both caches at a fresh directory so no test writes the shared root."""
    root = tmp_path / "operands"
    monkeypatch.setenv("NOVA_OPERAND_CACHE", str(root))
    return root


def test_cache_root_honours_the_environment_override(tmp_path, monkeypatch):
    """The helper resolves the configured root rather than a hard-coded path."""
    monkeypatch.setenv("NOVA_OPERAND_CACHE", str(tmp_path / "elsewhere"))
    assert operand_cache_root() == tmp_path / "elsewhere"
    monkeypatch.delenv("NOVA_OPERAND_CACHE")
    assert str(operand_cache_root()).startswith("/work/projects/imas_gpu")


def test_values_persist_by_default_and_hit_on_the_second_build():
    """A first build stores the entry; a second call loads it bitwise equal."""
    configure_dtypes()
    block = carrier()
    rule = triangle_rule(block.geometry)
    values, first = build_values(block.machine, rule, block.field, block.reference)
    assert first["cache_hit"] is False
    assert first["build_seconds"] > 0.0
    stored = Path(first["cache_path"])
    assert stored.is_file()
    assert stored.parent.parent.name == "sha256"
    assert stored.parent.name == first["digest"].removeprefix("sha256:")
    assert (stored.parent / "manifest.json").is_file()

    _, second = build_values(block.machine, rule, block.field, block.reference)
    assert second["cache_hit"] is True
    assert second["build_seconds"] is None
    assert second["stored_build_seconds"] == first["build_seconds"]
    assert second["load_seconds"] is not None
    assert second["matrix_ready_seconds"] == second["load_seconds"]
    assert second["digest"] == first["digest"]

    loaded = np.asarray(np.load(stored, mmap_mode="r"))
    assert loaded.tobytes() == np.asarray(values.matrix).tobytes()


def test_values_miss_on_a_perturbed_node_coordinate():
    """Moving one polygon vertex changes the key and rebuilds rather than loading."""
    configure_dtypes()
    block = carrier()
    rule = triangle_rule(block.geometry)
    points = np.asarray(rule.points).reshape(-1, 2)
    centres = np.asarray(rule.centre)

    moved = [np.array(polygon) for polygon in block.polygons]
    moved[0][0, 0] += 1.0e-9
    moved = tuple(moved)
    original = values_digest(points, block.polygons, centres)
    assert values_digest(points, moved, centres) != original

    _, first = build_values(block.machine, rule, block.field, block.reference)
    shifted = HexCarrier(
        block.centres, moved, block.geometry, block.field, block.reference
    )
    _, second = build_values(shifted.machine, rule, block.field, block.reference)
    assert second["digest"] != first["digest"]
    assert second["cache_hit"] is False


def test_values_miss_on_a_perturbed_source_byte(monkeypatch):
    """Flipping one byte of a hashed source produces a new key and a rebuild."""
    configure_dtypes()
    block = carrier()
    rule = triangle_rule(block.geometry)
    points = np.asarray(rule.points).reshape(-1, 2)
    centres = np.asarray(rule.centre)
    original = values_digest(points, block.polygons, centres)

    chunks = list(value_module.source_bytes())
    corrupted = bytearray(chunks[0])
    corrupted[0] ^= 0x01
    monkeypatch.setattr(
        value_module, "source_bytes", lambda: (bytes(corrupted), *chunks[1:])
    )
    assert values_digest(points, block.polygons, centres) != original

    _, first = build_values(block.machine, rule, block.field, block.reference)
    assert first["cache_hit"] is False
    _, second = build_values(block.machine, rule, block.field, block.reference)
    assert second["cache_hit"] is True


def _concurrent_build(root: str, index: int, queue):
    """Build one shared key in a child process and report what it read back."""
    os.environ["NOVA_OPERAND_CACHE"] = root
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    configure_dtypes()
    block = carrier()
    values, receipt = build_values(
        block.machine, triangle_rule(block.geometry), block.field, block.reference
    )
    stored = np.asarray(np.load(receipt["cache_path"], mmap_mode="r"))
    queue.put((index, np.asarray(values.matrix).tobytes(), stored.tobytes()))


@pytest.mark.slow
def test_two_concurrent_processes_never_see_a_partial_matrix(tmp_path):
    """Two builders of one key leave a complete entry and agree bitwise."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_build, args=(str(tmp_path / "operands"), index, queue)
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=1800) for _ in processes]
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    built = [entry[1] for entry in sorted(results)]
    stored = [entry[2] for entry in sorted(results)]
    assert built[0] == built[1]
    assert stored[0] == stored[1]
    assert stored[0] == built[0]


@pytest.mark.slow
def test_jet_matrix_persists_by_default():
    """``build_jet_matrix`` writes a published entry on its default arm."""
    configure_dtypes()
    from scripts.prototypes.read_route.jets import build_jet_matrix, read_operand

    block = carrier()
    coupling = topology.BiotMomentCoupling.from_polygons(block.polygons)
    columns = tuple(block.reference[:, index] for index in range(3))
    receipt = {}
    matrix, _, _, _, _ = build_jet_matrix(
        coupling, block.geometry, block.field, columns, receipt=receipt
    )
    assert receipt["cache_hit"] is False
    directory = Path(receipt["cache_path"]).parent
    assert (directory / "manifest.json").is_file()
    entry = read_operand("jets", receipt["digest"], tuple(np.asarray(matrix).shape))
    assert entry is not None
    assert np.asarray(entry.matrix).tobytes() == np.asarray(matrix).tobytes()
    assert entry.build_seconds == receipt["build_seconds"]

    hit = {}
    build_jet_matrix(coupling, block.geometry, block.field, columns, receipt=hit)
    assert hit["cache_hit"] is True
    assert hit["build_seconds"] is None
    assert hit["load_seconds"] is not None
    assert hit["stored_build_seconds"] == receipt["build_seconds"]
    assert hit["digest"] == receipt["digest"]