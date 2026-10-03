"""Measure contour-tree sweep placements without changing the production route.

The benchmark holds the carrier and receipt contract constant while comparing
serial host calls, a ``vmap`` of the existing implementation, and a device
prototype that compresses every union-find parent in parallel at each visit.
The prototype intentionally lives here: a receipt-identical result is a
precondition for considering a production algorithm change.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.contour_tree_brute_force import (
    Fixture,
    certificate_rung_fixtures,
    mast_fixtures,
    receipt_from_tree,
)
from nova.equilibrium.contour_tree import (
    ContourTreeArrays,
    ContourTreeResult,
    _append_edges,
    _carrier_adjacency,
    _insert_split_nodes,
    build_contour_tree,
)
from nova.jax.config import configure_dtypes


BATCH_SIZES = (1, 8, 64)
PLACEMENTS = ("cpu", "vmap", "parallel")
MESH_FIELDS = (
    "vertex_psi",
    "vertex_valid",
    "vertex_is_wall",
    "edges",
    "edge_valid",
)


@dataclass(frozen=True)
class Timing:
    """Cold-minus-warm compilation and warm execution walls for one carrier."""

    placement: str
    fixture: str
    vertices: int
    batch_size: int
    compile_seconds: float
    execute_seconds: float


def _parallel_roots(parents: jax.Array) -> jax.Array:
    """Compress every parent pointer at once by pointer jumping.

    This is the prototype's parallel union-find operation.  It produces the
    same representative vector as following only a visit's adjacent roots,
    but maps the full parent table in parallel before that visit consumes it.
    """

    def unresolved(table: jax.Array) -> jax.Array:
        return jnp.any(table != table[table])

    return jax.lax.while_loop(unresolved, lambda table: table[table], parents)


def _parallel_sweep_tree(
    values: jax.Array,
    vertex_valid: jax.Array,
    vertex_is_wall: jax.Array,
    neighbours: jax.Array,
    neighbour_valid: jax.Array,
    descending: bool,
    node_capacity: int,
    edge_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build one tree with full-table pointer-jumping union-find compression."""

    vertex_count = values.size
    neighbour_count = neighbours.shape[1]
    index = jnp.arange(vertex_count, dtype=jnp.int32)
    order = jnp.lexsort((index, -values if descending else values))
    last_usable_rank = jnp.max(
        jnp.where(vertex_valid[order], index, jnp.asarray(-1, dtype=jnp.int32))
    )
    parents = index
    active = jnp.zeros(vertex_count, dtype=bool)
    births = index
    node_valid = jnp.zeros(node_capacity, dtype=bool)
    node_type = jnp.full(node_capacity, -1, dtype=jnp.int32)
    output_edges = jnp.zeros((edge_capacity, 2), dtype=jnp.int32)
    output_valid = jnp.zeros(edge_capacity, dtype=bool)
    overflow = jnp.asarray(False)
    wall_flag = jnp.zeros(vertex_count, dtype=bool)

    def visit(rank: int, state):
        (
            parents,
            active,
            births,
            node_valid,
            node_type,
            output_edges,
            output_valid,
            overflow,
            wall_flag,
        ) = state
        parents = _parallel_roots(parents)
        vertex = order[rank].astype(jnp.int32)
        usable = vertex_valid[vertex]
        adjacent = neighbour_valid[vertex]
        neighbour = jnp.where(adjacent, neighbours[vertex], vertex)
        neighbour_root = parents[neighbour]
        root_active = adjacent & active[neighbour]
        earlier = jnp.arange(neighbour_count)[:, None] < jnp.arange(neighbour_count)
        repeated = jnp.any(
            earlier
            & root_active[:, None]
            & root_active[None, :]
            & (neighbour_root[:, None] == neighbour_root[None, :]),
            axis=0,
        )
        representative = root_active & ~repeated
        neighbour_wall = jnp.any(representative & wall_flag[neighbour_root])
        component_count = jnp.sum(representative, dtype=jnp.int32)
        first_wall = usable & descending & vertex_is_wall[vertex] & ~neighbour_wall
        final_vertex = usable & (rank == last_usable_rank)
        extremum = component_count == 0
        saddle = component_count >= 2
        terminal = final_vertex & (component_count == 1)
        event = usable & (extremum | saddle | terminal | first_wall)
        signed_maximum = (descending & extremum) | ((not descending) & terminal)
        signed_minimum = ((not descending) & extremum) | (descending & terminal)
        critical = jnp.where(
            saddle | first_wall,
            1,
            jnp.where(signed_maximum, 2, jnp.where(signed_minimum, 0, 1)),
        ).astype(jnp.int32)
        node_valid = node_valid.at[vertex].set(node_valid[vertex] | event)
        node_type = node_type.at[vertex].set(
            jnp.where(event, critical, node_type[vertex])
        )
        connect = usable & (saddle | terminal | first_wall)
        sources = births[neighbour_root]
        output_edges, output_valid, overflow = _append_edges(
            output_edges,
            output_valid,
            overflow,
            sources,
            representative & connect,
            vertex,
        )
        parents = parents.at[neighbour_root].set(
            jnp.where(root_active & usable, vertex, parents[neighbour_root])
        )
        parents = parents.at[vertex].set(vertex)
        first_root = jnp.argmax(representative).astype(jnp.int32)
        inherited = births[neighbour_root[first_root]]
        births = births.at[vertex].set(
            jnp.where(extremum | saddle | first_wall, vertex, inherited)
        )
        active = active.at[vertex].set(usable)
        wall_flag = wall_flag.at[vertex].set(
            usable & (vertex_is_wall[vertex] | neighbour_wall)
        )
        return (
            parents,
            active,
            births,
            node_valid,
            node_type,
            output_edges,
            output_valid,
            overflow,
            wall_flag,
        )

    result = jax.lax.fori_loop(
        0,
        vertex_count,
        visit,
        (
            parents,
            active,
            births,
            node_valid,
            node_type,
            output_edges,
            output_valid,
            overflow,
            wall_flag,
        ),
    )
    return result[3], result[4], result[5], result[6], result[7], order


@jax.jit
def build_parallel_union_find(
    vertex_psi: jax.Array,
    vertex_valid: jax.Array,
    vertex_is_wall: jax.Array,
    edges: jax.Array,
    edge_valid: jax.Array,
    sigma: jax.Array,
) -> ContourTreeArrays:
    """Build a receipt with the benchmark-only parallel union-find sweep."""

    node_capacity = ContourTreeResult.node_capacity
    edge_capacity = ContourTreeResult.edge_capacity
    vertex_count = vertex_psi.size
    signed = jnp.asarray(sigma, dtype=vertex_psi.dtype) * vertex_psi
    neighbours, neighbour_valid, adjacency_overflow = _carrier_adjacency(
        edges, edge_valid, vertex_count
    )
    join = _parallel_sweep_tree(
        signed,
        vertex_valid,
        vertex_is_wall,
        neighbours,
        neighbour_valid,
        True,
        vertex_count,
        edges.shape[0],
    )
    split = _parallel_sweep_tree(
        signed,
        vertex_valid,
        vertex_is_wall,
        neighbours,
        neighbour_valid,
        False,
        vertex_count,
        edges.shape[0],
    )
    join_nodes, join_types, join_edges, join_valid, join_overflow, _ = join
    split_nodes, split_types, _, _, split_overflow, _ = split
    node_valid = join_nodes | split_nodes
    critical_type = jnp.where(join_nodes, join_types, split_types)
    critical_type = jnp.where(
        critical_type == 1,
        critical_type,
        jnp.where(sigma > 0, critical_type, 2 - critical_type),
    )
    node_valid, merged_edges, merged_valid, overflow = _insert_split_nodes(
        node_valid,
        join_edges,
        join_valid,
        join_overflow | split_overflow,
        split_nodes & ~join_nodes,
        signed,
    )
    slots = jnp.arange(node_capacity, dtype=jnp.int32)
    join_order = join[-1]
    terminal_rank = jnp.max(
        jnp.where(
            vertex_valid[join_order],
            jnp.arange(vertex_count, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
        )
    )
    terminal = join_order[terminal_rank]
    virtual = jnp.argmin(
        jnp.where(vertex_valid, vertex_count, jnp.arange(vertex_count, dtype=jnp.int32))
    )
    replace_terminal = jnp.any(~vertex_valid) & join_nodes[terminal]
    node_valid = node_valid.at[terminal].set(
        jnp.where(replace_terminal, False, node_valid[terminal])
    )
    node_valid = node_valid.at[virtual].set(node_valid[virtual] | replace_terminal)
    critical_type = critical_type.at[virtual].set(
        jnp.where(replace_terminal, 3, critical_type[virtual])
    )
    merged_edges = jnp.where(
        (merged_edges == terminal) & replace_terminal, virtual, merged_edges
    )
    node_count = jnp.sum(node_valid, dtype=jnp.int32)
    node_source = jnp.nonzero(node_valid, size=node_capacity, fill_value=0)[0]
    compact_node_valid = slots < node_count
    edge_count = jnp.sum(merged_valid, dtype=jnp.int32)
    edge_source = jnp.nonzero(merged_valid, size=edge_capacity, fill_value=0)[0]
    compact_edge_valid = jnp.arange(edge_capacity) < edge_count
    carrier_vertex = jnp.arange(vertex_count, dtype=jnp.int32)
    node_matches = (
        node_source[:, None] == carrier_vertex[None, :]
    ) & compact_node_valid[:, None]
    vertex_node = jnp.argmax(node_matches, axis=0).astype(jnp.int32)
    compact_edges = vertex_node[merged_edges[edge_source]]
    edge_nodes_valid = jnp.all(
        jnp.any(node_matches, axis=0)[merged_edges[edge_source]], axis=1
    )
    overflow = overflow | adjacency_overflow | (node_count > node_capacity)
    overflow = overflow | (edge_count > edge_capacity)
    overflow = overflow | jnp.any(compact_edge_valid & ~edge_nodes_valid)
    return ContourTreeArrays(
        node_vertex=node_source,
        node_psi=jnp.where(
            compact_node_valid,
            vertex_psi[node_source],
            jnp.zeros((), vertex_psi.dtype),
        ),
        node_valid=compact_node_valid,
        critical_type=jnp.where(compact_node_valid, critical_type[node_source], -1),
        edges=compact_edges,
        edge_valid=compact_edge_valid,
        overflow=overflow,
    )


def _arguments(fixture: Fixture, batch_size: int) -> tuple[jax.Array, ...]:
    """Repeat one physical carrier into a same-shaped batch."""

    mesh = fixture.mesh
    return tuple(
        jnp.broadcast_to(value, (batch_size,) + value.shape)
        for value in (
            mesh.vertex_psi,
            mesh.vertex_valid,
            mesh.vertex_is_wall,
            mesh.edges,
            mesh.edge_valid,
            jnp.asarray(1, dtype=jnp.int32),
        )
    )


def _write_fixture_inputs(fixtures: tuple[Fixture, ...], directory: Path) -> None:
    """Persist host-built carrier arrays so every backend receives identical bits."""

    directory.mkdir(parents=True, exist_ok=True)
    for fixture in fixtures:
        np.savez_compressed(
            directory / f"{fixture.name}.npz",
            **{name: np.asarray(getattr(fixture.mesh, name)) for name in MESH_FIELDS},
        )


def _with_fixture_inputs(
    fixtures: tuple[Fixture, ...], directory: Path
) -> tuple[Fixture, ...]:
    """Replace backend-built meshes with the CPU-persisted carrier arrays."""

    loaded = []
    for fixture in fixtures:
        path = directory / f"{fixture.name}.npz"
        if not path.is_file():
            raise RuntimeError(f"missing persisted carrier input: {path}")
        with np.load(path) as payload:
            mesh = SimpleNamespace(
                **{name: jnp.asarray(payload[name]) for name in MESH_FIELDS}
            )
        loaded.append(Fixture(fixture.name, mesh))
    return tuple(loaded)


def _first_receipt(result: ContourTreeArrays) -> dict[str, object]:
    """Return one lane after asserting all replica lanes are bit-identical."""

    lanes = [np.asarray(value) for value in jax.tree.leaves(result)]
    for value in lanes:
        if not np.array_equal(value, np.broadcast_to(value[0], value.shape)):
            raise RuntimeError("replicated batch returned non-identical receipt lanes")
    first = jax.tree.map(lambda value: value[0], result)
    return receipt_from_tree(first)


def _block(result: ContourTreeArrays) -> ContourTreeArrays:
    """Synchronise the receipt's final scalar, thereby covering all device work."""

    result.overflow.block_until_ready()
    return result


def _measure_device(
    fixture: Fixture,
    batch_size: int,
    placement: str,
    implementation,
) -> tuple[Timing, dict[str, object]]:
    """Measure a compiled, same-carrier ``vmap`` on the active backend."""

    batched = jax.jit(jax.vmap(implementation))
    arguments = _arguments(fixture, batch_size)
    started = time.perf_counter()
    _block(batched(*arguments))
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm = _block(batched(*arguments))
    execute_seconds = time.perf_counter() - started
    return (
        Timing(
            placement=placement,
            fixture=fixture.name,
            vertices=int(np.count_nonzero(np.asarray(fixture.mesh.vertex_valid))),
            batch_size=batch_size,
            compile_seconds=max(cold_seconds - execute_seconds, 0.0),
            execute_seconds=execute_seconds,
        ),
        _first_receipt(warm),
    )


def _measure_cpu(fixture: Fixture, batch_size: int) -> tuple[Timing, dict[str, object]]:
    """Measure host-side single-field calls, serially, for one requested batch."""

    mesh = fixture.mesh
    arguments = (
        mesh.vertex_psi,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        jnp.asarray(1, dtype=jnp.int32),
    )
    started = time.perf_counter()
    _block(build_contour_tree(*arguments))
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    results = [_block(build_contour_tree(*arguments)) for _ in range(batch_size)]
    execute_seconds = time.perf_counter() - started
    receipts = [receipt_from_tree(result) for result in results]
    if any(receipt != receipts[0] for receipt in receipts[1:]):
        raise RuntimeError("serial replicas returned non-identical receipts")
    return (
        Timing(
            placement="cpu",
            fixture=fixture.name,
            vertices=int(np.count_nonzero(np.asarray(mesh.vertex_valid))),
            batch_size=batch_size,
            compile_seconds=max(cold_seconds - execute_seconds / batch_size, 0.0),
            execute_seconds=execute_seconds,
        ),
        receipts[0],
    )


def _measure(
    fixtures: tuple[Fixture, ...], placements: tuple[str, ...]
) -> tuple[list[Timing], dict[str, dict[str, object]]]:
    """Collect a receipt and timing for every fixture, placement, and batch."""

    rows: list[Timing] = []
    receipts: dict[str, dict[str, object]] = {placement: {} for placement in placements}
    implementations = {
        "vmap": build_contour_tree,
        "parallel": build_parallel_union_find,
    }
    for fixture in fixtures:
        for batch_size in BATCH_SIZES:
            key = f"{fixture.name}/batch-{batch_size}"
            for placement in placements:
                if placement == "cpu":
                    row, receipt = _measure_cpu(fixture, batch_size)
                else:
                    row, receipt = _measure_device(
                        fixture, batch_size, placement, implementations[placement]
                    )
                rows.append(row)
                receipts[placement][key] = receipt
    return rows, receipts


def _write_measurement(
    directory: Path, placement: str, rows: list[Timing], receipts: dict[str, object]
) -> Path:
    """Persist the numeric result and its comparator-ready receipt file."""

    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / f"{placement}-receipts.json"
    receipt_path.write_text(json.dumps(receipts, sort_keys=True), encoding="utf-8")
    path = directory / f"{placement}-measurement.json"
    path.write_text(
        json.dumps(
            {
                "placement": placement,
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "rows": [asdict(row) for row in rows],
                "receipt_path": str(receipt_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _render(rows: list[dict[str, object]], figure: Path) -> None:
    """Draw execute walls against carrier size with direct placement labels."""

    import matplotlib.pyplot as plt

    try:
        plt.style.use("data-ink")
    except OSError:
        pass
    figure.parent.mkdir(parents=True, exist_ok=True)
    colours = {"cpu": "#4c78a8", "vmap": "#d18438", "parallel": "#547a48"}
    labels = {
        "cpu": "host serial",
        "vmap": "device vmap",
        "parallel": "device union-find",
    }
    image, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=100, sharey=False)
    for axis, batch_size in zip(axes, BATCH_SIZES, strict=True):
        for placement in PLACEMENTS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["placement"] == placement and row["batch_size"] == batch_size
                ),
                key=lambda row: (row["vertices"], row["fixture"]),
            )
            if not selected:
                continue
            vertices = [row["vertices"] for row in selected]
            walls = [row["execute_seconds"] for row in selected]
            axis.plot(vertices, walls, color=colours[placement], linewidth=2.6)
            axis.text(
                vertices[-1],
                walls[-1],
                labels[placement],
                color=colours[placement],
                fontsize=11,
                ha="right",
                va="bottom",
            )
        axis.set_xlabel("live vertices")
        axis.set_title(f"batch {batch_size}")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("warm execute wall [s]")
    image.tight_layout()
    image.savefig(figure, bbox_inches="tight")
    plt.close(image)


def _combine(paths: list[Path], output: Path, figure: Path) -> None:
    """Combine device and host measurements and render the one evidence figure."""

    measurements = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = [row for measurement in measurements for row in measurement["rows"]]
    available = {measurement["placement"] for measurement in measurements}
    if available != set(PLACEMENTS):
        raise RuntimeError(f"need every placement, received {sorted(available)}")
    merged_receipts: dict[str, dict[str, object]] = {
        placement: {} for placement in PLACEMENTS
    }
    for measurement in measurements:
        placement = measurement["placement"]
        receipt_path = Path(measurement["receipt_path"])
        receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
        duplicate = set(merged_receipts[placement]) & set(receipts)
        if duplicate:
            raise RuntimeError(
                f"duplicate receipt keys for {placement}: {sorted(duplicate)}"
            )
        merged_receipts[placement].update(receipts)
    for placement, receipts in merged_receipts.items():
        (output.parent / f"{placement}-receipts.json").write_text(
            json.dumps(receipts, sort_keys=True), encoding="utf-8"
        )
    _render(rows, figure)
    output.write_text(
        json.dumps(
            {"measurements": measurements, "rows": rows, "figure": str(figure)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Run one backend arm or combine previously measured arms."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--placement", choices=("cpu", "device"))
    parser.add_argument(
        "--fixture",
        action="append",
        help="measure this exact fixture name; repeat to form a bounded subset",
    )
    parser.add_argument(
        "--write-input-dir",
        type=Path,
        help="persist CPU-built carrier arrays for a cross-device arm",
    )
    parser.add_argument(
        "--fixture-input-dir",
        type=Path,
        help="read CPU-persisted carrier arrays instead of backend-built meshes",
    )
    parser.add_argument("--combine", nargs="+", type=Path)
    parser.add_argument("--combined-output", type=Path)
    parser.add_argument("--figure", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.combine is not None:
        if arguments.combined_output is None or arguments.figure is None:
            parser.error("--combine requires --combined-output and --figure")
        _combine(arguments.combine, arguments.combined_output, arguments.figure)
        print(arguments.combined_output)
        return 0
    configure_dtypes()
    fixtures = certificate_rung_fixtures() + mast_fixtures()
    if arguments.fixture:
        selected = set(arguments.fixture)
        fixtures = tuple(item for item in fixtures if item.name in selected)
        missing = selected - {item.name for item in fixtures}
        if missing:
            parser.error(f"unknown fixture names: {sorted(missing)}")
    if arguments.write_input_dir is not None:
        _write_fixture_inputs(fixtures, arguments.write_input_dir)
        print(arguments.write_input_dir)
        return 0
    if arguments.output_dir is None or arguments.placement is None:
        parser.error("a measurement needs --output-dir and --placement")
    if arguments.fixture_input_dir is not None:
        fixtures = _with_fixture_inputs(fixtures, arguments.fixture_input_dir)
    placements = ("cpu",) if arguments.placement == "cpu" else ("vmap", "parallel")
    rows, receipts = _measure(fixtures, placements)
    for placement in placements:
        selected = [row for row in rows if row.placement == placement]
        print(
            _write_measurement(
                arguments.output_dir, placement, selected, receipts[placement]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
