"""Compare the tile backends of the polygon operator build: numpy pool vs traced.

    python benchmarks/tiled_backend.py --device cpu
    python benchmarks/tiled_backend.py --device gpu
    python benchmarks/tiled_backend.py --measure jax-vmap --cells 320 --tile 40

The question this answers is whether one traced kernel can serve both a CPU
build and a GPU build, or whether the CPU needs a separate implementation. So
every variant builds the SAME operator -- a hexagonally-tiled plasma-like cell
set coupled to its own cell centres, at the exact-everywhere 16x48 quadrature
rule -- and the only thing that changes is how the tiles are evaluated:

    numpy-<n>   the shipped path, tiles spread over an n-process pool
    jax-scan    one compiled kernel, quadrature blocks walked with scan
    jax-vmap    the same kernel, blocks mapped with vmap
    parity      both backends into two stores, compared over every pair

The CLOSED-form reduction is measured through the same harness, because the
question it answers is the same one -- can the accurate kernel reach a device --
and only a shared harness makes the two comparable:

    closed-host    the shipped host route, section by section, WITH its two
                   value-dependent shortcuts (a pole family it can prove no
                   column needs, and a corner term a closed chain cancels)
    closed-numpy   the same reduction over tiles through the packed driver, which
                   has no shortcuts because a traced value cannot be inspected --
                   so this isolates what the shortcuts are worth
    closed-scan    the packed driver traced, blocks walked with scan
    closed-vmap    the packed driver traced, blocks mapped with vmap
    closed-parity  the traced closed kernel against the numpy packed driver

Once per build is not once, and for the closed form the compile is half the
build -- so what a compile is paid PER is measured too, on either kernel:

    jax-positions    the same sections built at several POSITIONS in one
    closed-positions process, which is what a geometry scan is; the first
                     position carries the compile and the rest report what a
                     build costs once the evaluator is warm
    jax-cache        two fresh processes sharing one on-disk compilation cache:
    closed-cache     the first compiles, the second must not

Timings are one variant per FRESH process, because a compiled kernel and a warm
allocator both make a second measurement in the same interpreter look faster
than a build does. The parent process spawns those children and reports the
median. Every child runs with the persistent compilation cache OFF unless the
variant is about the cache, so a compile figure is always a compile and never a
disk read. Traced variants report the compile separately from the steady state:
the compile is paid once per build, so a build long enough to matter amortises
it, and quoting only the cold total would hide the per-pair rate that scales.

Run on a compute node -- login-node timings of this kernel spread 5x.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from benchmarks.tiled_assembly import hex_mesh
from nova.biot.polygon import pad_batch
from nova.biot.tiledassembly import (
    COMPONENTS,
    TilePlan,
    assemble,
    compilation_cache,
    tile_coupling,
    tile_evaluator,
)

CPU_VARIANTS = ("numpy-1", "numpy-8", "numpy-16", "jax-scan", "jax-vmap")
GPU_VARIANTS = ("numpy-8", "jax-vmap", "jax-scan")

# Positions a geometry scan is measured over.  Four is enough to separate the
# first build from the rest and to show the rest do not drift.
_POSITIONS = 4

# How far the pack moves between positions, in metres: far enough that the
# operator really changes, near enough that it stays the same problem.
_STEP = 0.013

# Persistent-cache traffic seen by this process, filled by the monitoring
# listeners that JAX calls on a cache hit.
_CACHE_EVENTS = {"hits": 0, "saved_seconds": 0.0}


def build_plan(tile: int) -> TilePlan:
    """Return the tile shape every variant builds, at the exact-everywhere rule."""
    return TilePlan(
        target_tile=tile, source_tile=tile, block=16, n_panels=16, n_nodes=48
    )


def new_store(path, shape, plan: TilePlan):
    """Return a zarr group chunked to the tile, one array per component."""
    import zarr

    store = zarr.open_group(str(path), mode="w")
    for name in COMPONENTS:
        store.create_array(
            name,
            shape=shape,
            chunks=(plan.target_tile, plan.source_tile),
            dtype="float64",
        )
    return store


def deviation(got, want) -> tuple[float, float]:
    """Return the worst absolute and worst relative deviation of one component.

    The relative figure is taken only where the reference is above a millionth of
    the component's own peak: the operator spans many orders of magnitude and a
    pointwise ratio against a near-zero entry measures the reference's own
    cancellation, not the backend's agreement.
    """
    absolute = np.abs(got - want)
    peak = float(np.max(np.abs(want)))
    significant = np.abs(want) >= 1e-6 * peak
    relative = absolute[significant] / np.abs(want)[significant]
    return float(absolute.max()), float(relative.max())


def device_report() -> dict:
    """Return what the traced kernel is actually running on, and its headroom."""
    import jax

    device = jax.devices()[0]
    record = {
        "platform": device.platform,
        "device": str(device),
        "visible_cores": len(os.sched_getaffinity(0)),
    }
    try:
        stats = device.memory_stats() or {}
    except Exception:  # a CPU device has no allocator to report
        stats = {}
    for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
        if key in stats:
            record[key] = int(stats[key])
    return record


def watch_cache() -> None:
    """Start counting persistent-cache hits, before anything is compiled.

    A warm process and a cold one differ only in how long the compile takes,
    which a slow filesystem or a busy node could equally explain -- so the hit is
    counted rather than inferred from the clock.
    """
    import jax.monitoring as monitoring

    def hit(event, **kwargs):
        if event == "/jax/compilation_cache/cache_hits":
            _CACHE_EVENTS["hits"] += 1

    def saved(event, duration_secs, **kwargs):
        if event == "/jax/compilation_cache/compile_time_saved_sec":
            _CACHE_EVENTS["saved_seconds"] += duration_secs

    monitoring.register_event_listener(hit)
    monitoring.register_event_duration_secs_listener(saved)


def store_every_compile() -> None:
    """Lower the cache's worth-keeping threshold, when a directory was named.

    JAX stores only compiles above a second, which is the right default and the
    wrong one for a probe whose subject IS the store: the quadrature kernel
    compiles in about that time, so whether its executable was kept would be a
    coin toss. Only a run that explicitly named a cache directory is affected --
    an ordinary timing has the cache off and reaches none of this.
    """
    if not os.environ.get("NOVA_COMPILATION_CACHE"):
        return
    directory = compilation_cache()
    if directory is not None:
        compilation_cache(directory, min_compile_seconds=0.0)


def cache_report() -> dict:
    """Return where the on-disk cache is and what this process took from it."""
    import jax

    return {
        "cache_dir": jax.config.jax_compilation_cache_dir,
        "cache_hits": _CACHE_EVENTS["hits"],
        "compile_seconds_saved": _CACHE_EVENTS["saved_seconds"],
    }


def measure_numpy(workers: int, cells: int, tile: int) -> dict:
    """Return the wall time of a pooled numpy build of the whole operator."""
    sections, target_r, target_z = hex_mesh(cells)
    plan = build_plan(tile)
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tiled-backend-"))
    try:
        start = time.perf_counter()
        assemble(
            directory / "coupling.zarr",
            target_r,
            target_z,
            sections,
            plan=plan,
            workers=workers,
        )
        seconds = time.perf_counter() - start
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    pairs = target_r.size * len(sections)
    return {
        "seconds": seconds,
        "pairs": pairs,
        "us_per_pair": 1e6 * seconds / pairs,
        "tiles": plan.tile_count(target_r.size, len(sections)),
    }


def measure_host_closed(cells: int, tile: int) -> dict:
    """Return the per-pair rate of the shipped host closed form over the mesh.

    Section by section, all targets at once, through the driver that keeps its
    value-dependent shortcuts -- the rate a near-band host evaluation would run at.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    sections, target_r, target_z = hex_mesh(cells)
    start = time.perf_counter()
    for section in sections:
        polygon_analytic_greens(target_r, target_z, section)
    seconds = time.perf_counter() - start
    pairs = target_r.size * len(sections)
    return {
        "seconds": seconds,
        "pairs": pairs,
        "us_per_pair": 1e6 * seconds / pairs,
        "tiles": 1,
    }


def measure_packed_closed(cells: int, tile: int) -> dict:
    """Return the per-pair rate of the packed closed driver on numpy, over tiles.

    The same arithmetic as the traced kernel and the same absence of shortcuts, so
    the gap to :func:`measure_host_closed` is what the shortcuts buy and the gap to
    the traced variants is what the compiler and the device buy.
    """
    from nova.biot.polygonanalytic import packed_analytic_greens

    sections, target_r, target_z = hex_mesh(cells)
    plan = build_plan(tile)
    edge, weight, norm = pad_batch(sections)
    bounds = list(plan.tiles(target_r.size, len(sections)))
    start = time.perf_counter()
    for rows, columns in bounds:
        pair_r, pair_z, pair_edge, pair_weight, pair_norm = _pair_geometry(
            target_r[rows],
            target_z[rows],
            edge[:, :, columns],
            weight[:, columns],
            norm[columns],
        )
        packed_analytic_greens(np, pair_r, pair_z, pair_edge, pair_weight, pair_norm)
    seconds = time.perf_counter() - start
    pairs = target_r.size * len(sections)
    return {
        "seconds": seconds,
        "pairs": pairs,
        "us_per_pair": 1e6 * seconds / pairs,
        "tiles": len(bounds),
    }


def _pair_geometry(target_r, target_z, edge, weight, norm):
    """Return one tile's geometry flattened onto its pair list.

    The closed form is a function of a PAIR -- one target against one section --
    where the quadrature kernel broadcasts a target row against a source column, so
    a tile is presented to it as a flat pair vector with each pair's own section
    beside it.
    """
    rows, columns = np.divmod(np.arange(target_r.size * norm.size), norm.size)
    return (
        target_r[rows],
        target_z[rows],
        edge[:, :, columns],
        weight[:, columns],
        norm[columns],
    )


def measure_traced(
    batched: bool, cells: int, tile: int, kernel: str = "quadrature"
) -> dict:
    """Return compile time and steady-state build time of the traced backend.

    The first tile carries the compile, so it is timed alone and then the whole
    operator is built with the compiled kernel. Both phases write into the zarr
    store, so the steady-state figure is comparable with the numpy build rather
    than being a bare kernel rate.
    """
    watch_cache()
    sections, target_r, target_z = hex_mesh(cells)
    plan = build_plan(tile)
    edge, weight, norm = pad_batch(sections)
    shape = (target_r.size, len(sections))
    bounds = list(plan.tiles(*shape))
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tiled-backend-"))
    try:
        store = new_store(directory / "coupling.zarr", shape, plan)
        evaluate = tile_evaluator(plan, batched=batched, kernel=kernel)

        def write(rows, columns):
            tile_result = evaluate(
                target_r[rows],
                target_z[rows],
                edge[:, :, columns],
                weight[:, columns],
                norm[columns],
            )
            for name, component in zip(COMPONENTS, tile_result):
                store[name][rows, columns] = component

        start = time.perf_counter()
        write(*bounds[0])
        compile_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for rows, columns in bounds:
            write(rows, columns)
        seconds = time.perf_counter() - start
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    pairs = target_r.size * len(sections)
    return {
        "seconds": seconds,
        "compile_seconds": compile_seconds,
        "cold_seconds": compile_seconds + seconds,
        "pairs": pairs,
        "us_per_pair": 1e6 * seconds / pairs,
        "tiles": len(bounds),
        "compile_count": evaluate.compile_count,
        **cache_report(),
        **device_report(),
    }


def measure_positions(cells: int, tile: int, kernel: str) -> dict:
    """Return what a geometry scan pays at its first position and at the rest.

    The same sections translated in R and built again -- a winding pack swept
    through positions, which is a real use and the one that turns a per-build
    compile into the whole cost. Each position goes through ``assemble``, so this
    is the caller's own loop rather than a bare kernel rate; the first build
    carries the compile and every later one finds the evaluator warm, so the gap
    between them IS the compile and a fresh evaluator per position would pay the
    first figure at every one of them.
    """
    watch_cache()
    sections, target_r, target_z = hex_mesh(cells)
    plan = build_plan(tile)
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tiled-positions-"))
    seconds = []
    try:
        for index in range(_POSITIONS):
            shift = _STEP * index
            start = time.perf_counter()
            assemble(
                directory / f"position-{index}.zarr",
                target_r + shift,
                target_z,
                [section + np.array([shift, 0.0]) for section in sections],
                plan=plan,
                backend="jax",
                batched=True,
                kernel=kernel,
            )
            seconds.append(time.perf_counter() - start)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    evaluate = tile_evaluator(plan, batched=True, kernel=kernel)
    later = statistics.median(seconds[1:])
    pairs = target_r.size * len(sections)
    return {
        # named as the traced variants name them, so the table reads the same:
        # the steady state is what a position costs warm, and the compile is
        # what the first one paid over it
        "seconds": later,
        "compile_seconds": seconds[0] - later,
        "cold_seconds": seconds[0],
        "position_seconds": seconds,
        "positions": _POSITIONS,
        "pairs": pairs,
        "us_per_pair": 1e6 * later / pairs,
        "tiles": plan.tile_count(target_r.size, len(sections)),
        "compile_count": evaluate.compile_count,
        **cache_report(),
        **device_report(),
    }


def measure_closed_parity(cells: int, tile: int) -> dict:
    """Return the traced closed kernel's deviation from the same driver on numpy."""
    from nova.biot.polygonanalytic import packed_analytic_greens

    sections, target_r, target_z = hex_mesh(cells)
    plan = build_plan(tile)
    edge, weight, norm = pad_batch(sections)
    evaluate = tile_evaluator(plan, batched=True, kernel="closed")
    worst_abs = worst_rel = 0.0
    for rows, columns in plan.tiles(target_r.size, len(sections)):
        arguments = (
            target_r[rows],
            target_z[rows],
            edge[:, :, columns],
            weight[:, columns],
            norm[columns],
        )
        want = packed_analytic_greens(np, *_pair_geometry(*arguments))
        for got, reference in zip(evaluate(*arguments), want):
            absolute, relative = deviation(
                np.asarray(got).ravel(), np.asarray(reference).ravel()
            )
            worst_abs = max(worst_abs, absolute)
            worst_rel = max(worst_rel, relative)
    return {
        "pairs": target_r.size * len(sections),
        "worst_abs": worst_abs,
        "worst_rel": worst_rel,
        **device_report(),
    }


def measure_parity(cells: int, tile: int) -> dict:
    """Return the deviation between the two backends over every pair, and in zarr."""
    sections, target_r, target_z = hex_mesh(cells)
    plan = build_plan(tile)
    edge, weight, norm = pad_batch(sections)
    evaluate = tile_evaluator(plan, batched=True)
    worst_abs = worst_rel = 0.0
    for rows, columns in plan.tiles(target_r.size, len(sections)):
        arguments = (
            target_r[rows],
            target_z[rows],
            edge[:, :, columns],
            weight[:, columns],
            norm[columns],
        )
        for got, want in zip(evaluate(*arguments), tile_coupling(*arguments)):
            absolute, relative = deviation(got, want)
            worst_abs = max(worst_abs, absolute)
            worst_rel = max(worst_rel, relative)
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tiled-backend-"))
    try:
        import zarr

        stores = {}
        for backend in ("numpy", "jax"):
            path = directory / f"{backend}.zarr"
            assemble(path, target_r, target_z, sections, plan=plan, backend=backend)
            stores[backend] = zarr.open_group(str(path), mode="r")
        store_abs = store_rel = 0.0
        for name in COMPONENTS:
            want = np.asarray(stores["numpy"][name][:])
            absolute, relative = deviation(np.asarray(stores["jax"][name][:]), want)
            store_abs = max(store_abs, absolute)
            store_rel = max(store_rel, relative)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    return {
        "pairs": target_r.size * len(sections),
        "worst_abs": worst_abs,
        "worst_rel": worst_rel,
        "store_worst_abs": store_abs,
        "store_worst_rel": store_rel,
        **device_report(),
    }


def measure(variant: str, cells: int, tile: int) -> dict:
    """Return one measurement record for one variant.

    Matched most specific first: a kernel prefix alone would swallow the variants
    that share it, and a variant quietly measured as another one is worse than an
    unknown variant, which at least says so.
    """
    if variant.endswith("-positions"):
        return measure_positions(
            cells, tile, "closed" if variant.startswith("closed") else "quadrature"
        )
    if variant.startswith("numpy-"):
        return measure_numpy(int(variant.split("-")[1]), cells, tile)
    if variant in ("jax-scan", "jax-vmap"):
        return measure_traced(variant.endswith("vmap"), cells, tile)
    if variant == "closed-host":
        return measure_host_closed(cells, tile)
    if variant == "closed-numpy":
        return measure_packed_closed(cells, tile)
    if variant in ("closed-scan", "closed-vmap"):
        return measure_traced(variant.endswith("vmap"), cells, tile, kernel="closed")
    if variant == "closed-parity":
        return measure_closed_parity(cells, tile)
    if variant == "parity":
        return measure_parity(cells, tile)
    raise SystemExit(f"unknown variant {variant}")


def run_child(
    variant: str, device: str, cells: int, tile: int, cache: str = "off"
) -> dict:
    """Return the record from a fresh process, so nothing is warm on entry.

    ``cache`` is where the child may keep compiled executables. It defaults to
    OFF, because a compile figure read out of a shared cache would be a disk read
    wearing a compile's name; only the cache probe passes a directory.
    """
    environment = dict(
        os.environ,
        TMPDIR=os.environ.get("TMPDIR", "/tmp"),
        NOVA_COMPILATION_CACHE=cache,
    )
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)
    if device == "cpu":
        environment["JAX_PLATFORMS"] = "cpu"
    else:
        environment.pop("JAX_PLATFORMS", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--measure",
            variant,
            "--cells",
            str(cells),
            "--tile",
            str(tile),
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    if completed.returncode != 0:
        raise SystemExit(f"{variant} failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def cache_probe(variant: str, device: str, cells: int, tile: int) -> dict:
    """Return the compile of two fresh processes that share one on-disk cache.

    The first process finds the cache empty and compiles; the second must find
    its executable there. Both are the same build in the same shape, so the only
    difference between them is where the executable came from -- and the hit is
    counted from JAX's own event rather than read off the clock.
    """
    measured = variant.replace("-cache", "-vmap")
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tiled-cache-"))
    try:
        runs = [
            run_child(measured, device, cells, tile, cache=str(directory))
            for _ in range(2)
        ]
        stored = sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file()
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    cold, warm = runs
    return {
        "variant": variant,
        "measured": measured,
        "cold_compile_seconds": cold["compile_seconds"],
        "warm_compile_seconds": warm["compile_seconds"],
        "cold_hits": cold["cache_hits"],
        "warm_hits": warm["cache_hits"],
        "compile_seconds_saved": warm["compile_seconds_saved"],
        "cache_bytes": stored,
        "pairs": cold["pairs"],
        "runs": runs,
    }


def cache_table(rows: list[dict]) -> str:
    """Return the table the cross-process claim is read from."""
    header = (
        "| variant | cold compile s | warm compile s | saved s | hits | cache MB |\n"
        "|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['cold_compile_seconds']:.2f} "
            f"| {row['warm_compile_seconds']:.2f} "
            f"| {row['compile_seconds_saved']:.2f} "
            f"| {row['cold_hits']} then {row['warm_hits']} "
            f"| {row['cache_bytes'] / (1 << 20):.1f} |"
        )
    return "\n".join(lines)


def table(rows: list[dict]) -> str:
    """Return the markdown table the decision is read from."""
    header = (
        "| variant | seconds (median) | us/pair | compile s | cold s | tiles |\n"
        "|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in rows:
        if "seconds" not in row:  # a parity check reports deviations, not a rate
            continue
        traced = "compile_seconds" in row
        lines.append(
            f"| {row['variant']} | {row['seconds']:.2f} "
            f"| {row['us_per_pair']:.1f} "
            f"| {f'{row["compile_seconds"]:.2f}' if traced else '-'} "
            f"| {f'{row["cold_seconds"]:.2f}' if traced else '-'} "
            f"| {row['tiles']} |"
        )
    return "\n".join(lines)


def sweep(
    device: str, cells: int, tile: int, repeat: int, variants, parity: bool = True
) -> dict:
    """Return the median record of every variant, plus the parity check."""
    rows = []
    for variant in variants:
        if variant.endswith("-cache"):  # a probe over two processes, not a timing
            rows.append(cache_probe(variant, device, cells, tile))
            continue
        records = [run_child(variant, device, cells, tile) for _ in range(repeat)]
        # a parity variant reports deviations rather than a rate, so only the keys it
        # actually carries are aggregated -- and the deviations are a worst case over
        # every pair, so the worst run is the summary rather than the median
        median = {
            key: statistics.median([record[key] for record in records])
            for key in ("seconds", "us_per_pair", "compile_seconds", "cold_seconds")
            if key in records[0]
        }
        median.update(
            {
                key: max(record[key] for record in records)
                for key in (
                    "worst_abs",
                    "worst_rel",
                    "store_worst_abs",
                    "store_worst_rel",
                )
                if key in records[0]
            }
        )
        rows.append(
            {
                "variant": variant,
                **median,
                "tiles": records[0].get("tiles", 1),
                "pairs": records[0]["pairs"],
                "runs": records,
            }
        )
    summary = {
        "device": device,
        "cells": cells,
        "tile": tile,
        "repeat": repeat,
        "rows": rows,
    }
    if parity:
        summary["parity"] = run_child("parity", device, min(cells, 120), tile)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--cells", type=int, default=320)
    parser.add_argument("--tile", type=int, default=40)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--variants")
    parser.add_argument(
        "--parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cross-check the backends; skip it when only a compile is in question",
    )
    parser.add_argument("--measure", help="run one variant in this process")
    parser.add_argument("--json", type=pathlib.Path)
    arguments = parser.parse_args()
    if arguments.measure:
        store_every_compile()
        print(json.dumps(measure(arguments.measure, arguments.cells, arguments.tile)))
        raise SystemExit(0)
    names = (
        arguments.variants.split(",")
        if arguments.variants
        else (CPU_VARIANTS if arguments.device == "cpu" else GPU_VARIANTS)
    )
    summary = sweep(
        arguments.device,
        arguments.cells,
        arguments.tile,
        arguments.repeat,
        names,
        parity=arguments.parity,
    )
    if arguments.json:
        arguments.json.write_text(json.dumps(summary, indent=2))
    print(table(summary["rows"]))
    probes = [row for row in summary["rows"] if "cold_compile_seconds" in row]
    if probes:
        print()
        print(cache_table(probes))
    if "parity" in summary:
        print()
        print(json.dumps(summary["parity"], indent=2))
