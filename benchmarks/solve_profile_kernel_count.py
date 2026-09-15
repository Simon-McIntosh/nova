"""Per-solve kernel-count and program-size profile of a whole-cell solve.

One whole-cell (chord-clipped) forward solve of the weak-rotation reactor row is
measured per cell count.  The first solve warm-starts the JIT and the persistent
compilation cache; a second solve runs under ``jax.profiler.trace`` writing a
Perfetto trace.  From the trace the benchmark reports, per cell count:

* total device kernels launched and the number of distinct kernel programs,
* summed kernel wall and the serial host-to-device launch gap between kernels,
* the ten heaviest kernels by summed wall,
* the share of wall spent in loop bodies (kernels re-issued more than once).

Beside the traced solve, the compiled program reports its HLO instruction count
(``compiled.as_text()`` op count), the serialized executable size in bytes, the
generated-code size, and the compile time.  A lower-only arm with the Newton
budget doubled reports the same StableHLO instruction count, which separates an
unrolled Newton (count doubles) from a scanned one (count stays flat).  One SVG
of cumulative kernel wall against kernel index is written per cell count.

The row construction mirrors the certificate driver's ``_measure`` sequence so
the measured program is the production one and the persistent compilation cache
is hit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium import ForwardProfile
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery

from benchmarks import solovev_certificate as solovev

CASE_NAME_DEFAULT = "weak-rotation-reactor-static"

_OP_LINE = re.compile(r"^\s+%[A-Za-z0-9_.-]+\s*=", re.M)
_OP_INDENT = re.compile(r"^(?P<indent>\s+)%(?P<name>[A-Za-z0-9_.-]+)\s*=", re.M)
_LOOP_OP = re.compile(r"\b(while|for|scan)\(")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_hlo_ops(text: str) -> dict[str, int | float]:
    """Count HLO instructions and classify the loop-structure evidence.

    Op lines appear as ``  %name = ...`` at the owning computation's top level
    and at deeper indentation inside nested regions (while/for bodies).  The
    loop-body op share is the deeper-ops fraction; the loop op count records how
    many structured loops survive in the reported program.
    """
    indent_matches = list(_OP_INDENT.finditer(text))
    top_level = sum(1 for m in indent_matches if len(m.group("indent")) <= 2)
    nested = sum(1 for m in indent_matches if len(m.group("indent")) > 2)
    return {
        "op_count": len(indent_matches),
        "top_level_ops": top_level,
        "nested_region_ops": nested,
        "loop_body_op_share": nested / max(len(indent_matches), 1),
        "loop_instruction_count": len(_LOOP_OP.findall(text)),
    }


def _program_cache_entry(profile: ForwardProfile) -> tuple[tuple[object, ...], object]:
    """Return the production accelerated program and its cache key."""
    matches = [
        (key, program)
        for key, program in profile._accelerated_program_cache.items()
        if key[0] in {"newton_krylov", "reduced_newton", "picard", "anderson"}
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected one production accelerated program in the profile cache, "
            f"found {len(matches)}"
        )
    return matches[0]


def _latest_trace(trace_root: Path) -> Path:
    traces = sorted(
        trace_root.glob("plugins/profile/*/perfetto_trace.json.gz"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not traces:
        raise RuntimeError("JAX profiler did not write a Perfetto trace")
    return traces[-1]


def _trace_summary(trace_path: Path, prefix: str) -> dict[str, object]:
    """Parse one Perfetto trace into kernel, gap and cumulative measurements."""
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    events = payload.get("traceEvents", [])
    annotations = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("name") == prefix
    ]
    if len(annotations) != 1:
        raise RuntimeError(
            f"expected one {prefix!r} annotation, found {len(annotations)}"
        )
    annotation = annotations[0]
    start = float(annotation["ts"])
    stop = start + float(annotation.get("dur", 0.0))
    gpu_pids = {
        event.get("pid")
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and "GPU" in str(event.get("args", {}).get("name", ""))
    }
    gpu_events = [
        event
        for event in events
        if event.get("pid") in gpu_pids
        and event.get("ph") == "X"
        and start <= float(event.get("ts", -1.0)) <= stop
        and not str(event.get("name", "")).startswith(("Memcpy", "Memset"))
    ]
    if not gpu_events:
        raise RuntimeError("trace annotation contains no GPU compute events")

    grouped: dict[str, list[float]] = {}
    ordered: list[tuple[float, float]] = []
    for event in gpu_events:
        name = str(event.get("name", "unknown"))
        ts = float(event.get("ts", 0.0)) / 1.0e6
        dur = float(event.get("dur", 0.0)) / 1.0e6
        grouped.setdefault(name, []).append(dur)
        ordered.append((ts, dur))
    total = sum(sum(values) for values in grouped.values())

    rows = sorted(
        [
            {
                "kernel": name,
                "call_count": len(values),
                "summed_wall_s": sum(values),
                "median_call_us": 1.0e6 * float(np.median(values)),
                "share_of_summed_gpu_time": sum(values) / max(total, 1.0e-30),
            }
            for name, values in grouped.items()
        ],
        key=lambda row: row["summed_wall_s"],
        reverse=True,
    )

    ordered.sort(key=lambda item: item[0])
    serial_gap_s = 0.0
    prev_end: float | None = None
    for ts, dur in ordered:
        if prev_end is not None and ts > prev_end:
            serial_gap_s += ts - prev_end
        prev_end = ts + dur
    ordered_durations = [dur for _ts, dur in ordered]
    cumulative = np.cumsum(ordered_durations).tolist()

    annotation_wall_s = float(annotation.get("dur", 0.0)) / 1.0e6
    kernel_wall_s = sum(row["summed_wall_s"] for row in rows)
    repeated_wall_s = sum(row["summed_wall_s"] for row in rows if row["call_count"] > 1)
    return {
        "trace_path": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "trace_size_bytes": trace_path.stat().st_size,
        "annotation_wall_s": annotation_wall_s,
        "device_kernel_launches": len(gpu_events),
        "distinct_kernel_programs": len(rows),
        "summed_kernel_wall_s": kernel_wall_s,
        "launch_gap_serial_s": serial_gap_s,
        "launch_gap_share_of_annotation_wall": (
            serial_gap_s / max(annotation_wall_s, 1.0e-30)
        ),
        "annotation_wall_minus_kernel_wall_s": annotation_wall_s - kernel_wall_s,
        "kernel_wall_share_of_annotation_wall": (
            kernel_wall_s / max(annotation_wall_s, 1.0e-30)
        ),
        "loop_body_kernel_wall_share": repeated_wall_s / max(kernel_wall_s, 1.0e-30),
        "ranked_kernels": rows[:10],
        "cumulative_index": list(range(1, len(ordered) + 1)),
        "cumulative_wall_s": cumulative,
        "method": (
            "GPU complete events whose start timestamps fall within the "
            "device-synchronised TraceAnnotation; launch gap is the serial idle "
            "between consecutive events on the assumed single compute stream"
        ),
    }


def _write_svg(
    cells: int,
    trace: dict[str, object],
    path: Path,
) -> None:
    indices = np.asarray(trace["cumulative_index"], dtype=np.float64)
    wall = np.asarray(trace["cumulative_wall_s"], dtype=np.float64)
    step = max(1, len(indices) // 4000)
    figure, axis = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    axis.plot(
        indices[::step],
        wall[::step],
        drawstyle="steps-post",
        color="tab:blue",
        linewidth=0.9,
    )
    axis.axhline(
        trace["annotation_wall_s"],
        color="0.35",
        linewidth=0.8,
        linestyle="--",
        label="annotation wall",
    )
    axis.set_xlabel("kernel launch index")
    axis.set_ylabel("cumulative GPU kernel wall [s]")
    axis.set_title(
        f"whole-cell solve kernel time vs kernel index - {CASE_NAME_DEFAULT}, "
        f"{cells} cells"
    )
    axis.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _measure_program(
    profile: ForwardProfile,
    request,
    run_dir: Path,
    cells: int,
    *,
    traced: bool,
) -> dict[str, object]:
    initial_flux = request.seed_policy.resolve(profile, current=request.current)
    external = profile.operator.external(request.current, request.prescribed_current)

    program_key, program = _program_cache_entry(profile)
    route = program_key[0]
    lowered = program.lower(initial_flux, external)
    stablehlo_text = lowered.as_text(dialect="stablehlo")

    compile_started = perf_counter()
    compiled = lowered.compile()
    compile_seconds = perf_counter() - compile_started
    try:
        hlo_text = compiled.as_text()
        hlo_census = _count_hlo_ops(hlo_text)
        hlo_unavailable = None
    except jax.errors.JaxRuntimeError as error:
        # The backend-lowered HLO module at 1000 cells exceeds the 2 GiB
        # protobuf transport limit of as_text(); the StableHLO text lowered
        # above is smaller (pre-backend-lowering) and still serializes, so the
        # op census falls back to that source and the missing-module reason is
        # recorded on the receipt.
        hlo_text = None
        hlo_census = None
        hlo_unavailable = str(error)
    stablehlo_census = _count_hlo_ops(stablehlo_text)

    runner = compiled.runtime_executable()
    # Executable-size metrics are GPU-defined: the CPU backend cannot
    # serialize an AOT result, so cold CPU smokes record None rather than
    # failing the measurement.
    if any(device.platform == "gpu" for device in jax.devices()):
        try:
            serialized = runner.serialize()
        except jax.errors.JaxRuntimeError as error:
            # The 1000-cell executable's GpuExecutableProto is ~4 GiB, above
            # the 2 GiB protobuf transport limit of serialize(); the size
            # query below is a direct query and still works, so record the
            # serialization shortage instead of failing the measurement.
            serialized = None
            serialize_unavailable = str(error)
        else:
            serialize_unavailable = None
        try:
            generated_code_bytes = runner.size_of_generated_code_in_bytes()
        except AttributeError:
            generated_code_bytes = None
    else:
        serialized = None
        serialize_unavailable = None
        generated_code_bytes = None

    record: dict[str, object] = {
        "cells": cells,
        "route": route,
        "compile_seconds": compile_seconds,
        "hlo": hlo_census,
        "hlo_as_text_length": None if hlo_text is None else len(hlo_text),
        "hlo_sha256": (
            None
            if hlo_text is None
            else hashlib.sha256(hlo_text.encode("utf-8")).hexdigest()
        ),
        "hlo_unavailable": hlo_unavailable,
        "stablehlo": stablehlo_census,
        "serialized_executable_bytes": (
            None if serialized is None else len(serialized)
        ),
        "serialize_unavailable": serialize_unavailable,
        "generated_code_bytes": generated_code_bytes,
    }

    if traced:
        prefix = f"whole-cell-{cells}"
        # The profiler's CUDA capture cannot allocate fresh device buffers
        # inside the annotated region, so the compiled program is executed
        # once untraced first — allocating all of its internal buffers and
        # confirming the AOT artifact runs — and the captured region then
        # dispatches the same executable on the already-resident inputs. A
        # first failed capture is retried once into a fresh directory; a
        # second failure propagates.
        jax.block_until_ready(compiled(initial_flux, external))
        jax.block_until_ready(initial_flux)
        jax.block_until_ready(external)
        trace_root = run_dir / "traces" / f"cells-{cells}"
        trace_root.mkdir(parents=True, exist_ok=True)
        final_root: Path | None = None
        for attempt in (1, 2):
            attempt_root = trace_root / f"attempt-{attempt}"
            attempt_root.mkdir(parents=True, exist_ok=True)
            try:
                with jax.profiler.trace(str(attempt_root), create_perfetto_trace=True):
                    with jax.profiler.TraceAnnotation(prefix):
                        traced_flux = compiled(initial_flux, external)
                        jax.block_until_ready(traced_flux)
            except (jax.errors.JaxRuntimeError, RuntimeError, OSError) as error:
                if attempt == 2:
                    raise
                print(
                    f"TRACE_CAPTURE_FAILED cells={cells} attempt={attempt} "
                    f"error={type(error).__name__}",
                    flush=True,
                )
            else:
                final_root = attempt_root
                break
        if final_root is None:
            raise RuntimeError(f"profiler capture failed for cells={cells}")
        record["trace"] = _trace_summary(_latest_trace(final_root), prefix)

    return record


def _doubled_arm(profile: ForwardProfile, request) -> dict[str, object]:
    """Lower the production solve with the Newton budget doubled."""
    options = request.policy.kernel_options()
    doubled_options = dict(options, newton_steps=2 * int(options["newton_steps"]))
    initial_flux = request.seed_policy.resolve(profile, current=request.current)
    external = profile.operator.external(request.current, request.prescribed_current)
    doubled_program = profile._accelerated_history_program(
        request.route,
        requested_class=None,
        target_current=request.target_current,
        **doubled_options,
    )
    lowered = doubled_program.lower(initial_flux, external)
    try:
        census = _count_hlo_ops(lowered.as_text(dialect="stablehlo"))
        census_unavailable = None
    except jax.errors.JaxRuntimeError as error:
        # The doubled-Newton arm is a superset of the primary program, so at
        # 1000 cells its StableHLO text can itself exceed the 2 GiB protobuf
        # transport limit; record the shortage rather than failing the job.
        census = None
        census_unavailable = str(error)
    return {
        "doubled_newton_steps": doubled_options["newton_steps"],
        "base_newton_steps": options["newton_steps"],
        "doubled_stablehlo_op_count": (None if census is None else census["op_count"]),
        "doubled_stablehlo_unavailable": census_unavailable,
    }


def measure_case(
    case_name: str,
    cells: int,
    *,
    run_dir: Path,
    traced: bool,
) -> dict[str, object]:
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )

    carrier_case, source_case, exact = solovev._case(case_name)
    # The certificate fixture keys its machine cache on the negative
    # requested-cell count; the delivered mesh carries abs(count) nodes.
    requested_cells = -cells
    machine = solovev._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = solovev._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, fixture_exterior, fixture_cache = (
        oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty_operator, oracle_state
        )
    )
    operator = oracle_fixture.forward_operator(source_case, machine, fixture_exterior)
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(operator, mesh, newton_steps=recovery.NEWTON_STEPS)
    target_current, current_centroid, current_receipt = (
        solovev._closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
    )
    seed, _requested_class, _seed_receipt = solovev._production_seed(
        profile, case_name, target_current, current_centroid, current_receipt
    )
    request = solovev._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{case_name}:{requested_cells}",
    )

    warm_started = perf_counter()
    warm_receipt = profile.solve(request)
    jax.block_until_ready(warm_receipt.equilibrium.flux)
    warm_seconds = perf_counter() - warm_started

    row: dict[str, object] = {
        "case": case_name,
        "requested_cells": cells,
        "mesh_nodes": len(machine.node),
        "whole_cell_clip_mode": support_clip_mode(),
        "warm_solve_seconds": warm_seconds,
        "machine_cache": machine.cache,
        "fixture_exterior_cache": fixture_cache,
        "compilation_cache": {
            "directory": str(cache.directory),
            "version": cache.version_key,
        },
        **_measure_program(profile, request, run_dir, cells, traced=traced),
        **_doubled_arm(profile, request),
    }
    return row


def _op_count(row: dict[str, object]) -> int | None:
    """Prefer the compiled-HLO census, falling back to StableHLO at 1000 cells."""
    census = row["hlo"] if row["hlo"] is not None else row["stablehlo"]
    return None if census is None else int(census["op_count"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=CASE_NAME_DEFAULT)
    parser.add_argument("--cells", type=int, required=True, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()

    set_support_clip_mode("chord")
    run_dir = arguments.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for cells in arguments.cells:
        row = measure_case(
            arguments.case,
            cells,
            run_dir=run_dir,
            traced=not arguments.smoke,
        )
        part = output_dir / f"solve-profile-{cells}.json"
        part.write_text(json.dumps(row, indent=2, default=str) + "\n", encoding="utf-8")
        rows.append(row)
        if arguments.smoke:
            print(
                f"SMOKE_COMPLETE cells={cells} warm_s={row['warm_solve_seconds']:.3g} "
                f"compile_s={row['compile_seconds']:.3g} "
                f"hlo_ops={_op_count(row)} ",
                flush=True,
            )
        else:
            trace = row["trace"]
            svg = output_dir / f"solve-profile-{cells}.svg"
            _write_svg(cells, trace, svg)
            print(
                f"PART_COMPLETE cells={cells} "
                f"launches={trace['device_kernel_launches']} "
                f"kernels={trace['distinct_kernel_programs']} "
                f"gap_s={trace['launch_gap_serial_s']:.4g} "
                f"hlo_ops={_op_count(row)}",
                flush=True,
            )
            print(f"FIGURE_WRITTEN path={svg}", flush=True)

    aggregate = {"case": arguments.case, "cells": arguments.cells, "rows": rows}
    receipt = output_dir / "solve-profile.json"
    receipt.write_text(
        json.dumps(aggregate, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"RECEIPT_WRITTEN path={receipt}", flush=True)


if __name__ == "__main__":
    main()
