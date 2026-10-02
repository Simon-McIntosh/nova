#!/usr/bin/env python3
"""Measure CUDA executable-graph lifetime across distinct JAX programs.

The parent process runs two fresh child processes so the command-buffer XLA
flag is fixed before JAX loads.  Each child compiles a ladder of matrix shapes,
executes every compiled program, drops half of the executable handles, clears
JAX's compilation caches, and executes one retained handle again.  XLA's
``gpu_command_buffer`` VLOG is the count instrument; child markers associate
each runtime count with the number of programs compiled.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


COUNT_PATTERNS = (
    re.compile(r"remaining alive executable graphs:\s*(\d+)"),
    re.compile(r"alive executable graphs:\s*(\d+)"),
    re.compile(r"total of\s+(\d+)\s+alive graphs in the process"),
)
EVENT_PREFIX = "CUDA_GRAPH_EVENT "


def _event(**values: object) -> None:
    print(EVENT_PREFIX + json.dumps(values, sort_keys=True), flush=True)


def _child(programs: int, start_size: int, size_step: int) -> int:
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"CUDA graph measurement requires gpu, got {jax.default_backend()}"
        )

    executables = []
    _event(kind="start", backend=jax.default_backend(), device=str(jax.devices()[0]))
    for index in range(programs):
        size = start_size + index * size_step

        def kernel(left, right):
            product = left @ right
            return jnp.tanh(product) + jnp.sin(product)

        left = jnp.full((size, size), 0.125, dtype=jnp.float32)
        right = jnp.eye(size, dtype=jnp.float32)
        executable = jax.jit(kernel).lower(left, right).compile()
        for _ in range(3):
            result = executable(left, right)
            result.block_until_ready()
        executables.append(executable)
        _event(kind="program", programs_compiled=index + 1, shape=[size, size])

    drop_count = programs // 2
    _event(
        kind="before_drop",
        programs_compiled=programs,
        executable_handles=len(executables),
    )
    del executables[:drop_count]
    jax.clear_caches()
    gc.collect()
    jax.effects_barrier()
    time.sleep(0.5)
    retained_size = start_size + (programs - 1) * size_step
    retained_left = jnp.full((retained_size, retained_size), 0.125, dtype=jnp.float32)
    retained_right = jnp.eye(retained_size, dtype=jnp.float32)
    executables[-1](retained_left, retained_right).block_until_ready()
    _event(
        kind="after_drop",
        programs_compiled=programs,
        executable_handles=len(executables),
        dropped_handles=drop_count,
    )
    return 0


def _count_from_line(line: str) -> int | None:
    for pattern in COUNT_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def _run_arm(
    script: Path,
    output_dir: Path,
    *,
    arm: str,
    programs: int,
    start_size: int,
    size_step: int,
) -> dict[str, object]:
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "0"
    env["TF_CPP_VMODULE"] = (
        "command_buffer_thunk=5,cuda_command_buffer=5,gpu_command_buffer=5"
    )
    if arm == "command_buffers_disabled":
        env["XLA_FLAGS"] = (
            "--xla_gpu_graph_min_graph_size=1 --xla_gpu_enable_command_buffer="
        )
    else:
        env["XLA_FLAGS"] = "--xla_gpu_graph_min_graph_size=1"

    command = [
        sys.executable,
        str(script),
        "--child",
        "--programs",
        str(programs),
        "--start-size",
        str(start_size),
        "--size-step",
        str(size_step),
    ]
    completed = subprocess.run(
        command,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path = output_dir / f"{arm}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")

    latest_count: int | None = None
    count_observations = 0
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        count = _count_from_line(line)
        if count is not None:
            latest_count = count
            count_observations += 1
        if not line.startswith(EVENT_PREFIX):
            continue
        event = json.loads(line.removeprefix(EVENT_PREFIX))
        if event["kind"] in {"program", "before_drop", "after_drop"}:
            event["alive_executable_graphs"] = latest_count
            rows.append(event)

    arm_receipt: dict[str, object] = {
        "arm": arm,
        "command": command,
        "xla_flags": env.get("XLA_FLAGS", "<default>"),
        "exit_status": completed.returncode,
        "log": str(log_path),
        "count_observations": count_observations,
        "rows": rows,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"{arm} exited {completed.returncode}; read {log_path}")
    if arm == "command_buffers_enabled" and count_observations == 0:
        raise RuntimeError(
            "enabled arm emitted no gpu_command_buffer count; "
            "the instrument did not observe its positive control"
        )
    if arm == "command_buffers_disabled" and count_observations != 0:
        raise RuntimeError(
            f"disabled arm emitted {count_observations} graph counts; "
            "command buffers were not disabled"
        )
    return arm_receipt


def _classify(
    enabled: dict[str, object], disabled: dict[str, object]
) -> dict[str, object]:
    program_rows = [row for row in enabled["rows"] if row["kind"] == "program"]
    after_drop = next(row for row in enabled["rows"] if row["kind"] == "after_drop")
    counts = [row["alive_executable_graphs"] for row in program_rows]
    if (
        any(count is None for count in counts)
        or after_drop["alive_executable_graphs"] is None
    ):
        raise RuntimeError(
            "enabled arm did not expose a count at every required marker"
        )
    initial = int(counts[0])
    peak = max(int(count) for count in counts)
    final = int(counts[-1])
    post_drop = int(after_drop["alive_executable_graphs"])
    growth = final - initial
    release = final - post_drop
    control_rows = [row for row in disabled["rows"] if row["kind"] == "program"]
    control_counts = [row["alive_executable_graphs"] or 0 for row in control_rows]
    if growth > 0 and release > 0:
        verdict = "per_compiled_executable_and_released_when_executables_are_dropped"
    elif growth > 0:
        verdict = "process_cumulative_across_dropped_executables"
    else:
        verdict = "no_growth_observed"
    return {
        "verdict": verdict,
        "initial_alive_executable_graphs": initial,
        "final_alive_executable_graphs": final,
        "peak_alive_executable_graphs": peak,
        "growth_across_program_ladder": growth,
        "alive_executable_graphs_after_drop": post_drop,
        "graphs_released_after_drop": release,
        "disabled_control_maximum": max(control_counts, default=0),
    }


def _plot(receipt: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    try:
        plt.style.use("data-ink")
    except OSError:
        plt.rcParams.update(
            {
                "axes.grid": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.labelsize": 22,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "lines.linewidth": 2.6,
            }
        )
    enabled, disabled = receipt["arms"]
    enabled_rows = [row for row in enabled["rows"] if row["kind"] == "program"]
    disabled_rows = [row for row in disabled["rows"] if row["kind"] == "program"]
    x_enabled = [row["programs_compiled"] for row in enabled_rows]
    y_enabled = [row["alive_executable_graphs"] for row in enabled_rows]
    x_disabled = [row["programs_compiled"] for row in disabled_rows]
    y_disabled = [row["alive_executable_graphs"] or 0 for row in disabled_rows]
    post_drop = next(row for row in enabled["rows"] if row["kind"] == "after_drop")

    figure, axis = plt.subplots(figsize=(14, 7), dpi=100)
    enabled_colour = "#31688e"
    neutral = "#777777"
    axis.plot(x_enabled, y_enabled, color=enabled_colour, linewidth=3.0)
    axis.plot(x_disabled, y_disabled, color=neutral, linewidth=2.4, linestyle="--")
    axis.scatter(
        [post_drop["programs_compiled"]],
        [post_drop["alive_executable_graphs"]],
        color=enabled_colour,
        s=110,
        zorder=3,
    )
    axis.annotate(
        f"after dropping {post_drop['dropped_handles']} handles",
        (post_drop["programs_compiled"], post_drop["alive_executable_graphs"]),
        xytext=(-16, -30),
        textcoords="offset points",
        ha="right",
        color=enabled_colour,
        fontsize=20,
    )
    axis.text(
        x_enabled[-1],
        y_enabled[-1],
        "  command buffers enabled",
        color=enabled_colour,
        fontsize=20,
        va="center",
    )
    axis.text(
        x_disabled[-1],
        y_disabled[-1],
        "  command buffers disabled",
        color=neutral,
        fontsize=20,
        va="center",
    )
    axis.set_xlabel("distinct programs compiled")
    axis.set_ylabel("alive executable CUDA graphs")
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    axis.tick_params(width=1.2)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--programs", type=int, default=20)
    parser.add_argument("--start-size", type=int, default=64)
    parser.add_argument("--size-step", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.child:
        return _child(arguments.programs, arguments.start_size, arguments.size_step)
    if arguments.output_dir is None:
        parser.error("--output-dir is required in parent mode")

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    receipt_path = output_dir / "receipt.json"
    receipt: dict[str, object] = {
        "schema_version": 1,
        "measurement": (
            "CUDA executable-graph lifetime across distinct compiled programs"
        ),
        "revision": subprocess.check_output(
            ["git", "-C", str(script.parents[4]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "instrument": str(script),
        "working_directory": str(Path.cwd().resolve()),
        "programs": arguments.programs,
        "arms": [],
    }
    for arm in ("command_buffers_enabled", "command_buffers_disabled"):
        arm_receipt = _run_arm(
            script,
            output_dir,
            arm=arm,
            programs=arguments.programs,
            start_size=arguments.start_size,
            size_step=arguments.size_step,
        )
        receipt["arms"].append(arm_receipt)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    receipt["summary"] = _classify(*receipt["arms"])
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    _plot(receipt, output_dir / "alive-graph-count.png")
    print(json.dumps(receipt["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
