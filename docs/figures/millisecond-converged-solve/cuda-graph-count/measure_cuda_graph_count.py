#!/usr/bin/env python3
"""Measure CUDA executable-graph lifetime across distinct JAX programs.

The parent process runs two fresh child processes so the command-buffer XLA
flag is fixed before JAX loads. Each child compiles a ladder of matrix shapes,
executes every compiled program, then drops executable handles and clears JAX's
compilation caches. XLA's ``gpu_command_buffer`` VLOG is the count instrument.
The post-drop marker is emitted only after synchronization with no intervening
program execution, so its count is a settled re-read rather than execution churn.
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
    _event(
        kind="settled_reread",
        programs_compiled=programs,
        executable_handles=len(executables),
        dropped_handles=drop_count,
        programs_executed_after_drop=0,
    )
    return 0


def _count_from_line(line: str) -> int | None:
    for pattern in COUNT_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def _parse_arm_output(
    text: str,
    *,
    arm: str,
    command: list[str],
    xla_flags: str,
    log_path: Path,
    exit_status: int,
) -> dict[str, object]:
    latest_count: int | None = None
    last_count_after_drop: int | None = None
    drop_started = False
    count_events: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        count = _count_from_line(line)
        if count is not None:
            latest_count = count
            if drop_started:
                last_count_after_drop = count
            if "Destroying GPU command buffer executable graph" in line:
                action = "destroy"
            elif "Instantiated executable graph" in line:
                action = "instantiate"
            else:
                action = "update"
            count_events.append(
                {"log_line": line_number, "action": action, "count": count}
            )
        if not line.startswith(EVENT_PREFIX):
            continue
        event = json.loads(line.removeprefix(EVENT_PREFIX))
        if event["kind"] == "before_drop":
            drop_started = True
            last_count_after_drop = None
        if event["kind"] in {"program", "before_drop", "settled_reread"}:
            event["alive_executable_graphs"] = latest_count
            event["log_line"] = line_number
            if event["kind"] == "settled_reread":
                if last_count_after_drop is None:
                    raise RuntimeError(
                        "no XLA graph count was emitted between before_drop and "
                        "the settled reread marker"
                    )
                event["alive_executable_graphs"] = last_count_after_drop
                event["count_source"] = (
                    "last XLA count emitted after before_drop and before "
                    "settled_reread, with no program executed in that interval"
                )
            rows.append(event)

    arm_receipt: dict[str, object] = {
        "arm": arm,
        "command": command,
        "xla_flags": xla_flags,
        "exit_status": exit_status,
        "log": str(log_path.resolve()),
        "count_observations": len(count_events),
        "count_events": count_events,
        "rows": rows,
    }
    return arm_receipt


def _run_arm(
    script: Path,
    output_dir: Path,
    log_dir: Path,
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
    log_path = log_dir / f"{arm}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    arm_receipt = _parse_arm_output(
        completed.stdout,
        arm=arm,
        command=command,
        xla_flags=env.get("XLA_FLAGS", "<default>"),
        log_path=log_path,
        exit_status=completed.returncode,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{arm} exited {completed.returncode}; read {log_path}")
    if arm == "command_buffers_enabled" and arm_receipt["count_observations"] == 0:
        raise RuntimeError(
            "enabled arm emitted no gpu_command_buffer count; "
            "the instrument did not observe its positive control"
        )
    if arm == "command_buffers_disabled" and arm_receipt["count_observations"] != 0:
        raise RuntimeError(
            f"disabled arm emitted {arm_receipt['count_observations']} graph counts; "
            "command buffers were not disabled"
        )
    return arm_receipt


def _classify(
    enabled: dict[str, object], disabled: dict[str, object]
) -> dict[str, object]:
    program_rows = [row for row in enabled["rows"] if row["kind"] == "program"]
    settled_reread = next(
        row for row in enabled["rows"] if row["kind"] == "settled_reread"
    )
    counts = [row["alive_executable_graphs"] for row in program_rows]
    if (
        any(count is None for count in counts)
        or settled_reread["alive_executable_graphs"] is None
    ):
        raise RuntimeError(
            "enabled arm did not expose a count at every required marker"
        )
    initial = int(counts[0])
    peak = max(int(count) for count in counts)
    final = int(counts[-1])
    post_drop = int(settled_reread["alive_executable_graphs"])
    growth = final - initial
    release = final - post_drop
    control_rows = [row for row in disabled["rows"] if row["kind"] == "program"]
    control_counts = [row["alive_executable_graphs"] or 0 for row in control_rows]
    if growth == 0 and release > 0:
        verdict = "no_growth_with_demonstrated_release"
    elif growth == 0:
        verdict = "no_growth_without_demonstrated_release"
    elif release > 0:
        verdict = "growth_with_demonstrated_release"
    else:
        verdict = "growth_without_demonstrated_release"
    return {
        "verdict": verdict,
        "initial_alive_executable_graphs": initial,
        "final_alive_executable_graphs": final,
        "peak_alive_executable_graphs": peak,
        "transient_peak_alive_executable_graphs": max(
            event["count"] for event in enabled["count_events"]
        ),
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
    settled_reread = next(
        row for row in enabled["rows"] if row["kind"] == "settled_reread"
    )
    post_drop = {
        "programs_compiled": settled_reread["programs_compiled"],
        "dropped_handles": settled_reread["dropped_handles"],
        "alive_executable_graphs": receipt["summary"][
            "alive_executable_graphs_after_drop"
        ],
    }

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
        f"settled reread after dropping {post_drop['dropped_handles']} handles",
        (post_drop["programs_compiled"], post_drop["alive_executable_graphs"]),
        xytext=(-30, 35),
        textcoords="offset points",
        ha="right",
        va="bottom",
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
    axis.annotate(
        "  command buffers disabled",
        (x_disabled[-1], y_disabled[-1]),
        xytext=(8, 12),
        textcoords="offset points",
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
    parser.add_argument("--log-dir", type=Path)
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
    log_dir = (arguments.log_dir or output_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    receipt_path = output_dir / "receipt.json"
    current_revision = subprocess.check_output(
        ["git", "-C", str(script.parents[4]), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt: dict[str, object] = {
        "schema_version": 1,
        "measurement": (
            "CUDA executable-graph lifetime across distinct compiled programs"
        ),
        "revision": current_revision,
        "instrument": str(script),
        "working_directory": str(Path.cwd().resolve()),
        "programs": arguments.programs,
        "arms": [],
    }
    for arm in ("command_buffers_enabled", "command_buffers_disabled"):
        arm_receipt = _run_arm(
            script,
            output_dir,
            log_dir,
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
