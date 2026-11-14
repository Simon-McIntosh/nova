"""Audit default-allocator width limits and a separate async-allocator arm."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess

from scripts.prototypes.bank_width_probe.run import classify_outcome


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    assert summary["complete"] and summary["inert_control"].startswith("inert lever:")
    revision = summary["revision"]
    source = args.run / "measured-source/nova/equilibrium/support_read.py"
    assert source.read_bytes() == subprocess.check_output(
        [
            "git",
            "-C",
            str(args.root),
            "show",
            revision + ":nova/equilibrium/support_read.py",
        ]
    )
    rows, accepted = {}, []
    for arm in summary["arms"]:
        record = json.loads(Path(arm["receipt"]).read_text())
        log = Path(arm["log"]).read_text()
        assert record["revision"] == revision
        assert (
            record["module_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        assert record["module"] == summary["measurement_module"]
        assert record["cwd"] == summary["measurement_cwd"]
        assert record["inputs_sha256"] == summary["input_sha256"]
        assert (
            hashlib.sha256(Path(record["inputs_path"]).read_bytes()).hexdigest()
            == summary["input_sha256"]
        )
        assert record["device"] == "NVIDIA H200 NVL" and record["x64"]
        assert log.splitlines()[-1] == f"EXIT={arm['exit_status']}"
        assert classify_outcome(arm["exit_status"], record, log) == arm["outcome"]
        assert arm["admitted"] == (arm["outcome"] == "executed")
        assert arm["working_bytes"] == record["programs"]["solve"]["working_bytes"]
        assert arm["allocator_setting"] == (
            "unset" if arm["allocator"] == "default" else "cuda_malloc_async"
        )
        key = (arm["allocator"], arm["width"])
        if key not in rows:
            rows[key] = dict(
                allocator=arm["allocator"],
                width=arm["width"],
                admitted=arm["admitted"],
                outcome=arm["outcome"],
                working_bytes=arm["working_bytes"],
                limit_bytes=arm["device_limit_bytes"],
                process_walls_ms=[],
                exploratory_wall_ms=None,
                compile_seconds=[],
            )
        row = rows[key]
        assert row["outcome"] == arm["outcome"]
        assert row["working_bytes"] == arm["working_bytes"]
        assert row["limit_bytes"] == arm["device_limit_bytes"]
        row["compile_seconds"].append(arm["compile_seconds"])
        if arm["admitted"]:
            accepted.append(record)
            assert max(record["fresh_residual"]) <= 1e-10
            for name, bound in (("axis", 5e-4), ("saddle", 5e-4), ("boundary", 1e-4)):
                assert len(record["precision"][name]) == arm["width"]
                assert max(record["precision"][name]) <= bound
            if arm["process"] != 0:
                row["process_walls_ms"].append(arm["batch_ms"])
            else:
                row["exploratory_wall_ms"] = arm["batch_ms"]
        elif arm["outcome"] == "compiled-memory-refusal":
            assert arm["working_bytes"] > 0.9 * arm["device_limit_bytes"]
    largest, refused = summary["largest_admitted_width"], summary["first_refused_width"]
    for row in rows.values():
        walls = row["process_walls_ms"]
        repeated = row["allocator"] == "default" and row["width"] in {64, largest}
        assert len(walls) == (2 if repeated else 0)
        if row["admitted"]:
            values = walls or [row["exploratory_wall_ms"]]
            row.update(
                batch_ms=mean(values),
                per_slice_ms=mean(values) / row["width"],
                slices_per_second=1000 * row["width"] / mean(values),
                spread=abs(walls[0] - walls[1]) / min(walls) if walls else None,
            )
            assert row["spread"] is None or row["spread"] <= 0.05
        else:
            row.update(
                batch_ms=None, per_slice_ms=None, slices_per_second=None, spread=None
            )
    assert rows["default", largest]["admitted"]
    assert not rows["default", refused]["admitted"] and refused == largest + 1
    assert rows["default", 256]["outcome"] == "compiled-memory-refusal"
    assert rows["default", 96]["outcome"] == "runtime-oom"
    assert ("async", 96) in rows
    result = dict(
        revision=revision,
        largest_admitted_width=largest,
        first_refused_width=refused,
        widths=[rows[k] for k in sorted(rows)],
        max_residual=max(max(r["fresh_residual"]) for r in accepted),
        max_precision={
            k: max(max(r["precision"][k]) for r in accepted)
            for k in ("axis", "saddle", "boundary")
        },
    )
    (args.run / "audited.json").write_text(json.dumps(result, indent=2))
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100)
    defaults = [rows[k] for k in sorted(rows) if k[0] == "default"]
    admitted = [r for r in defaults if r["admitted"]]
    denied = [r for r in defaults if not r["admitted"]]
    asynchronous = rows["async", 96]
    blue, red, grey = "#1f5d92", "#b63636", "#929292"
    memory, wall = axes
    memory.plot(
        [r["width"] for r in defaults],
        [r["working_bytes"] / 1e9 for r in defaults],
        color=grey,
        linewidth=2.6,
        zorder=1,
    )
    for data, marker, colour in ((admitted, "o", blue), (denied, "x", red)):
        memory.scatter(
            [r["width"] for r in data],
            [r["working_bytes"] / 1e9 for r in data],
            marker=marker,
            color=colour,
            s=55,
            zorder=3,
        )
    memory.scatter(
        [96],
        [asynchronous["working_bytes"] / 1e9],
        marker="D",
        facecolors="none",
        edgecolors="#222222",
        s=130,
        linewidths=1.8,
        zorder=4,
    )
    limit = rows["default", largest]["limit_bytes"] / 1e9
    for value, label in ((limit, "allocator limit"), (0.9 * limit, "90% ceiling")):
        memory.axhline(value, color="#555555", linestyle=":", linewidth=1.2)
        memory.text(
            170,
            value + (3 if value == limit else -5),
            label,
            fontsize=12,
            color="#555555",
        )
    memory.set(
        xlabel="Independent pulses",
        ylabel="Compiled working memory [GB]",
        xlim=(56, 264),
    )
    memory.set_xticks([64, 96, 128, 160, 192, 224, 256])
    wall.plot(
        [r["width"] for r in admitted],
        [r["per_slice_ms"] for r in admitted],
        color=blue,
        marker="o",
        linewidth=3,
        zorder=3,
    )
    if asynchronous["admitted"]:
        wall.scatter(
            [96],
            [asynchronous["per_slice_ms"]],
            marker="D",
            facecolors="none",
            edgecolors="#222222",
            s=130,
            linewidths=1.8,
            zorder=4,
        )
    wall.set(xlabel="Independent pulses", ylabel="Warm throughput cost [ms/slice]")
    wall.margins(x=0.1, y=0.3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.81, wspace=0.35)
    fig.legend(
        handles=[
            Line2D(
                [], [], color=blue, marker="o", linewidth=3, label="default: executes"
            ),
            Line2D(
                [],
                [],
                color=red,
                marker="x",
                linestyle="none",
                label="default: refused",
            ),
            Line2D(
                [],
                [],
                color="#222222",
                marker="D",
                markerfacecolor="none",
                linestyle="none",
                label="async allocator: width 96",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    save_checked(fig, args.figure)
    fig.savefig(args.run / "width-boundary.png", dpi=100)
    plt.close(fig)
    print(
        "SUMMARY",
        json.dumps(
            dict(
                largest_admitted_width=largest,
                first_refused_width=refused,
                default=rows["default", largest],
                reference=rows["default", 64],
                asynchronous=asynchronous,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
