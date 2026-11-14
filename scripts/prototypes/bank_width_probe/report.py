"""Validate width receipts and draw memory admission beside warm throughput."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess


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
    rows, records = {}, []
    for arm in summary["arms"]:
        record = json.loads(Path(arm["receipt"]).read_text())
        records.append(record)
        assert record["complete"] and record["revision"] == revision
        assert (
            record["module_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        assert record["inputs_sha256"] == summary["input_sha256"]
        assert (
            hashlib.sha256(Path(record["inputs_path"]).read_bytes()).hexdigest()
            == summary["input_sha256"]
        )
        assert record["device"] == "NVIDIA H200 NVL" and record["x64"]
        assert (
            Path(arm["log"]).read_text().splitlines()[-1]
            == f"EXIT={arm['exit_status']}"
        )
        width = arm["width"]
        if width not in rows:
            rows[width] = dict(
                width=width,
                admitted=arm["admitted"],
                working_bytes=arm["working_bytes"],
                limit_bytes=arm["device_limit_bytes"],
                process_walls_ms=[],
                compile_seconds=[],
            )
        row = rows[width]
        assert row["admitted"] == arm["admitted"]
        assert row["working_bytes"] == arm["working_bytes"]
        row["compile_seconds"].append(arm["compile_seconds"])
        if arm["admitted"]:
            assert max(record["fresh_residual"]) <= 1e-10
            for name, bound in (("axis", 5e-4), ("saddle", 5e-4), ("boundary", 1e-4)):
                assert len(record["precision"][name]) == width
                assert max(record["precision"][name]) <= bound
            if arm["process"] != 0:
                row["process_walls_ms"].append(arm["batch_ms"])
        else:
            assert arm["exit_status"] == 77 and record["memory_limited"]
            assert arm["working_bytes"] > 0.9 * arm["device_limit_bytes"]
    for row in rows.values():
        if row["admitted"]:
            walls = row["process_walls_ms"]
            assert len(walls) == 2
            row.update(
                batch_ms=mean(walls),
                per_slice_ms=mean(walls) / row["width"],
                slices_per_second=1000 * row["width"] / mean(walls),
                spread=abs(walls[0] - walls[1]) / min(walls),
            )
            assert row["spread"] <= 0.05
        else:
            row.update(
                batch_ms=None, per_slice_ms=None, slices_per_second=None, spread=None
            )
    assert {64, 96, 128, 160, 192, 256} <= rows.keys()
    largest, refused = summary["largest_admitted_width"], summary["first_refused_width"]
    assert rows[largest]["admitted"] and not rows[refused]["admitted"]
    assert refused == largest + 1 and not rows[256]["admitted"]
    accepted = [r for r in records if not r.get("memory_limited")]
    result = dict(
        revision=revision,
        largest_admitted_width=largest,
        first_refused_width=refused,
        widths=[rows[w] for w in sorted(rows)],
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
    admitted = [row for row in result["widths"] if row["admitted"]]
    denied = [row for row in result["widths"] if not row["admitted"]]
    blue, red, grey = "#1f5d92", "#b63636", "#929292"
    memory, wall = axes
    memory.plot(
        [r["width"] for r in result["widths"]],
        [r["working_bytes"] / 1e9 for r in result["widths"]],
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
    ceiling = rows[largest]["limit_bytes"] * 0.90 / 1e9
    memory.axhline(ceiling, color="#555555", linestyle=":", linewidth=1.2)
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
    wall.set(xlabel="Independent pulses", ylabel="Warm throughput cost [ms/slice]")
    wall.margins(x=0.1, y=0.3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.81, wspace=0.35)
    fig.legend(
        handles=[
            Line2D([], [], color=blue, marker="o", linewidth=3, label="admitted"),
            Line2D([], [], color=red, marker="x", linestyle="none", label="refused"),
            Line2D([], [], color=grey, linewidth=2.6, label="compiled bytes"),
            Line2D(
                [],
                [],
                color="#555555",
                linestyle=":",
                linewidth=1.2,
                label="90% admission ceiling",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    save_checked(fig, args.figure)
    fig.savefig(args.run / "width-boundary.png", dpi=100)
    plt.close(fig)
    headline = rows[largest]
    print(
        "SUMMARY",
        json.dumps(
            dict(
                largest_admitted_width=largest,
                first_refused_width=refused,
                **{
                    k: headline[k]
                    for k in (
                        "working_bytes",
                        "batch_ms",
                        "per_slice_ms",
                        "slices_per_second",
                        "spread",
                    )
                },
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
