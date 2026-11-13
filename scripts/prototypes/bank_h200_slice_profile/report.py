"""Audit completed profile receipts and render their per-slice stage costs."""

import argparse
from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path
from statistics import mean


def stages(record):
    profile = record["programs"]["solve"]["profile"]
    totals = defaultdict(float)
    for name, row in profile["groups"].items():
        if name.startswith("carried read:"):
            label = "carried geometry"
        elif name.startswith("full read:"):
            label = "full geometry"
        elif name in ("assembly", "block GMRES and scaling"):
            label = name
        elif name in ("update residual", "read residual"):
            label = name
        elif name in ("read census", "read qualification"):
            label = "tracking and qualification"
        else:
            label = "other and fused"
        totals[label] += row["device_ms"]
    totals["host gaps and transfers"] = (
        profile["gap_ms"]
        - profile["overlap_ms"]
        + sum(v["device_ms"] for v in profile["copies"].values())
    )
    totals["predictor"] = record["programs"]["predictor"]["profile"]["profiled_ms"]
    expected = profile["profiled_ms"] + totals["predictor"]
    assert abs(sum(totals.values()) / expected - 1) <= 0.10
    return dict(totals)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    assert summary["complete"] and not summary["smoke"]
    current = defaultdict(list)
    historical = defaultdict(list)
    for arm in summary["arms"]:
        if arm["process"] == 0 or arm["exit_status"] == 77:
            continue
        record = json.loads(Path(arm["receipt"]).read_text())
        assert record["complete"] and arm["exit_status"] == 0
        assert record["profiler_control"]["h2d_bytes"] == 2097152
        assert record["profiler_control"]["kernels"] > 0
        if arm["read_only"]:
            historical[arm["label"]].append(record)
        else:
            current[arm["width"]].append(record)
            assert max(record["fresh_residual"]) <= 1e-10
            for name, limit in (("axis", 5e-4), ("saddle", 5e-4), ("boundary", 1e-4)):
                assert len(record["precision"][name]) == arm["width"]
                assert max(record["precision"][name]) <= limit
    audited = []
    for width, records in sorted(current.items()):
        assert len(records) == 2
        walls = [r["end_to_end_ms"] for r in records]
        assert abs(walls[0] - walls[1]) / min(walls) <= 0.05
        profiles = [stages(r) for r in records]
        names = sorted(set().union(*profiles))
        audited.append(
            dict(
                width=width,
                batch_ms=mean(walls),
                per_slice_ms=mean(walls) / width,
                stages_ms={
                    name: mean(p.get(name, 0) for p in profiles) for name in names
                },
                kernels=[
                    r["programs"]["solve"]["profile"]["kernel_launches"]
                    for r in records
                ],
                residual=max(max(r["fresh_residual"]) for r in records),
            )
        )
    assert set(historical) == {"historical", "main"}
    paired = {}
    for label, records in historical.items():
        assert len(records) == 2
        values = [r["programs"]["carried"]["median_ms"] for r in records]
        assert abs(values[0] - values[1]) / min(values) <= 0.05
        paired[label] = dict(
            wall_ms=mean(values),
            groups={
                name: mean(
                    r["programs"]["carried"]["profile"]["groups"]
                    .get(name, {})
                    .get("device_ms", 0)
                    for r in records
                )
                for name in set().union(
                    *(r["programs"]["carried"]["profile"]["groups"] for r in records)
                )
            },
        )
    (args.run / "audited.json").write_text(
        json.dumps(dict(widths=audited, carried=paired), indent=2)
    )
    headers = [
        "Width",
        "Batch wall [ms]",
        "Per slice [ms]",
        "Kernels",
        "Fresh residual",
    ]
    names = sorted(set().union(*(r["stages_ms"] for r in audited)))
    headers += [name + " [device ms/batch]" for name in names]
    rows = [
        [
            r["width"],
            f"{r['batch_ms']:.4f}",
            f"{r['per_slice_ms']:.4f}",
            str(r["kernels"]),
            f"{r['residual']:.4g}",
            *(f"{r['stages_ms'].get(name, 0):.4f}" for name in names),
        ]
        for r in audited
    ]

    def cells(tag, values):
        return (
            "<tr>"
            + "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in values)
            + "</tr>"
        )

    (args.run / "tables.html").write_text(
        "<table><thead>"
        + cells("th", headers)
        + "</thead><tbody>"
        + "".join(cells("td", r) for r in rows)
        + "</tbody></table>"
    )
    import matplotlib.pyplot as plt
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    plt.style.use("data-ink")
    rows_count = math.ceil((len(names) + 1) / 3)
    fig, axes = plt.subplots(
        rows_count, 3, figsize=(14, max(7.2, rows_count * 2.6)), dpi=100
    )
    widths = [r["width"] for r in audited]
    for ax in list(axes.flat)[len(names) + 1 :]:
        ax.set_axis_off()
    for ax, name in zip(
        list(axes.flat)[: len(names) + 1], ["whole slice", *names], strict=True
    ):
        values = [
            r["per_slice_ms"]
            if name == "whole slice"
            else r["stages_ms"][name] / r["width"]
            for r in audited
        ]
        ax.plot(widths, values, "o-", color="#1f5d92", linewidth=3.0, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, [str(w) for w in widths], fontsize=11)
        ax.tick_params(labelsize=11)
        ax.set_ylabel(name + " [ms/slice]", fontsize=13)
        ax.set_xlabel("batch width", fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.10, top=0.97, hspace=0.70, wspace=0.45
    )
    save_checked(fig, args.figure)
    import xml.etree.ElementTree as ET

    image = ET.parse(args.figure)
    image.getroot().set("height", f"{31.1 * fig.get_figheight() / 14:.3f}cm")
    image.write(args.figure, encoding="unicode", xml_declaration=True)
    fig.savefig(args.run / "per-slice-stages.png", dpi=100)
    plt.close(fig)
    print("SUMMARY", json.dumps(dict(widths=audited, carried=paired)), flush=True)


if __name__ == "__main__":
    main()
