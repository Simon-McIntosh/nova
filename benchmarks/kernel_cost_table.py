"""Turn collected kernel-cost records into the comparison table and figure.

    python benchmarks/kernel_cost_table.py <record-directory> [figure-directory]

Cost is the median across repeats of the same variant. Accuracy is per component
(psi, B_R, B_Z) against two references: the exact-everywhere polygon rule for
the reduced rules, and the closed-form rectangle kernel where the section is a
rectangle, which is an independent oracle rather than a self-comparison.

The rectangle comparison is taken outside the section's bounding box. Inside it,
and along the horizontal faces in particular, ``cylinder_greens`` is not usable
as an oracle: its corner antiderivative carries ``sign(z_corner - z_target)``
step terms with a dead-band, and on a face where that difference is zero the
returned flux jumps -- one benchmark target on the lower face returns a NEGATIVE
psi where every neighbouring target, and the polygon kernel, give a smooth
positive value. That is a defect in the rectangle kernel, not in what is being
measured against it.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

COMPONENTS = ("psi", "br", "bz")


def load(directory: pathlib.Path) -> dict[str, dict]:
    """Return {variant: record} with cost as the median over repeats."""
    grouped: dict[str, list[dict]] = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        grouped.setdefault(record["variant"], []).append(record)
    out = {}
    for variant, records in grouped.items():
        best = dict(records[0])
        best["us_per_pair"] = float(np.median([r["us_per_pair"] for r in records]))
        best["repeats"] = len(records)
        best["spread"] = float(
            np.ptp([r["us_per_pair"] for r in records])
            / max(best["us_per_pair"], 1e-30)
        )
        out[variant] = best
    return out


def error(record: dict, reference: dict, mask=None) -> dict[str, float]:
    """Return the max relative error per component, scaled by the reference peak."""
    out = {}
    for component in COMPONENTS:
        got = np.asarray(record[component])
        want = np.asarray(reference[component])
        if mask is not None:
            got, want = got[mask], want[mask]
        scale = np.max(np.abs(want))
        out[component] = float(np.max(np.abs(got - want)) / scale)
    return out


def rectangle_oracle(radii) -> tuple[dict, np.ndarray]:
    """Return the closed-form rectangle kernel and the targets it is valid on."""
    from nova.biot.greens import cylinder_greens

    from benchmarks.kernel_cost import R0, RECT, Z0, targets

    tr, tz = targets()
    psi, br, bz = cylinder_greens(tr, tz, R0, Z0, *RECT)
    outside = np.hypot(tr - R0, tz - Z0) > 1.05 * np.hypot(*RECT) / 2.0
    return {"psi": psi, "br": br, "bz": bz, "radii": np.asarray(radii)}, outside


def table(records: dict[str, dict]) -> str:
    """Return the cost/accuracy comparison as fixed-width text."""
    point = records["point"]["us_per_pair"]
    exact = records["polygon_hex_16x48"]
    oracle, outside = rectangle_oracle(exact["radii"])
    lines = [
        f"{'method':34s} {'us/pair':>10s} {'x point':>9s} "
        f"{'d psi':>10s} {'d Br':>10s} {'d Bz':>10s}  reference"
    ]
    rectangle_family = {
        "point",
        "cylinder_rect",
        "hybrid_rect",
        "polygon_rect_16x48",
        "polygon_rect_complex_step_16x48",
    }
    for variant, record in records.items():
        if variant in rectangle_family:
            reference, mask, label = oracle, outside, "rectangle oracle, outside"
        else:
            reference, mask, label = exact, None, "polygon 16x48"
        deviation = error(record, reference, mask)
        lines.append(
            f"{variant:34s} {record['us_per_pair']:10.3f} "
            f"{record['us_per_pair'] / point:9.1f} "
            f"{deviation['psi']:10.2e} {deviation['br']:10.2e} "
            f"{deviation['bz']:10.2e}  {label}"
        )
    return "\n".join(lines)


def figure(records: dict[str, dict], path: pathlib.Path) -> None:
    """Write the cost-versus-accuracy scatter and the error-versus-distance panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exact = records["polygon_hex_16x48"]
    radii = np.asarray(exact["radii"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # the reference rule and the complex-step formulation of it are the same
    # integral, so they sit at zero error and would only bend the curve back on
    # itself; they are annotated separately below.
    rules = {
        key: value
        for key, value in records.items()
        if key.startswith("polygon_hex_")
        and "x" in key.split("_")[-1]
        and "complex_step" not in key
        and key != "polygon_hex_16x48"
    }
    for component, marker in zip(COMPONENTS, "os^"):
        cost, worst = [], []
        for record in rules.values():
            cost.append(record["us_per_pair"])
            worst.append(max(error(record, exact)[component], 1e-16))
        order = np.argsort(cost)
        axes[0].plot(
            np.array(cost)[order],
            np.array(worst)[order],
            marker + "-",
            label=f"polygon rule, {component}",
        )
    for variant, colour in (
        ("point", "C3"),
        ("cylinder_rect", "C4"),
        ("hybrid_rect", "C5"),
    ):
        record = records[variant]
        oracle, outside = rectangle_oracle(radii)
        worst = max(error(record, oracle, outside).values())
        axes[0].plot(
            record["us_per_pair"], worst, "*", ms=14, color=colour, label=variant
        )
    for variant, label in (
        ("polygon_hex_16x48", "exact 16x48\n(closed-form gradient)"),
        ("polygon_hex_complex_step_16x48", "exact 16x48\n(complex step)"),
    ):
        cost = records[variant]["us_per_pair"]
        axes[0].axvline(cost, color="0.6", ls="--", lw=0.8)
        axes[0].text(cost, 3e-15, label, fontsize=6, rotation=90, va="bottom")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("cost [us per target-source pair]")
    axes[0].set_ylabel("max error, relative to peak")
    axes[0].set_title("cost versus accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    order = np.argsort(radii)
    for variant in (
        "polygon_hex_8x24",
        "polygon_hex_4x16",
        "polygon_hex_2x12",
        "polygon_hex_1x8",
    ):
        if variant not in records:
            continue
        got = np.asarray(records[variant]["bz"])
        want = np.asarray(exact["bz"])
        axes[1].plot(
            radii[order],
            np.abs(got - want)[order] / np.max(np.abs(want)),
            label=variant.replace("polygon_hex_", "Bz, "),
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("target distance [section radii]")
    axes[1].set_ylabel("Bz error, relative to peak")
    axes[1].set_title("reduced quadrature: error falls with distance")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    directory = pathlib.Path(sys.argv[1])
    records = load(directory)
    print(table(records))
    if len(sys.argv) > 2:
        out = pathlib.Path(sys.argv[2]) / "kernel_cost.png"
        figure(records, out)
        print(f"\nfigure: {out}")
