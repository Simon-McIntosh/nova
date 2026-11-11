"""Draw the manifest-pinned WEST differential material law.

Run from the repository root with ``python -m scripts.magnetisation_law.figure``.
``--output`` selects the SVG and ``--png`` optionally writes an inspection PNG.
The shared WEST fixture honours ``NOVA_WEST_SOURCE_MANIFEST`` and supplies the
curation manifest's source and dictionary pin.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.imas.machine import StaticMachineDescription  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402
from nova.media.ink import DEFAULT_INK, trace_axes  # noqa: E402
from scripts.edge_sheet.figure import (  # noqa: E402
    HEIGHT_INCHES,
    WIDTH_INCHES,
    save_checked,
)
from tests.test_imas_iron_read import west  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "scripts/magnetisation_high_permeability/data-ink.mplstyle"
OUTPUT = ROOT / "docs/figures/iron-core-magnetisation/icm-law-fit/differential-law.svg"
PRODUCTION = "#7b4fb3"
MODEL = "#222222"
POPULATION = "#929292"


def render(output: Path, png: Path | None = None) -> None:
    """Render the production derivative, preserving its knot locations."""
    configure_dtypes()
    assert jax.config.jax_enable_x64
    uri, dd_version, _ = west.__wrapped__()
    machine = StaticMachineDescription.read_iron_core(uri, dd_version=dd_version)
    material = machine.iron_segments[0].material
    b, mu = map(np.asarray, (material.b_field, material.permeability_relative))
    knots, coefficients, ends = iron._fit_law(b, mu)
    law = tuple(jnp.asarray(x) for x in (knots, coefficients, ends))

    def h(field):
        return field * (1 / iron._MU0 - iron._coefficient(field, *law))

    def derivative(field):
        return np.asarray(jax.vmap(jax.grad(h))(jnp.asarray(field))) / 1000

    sample = np.unique(
        np.concatenate(
            [
                np.linspace(start, stop, 161)
                for start, stop in zip(knots[:-1], knots[1:], strict=True)
            ]
        )
    )
    inserted = np.asarray(sorted(set(knots) - set(b)))
    values = derivative(sample)
    assert values.min() > 0
    assert coefficients[2, 0] == 1 / (iron._MU0 * mu[0])
    print("SOURCE:", uri, "DD:", dd_version)
    print("MODULE:", iron.__file__)
    print("INSERTED_T:", inserted.tolist())
    print("SAMPLED_MIN_kA_per_m_T:", values.min())
    print("TAIL_kA_per_m_T:", float(ends[-1] / 1000))

    plt.style.use(STYLE)
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=100)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.14, top=0.84, wspace=0.27)
    for axis in axes:
        trace_axes(axis, DEFAULT_INK.variant(label_fontsize=14))
        for side in ("left", "bottom"):
            axis.spines[side].set_linewidth(1.2)
        axis.plot(
            sample,
            values,
            color=MODEL,
            linestyle="--",
            linewidth=3.0,
            zorder=2,
            label="fit",
        )
        axis.plot(
            [b[-1], 12],
            [ends[-1] / 1000] * 2,
            color=POPULATION,
            linewidth=2.4,
            zorder=2,
            label="held tail",
        )
        axis.plot(
            b,
            derivative(b),
            linestyle="none",
            marker="o",
            markersize=7,
            color=POPULATION,
            zorder=3,
            label="authored knots",
        )
        axis.plot(
            inserted,
            derivative(inserted),
            linestyle="none",
            marker="D",
            markersize=9,
            color=PRODUCTION,
            zorder=4,
            label="inserted breakpoints",
        )
        axis.set_xlabel("B [T]")
        axis.set_ylabel("dH/dB [kA/(m·T)]")
    axes[0].set_xlim(-0.15, 12.2)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.04, 2000)
    axes[0].set_xticks([0, 2, 4, 6, 8, 10, 12])
    axes[0].set_yticks(
        [0.1, 1, 10, 100, 1000], labels=["0.1", "1", "10", "100", "1000"]
    )
    axes[0].minorticks_off()
    axes[1].set_xlim(-0.015, 0.515)
    axes[1].set_ylim(0, 4.1)
    axes[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axes[1].set_yticks([0, 1, 2, 3, 4])
    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 2, 3, 1]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if png is not None:
        png.parent.mkdir(parents=True, exist_ok=True)
    names = save_checked(fig, str(output), str(png) if png else None)
    # Physical SVG dimensions retain the slide aspect without changing its viewBox.
    tree = ET.parse(output)
    root = tree.getroot()
    root.set("width", f"{16 * WIDTH_INCHES / HEIGHT_INCHES:.6f}cm")
    root.set("height", "16cm")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(
        f"TEXT_LAYOUT_PASS: {len(names)} text artists through {save_checked.__module__}"
    )
    print("SVG_SAVED:", output)
    if png:
        print("PNG_SAVED:", png)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--png", type=Path)
    arguments = parser.parse_args()
    render(arguments.output, arguments.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
