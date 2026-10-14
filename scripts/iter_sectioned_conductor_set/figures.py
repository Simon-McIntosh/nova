"""Draw ITER section geometry and coupled terminal-impedance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from nova.circuit.conductor import subdivide_passive_thickness
from nova.media.ink import poloidal_axes, trace_axes
from scripts.iter_sectioned_conductor_set.build import source_conductors


ACTIVE = "#3b6178"
PASSIVE = "#777777"
BARE = "#555555"


def geometry_figure(path: Path, max_layer_thickness: float) -> None:
    conductors, coil_names, owner, thickness, _resistance, _names = source_conductors()
    n_section = len(owner)
    passive = np.arange(n_section, n_section + len(thickness))
    refined = subdivide_passive_thickness(
        conductors, passive, thickness, max_layer_thickness
    )
    fig, ax = plt.subplots(figsize=(14, 17), dpi=100)
    poloidal_axes(ax)
    for stripe in refined.conductors.polygon_sections:
        ax.add_patch(
            Polygon(
                stripe.vertices,
                closed=True,
                fill=False,
                edgecolor=PASSIVE,
                linewidth=0.25,
            )
        )
    for index in range(n_section):
        r, z = refined.conductors.r[index], refined.conductors.z[index]
        width, height = refined.conductors.dr[index], refined.conductors.dz[index]
        ax.add_patch(
            Rectangle(
                (r - width / 2, z - height / 2),
                width,
                height,
                fill=False,
                edgecolor=ACTIVE,
                linewidth=0.65,
            )
        )
    for index, name in enumerate(coil_names):
        if name.startswith("CS") and name != "CS3U":
            continue
        rows = np.flatnonzero(owner == index)
        r = float(np.mean(conductors.r[rows]))
        z = float(np.mean(conductors.z[rows]))
        if name == "CS3U":
            ax.text(0.25, 8.5, "CS modules", fontsize=15, color=ACTIVE, ha="left")
        elif name.startswith("PF"):
            right = float(np.max(conductors.r[rows] + conductors.dr[rows] / 2))
            ax.text(right + 0.15, z, name, fontsize=15, color=ACTIVE, va="center")
        else:
            ax.text(r, z, name, fontsize=15, color=ACTIVE, ha="center", va="center")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-14, 17)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def impedance_figure(path: Path, receipt: dict) -> None:
    frequency = np.asarray(receipt["frequency_hz"])
    screened = np.asarray(receipt["screened_real_ohm"]) + 1j * np.asarray(
        receipt["screened_imag_ohm"]
    )
    bare = np.asarray(receipt.get("bare_real_ohm", 0.0)) + 1j * np.asarray(
        receipt["bare_imag_ohm"]
    )
    fig, axes = plt.subplots(7, 2, figsize=(14, 19), dpi=100)
    for index, (name, ax) in enumerate(
        zip(receipt["coil_names"], axes.flat, strict=True)
    ):
        trace_axes(ax)
        ax.plot(
            frequency, np.abs(bare[:, index]), color=BARE, linewidth=2.4, linestyle="--"
        )
        ax.plot(frequency, np.abs(screened[:, index]), color=ACTIVE, linewidth=3.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.text(0.04, 0.9, name, transform=ax.transAxes, fontsize=20, color=ACTIVE)
        if index % 2 == 0:
            ax.set_ylabel("|Z| [Ω]", fontsize=18)
        if index >= 12:
            ax.set_xlabel("frequency [Hz]", fontsize=18)
        ax.tick_params(labelsize=16)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    args.directory.mkdir(parents=True, exist_ok=True)
    geometry_figure(args.directory / "conductors.svg", 0.05)
    impedance_figure(args.directory / "terminal-impedance.svg", receipt)
    print(f"SUMMARY wrote two figures for {len(receipt['coil_names'])} coils")


if __name__ == "__main__":
    main()
