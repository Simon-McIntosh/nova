"""Draw the measured bank state beside its analytic poloidal flux."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nova.database import content_store
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_scattered_contours, draw_wall
from tests.equilibrium.test_topology_read import _analytic_inputs


STORE = Path("/work/projects/imas_gpu/sophelio/nova-assembled-step-bank")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((STORE / "receipt.json").read_text())
    root = content_store.verified_object_root(STORE, create=False)
    destination = content_store.verified_destination(
        root, content_store.digest_hex(receipt["digest"])
    )
    with np.load(destination / "bank.npz", allow_pickle=False) as bank:
        count = int(bank["active_nodes"])
        nodes = bank["nodes"][:count]
        rows = bank["rows"][:count, :, :, 0]
        exterior = bank["exterior"][:count, 0]
    with np.load(args.state, allow_pickle=False) as state:
        moments = state["moments"].T
        solved_axis = state["axis"]
        solved_boundary = state["boundary"]
        residual = float(state["residual_sup"])
        converged = bool(state["converged"])
    bank_flux = exterior + np.einsum("ncm,cm->n", rows, moments)
    _, analytic, wall, oracle_axis, oracle_saddle = _analytic_inputs("diverted")
    oracle_flux = np.asarray(jax.vmap(analytic.value)(jnp.asarray(nodes)))
    lower = max(np.min(bank_flux), np.min(oracle_flux))
    upper = min(np.max(bank_flux), np.max(oracle_flux))
    levels = np.linspace(lower, upper, 10)[1:-1]
    fig, axis = plt.subplots(figsize=(14, 7.2), dpi=100)
    axis.set_position((0.03, 0.02, 0.94, 0.88))
    poloidal_axes(axis, DEFAULT_INK)
    reference = draw_scattered_contours(
        axis,
        nodes[:, 0],
        nodes[:, 1],
        oracle_flux,
        levels,
        wall,
        color="#555555",
        linewidth=2.4,
        zorder=2,
    )
    reference.set_linestyles("dashed")
    draw_scattered_contours(
        axis,
        nodes[:, 0],
        nodes[:, 1],
        bank_flux,
        levels,
        wall,
        color=DEFAULT_INK.flux_color,
        linewidth=3.0,
        zorder=3,
    )
    draw_wall(axis, np.asarray(wall)[:, 0], np.asarray(wall)[:, 1])
    reference_style = DEFAULT_INK.variant(
        axis_color="#555555",
        xpoint_color="#555555",
        axis_markersize=13,
        xpoint_markersize=14,
    )
    solved_style = DEFAULT_INK.variant(
        axis_color=DEFAULT_INK.flux_color,
        xpoint_color=DEFAULT_INK.flux_color,
        axis_markersize=8.5,
        xpoint_markersize=9.5,
    )
    draw_nulls(
        axis,
        oracle_axis,
        np.asarray(oracle_saddle).reshape(1, 2),
        style=reference_style,
        contain=wall,
    )
    draw_nulls(
        axis,
        solved_axis,
        np.asarray(solved_boundary).reshape(1, 2),
        style=solved_style,
        contain=wall,
    )
    fig.legend(
        handles=(
            Line2D(
                [],
                [],
                color=DEFAULT_INK.flux_color,
                linewidth=3.0,
                marker="^",
                label="stalled bank state",
            ),
            Line2D(
                [],
                [],
                color="#555555",
                linewidth=2.4,
                linestyle="dashed",
                marker="^",
                label="analytic oracle",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg", metadata={"Date": None})
    fig.savefig(args.preview, format="png", dpi=100, metadata={"Software": None})
    plt.close(fig)
    source = args.output.read_text()
    source = source.replace(
        'width="1008pt" height="518.4pt"', 'width="31.1cm" height="16cm"', 1
    )
    args.output.write_text(source)
    print(
        f"PANEL converged={converged} residual_sup={residual:.9g} levels={len(levels)}"
    )
    print(
        f"RANGE solved=[{bank_flux.min():.9g},{bank_flux.max():.9g}] "
        f"oracle=[{oracle_flux.min():.9g},{oracle_flux.max():.9g}]"
    )


if __name__ == "__main__":
    main()
