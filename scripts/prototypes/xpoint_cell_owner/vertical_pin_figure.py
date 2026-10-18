"""Render the vertical-pin evidence figure and its measured histories.

The carrier is the elongated analytic equilibrium used by the pin test: the
current centroid ``(R, Z)`` on a map whose vertical row reads the elongation
force ``c * Z * (Z - Z_free)`` plus the external vertical-field source
``-beta * u``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from nova.equilibrium import solve  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402


def despine(axes):
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)


KAPPA = 1.8
Z_FREE = 0.5
R_TARGET = 0.0
COUPLING = 0.6
FIELD_LEVERAGE = 1.0
TARGETS = (0.7, 0.9)
SEED = jnp.asarray([1.2, 0.62])
CORRECTIONS = 12

FREE_COLOUR = "#222222"
TARGET_COLOURS = ("#1f6f6f", "#8a5a00")


def free_map(state, operands):
    del operands
    radius, height = state[0], state[1]
    return jnp.stack(
        (
            (radius - R_TARGET) + COUPLING * height * height,
            COUPLING * height * (height - Z_FREE),
        )
    )


def pinned_map(state, operands):
    del operands
    radius, height, unknown = state[0], state[1], state[2]
    return jnp.stack(
        (
            (radius - R_TARGET) + COUPLING * height * height,
            COUPLING * height * (height - Z_FREE) - FIELD_LEVERAGE * unknown,
        )
    )


def centroid_height(state, operands):
    del operands
    return state[1]


def free_history():
    heights = []
    for step in range(1, CORRECTIONS + 1):
        result = solve.fixed_point(free_map, SEED, jnp.ones(1), max_corrections=step)
        heights.append(float(result.state[1]))
    return heights


def pinned_history(target):
    heights = []
    for step in range(1, CORRECTIONS + 1):
        pin = solve.VerticalPin(
            target=jnp.asarray(float(target)),
            pin_scale=jnp.asarray(1.0),
            initial=jnp.asarray(0.0),
        )
        result = solve.pinned_fixed_point(
            pinned_map,
            centroid_height,
            SEED,
            jnp.ones(1),
            pin,
            max_corrections=step,
        )
        heights.append(float(centroid_height(result.state, jnp.ones(1))))
    return heights


def render(path, histories):
    plt.style.use("data-ink")
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=100)
    steps = np.arange(1, CORRECTIONS + 1)
    left.plot(
        steps, histories["free"], color=FREE_COLOUR, linewidth=2.4, label="unpinned"
    )
    for (label, values), colour in zip(histories["pinned"].items(), TARGET_COLOURS):
        left.plot(steps, values, color=colour, linewidth=3.0, label=label)
        left.axhline(
            float(label.split("=")[-1]), color=colour, linewidth=1.2, linestyle=":"
        )
    left.set_xlabel("Newton correction")
    left.set_ylabel("centroid height [m]")
    left.legend(frameon=False, fontsize=14)
    despine(left)

    grid_r = np.linspace(-0.4, 1.4, 160)
    grid_z = np.linspace(0.0, 1.2, 160)
    rr, zz = np.meshgrid(grid_r, grid_z)
    residual = COUPLING * zz * (zz - Z_FREE)
    levels = np.linspace(np.nanmin(residual), np.nanmax(residual), 12)
    right.contour(rr, zz, residual, levels=levels, colors=FREE_COLOUR, linewidths=1.0)
    right.axhline(Z_FREE, color=TARGET_COLOURS[0], linewidth=1.2, linestyle=":")
    right.set_xlabel("R [m]")
    right.set_ylabel("Z [m]")
    despine(right)
    figure.tight_layout()
    figure.savefig(path, transparent=True)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    histories = {
        "free": free_history(),
        "pinned": {f"target={t:.2f}": pinned_history(t) for t in TARGETS},
    }
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    render(args.figure, histories)
    args.receipt.write_text(json.dumps(histories, indent=2) + "\n")
    print(json.dumps({"figure": str(args.figure), "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
