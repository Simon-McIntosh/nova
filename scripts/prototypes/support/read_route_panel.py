"""Draw one solved diverted state read through the support route.

Line contours of the Bell field the read contracts and of the closed form on
shared levels, both null sets, the wall and the read's support boundary.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import nova

from nova.equilibrium import fixed_points as fixed
from nova.media.ink import DEFAULT_INK
from nova.media.poloidal import draw_nulls, draw_wall
from scripts.prototypes.support.read_route_rows import _case, _identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=2000)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"NOVA_MODULE={nova.__file__} DEVICES={jax.devices()}", flush=True)
    assert jax.config.jax_enable_x64 is True
    case, source, exact = _case("diverted", args.cells)
    row, (operands, _cache, support, booked, _psi, _step) = _identity(
        case, source, exact
    )
    jets = case.jets
    bell = fixed.field_from_jets(jets, operands.sample_index, operands.mesh)
    wall = np.asarray(
        __import__(
            "tests.equilibrium.test_topology_read", fromlist=["x"]
        )._analytic_inputs("diverted")[2]
    )
    low, high = wall.min(axis=0), wall.max(axis=0)
    radial = np.linspace(low[0], high[0], 220)
    vertical = np.linspace(low[1], high[1], 300)
    grid = np.stack(np.meshgrid(radial, vertical, indexing="xy"), axis=-1).reshape(
        -1, 2
    )
    points = jnp.asarray(grid)
    solved = np.asarray(jax.jit(jax.vmap(bell.value))(points))
    closed = np.asarray(jax.jit(jax.vmap(case.analytic.value))(points))
    shape = (len(vertical), len(radial))
    axis_flux = float(support.reading.axis_flux)
    levels = np.sort(
        np.concatenate(
            (axis_flux * np.linspace(0.1, 0.9, 5), [0.0], axis_flux * [-0.1, -0.2])
        )
    )
    figure, axes = plt.subplots(figsize=(14, 9), dpi=100)
    axes.contour(
        radial,
        vertical,
        closed.reshape(shape),
        levels=levels,
        colors="#555555",
        linewidths=2.4,
        linestyles="dashed",
    )
    axes.contour(
        radial,
        vertical,
        solved.reshape(shape),
        levels=levels,
        colors="#2c5a8c",
        linewidths=2.6,
    )
    centre = np.asarray(case.geometry.centre)
    membership = np.asarray(support.reading.membership)
    mesh = tri.Triangulation(centre[:, 0], centre[:, 1])
    axes.tricontour(
        mesh,
        np.nan_to_num(membership),
        levels=[0.5],
        colors="#2f7f5f",
        linewidths=3.0,
    )
    draw_wall(axes, wall[:, 0], wall[:, 1], style=DEFAULT_INK)
    analytic_style = DEFAULT_INK.variant(
        axis_color="#555555",
        xpoint_color="#555555",
        axis_markersize=14,
        xpoint_markersize=15,
    )
    solved_style = DEFAULT_INK.variant(
        axis_color="#2c5a8c",
        xpoint_color="#2c5a8c",
        axis_markersize=7,
        xpoint_markersize=8,
    )
    oracle_axis = np.asarray(case.roots[0])
    oracle_saddle = np.asarray(case.roots[1])
    draw_nulls(
        axes,
        magnetic_axis=oracle_axis,
        x_points=oracle_saddle[None],
        style=analytic_style,
    )
    valid = np.asarray(support.reading.x_point_valid)
    draw_nulls(
        axes,
        magnetic_axis=np.asarray(support.reading.axis),
        x_points=np.asarray(support.reading.x_points)[valid],
        style=solved_style,
    )
    axes.set_aspect("equal", adjustable="box")
    axes.set_axis_off()
    axes.margins(0.03)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="svg", facecolor="white")
    plt.close(figure)
    gradient = np.linalg.norm(np.asarray(support.stationary.jet.gradient), axis=-1)
    receipt = dict(
        row,
        figure=str(args.output),
        stationary_gradient_norm=gradient.tolist(),
        axis_position_error=float(
            np.linalg.norm(np.asarray(support.reading.axis) - oracle_axis)
        ),
        x_point_position_error=float(
            np.linalg.norm(np.asarray(support.stationary.position)[1] - oracle_saddle)
        ),
        pitch=float(case.pitch),
        levels=levels.tolist(),
    )
    args.data.mkdir(parents=True, exist_ok=True)
    (args.data / "panel.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("PANEL " + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
