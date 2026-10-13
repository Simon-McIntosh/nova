"""Plot measured Newton residuals and a converged diverted flux comparison."""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes, trace_axes
from nova.media.poloidal import draw_nulls, draw_wall


def render(row, directory, destination):
    """Render one recorded diverted row using a common physical level array."""
    routes = row["routes"]
    if row["kind"] != "diverted" or not all(
        route in routes and routes[route].get("status") == "completed"
        for route in ("dense", "krylov")
    ):
        return None
    converged = [name for name in ("dense", "krylov") if routes[name]["qualified"]]
    if not converged:
        return None
    selected = min(
        converged, key=lambda name: routes[name]["execution"]["solve_seconds"]
    )
    colours = {"krylov": DEFAULT_INK.flux_color, "dense": "#222222"}
    styles = {"krylov": "-", "dense": "--"}
    stored = np.load(
        directory / f"{row['kind']}-{row['requested_cells']}-{selected}-analytic.npz"
    )
    grid, wall = stored["grid"], stored["wall"]
    count = len(grid)
    analytic, solved = stored["analytic"][:count], stored["flux"][:count]
    levels = np.linspace(float(np.min(analytic)), float(np.max(analytic)), 13)[1:-1]
    with plt.rc_context(
        {
            "font.size": 20,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.linewidth": 1.2,
            "axes.grid": False,
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=100)
        trace_axes(axes[0])
        poloidal_axes(axes[1])
        for name in ("krylov", "dense"):
            values = routes[name]
            history = np.asarray(values["history"])[: values["iterations"] + 1]
            axes[0].semilogy(
                np.arange(len(history)),
                history,
                color=colours[name],
                linestyle=styles[name],
                linewidth=3,
                marker="o",
                label=name,
            )
        axes[0].axhline(1e-12, color="#777777", linewidth=1.2, linestyle=":")
        axes[0].set_xlabel("Accepted Newton iteration")
        axes[0].set_ylabel("Scaled moment residual")
        axes[0].legend(frameon=False, fontsize=20)
        triangulation = tri.Triangulation(grid[:, 0], grid[:, 1])
        axes[1].tricontour(
            triangulation,
            analytic,
            levels=levels,
            colors="#777777",
            linewidths=2.4,
            linestyles="--",
        )
        axes[1].tricontour(
            triangulation,
            solved,
            levels=levels,
            colors=colours[selected],
            linewidths=3,
        )
        draw_wall(axes[1], units=wall, linewidth=1.2)
        reference_style = replace(
            DEFAULT_INK,
            axis_color="#777777",
            xpoint_color="#777777",
            axis_markersize=12,
            xpoint_markersize=13,
        )
        solved_style = replace(
            DEFAULT_INK,
            axis_color=colours[selected],
            xpoint_color=colours[selected],
            axis_markersize=8,
            xpoint_markersize=9,
        )
        reference_roots, solved_roots = stored["roots"], stored["solved_roots"]
        draw_nulls(
            axes[1], reference_roots[0], reference_roots[1:], style=reference_style
        )
        draw_nulls(axes[1], solved_roots[0], solved_roots[1:], style=solved_style)
        axes[1].text(
            0.02,
            0.97,
            selected,
            color=colours[selected],
            transform=axes[1].transAxes,
            va="top",
        )
        axes[1].text(
            0.02,
            0.90,
            "closed form",
            color="#777777",
            transform=axes[1].transAxes,
            va="top",
        )
        axes[1].set_xlim(np.min(wall[:, 0]), np.max(wall[:, 0]))
        axes[1].set_ylim(np.min(wall[:, 1]), np.max(wall[:, 1]))
        figure.tight_layout()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, format="svg")
        plt.close(figure)
    return dict(
        path=str(destination),
        selected=selected,
        levels=levels.tolist(),
        residual=routes[selected]["residual"],
        converged=True,
        flux_error=routes[selected]["flux_error"],
        reference_nulls=reference_roots.tolist(),
        solved_nulls=solved_roots.tolist(),
    )
