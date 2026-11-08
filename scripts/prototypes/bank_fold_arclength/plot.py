"""Render accepted branch points and a continued equilibrium with its oracle."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()
import argparse
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from nova.equilibrium import fixed_points, support_read
from nova.equilibrium.wall_mask import WallUnit
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall
from scripts.prototypes.bank_newton_step.measure import machine

BRANCH = "#2e6fa8"
REFERENCE = "#303030"


def checked_save(fig, path, preview):
    """Refuse canvas clipping and text collisions before saving either artifact."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        artist
        for artist in fig.findobj(matplotlib.text.Text)
        if artist.get_visible() and artist.get_text()
    ]
    boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in texts]
    for name, box in boxes:
        if not fig.bbox.contains(*box.get_points()[0]) or not fig.bbox.contains(
            *box.get_points()[1]
        ):
            raise ValueError(f"clipped text: {name}")
    for i, (name, box) in enumerate(boxes):
        for other_name, other_box in boxes[i + 1 :]:
            if box.overlaps(other_box):
                raise ValueError(f"text collision: {name} / {other_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    source = re.sub(
        r'width="[^"]+" height="[^"]+"',
        'width="31.1cm" height="16cm"',
        path.read_text(),
        count=1,
    )
    path.write_text(source)
    fig.savefig(preview, dpi=100)
    print(
        "FIGURE",
        path.name,
        "text_bounds=pass text_collisions=pass direct_labels=none",
        flush=True,
    )
    plt.close(fig)


def branch_figure(receipt, output, preview):
    rows = receipt["rows"]
    length = np.array([r["arclength"] for r in rows])
    drive = np.array([r["drive"] for r in rows])
    peak = np.argmax(drive)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2))
    for ax, quantity in zip(axes.flat, ("drive", "singular", "drive", "augmented")):
        ax.plot(
            length,
            [row[quantity] for row in rows],
            color=BRANCH,
            lw=3,
            marker="o",
            ms=4,
            zorder=3,
        )
        ax.axvline(length[peak], color="#777777", ls=":", lw=1.2, zorder=1)
        ax.set_xlabel("arclength [dimensionless]")
        ax.set_xlim(0, 0.07)
        ax.set_xticks([0, 0.02, 0.04, 0.06])
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        if quantity != "drive":
            ax.set_yscale("log")
        else:
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0, 0].set_ylabel("drive")
    axes[0, 0].set_ylim(1, 1.01)
    axes[0, 0].set_yticks([1, 1.004, 1.008])
    axes[0, 1].set_ylim(2e-5, 1e-3)
    axes[0, 1].set_yticks([3e-5, 1e-4, 3e-4], ["0.00003", "0.0001", "0.0003"])
    axes[1, 1].set_ylim(1.8e-4, 8e-4)
    axes[1, 1].set_yticks([2e-4, 4e-4, 7e-4], ["0.0002", "0.0004", "0.0007"])
    for ax in (axes[0, 1], axes[1, 1]):
        ax.yaxis.set_minor_locator(NullLocator())
    axes[1, 0].set_ylabel("drive, near turning point")
    axes[1, 0].set_xlim(0.0625, 0.068)
    axes[1, 0].set_ylim(1.00870, 1.00896)
    axes[1, 0].set_yticks([1.0087, 1.0088, 1.0089])
    axes[1, 0].set_xticks([0.063, 0.065, 0.067])
    axes[0, 1].set_ylabel("bank σmin")
    axes[1, 1].set_ylabel("augmented σmin")
    fig.legend(
        handles=[
            Line2D([], [], color=BRANCH, lw=3, marker="o", label="accepted branch"),
            Line2D(
                [], [], color="#777777", lw=1.2, ls=":", label="sampled drive maximum"
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(
        left=0.10, right=0.975, bottom=0.11, top=0.89, wspace=0.38, hspace=0.47
    )
    checked_save(fig, output, preview)


def scene_data(states_path, output):
    """Sample the bank's Bell field and analytic seed on one physical grid."""
    assert jax.config.jax_enable_x64
    op, cache, roots, convention, policy, profile, quadrature, _, analytic = machine(
        "diverted"
    )
    with np.load(states_path) as archive:
        final = archive["states"][-1, 0]
        scale = archive["scale"]
        seed = jnp.asarray(archive["seeds"][0])
    n = len(scale) // 3
    state = tuple(
        jnp.asarray(part) for part in np.split(final[:-1] * scale, (n, 2 * n))
    )
    read = jax.jit(
        lambda state: support_read.read(
            op, cache, state, seed, seed, convention, policy
        )
    )(state)
    assert bool(read.valid)
    from scripts.prototypes.bank_warm_start.streams import scaled_profile

    residual = jax.jit(
        lambda state: support_read.step_residual(
            op,
            state,
            state,
            read,
            scaled_profile(profile, jnp.asarray(final[-1])),
            quadrature,
        )
    )(state)
    norm = float(jnp.max(jnp.abs(jnp.concatenate(residual.residual))))
    assert norm <= 1e-10
    points = np.asarray(op.geometry.wall_points)
    radius = np.linspace(points[:, 0].min(), points[:, 0].max(), 220)
    height = np.linspace(points[:, 1].min(), points[:, 1].max(), 260)
    rr, zz = np.meshgrid(radius, height)
    samples = jnp.asarray(np.column_stack((rr.ravel(), zz.ravel())))
    jets = support_read.nodal_jets(op, state)
    field = fixed_points.field_from_jets(jets, op.sample_index, op.mesh)
    solved = jax.jit(jax.vmap(field.value))(samples)
    reference = jax.jit(jax.vmap(analytic.value))(samples)
    _, counts = np.unique(np.asarray(op.geometry.wall_unit), return_counts=True)
    reading = read.reading
    print(
        "SCENE",
        json.dumps(
            dict(
                drive=float(final[-1]),
                residual=norm,
                axis=np.asarray(reading.axis).tolist(),
                xpoint=np.asarray(reading.boundary).tolist(),
                reference_nulls=np.asarray(roots).tolist(),
                flux_span=float(reading.boundary_flux - reading.axis_flux),
            )
        ),
        flush=True,
    )
    zooms = {}
    for name, solved_null, reference_null in (
        ("axis", np.asarray(reading.axis), np.asarray(roots[0])),
        ("saddle", np.asarray(reading.boundary), np.asarray(roots[1])),
    ):
        rzoom = np.linspace(
            min(solved_null[0], reference_null[0]) - 0.025,
            max(solved_null[0], reference_null[0]) + 0.025,
            160,
        )
        zzoom = np.linspace(
            min(solved_null[1], reference_null[1]) - 0.018,
            max(solved_null[1], reference_null[1]) + 0.018,
            160,
        )
        rgrid, zgrid = np.meshgrid(rzoom, zzoom)
        local = jnp.asarray(np.column_stack((rgrid.ravel(), zgrid.ravel())))
        zooms[name + "_r"] = rzoom
        zooms[name + "_z"] = zzoom
        zooms[name + "_solved"] = np.asarray(
            jax.jit(jax.vmap(field.value))(local)
        ).reshape(rgrid.shape)
        zooms[name + "_reference"] = np.asarray(
            jax.jit(jax.vmap(analytic.value))(local)
        ).reshape(rgrid.shape)
    np.savez(
        output,
        **zooms,
        r=radius,
        z=height,
        solved=np.asarray(solved).reshape(rr.shape),
        reference=np.asarray(reference).reshape(rr.shape),
        wall=points,
        offsets=np.r_[0, np.cumsum(counts)],
        axis=np.asarray(reading.axis),
        saddle=np.asarray(reading.boundary),
        roots=np.asarray(roots),
        levels=np.linspace(float(reading.axis_flux), float(reading.boundary_flux), 13)[
            1:
        ],
    )


def poloidal_figure(scene, output, preview):
    with np.load(scene) as archive:
        data = dict(archive)
    points = data["wall"]
    offsets = data["offsets"]
    units = tuple(
        WallUnit(
            points[a:b, 0],
            points[a:b, 1],
            kind="vessel" if i == 0 else "material",
            closed=True,
        )
        for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:]))
    )
    fig = plt.figure(figsize=(14, 7.2))
    axes = [
        fig.add_axes(bounds)
        for bounds in (
            (0.025, 0.06, 0.44, 0.83),
            (0.51, 0.51, 0.46, 0.36),
            (0.51, 0.06, 0.46, 0.36),
        )
    ]
    styles = [
        (
            BRANCH,
            DEFAULT_INK.variant(
                axis_color=BRANCH,
                xpoint_color=BRANCH,
                axis_markersize=7.5,
                xpoint_markersize=8.5,
            ),
        ),
        (
            REFERENCE,
            DEFAULT_INK.variant(
                axis_color=REFERENCE,
                xpoint_color=REFERENCE,
                axis_markersize=6.5,
                xpoint_markersize=7.5,
            ),
        ),
    ]
    for panel, ax in enumerate(axes):
        poloidal_axes(ax)
        for key, (color, style) in zip(("reference", "solved"), styles[::-1]):
            prefix = "" if panel == 0 else ("axis_" if panel == 1 else "saddle_")
            contours = draw_flux_contours(
                ax,
                data[prefix + "r"],
                data[prefix + "z"],
                data[prefix + key],
                data["levels"],
                wall=units,
                color=color,
                linewidth=2.0,
            )
            if key == "reference":
                contours.set_linestyle("--")
                draw_nulls(
                    ax,
                    magnetic_axis=data["roots"][0],
                    x_points=data["roots"][1:],
                    style=style,
                )
            else:
                draw_nulls(
                    ax,
                    magnetic_axis=data["axis"],
                    x_points=data["saddle"][None, :],
                    style=style,
                )
        draw_wall(ax, units=units, color="#777777", linewidth=1.2)
    axes[1].set_xlim(
        min(data["axis"][0], data["roots"][0, 0]) - 0.025,
        max(data["axis"][0], data["roots"][0, 0]) + 0.025,
    )
    axes[1].set_ylim(
        min(data["axis"][1], data["roots"][0, 1]) - 0.018,
        max(data["axis"][1], data["roots"][0, 1]) + 0.018,
    )
    axes[2].set_xlim(
        min(data["saddle"][0], data["roots"][1, 0]) - 0.025,
        max(data["saddle"][0], data["roots"][1, 0]) + 0.025,
    )
    axes[2].set_ylim(
        min(data["saddle"][1], data["roots"][1, 1]) - 0.018,
        max(data["saddle"][1], data["roots"][1, 1]) + 0.018,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=BRANCH, lw=2, label="continued state"),
            Line2D([], [], color=REFERENCE, lw=2, ls="--", label="analytic seed"),
            Line2D([], [], color="#777777", lw=1.2, label="wall"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    checked_save(fig, output, preview)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plt.style.use("data-ink")
    plt.rcParams["svg.fonttype"] = "none"
    scene = args.run / "scene.npz"
    if not scene.exists():
        scene_data(args.run / "first.npz", scene)
    else:
        with np.load(scene) as archive:
            refined = "saddle_solved" in archive
        if not refined:
            scene_data(args.run / "first.npz", scene)
    receipt = json.loads((args.run / "first.json").read_text())
    branch_figure(receipt, args.output / "branch.svg", args.run / "branch.png")
    poloidal_figure(scene, args.output / "poloidal.svg", args.run / "poloidal.png")


if __name__ == "__main__":
    main()
