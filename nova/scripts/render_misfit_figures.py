"""Render the vacuum-misfit separation figures from the banked evidence.

Reads only precomputed arrays under ``$HUNT`` and writes SVG into the document's
figure directory.  Visual constants follow the imas-ink ``InkStyle`` palette so
these figures sit beside the rest of the project's plots without restyling:
flux contours in ``#3366cc``, coil outlines in ``#888888``, wall/axes in black,
probe markers in ``#888888`` with a second colour ``#cc7722`` for the channels
under discussion, and the sienna reference family ``#b85c38`` for the negative
half of a signed field.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

HUNT = Path("/home/ITER/mcintos/.cache/nova-mast/misfit-hunt")
FIGDIR = Path(
    "/home/ITER/mcintos/.cache/reckon-worktrees/nova-a0f1e0938fc2/s5-followup/"
    "doc-nova/docs/figures/mast-misfit-hunt"
)

# imas-ink InkStyle palette
FLUX = "#3366cc"
SIENNA = "#b85c38"
RUST = "#a02c00"
RED = "#cc0000"
COIL_EDGE = "#888888"
PROBE = "#888888"
PROBE_MARK = "#cc7722"
WALL = "#000000"
GREY = "#999999"

# Two families under discussion: the gain-error probe and the P4/P5-local set.
GAIN_CHANNEL = "obr17"
LOCAL_CHANNELS = ("obv06", "obv13", "obv14")
WORST = ("obr17", "obv06", "obv14", "obr04", "obr10", "ccbv25")

# Signed-field map built from the ink palette so zero flux reads as white.
SIGNED = LinearSegmentedColormap.from_list("signed_flux", [SIENNA, "#ffffff", FLUX])
SEQUENTIAL = LinearSegmentedColormap.from_list("level", ["#ffffff", FLUX, "#12264f"])

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
    }
)


def load_cross_sections() -> dict:
    npz = np.load(HUNT / "figures" / "cross_sections.npz", allow_pickle=True)
    return {k: npz[k] for k in npz.files}


# Hand-placed label offsets (points) so no two coil names collide.
LABEL_OFFSET = {
    "sol": (8, 0, "left"),
    "p2_inner_lower": (-8, -7, "right"),
    "p2_outer_lower": (9, 5, "left"),
    "p2_inner_upper": (-8, -7, "right"),
    "p2_outer_upper": (9, 5, "left"),
    "p3_lower": (-9, -6, "right"),
    "p3_upper": (-9, 6, "right"),
    "p4_lower": (7, -8, "left"),
    "p4_upper": (7, 8, "left"),
    "p5_lower": (9, 0, "left"),
    "p5_upper": (9, 0, "left"),
    "p6_lower": (-9, 3, "right"),
    "p6_upper": (-9, -3, "right"),
}


def draw_conductors(ax, xs: dict, *, labels: bool = False) -> None:
    """Outline every active coil and every case plate."""
    for key in xs:
        if key.startswith("coil__"):
            poly = xs[key]
            ax.fill(
                poly[:, 0],
                poly[:, 1],
                facecolor="none",
                edgecolor=COIL_EDGE,
                linewidth=0.5,
                zorder=3,
            )
        elif key.startswith("plate__"):
            poly = xs[key]
            ax.fill(
                poly[:, 0],
                poly[:, 1],
                facecolor="none",
                edgecolor=GREY,
                linewidth=0.35,
                linestyle=(0, (2, 1.5)),
                zorder=3,
            )
    if labels:
        for key in xs:
            if not key.startswith("coil__"):
                continue
            family = key.split("__")[1]
            poly = xs[key]
            r = poly[:, 0].mean()
            z = poly[:, 1].mean()
            dx, dy, ha = LABEL_OFFSET.get(family, (6, 0, "left"))
            ax.annotate(
                family.replace("_", " "),
                (r, z),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=6.0,
                color="#333333",
                ha=ha,
                va="center",
                zorder=7,
            )


def draw_probes(ax, xs: dict, *, ticks: bool = True, highlight=WORST) -> None:
    """Plot every probe with its (undirected) sensitive axis as a centred tick."""
    r = xs["probe_r"]
    z = xs["probe_z"]
    channel = xs["probe_channel"]
    cos = xs["probe_radial_cosine"]
    sin = xs["probe_axial_sine"]
    marked = np.isin(channel, list(highlight))

    ax.scatter(
        r[~marked],
        z[~marked],
        s=2.5**2,
        c=PROBE,
        marker="o",
        linewidths=0,
        zorder=6,
    )
    ax.scatter(
        r[marked],
        z[marked],
        s=4.6**2,
        facecolor=PROBE_MARK,
        edgecolor="white",
        linewidths=0.5,
        marker="o",
        zorder=8,
    )
    if ticks:
        half = 0.06
        segs = [
            [
                (r[i] - half * cos[i], z[i] - half * sin[i]),
                (r[i] + half * cos[i], z[i] + half * sin[i]),
            ]
            for i in range(r.size)
        ]
        from matplotlib.collections import LineCollection

        ax.add_collection(
            LineCollection(
                [s for i, s in enumerate(segs) if not marked[i]],
                colors=PROBE,
                linewidths=0.55,
                zorder=6,
            )
        )
        ax.add_collection(
            LineCollection(
                [s for i, s in enumerate(segs) if marked[i]],
                colors=PROBE_MARK,
                linewidths=1.1,
                zorder=8,
            )
        )


def flux_panel(ax, npz, xs, *, title: str, probes: bool = True) -> None:
    """Draw one signed vacuum-flux map with the conductors over it."""
    r_axis = npz["r_grid"][:, 0]
    z_axis = npz["z_grid"][0, :]
    psi = npz["psi"].T  # (z, r) for contour
    scale = float(np.nanmax(np.abs(psi)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    filled = ax.contourf(
        r_axis,
        z_axis,
        psi,
        levels=np.linspace(-scale, scale, 33),
        cmap=SIGNED,
        norm=norm,
        zorder=1,
    )
    filled.set_rasterized(True)
    ax.contour(
        r_axis,
        z_axis,
        psi,
        levels=np.linspace(-scale, scale, 17),
        colors="#33333366",
        linewidths=0.3,
        zorder=2,
    )
    draw_conductors(ax, xs)
    if probes:
        draw_probes(ax, xs, ticks=False)
    ax.set_title(title, pad=3, fontsize=7.0, linespacing=1.35)
    ax.set_aspect("equal")
    ax.set_xlim(0.05, 2.05)
    ax.set_ylim(-2.25, 2.25)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-2, -1, 0, 1, 2])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def fig_cross_sections(xs: dict) -> None:
    """Full-machine conductor and probe layout, plus the outboard window."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.6), width_ratios=[1.0, 0.78])

    ax = axes[0]
    draw_conductors(ax, xs, labels=True)
    draw_probes(ax, xs)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 2.45)
    ax.set_ylim(-2.35, 2.35)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("all 13 active coils, 24 case plates, 78 probes", pad=4)
    ax.add_patch(
        plt.Rectangle(
            (1.4, -1.0),
            0.6,
            2.0,
            facecolor="none",
            edgecolor=RUST,
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            zorder=9,
        )
    )
    ax.annotate(
        "outboard\nwindow",
        (2.0, -1.0),
        textcoords="offset points",
        xytext=(4, -4),
        ha="left",
        va="top",
        fontsize=6.5,
        color=RUST,
        linespacing=1.3,
    )

    ax = axes[1]
    draw_conductors(ax, xs, labels=True)
    draw_probes(ax, xs)
    channel = xs["probe_channel"]
    for name in WORST:
        idx = np.where(channel == name)[0]
        if not idx.size:
            continue
        i = idx[0]
        ax.annotate(
            name,
            (xs["probe_r"][i], xs["probe_z"][i]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=7.0,
            fontweight="bold",
            color=RUST,
            zorder=10,
        )
    ax.set_aspect("equal")
    ax.set_xlim(1.4, 2.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("R [m]")
    ax.set_title("outboard window — where the worst channels sit", pad=4)
    for spine in ax.spines.values():
        spine.set_color(RUST)

    handles = [
        Line2D([], [], color=COIL_EDGE, lw=0.8, label="active coil outline"),
        Line2D(
            [], [], color=GREY, lw=0.7, linestyle=(0, (2, 1.5)), label="case plate"
        ),
        Line2D(
            [],
            [],
            color=PROBE,
            marker="o",
            lw=0.8,
            markersize=3,
            label="probe + sensitive axis",
        ),
        Line2D(
            [],
            [],
            color=PROBE_MARK,
            marker="o",
            lw=1.4,
            linestyle="-",
            markersize=4.5,
            markeredgecolor="white",
            label="channel carrying the misfit",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.savefig(FIGDIR / "conductor_probe_layout.svg")
    plt.close(fig)


def fig_static_flux(xs: dict, manifest: list) -> None:
    """Eleven sustained single-coil vacuum-flux maps."""
    cases = sorted(
        (r for r in manifest if r["kind"] == "sustained_single_coil"),
        key=lambda r: r["family"],
    )
    ncol = 4
    fig, axes = plt.subplots(3, ncol, figsize=(10.2, 11.4))
    flat = axes.ravel()
    for ax, case in zip(flat, cases, strict=False):
        npz = np.load(case["npz"])
        drive = case["drive_ampere_turns"][case["family"]]
        flux_panel(
            ax,
            npz,
            xs,
            title="%s\nshot %d,  t = %.3f s\n%+.0f kA-turns"
            % (
                case["family"].replace("_", " "),
                case["shot"],
                case["time_s"],
                drive / 1e3,
            ),
        )
    for ax in flat[len(cases) :]:
        ax.axis("off")
    for i, ax in enumerate(flat[: len(cases)]):
        if i % ncol == 0:
            ax.set_ylabel("Z [m]")
        else:
            ax.set_yticklabels([])
        if i >= len(cases) - ncol:
            ax.set_xlabel("R [m]")
        else:
            ax.set_xticklabels([])
    fig.subplots_adjust(hspace=0.28, wspace=0.12)

    key = flat[len(cases)]
    key.axis("off")
    key.text(
        0.02,
        0.92,
        "Poloidal flux ψ of the described coil set,\n"
        "driven by the recorded currents at that\n"
        "sample.  Each panel is scaled to its own\n"
        "peak, so the pattern is comparable but the\n"
        "amplitude is not.\n\n"
        "blue   ψ > 0\n"
        "sienna ψ < 0\n\n"
        "grey outlines   active coil turns\n"
        "dotted outlines coil case plates\n"
        "orange markers  the six channels that\n"
        "carry the misfit",
        transform=key.transAxes,
        fontsize=6.8,
        va="top",
        linespacing=1.6,
    )
    fig.suptitle(
        "Sustained single-coil vacuum flux — one identifiable drive per shot",
        fontsize=9.5,
        y=0.995,
    )
    fig.savefig(FIGDIR / "static_flux_gallery.svg", dpi=170)
    plt.close(fig)


def fig_pulsed_flux(xs: dict, manifest: list) -> None:
    """The two pulsed excitations as time sequences."""
    seq = [r for r in manifest if r["kind"] == "pulsed_excitation_sequence"]
    shots = sorted({r["shot"] for r in seq})
    fig, axes = plt.subplots(len(shots), 6, figsize=(11.6, 8.2))
    for row, shot in enumerate(shots):
        steps = sorted((r for r in seq if r["shot"] == shot), key=lambda r: r["time_s"])
        for col, case in enumerate(steps):
            ax = axes[row, col]
            npz = np.load(case["npz"])
            drive = case["drive_ampere_turns"][case["family"]]
            flux_panel(
                ax,
                npz,
                xs,
                title="t = %.4f s\n%+.0f kA-turns" % (case["time_s"], drive / 1e3),
                probes=False,
            )
            if col == 0:
                ax.set_ylabel(
                    "shot %d — %s\n\nZ [m]" % (shot, case["family"].replace("_", " "))
                )
            else:
                ax.set_yticklabels([])
            if row == len(shots) - 1:
                ax.set_xlabel("R [m]")
            else:
                ax.set_xticklabels([])
    fig.subplots_adjust(hspace=0.24, wspace=0.10)
    fig.suptitle(
        "Pulsed excitation — the rate-rich cases.  The pattern is the sustained one "
        "rescaled, which is why the residual tracks level and not rate.",
        fontsize=9.0,
        y=0.995,
    )
    fig.savefig(FIGDIR / "pulsed_flux_sequence.svg", dpi=170)
    plt.close(fig)


def fig_gate_flux(xs: dict, manifest: list) -> None:
    """The gate shot's plasma-free sample beside the coil Green's columns."""
    gate = next(r for r in manifest if r["kind"] == "gate_plasma_free")
    greens = np.load(HUNT / "figures" / "coil_flux_greens.npz")
    families = list(greens["family_names"])
    npz = np.load(gate["npz"])

    fig, axes = plt.subplots(1, 4, figsize=(11.4, 5.0))
    flux_panel(
        axes[0],
        npz,
        xs,
        title="gate shot %d\nplasma-free sample  t=%.4f s"
        % (gate["shot"], gate["time_s"]),
    )
    axes[0].set_ylabel("Z [m]")
    axes[0].set_xlabel("R [m]")

    r_axis = greens["r_grid"][:, 0]
    z_axis = greens["z_grid"][0, :]
    for ax, family in zip(axes[1:], ("p4_upper", "p5_upper", "p5_lower"), strict=True):
        g = greens["flux_greens"][families.index(family)].T
        scale = float(np.nanmax(np.abs(g)))
        filled = ax.contourf(
            r_axis,
            z_axis,
            g,
            levels=np.linspace(-scale, scale, 33),
            cmap=SIGNED,
            norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
            zorder=1,
        )
        filled.set_rasterized(True)
        ax.contour(
            r_axis,
            z_axis,
            g,
            levels=np.linspace(-scale, scale, 17),
            colors="#33333366",
            linewidths=0.3,
            zorder=2,
        )
        draw_conductors(ax, xs)
        draw_probes(ax, xs, ticks=False)
        ax.set_title("%s unit column\n[Wb per A-turn]" % family.replace("_", " "), pad=3)
        ax.set_aspect("equal")
        ax.set_xlim(0.05, 2.05)
        ax.set_ylim(-2.25, 2.25)
        ax.set_xlabel("R [m]")
        ax.set_yticklabels([])
    fig.suptitle(
        "The gate's plasma-free field is a superposition of the same unit columns "
        "the inversion solves in — no new field shape appears at the gate",
        fontsize=8.5,
        y=0.99,
    )
    fig.savefig(FIGDIR / "gate_and_unit_columns.svg", dpi=170)
    plt.close(fig)


def fig_separation() -> None:
    """Level R² against rate R² for every scored channel."""
    rows = json.loads((HUNT / "separation_verdicts.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), width_ratios=[1.0, 1.25])

    ax = axes[0]
    lvl = np.array([r["median_levels_r2"] for r in rows])
    rate = np.array([r["median_rates_r2"] for r in rows])
    names = [r["channel"] for r in rows]
    marked = np.array([n in WORST for n in names])
    ax.scatter(
        rate[~marked], lvl[~marked], s=14, c=PROBE, linewidths=0, alpha=0.75, zorder=3
    )
    ax.scatter(
        rate[marked],
        lvl[marked],
        s=34,
        facecolor=PROBE_MARK,
        edgecolor="white",
        linewidths=0.6,
        zorder=5,
    )
    nudge = {
        "obr17": (5, 2),
        "obr04": (5, 3),
        "obr10": (6, -1),
        "ccbv25": (6, 4),
        "obv06": (6, -6),
        "obv14": (-6, -7),
    }
    for i, n in enumerate(names):
        if marked[i]:
            dx, dy = nudge.get(n, (5, -1))
            ax.annotate(
                n,
                (rate[i], lvl[i]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="right" if dx < 0 else "left",
                fontsize=6.4,
                color=RUST,
                zorder=6,
            )
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xlabel("median R² on supply RATES  (dI/dt)")
    ax.set_ylabel("median R² on supply LEVELS  (I)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("every channel sits far above the diagonal", pad=4)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    inc = np.array([r["rate_increment"] for r in rows])
    ax.hist(100.0 * inc, bins=np.linspace(0, 6, 31), color=PROBE, zorder=3)
    top = ax.get_ylim()[1]
    ax.axvline(100.0 * np.median(inc), color=FLUX, lw=1.1, zorder=5)
    ax.annotate(
        "median %.2f%%" % (100.0 * np.median(inc)),
        (100.0 * np.median(inc), top * 0.94),
        textcoords="offset points",
        xytext=(6, 0),
        fontsize=6.8,
        color=FLUX,
    )
    # Channels whose rate increment coincides share one label.
    grouped: dict[float, list[str]] = {}
    for i, n in enumerate(names):
        if n in WORST:
            grouped.setdefault(round(100.0 * inc[i], 1), []).append(n)
    for position, group in grouped.items():
        ax.plot(
            [position],
            [top * 0.06],
            marker="v",
            markersize=4.0,
            color=PROBE_MARK,
            zorder=6,
        )
        ax.annotate(
            ", ".join(sorted(group)),
            (position, top * 0.09),
            fontsize=5.8,
            color=RUST,
            rotation=90,
            ha="center",
            va="bottom",
            zorder=6,
        )
    ax.set_xlim(-0.1, 2.1)
    ax.set_xlabel("R² the supply RATES add on top of the levels [percentage points]")
    ax.set_ylabel("channels (of %d)" % len(rows))
    ax.set_title("rates add almost nothing once levels are in", pad=4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Level-versus-rate separation — the residual is a static field error, "
        "not an inductive one",
        fontsize=9.0,
        y=1.0,
    )
    fig.savefig(FIGDIR / "level_versus_rate.svg")
    plt.close(fig)


def fig_gain_invariance() -> None:
    """obr17's weight error is amplitude-independent; its fitted scale is unique."""
    from collections import defaultdict

    attribution = json.loads((HUNT / "attribution.json").read_text())
    grouped: dict[tuple[str, int], list] = defaultdict(list)
    for row in attribution["rows"]:
        if row["channel"] in (GAIN_CHANNEL,) + LOCAL_CHANNELS:
            grouped[(row["channel"], row["shot"])].append(row)

    by_channel: dict[str, list] = defaultdict(list)
    for (channel, _shot), rows in grouped.items():
        lead = max(rows, key=lambda r: r["leverage"])
        by_channel[channel].append((lead["residual_rms"], lead["weight_error"], rows[0]["class"]))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), width_ratios=[1.15, 1.0, 0.9])

    ax = axes[0]
    classes = {
        "sustained_single_coil": (FLUX, "o", "sustained single coil"),
        "sustained_symmetric_pair": (SIENNA, "s", "sustained symmetric pair"),
        "pulsed_excitation": (GREY, "^", "pulsed excitation"),
        "gate": (RUST, "D", "gate shot"),
    }
    for klass, (color, marker, label) in classes.items():
        pts = [(a, w) for a, w, k in by_channel[GAIN_CHANNEL] if k == klass]
        if not pts:
            continue
        ax.scatter(
            [1e3 * p[0] for p in pts],
            [p[1] for p in pts],
            s=18,
            c=color,
            marker=marker,
            linewidths=0,
            alpha=0.85,
            label=label,
            zorder=4,
        )
    ax.axhline(-0.5, color=RED, lw=0.9, linestyle=(0, (4, 2)), zorder=3)
    ax.annotate(
        "−0.50",
        (0.055, -0.5),
        textcoords="offset points",
        xytext=(0, 4),
        fontsize=7,
        color=RED,
    )
    ax.set_xscale("log")
    ax.set_xlabel("residual amplitude on that shot [mT]")
    ax.set_ylabel("fitted error on the described field")
    ax.set_title(
        "%s: one gain across a 765× amplitude range" % GAIN_CHANNEL, pad=4
    )
    ax.set_ylim(-1.15, 0.5)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    verdicts = {
        r["channel"]: r
        for r in json.loads((HUNT / "attribution_verdicts.json").read_text())
    }
    per_family = verdicts[GAIN_CHANNEL]["per_family"]
    families = sorted(per_family, key=lambda f: per_family[f])
    ax.barh(
        np.arange(len(families)),
        [per_family[f] for f in families],
        height=0.75,
        color=FLUX,
        zorder=3,
    )
    ax.axvline(-0.5, color=RED, lw=0.9, linestyle=(0, (4, 2)), zorder=4)
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels([f.replace("_", " ") for f in families], fontsize=6.2)
    ax.set_xlabel("fitted error on that family's field column")
    ax.set_title(
        "the same −0.5 whichever coil supplies the field", pad=4
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    scales = json.loads((HUNT / "inversion.json").read_text())["calibration_scales"]
    values = np.array(list(scales.values()))
    ax.hist(values, bins=np.linspace(0.4, 1.6, 49), color=PROBE, zorder=3)
    ax.axvline(scales[GAIN_CHANNEL], color=RUST, lw=1.2, zorder=5)
    ax.annotate(
        "%s = %.4f" % (GAIN_CHANNEL, scales[GAIN_CHANNEL]),
        (scales[GAIN_CHANNEL], ax.get_ylim()[1] * 0.9),
        textcoords="offset points",
        xytext=(6, 0),
        fontsize=6.8,
        color=RUST,
    )
    ax.set_xlabel("independently fitted per-channel scale")
    ax.set_ylabel("channels (of %d)" % values.size)
    ax.set_title("the only channel fitted near one half", pad=4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "A probe gain, not a field error: three independent routes agree on one half",
        fontsize=9.0,
        y=1.02,
    )
    fig.savefig(FIGDIR / "probe_gain_invariance.svg")
    plt.close(fig)


def fig_orthogonal_residual() -> None:
    """The share of each channel's residual no described coil column can reach."""
    from collections import defaultdict

    attribution = json.loads((HUNT / "attribution.json").read_text())
    per_shot: dict[tuple[str, int], dict] = {}
    for row in attribution["rows"]:
        per_shot[(row["channel"], row["shot"])] = row
    by_channel: dict[str, list] = defaultdict(list)
    for (channel, _shot), row in per_shot.items():
        if np.isfinite(row["fit_r2"]):
            by_channel[channel].append((row["class"], 1.0 - row["fit_r2"]))

    def survivors(name: str) -> dict[str, list[float]]:
        record = json.loads((HUNT / name).read_text())
        out: dict[str, list[float]] = defaultdict(list)
        for shot in record["shots"]:
            for channel, value in shot["channel_residual_t"].items():
                out[channel].append(value)
        return out

    free = survivors("inversion.json")
    solved = survivors("inversion_cases.json")
    described = {
        r["channel"]: r["median_residual_rms"]
        for r in json.loads((HUNT / "separation_verdicts.json").read_text())
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), width_ratios=[1.15, 1.0])

    ax = axes[0]
    classes = [
        ("sustained_single_coil", "single coil"),
        ("sustained_symmetric_pair", "symmetric pair"),
        ("pulsed_excitation", "pulsed"),
        ("gate", "gate"),
    ]
    channels = [GAIN_CHANNEL, *LOCAL_CHANNELS, "obr04", "obr10", "ccbv25"]
    width = 0.2
    colors = [FLUX, SIENNA, GREY, RUST]
    for j, ((klass, label), color) in enumerate(zip(classes, colors, strict=True)):
        heights = [
            100.0 * np.median([v for k, v in by_channel[c] if k == klass] or [np.nan])
            for c in channels
        ]
        ax.bar(
            np.arange(len(channels)) + (j - 1.5) * width,
            heights,
            width=width,
            color=color,
            label=label,
            zorder=3,
        )
    ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(channels, fontsize=6.8)
    ax.set_ylabel("share of residual orthogonal to\nevery described field column [%]")
    ax.set_title(
        "what no combination of described coil fields can reach", pad=4
    )
    ax.legend(loc="upper right", ncol=2, title="drive class", title_fontsize=6.2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    x = np.arange(len(channels))
    width = 0.27
    ax.bar(
        x - width,
        [1e3 * described[c] for c in channels],
        width=width,
        color=GREY,
        label="described drive map",
        zorder=3,
    )
    ax.bar(
        x,
        [1e3 * np.median(free[c]) for c in channels],
        width=width,
        color=FLUX,
        label="all 13 coil currents solved freely",
        zorder=3,
    )
    ax.bar(
        x + width,
        [1e3 * np.median(solved[c]) for c in channels],
        width=width,
        color="white",
        edgecolor=SIENNA,
        linewidth=0.7,
        hatch="////",
        label="cases solved too (not identifiable)",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(channels, fontsize=6.8)
    ax.set_ylabel("median residual on that channel [mT]")
    ax.set_title("what a free current solve can and cannot remove", pad=4)
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "A field the description cannot produce — orthogonal to every column, "
        "and surviving a free inversion",
        fontsize=9.0,
        y=1.01,
    )
    fig.savefig(FIGDIR / "orthogonal_and_surviving.svg")
    plt.close(fig)


def fig_identifiability() -> None:
    """The singular spectrum's rank cliff and the inverted drive weights."""
    inversion = json.loads((HUNT / "inversion.json").read_text())
    taxonomy = json.loads((HUNT / "inversion_taxonomy.json").read_text())
    coils = inversion["identifiability"]["coils_only"]["singular_values"]
    both = inversion["identifiability"]["coils_and_cases"]["singular_values"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), width_ratios=[0.95, 1.2])

    ax = axes[0]
    rel_coils = np.asarray(coils) / coils[0]
    rel_both = np.asarray(both) / both[0]
    ax.semilogy(
        np.arange(1, rel_coils.size + 1),
        rel_coils,
        marker="o",
        markersize=3.4,
        color=FLUX,
        label="13 coils, cases subtracted",
        zorder=4,
    )
    ax.semilogy(
        np.arange(1, rel_both.size + 1),
        rel_both,
        marker="s",
        markersize=3.0,
        color=SIENNA,
        label="21 columns, cases solved",
        zorder=4,
    )
    ax.axvspan(10.5, 21.5, color="#00000010", zorder=1)
    ax.annotate(
        "12× cliff after mode 10\n— three coil directions\nthe array cannot see",
        (11.0, 2.0e-4),
        fontsize=6.5,
        color=RUST,
        linespacing=1.4,
    )
    ax.set_xlabel("singular mode")
    ax.set_ylabel("singular value relative to the first")
    ax.set_title("the probe array resolves 10 of 13 coil directions", pad=4)
    ax.set_xticks([1, 5, 10, 15, 21])
    ax.legend(loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    verdict_color = {
        "confirmed": FLUX,
        "magnitude contradicted": RUST,
        "degenerate": GREY,
    }
    order = sorted(taxonomy, key=lambda r: r["conductor"])
    y = np.arange(len(order))
    ratios = np.array([r["inverted_ratio"] / r["assumed_weight"] for r in order])
    spreads = np.array([min(r["relative_spread"], 3.0) for r in order])
    ax.barh(
        y,
        ratios,
        height=0.7,
        color=[verdict_color[r["verdict"]] for r in order],
        zorder=3,
    )
    ax.errorbar(
        ratios,
        y,
        xerr=spreads * ratios,
        fmt="none",
        ecolor="#333333",
        elinewidth=0.7,
        capsize=1.8,
        zorder=5,
    )
    ax.axvline(1.0, color=RED, lw=0.9, linestyle=(0, (4, 2)), zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(
        ["%s (%.4g)" % (r["conductor"].replace("_", " "), r["assumed_weight"]) for r in order],
        fontsize=6.2,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.05, 60)
    ax.set_xlabel("inverted weight ÷ published weight  (bars: relative spread, clipped at 3)")
    ax.set_title("what the field actually pins", pad=4)
    handles = [
        Line2D([], [], color=c, lw=5, label=v) for v, c in verdict_color.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Identifiability sets what the inversion may claim — P4/P5 are pinned, "
        "P2/P3/P6 and the solenoid are not",
        fontsize=9.0,
        y=1.01,
    )
    fig.savefig(FIGDIR / "inversion_identifiability.svg")
    plt.close(fig)


def fig_parity() -> None:
    """Mirror-pair parity, with the corrected reading of what it implies."""
    rows = json.loads((HUNT / "antisymmetry_supported.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), width_ratios=[1.3, 0.85])

    ax = axes[0]
    groups = {"obv": PROBE_MARK, "obr": FLUX, "ccbv": GREY}
    for prefix, color in groups.items():
        sel = [r for r in rows if r["upper"].startswith(prefix)]
        if not sel:
            continue
        ax.scatter(
            [1e3 * r["anti"] for r in sel],
            [1e3 * r["sym"] for r in sel],
            s=26,
            c=color,
            linewidths=0,
            alpha=0.9,
            label="%s pairs (n=%d)" % (prefix, len(sel)),
            zorder=4,
        )
    lo, hi = 3e-2, 6.0
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.7, linestyle=(0, (3, 2)), zorder=2)
    ax.annotate("equal parity", (hi, hi), xytext=(-4, -10), textcoords="offset points",
                fontsize=6.2, color="#555555", ha="right")
    for name in ("obv06", "obr03", "obr04"):
        r = next((x for x in rows if x["upper"] == name), None)
        if r is None:
            continue
        ax.annotate(
            "%s/%s" % (r["upper"], r["lower"]),
            (1e3 * r["anti"], 1e3 * r["sym"]),
            textcoords="offset points",
            xytext=(6, 2),
            fontsize=6.4,
            color=RUST,
            zorder=6,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("antisymmetric part of the residual [mT]")
    ax.set_ylabel("symmetric part of the residual [mT]")
    ax.set_title("mirror-pair parity of the residual", pad=4)
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.axis("off")
    z = np.linspace(-1, 1, 200)
    inset = ax.inset_axes([0.05, 0.50, 0.9, 0.46])
    inset.plot(z, np.exp(-((z - 0.0) ** 2) / 0.35) * 0 + np.tanh(2.4 * z), color=FLUX,
               label="$B_R$  odd in Z")
    inset.plot(z, 1.0 / np.cosh(2.0 * z), color=SIENNA, label="$B_Z$  even in Z")
    inset.axhline(0, color="#888888", lw=0.5)
    inset.axvline(0, color="#888888", lw=0.5)
    inset.set_xlabel("Z / Z$_{coil}$", fontsize=6.5)
    inset.set_ylabel("field component", fontsize=6.5)
    inset.set_title("an up-down SYMMETRIC current pair", fontsize=7.0, pad=3)
    inset.tick_params(labelsize=5.5)
    inset.legend(fontsize=6.0, loc="lower right")
    inset.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.0,
        0.38,
        "A symmetric current makes $A_\\varphi$ even in Z, so\n"
        "$B_R=-\\partial_Z A_\\varphi$ is ODD and $B_Z$ is EVEN.\n\n"
        "Radial pairs therefore read ANTIsymmetric and\n"
        "vertical pairs read SYMMETRIC for the SAME source.\n"
        "Both hold here (obr03/obr17 anti-led, obv06/obv14\n"
        "symmetric-led by 3.9×), so the source current is\n"
        "up-down symmetric — the opposite of the earlier\n"
        "reading, which took an antisymmetric radial residual\n"
        "as evidence for an antisymmetric (mis-wired) current.",
        transform=ax.transAxes,
        fontsize=6.8,
        va="top",
        linespacing=1.5,
    )

    fig.suptitle(
        "The parity argument, corrected — the residual source is an up-down "
        "symmetric current",
        fontsize=9.0,
        y=1.01,
    )
    fig.savefig(FIGDIR / "mirror_parity.svg")
    plt.close(fig)


def fig_pair_relations() -> None:
    """Counterexample shots against each published circuit relation."""
    rows = json.loads((HUNT / "pair_independence.json").read_text())
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(len(rows))
    med = np.array([r["median_peak_ratio"] for r in rows])
    lo = np.array([r["min_peak_ratio"] for r in rows])
    ax.hlines(y, lo, med, color=GREY, lw=1.0, zorder=2)
    ax.scatter(med, y, s=34, c=FLUX, linewidths=0, label="median shot", zorder=4)
    ax.scatter(
        lo,
        y,
        s=40,
        facecolor=RUST,
        edgecolor="white",
        linewidths=0.5,
        marker="D",
        label="worst counterexample",
        zorder=5,
    )
    for i, r in enumerate(rows):
        ax.annotate(
            "%d of %d shots below ½" % (r["n_below_half"], r["n_shots"]),
            (1.25, i),
            fontsize=6.4,
            va="center",
            color="#444444",
        )
    ax.axvline(1.0, color=RED, lw=0.9, linestyle=(0, (4, 2)), zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1.5e-4, 12.0)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1.0])
    ax.set_yticks(y)
    ax.set_yticklabels(
        ["%s\n(predicted ratio 1)" % r["pair"] for r in rows], fontsize=6.6
    )
    ax.set_xlabel("partner current ÷ driven current, at the driven peak")
    ax.set_title(
        "Every published circuit relation has counterexample shots — the members "
        "are independently drivable",
        fontsize=8.5,
        pad=6,
    )
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGDIR / "circuit_relation_counterexamples.svg")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    xs = load_cross_sections()
    manifest = json.loads((HUNT / "figures" / "flux_manifest.json").read_text())

    fig_cross_sections(xs)
    print("layout done", flush=True)
    fig_static_flux(xs, manifest)
    print("static flux done", flush=True)
    fig_pulsed_flux(xs, manifest)
    print("pulsed flux done", flush=True)
    fig_gate_flux(xs, manifest)
    print("gate done", flush=True)
    fig_separation()
    print("separation done", flush=True)
    fig_gain_invariance()
    print("gain done", flush=True)
    fig_orthogonal_residual()
    print("orthogonal done", flush=True)
    fig_identifiability()
    print("identifiability done", flush=True)
    fig_parity()
    print("parity done", flush=True)
    fig_pair_relations()
    print("pairs done", flush=True)


if __name__ == "__main__":
    main()
