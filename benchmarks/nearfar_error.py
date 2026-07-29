"""Per-component error of the reduced polygon kernels against distance.

Two questions the exact-everywhere default cannot answer on its own, both of
which need the error resolved PER COMPONENT rather than on the flux alone --
flux is the least sensitive of the three, and a rule that holds psi to 1e-06 can
be losing two digits of B_Z at the same target:

*Quadrature order.* The phi integrand is analytic off the section boundary, so
convergence is spectral and the number of nodes a target needs falls with its
distance. This measures how far, so an adaptive-ORDER scheme can bin targets by
distance and keep the array shape fixed inside each bin.

*Near and far.* Replacing the finite-area kernel with a point filament beyond
some standoff is only defensible with an error bound. This measures the
finite-area correction itself -- exact polygon against point filament, per
component -- and reports the distance at which each component's correction
falls below a given tolerance. Those distances are evidence for a cutoff
decision, not a cutoff.

The bare filament is measured alongside a SECOND-MOMENT CORRECTED filament,
because the bare one turns out not to converge to the section at all. Spreading
current over a section of circumradius ``a`` shifts the coupling by the second
moment of the section weighted by the curvature of the ring Green's function,
and for a full toroidal ring that curvature is set by the MAJOR radius, not by
the distance to the target. So the relative correction does not fall off as
``(a/d)^2``; it flattens onto a floor of order ``(a/R0)^2`` that no standoff can
get under. Adding the second-moment term -- three extra Green's function
evaluations, still point-filament cost -- removes that floor, and only then does
an error-bounded cutoff exist at a useful tolerance.

    python benchmarks/nearfar_error.py <output.json>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from nova.biot.greens import greens_bz_br, greens_psi
from nova.biot.polygon import polygon_greens

R0, Z0 = 6.2, 0.0
CELL_RADIUS = 0.06
COMPONENTS = ("psi", "br", "bz")
RULES = [(16, 48), (8, 24), (8, 16), (4, 16), (4, 8), (2, 12), (2, 8), (1, 8), (1, 4)]
TOLERANCES = (1e-6, 1e-8, 1e-10)


def hexagon(radius=CELL_RADIUS):
    """Return the plasma cell section, a regular hexagon of circumradius ``radius``."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([R0 + radius * np.cos(angle), Z0 + radius * np.sin(angle)])


def ray_targets(radii, count=24):
    """Return targets on rings at each offset in section radii, all directions.

    Sampling every direction rather than one ray matters: the correction is not
    isotropic, and a cutoff has to hold in the worst direction, which for a
    hexagon is towards a vertex rather than a face.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    offset = np.asarray(radii)[:, None] * CELL_RADIUS
    return (
        (R0 + offset * np.cos(angle)).ravel(),
        (Z0 + offset * np.sin(angle)).ravel(),
        np.repeat(np.asarray(radii), count),
    )


def section_moments(vertices: np.ndarray) -> tuple[float, float, float]:
    """Return the area-normalised second moments ``(Irr, Izz, Irz)`` about the centroid.

    Exact polygon formulae (the shoelace second moments), not a sampled
    approximation: the correction term they feed is itself only ~1e-04, so a
    percent-level moment would swamp the thing being measured.
    """
    v = np.asarray(vertices, dtype=np.float64) - np.asarray(vertices).mean(axis=0)
    r, z = v[:, 0], v[:, 1]
    r_next, z_next = np.roll(r, -1), np.roll(z, -1)
    cross = r * z_next - r_next * z
    area = 0.5 * cross.sum()
    irr = np.sum((r**2 + r * r_next + r_next**2) * cross) / 12.0 / area
    izz = np.sum((z**2 + z * z_next + z_next**2) * cross) / 12.0 / area
    irz = (
        np.sum((r * z_next + 2.0 * r * z + 2.0 * r_next * z_next + r_next * z) * cross)
        / 24.0
        / area
    )
    return float(irr), float(izz), float(irz)


def corrected_filament(target_r, target_z, vertices, step=1e-4):
    """Return the point filament plus the section's second-moment term.

    The correction is the section second moment contracted with the curvature of
    the Green's function in the SOURCE position, taken by central differences on
    a step far larger than round-off and far smaller than the major radius.
    """
    centre = np.asarray(vertices, dtype=np.float64).mean(axis=0)
    irr, izz, irz = section_moments(vertices)

    def at(dr, dz):
        psi = greens_psi(target_r, target_z, centre[0] + dr, centre[1] + dz)
        bz, br = greens_bz_br(target_r, target_z, centre[0] + dr, centre[1] + dz)
        return np.array([psi, br, bz])

    value = at(0.0, 0.0)
    curvature_r = (at(step, 0.0) - 2.0 * value + at(-step, 0.0)) / step**2
    curvature_z = (at(0.0, step) - 2.0 * value + at(0.0, -step)) / step**2
    cross = (at(step, step) - at(step, -step) - at(-step, step) + at(-step, -step)) / (
        4.0 * step**2
    )
    corrected = value + 0.5 * (
        irr * curvature_r + izz * curvature_z + 2.0 * irz * cross
    )
    return dict(zip(COMPONENTS, corrected))


def worst_per_ring(error, radii, ring):
    """Return the worst error over directions on each ring."""
    return np.array([np.max(error[radii == value]) for value in ring])


def measure() -> dict:
    """Return the quadrature-order and finite-area-correction error tables."""
    section = hexagon()
    ring = np.geomspace(1.05, 60.0, 40)
    target_r, target_z, radii = ray_targets(ring)

    exact = dict(zip(COMPONENTS, polygon_greens(target_r, target_z, section, block=32)))
    scale = {name: float(np.max(np.abs(exact[name]))) for name in COMPONENTS}

    quadrature = {}
    for panels, nodes in RULES:
        rule = dict(
            zip(
                COMPONENTS,
                polygon_greens(
                    target_r,
                    target_z,
                    section,
                    n_panels=panels,
                    n_nodes=nodes,
                    block=32,
                ),
            )
        )
        quadrature[f"{panels}x{nodes}"] = {
            "nodes": panels * nodes,
            **{
                name: worst_per_ring(
                    np.abs(rule[name] - exact[name]) / scale[name], radii, ring
                ).tolist()
                for name in COMPONENTS
            },
        }

    psi_point = greens_psi(target_r, target_z, R0, Z0)
    bz_point, br_point = greens_bz_br(target_r, target_z, R0, Z0)
    point = {"psi": psi_point, "br": br_point, "bz": bz_point}
    # Normalise each component against a local magnitude that does not vanish:
    # flux against its own value (single-signed everywhere outside the section),
    # the field components against the poloidal field magnitude. Dividing B_R or
    # B_Z by itself would blow up wherever that component crosses zero, which it
    # does on every ring, and would report an unbounded correction where the
    # absolute error is in fact tiny.
    magnitude = np.hypot(exact["br"], exact["bz"])
    local = {"psi": np.abs(exact["psi"]), "br": magnitude, "bz": magnitude}
    far = {
        "bare_filament": point,
        "corrected_filament": corrected_filament(target_r, target_z, section),
    }
    correction = {
        label: {
            name: worst_per_ring(
                np.abs(model[name] - exact[name]) / local[name], radii, ring
            )
            for name in COMPONENTS
        }
        for label, model in far.items()
    }

    cutoff = {}
    for label, model in correction.items():
        cutoff[label] = {}
        for name in COMPONENTS:
            cutoff[label][name] = {}
            for tolerance in TOLERANCES:
                # the first ring beyond which EVERY further ring also clears the
                # tolerance -- a single ring dipping under it is not a cutoff
                valid = [
                    index
                    for index in np.flatnonzero(model[name] < tolerance)
                    if np.all(model[name][index:] < tolerance)
                ]
                cutoff[label][name][f"{tolerance:.0e}"] = (
                    float(ring[valid[0]]) if valid else float("inf")
                )

    return {
        "ring": ring.tolist(),
        "scale": scale,
        "quadrature": quadrature,
        "correction": {
            label: {name: value.tolist() for name, value in model.items()}
            for label, model in correction.items()
        },
        "cutoff_section_radii": cutoff,
        "tolerances": list(TOLERANCES),
    }


def figure(data: dict, path: pathlib.Path) -> None:
    """Write both evidence panels: quadrature order, and far-field correction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ring = np.asarray(data["ring"])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharex=True)

    for index, name in enumerate(COMPONENTS):
        for rule, record in data["quadrature"].items():
            if rule == "16x48":
                continue
            axes[index].plot(ring, np.maximum(record[name], 1e-17), label=rule)
        axes[index].axhline(1e-6, color="k", ls=":", lw=0.8)
        axes[index].set_xscale("log")
        axes[index].set_yscale("log")
        axes[index].set_xlabel("distance [section radii]")
        axes[index].set_title(f"{name}: reduced rule versus 16x48")
        axes[index].grid(alpha=0.3)
    axes[0].set_ylabel("error, relative to peak")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path.with_name("quadrature_order.png"), dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 5.4))
    styles = {"bare_filament": "-", "corrected_filament": "--"}
    for label, model in data["correction"].items():
        for name, colour in zip(COMPONENTS, ("C0", "C1", "C2")):
            axis.plot(
                ring,
                np.maximum(model[name], 1e-17),
                styles[label],
                color=colour,
                label=f"{label.replace('_', ' ')}, {name}",
            )
    for tolerance in data["tolerances"]:
        axis.axhline(tolerance, color="k", ls=":", lw=0.8)
        axis.text(ring[0], tolerance * 1.3, f"{tolerance:.0e}", fontsize=7)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("distance [section radii]")
    axis.set_ylabel("far-field error, relative to local magnitude")
    axis.set_title(
        "far field against the exact hexagon section:\n"
        "the bare filament flattens onto a floor, the corrected one does not"
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path.with_name("nearfar_correction.png"), dpi=140)


if __name__ == "__main__":
    destination = pathlib.Path(sys.argv[1])
    result = measure()
    destination.write_text(json.dumps(result, indent=2))
    figure(result, destination)
    print(json.dumps(result["cutoff_section_radii"], indent=2))
