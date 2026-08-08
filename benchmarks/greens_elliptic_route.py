"""Accuracy and cost of the two elliptic-integral routes at the Green's call sites.

The axisymmetric kernels in :mod:`nova.biot.greens` come in pairs: a numpy host
and an ``xp``-threaded twin that runs under numpy or a tracer.  The pairs are
NOT the same arithmetic, and they differ in two independent places:

* the POINT FILAMENT -- :func:`~nova.biot.greens.greens_psi` and
  :func:`~nova.biot.greens.greens_bz_br` reach ``K`` and ``E`` through
  ``scipy.special.ellipkm1`` / ``scipy.special.ellipe`` (Cephes), while
  :func:`~nova.biot.greens.traced_filament_greens` reaches them through the
  Bulirsch ``cel`` descent in :mod:`nova.biot.completeelliptic`.  The pole of
  interest is the coincident-ring limit ``k^2 -> 1``, where ``K`` grows like
  ``-log k'`` and the two routes' argument handling is what decides the answer;
* the RECTANGULAR SECTION -- :func:`~nova.biot.greens.corner_fields` and
  :func:`~nova.biot.greens.traced_corner_fields` already share the Bulirsch
  descent for all three kinds, so their ONLY difference is the ``zeta``
  quadrature: the host routes each element between a 48-node Gauss-Legendre
  rule and a 177-node tanh-sinh rule, the twin takes tanh-sinh unconditionally.

This module measures both differences over the argument regimes the call sites
actually reach, arbitrated where possible against a reference neither route can
be: the extended-precision filament in :mod:`benchmarks.near_field_elliptic`,
and an adaptive quadrature for ``zeta``.  Run as
``python benchmarks/greens_elliptic_route.py`` for the whole report, or with
``accuracy`` / ``cost`` / ``zeta`` for one section.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.integrate  # type: ignore[import-untyped]

from benchmarks.near_field_elliptic import reference
from nova.biot.greens import (
    corner_fields,
    cylinder_greens,
    greens_bz_br,
    greens_psi,
    traced_corner_fields,
    traced_cylinder_greens,
    traced_filament_greens,
)
from nova.biot.zeta import NEAR_PLANE_RATIO, traced_zeta, zeta

RING_R, RING_Z = 6.2, 0.0
"""Ring geometry of the rest of the suite: the ITER major radius."""

SECTION = (6.2, 0.0, 0.1, 0.08)
"""Representative rectangular winding-pack section ``(a, z0, da, dz)`` [m]."""

REPEATS = 7
"""Timing repeats; the MINIMUM is reported, which is the least contaminated by
whatever else shares the node."""


# --- argument regimes the call sites reach ----------------------------


def _spiral(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return targets on a spiral at the given distances from the ring point.

    A spiral rather than a radial line so every sample carries a different
    angle: the field brackets cancel differently above the ring, beside it, and
    in its own plane, and a radial sweep would report one of those three.

    The radius is held off the axis at a tenth of the ring's: a spiral wide
    enough to reach the far side of the machine otherwise walks through
    ``R = 0`` and out to negative radii, which is not a target a caller has.
    """
    angle = np.linspace(0.0, 8.0 * np.pi, offsets.size)
    radius = RING_R + offsets * np.cos(angle)
    return np.maximum(radius, 0.1 * RING_R), RING_Z + offsets * np.sin(angle)


def filament_regimes(count: int = 512) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the target sets the filament kernels are driven over.

    Named for the geometry, and each one is a configuration a caller reaches:

    ``coincident_limit``
        1 nm to 1 cm off the filament -- the modulus complement runs 1e-20 to
        1e-6, which is ``k^2 -> 1``.  Reached by a self-coupling read and by a
        plasma-cell filament that lands on a conductor.
    ``near``
        1 cm to 1 m, the band a finite-section kernel hands over to the point
        form at.
    ``standoff``
        1 m to 20 m -- diagnostic loops, passive-structure cells against the
        magnetic axis, the far side of the machine.
    ``ring_plane``
        the ring's own plane, where ``B_R`` is identically zero and the two
        field brackets cancel hardest.
    ``axis``
        the machine axis, where the kernels take their masked limits.
    """
    return {
        "coincident_limit": _spiral(np.geomspace(1e-9, 1e-2, count)),
        "near": _spiral(np.geomspace(1e-2, 1.0, count)),
        "standoff": _spiral(np.geomspace(1.0, 20.0, count)),
        "ring_plane": (
            RING_R + np.geomspace(1e-6, 12.0, count),
            np.zeros(count),
        ),
        "axis": (
            np.geomspace(1e-12, 1e-3, count),
            np.linspace(-3.0, 3.0, count),
        ),
    }


def section_regimes(count: int = 512) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the target sets the rectangular-section kernels are driven over.

    ``inside`` and ``corner_plane`` are the configurations the finite-section
    kernel exists for -- a target within the conductor, and one level with a
    corner where the antiderivative's branch terms change sign.  ``corner_plane``
    also straddles the ``zeta`` rule switch: the host takes tanh-sinh within
    ``NEAR_PLANE_RATIO`` of a corner's own level and Gauss-Legendre outside it,
    so a sweep through that ratio is where the two routes can part.
    """
    a, z0, da, dz = SECTION
    grid_r = np.linspace(a - 0.4 * da, a + 0.4 * da, int(np.sqrt(count)))
    grid_z = np.linspace(z0 - 0.4 * dz, z0 + 0.4 * dz, int(np.sqrt(count)))
    mesh_r, mesh_z = (array.ravel() for array in np.meshgrid(grid_r, grid_z))
    # gamma/r from far outside the switch down to the corner's own level
    gap = np.concatenate(
        [
            np.geomspace(2.0 * NEAR_PLANE_RATIO * a, 1e-9 * a, count // 2),
            -np.geomspace(2.0 * NEAR_PLANE_RATIO * a, 1e-9 * a, count // 2),
        ]
    )
    return {
        "inside": (mesh_r, mesh_z),
        "corner_plane": (np.full(gap.size, a + 0.3 * da), z0 + dz / 2.0 + gap),
        "near": _spiral(np.geomspace(0.6 * da, 4.0 * da, count)),
        "standoff": _spiral(np.geomspace(1.0, 20.0, count)),
    }


# --- accuracy ---------------------------------------------------------


def _scaled_error(value: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return ``|value - truth|`` over the largest magnitude in the regime.

    Normalised on the SET rather than element-wise: ``B_Z`` changes sign inside
    the standoff sweep and ``B_R`` is identically zero in the ring's own plane,
    so an element-wise ratio reports where the zero is and not what the routes
    did.  Against the regime's own scale it reports what a caller summing the
    coupling over that regime would actually carry.
    """
    value = np.asarray(value, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(value) & np.isfinite(truth)
    scale = np.abs(truth[finite]).max() if finite.any() else 0.0
    if scale == 0.0:
        return np.zeros(1)
    return np.abs(value[finite] - truth[finite]) / scale


def _deviation(one: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """Return ``(median, max)`` deviation of ``one`` from ``other``, set-normalised."""
    error = _scaled_error(one, other)
    return float(np.median(error)), float(error.max())


def filament_accuracy(count: int = 512) -> list[dict]:
    """Return per-regime, per-component agreement of the two filament routes.

    Each route is ALSO differenced against the extended-precision reference, so
    the table says which one is right rather than only that they differ.  The
    reference is the ``longdouble`` filament of
    :mod:`benchmarks.near_field_elliptic`, whose complement comes from the
    geometry and whose field brackets have their pole split off -- two things a
    double kernel cannot do for itself.
    """
    rows = []
    for name, (target_r, target_z) in filament_regimes(count).items():
        with np.errstate(divide="ignore", invalid="ignore"):
            host_psi = greens_psi(target_r, target_z, RING_R, RING_Z)
            host_bz, host_br = greens_bz_br(target_r, target_z, RING_R, RING_Z)
            twin_psi, twin_br, twin_bz = traced_filament_greens(
                np, target_r, target_z, RING_R, RING_Z
            )
            exact = reference(target_r, target_z, RING_R, RING_Z)
        for component, host, twin, truth in zip(
            ("psi", "bz", "br"),
            (host_psi, host_bz, host_br),
            (twin_psi, twin_bz, twin_br),
            exact,
        ):
            median, worst = _deviation(twin, host)
            truth = np.asarray(truth, dtype=np.float64)
            host_error = _scaled_error(host, truth)
            twin_error = _scaled_error(twin, truth)
            rows.append(
                {
                    "regime": name,
                    "component": component,
                    "median": median,
                    "max": worst,
                    "host_vs_exact": float(host_error.max()),
                    "twin_vs_exact": float(twin_error.max()),
                    "twin_closer": float(np.mean(twin_error < host_error)),
                }
            )
    return rows


def section_accuracy(count: int = 512) -> list[dict]:
    """Return per-regime agreement of the two rectangular-section routes.

    Both the four-corner ``cylinder_greens`` combination and the bare corner
    antiderivative, because the combination differences four corner values
    against each other and can either cancel a per-corner difference or amplify
    it.
    """
    a, z0, da, dz = SECTION
    rows = []
    for name, (target_r, target_z) in section_regimes(count).items():
        host = cylinder_greens(target_r, target_z, a, z0, da, dz)
        twin = traced_cylinder_greens(np, target_r, target_z, a, z0, da, dz)
        for component, one, other in zip(("psi", "br", "bz"), twin, host):
            median, worst = _deviation(one, other)
            rows.append(
                {
                    "regime": name,
                    "kernel": "cylinder",
                    "component": component,
                    "median": median,
                    "max": worst,
                }
            )
        stacks = _corner_stacks(target_r, target_z)
        host_corner = corner_fields(*stacks)
        twin_corner = traced_corner_fields(np, *stacks)
        for component, one, other in zip(
            ("aphi", "br", "bz"), twin_corner, host_corner
        ):
            median, worst = _deviation(one, other)
            rows.append(
                {
                    "regime": name,
                    "kernel": "corner",
                    "component": component,
                    "median": median,
                    "max": worst,
                }
            )
    return rows


def _corner_stacks(
    target_r: np.ndarray, target_z: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Return ``(rs, zs, r, z)`` shaped ``(T, 4)`` for the section's four corners."""
    a, z0, da, dz = SECTION
    rs = np.stack(
        [np.full(target_r.shape, a + d * da / 2.0) for d in (-1, 1, 1, -1)], axis=-1
    )
    zs = np.stack(
        [np.full(target_r.shape, z0 + d * dz / 2.0) for d in (-1, -1, 1, 1)], axis=-1
    )
    return (
        rs,
        zs,
        np.repeat(target_r[..., None], 4, axis=-1),
        np.repeat(target_z[..., None], 4, axis=-1),
    )


# --- the zeta quadrature, arbitrated ----------------------------------


def zeta_accuracy(samples: int = 24) -> list[dict]:
    """Return both ``zeta`` rules against adaptive quadrature, across the switch.

    The host's rule switch is the whole of the section kernels' route difference,
    so which rule is right where decides the section verdict.  The arbiter is
    :func:`scipy.integrate.quad` on the same integrand, run per element to its own
    tolerance -- an independent method, not a finer version of either rule.
    """
    a, _, da, _ = SECTION
    source_r = a + da / 2.0
    rows = []
    for ratio in np.geomspace(50.0, 1e-8, samples):
        gamma = ratio * NEAR_PLANE_RATIO * source_r
        target_r = np.array([a + 0.3 * da])
        arm = np.array([gamma])
        host = float(zeta(target_r, np.array([source_r]), arm, np.pi / 2.0)[0])
        twin = float(
            traced_zeta(np, target_r, np.array([source_r]), arm, np.pi / 2.0)[0]
        )

        def integrand(alpha, rs=float(target_r[0]), r=source_r, g=gamma):
            phi = np.pi - 2.0 * alpha
            return np.arcsinh(
                (rs - r * np.cos(phi)) / np.sqrt(g**2 + r**2 * np.sin(phi) ** 2)
            )

        exact, _ = scipy.integrate.quad(
            integrand, 0.0, np.pi / 2.0, epsabs=1e-14, epsrel=1e-14, limit=400
        )
        near_plane = abs(gamma) < NEAR_PLANE_RATIO * source_r
        rows.append(
            {
                "gamma_over_r": gamma / source_r,
                "rule": "tanh_sinh" if near_plane else "gauss",
                "host_error": abs(host - exact) / abs(exact),
                "twin_error": abs(twin - exact) / abs(exact),
            }
        )
    return rows


# --- cost -------------------------------------------------------------


def _fastest(call, *args) -> float:
    """Return the minimum wall seconds of :data:`REPEATS` calls."""
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        call(*args)
        best = min(best, time.perf_counter() - start)
    return best


def cost(sizes=(64, 512, 4096, 65536)) -> list[dict]:
    """Return per-element cost of both routes at the array sizes callers use.

    Sizes span one diagnostic loop set (64) through a plasma-cell coupling block
    (65536 pairs is a 256-cell mesh against itself).  Timed on the ``near``
    regime, which is where both the descent and the quadrature do their full
    work -- a far-field set exercises the same node counts, the fixed trip count
    being the point of both routines.
    """
    a, z0, da, dz = SECTION
    rows = []
    for size in sizes:
        target_r, target_z = _spiral(np.geomspace(1e-2, 1.0, size))

        def host_filament(tr=target_r, tz=target_z):
            greens_psi(tr, tz, RING_R, RING_Z)
            greens_bz_br(tr, tz, RING_R, RING_Z)

        def twin_filament(tr=target_r, tz=target_z):
            traced_filament_greens(np, tr, tz, RING_R, RING_Z)

        rows.append(
            {
                "kernel": "filament",
                "size": size,
                "host_us": 1e6 * _fastest(host_filament),
                "twin_us": 1e6 * _fastest(twin_filament),
            }
        )
        rows.append(
            {
                "kernel": "cylinder",
                "size": size,
                "host_us": 1e6
                * _fastest(cylinder_greens, target_r, target_z, a, z0, da, dz),
                "twin_us": 1e6
                * _fastest(
                    traced_cylinder_greens, np, target_r, target_z, a, z0, da, dz
                ),
            }
        )
    return rows


# --- report -----------------------------------------------------------


def _print(rows: list[dict], columns: tuple[str, ...], formats: tuple[str, ...]):
    """Print one table."""
    print("  ".join(f"{name:>16}" for name in columns))
    for row in rows:
        cells = []
        for name, spec in zip(columns, formats):
            cells.append(f"{row[name]:>16{spec}}")
        print("  ".join(cells))
    print()


def main(section: str = "all"):
    """Print the requested section of the report."""
    if section in ("all", "accuracy"):
        print("FILAMENT: twin vs host, and each vs the extended-precision reference")
        _print(
            filament_accuracy(),
            (
                "regime",
                "component",
                "median",
                "max",
                "host_vs_exact",
                "twin_vs_exact",
                "twin_closer",
            ),
            ("", "", ".3e", ".3e", ".3e", ".3e", ".2f"),
        )
        print("SECTION: twin vs host (both already on the same elliptic descent)")
        _print(
            section_accuracy(),
            ("regime", "kernel", "component", "median", "max"),
            ("", "", "", ".3e", ".3e"),
        )
    if section in ("all", "zeta"):
        print("ZETA: both rules against adaptive quadrature, across the rule switch")
        _print(
            zeta_accuracy(),
            ("gamma_over_r", "rule", "host_error", "twin_error"),
            (".3e", "", ".3e", ".3e"),
        )
    if section in ("all", "cost"):
        print("COST: microseconds per call, minimum of %d" % REPEATS)
        rows = cost()
        for row in rows:
            row["ratio"] = row["twin_us"] / row["host_us"]
            row["host_ns_per_element"] = 1e3 * row["host_us"] / row["size"]
            row["twin_ns_per_element"] = 1e3 * row["twin_us"] / row["size"]
        _print(
            rows,
            (
                "kernel",
                "size",
                "host_us",
                "twin_us",
                "ratio",
                "host_ns_per_element",
                "twin_ns_per_element",
            ),
            ("", "d", ".1f", ".1f", ".2f", ".1f", ".1f"),
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
