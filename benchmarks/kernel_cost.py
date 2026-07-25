"""Fresh-process per-pair cost and accuracy for every axisymmetric coupling kernel.

One variant per process: repeats inside a single interpreter warm the allocator
and the BLAS thread pool and understate the first-build cost that operator
assembly actually pays. Run as ``python benchmarks/kernel_cost.py <variant>``;
``list`` prints the variants. Output is one JSON record on stdout so a driver
can collect a table across processes, and ``kernel_cost_table.py`` turns the
collected records into the comparison table and figure.

Run it on a compute node. On a shared login node the same call has been
observed to vary five-fold, which is larger than every difference the table is
meant to resolve.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

R0, Z0 = 6.2, 0.0
CELL_RADIUS = 0.06  # hexagon circumradius of a ~560-cell ITER plasma mesh
RECT = (0.1, 0.08)  # a representative rectangular winding-pack section
N_TARGET = 512
CSTEP = 1e-30


def hexagon(r0=R0, z0=Z0, radius=CELL_RADIUS):
    """Return regular hexagon vertices, counter-clockwise."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def rectangle(r0=R0, z0=Z0, width=RECT[0], height=RECT[1]):
    """Return rectangle vertices, counter-clockwise."""
    return np.array(
        [
            [r0 - width / 2, z0 - height / 2],
            [r0 + width / 2, z0 - height / 2],
            [r0 + width / 2, z0 + height / 2],
            [r0 - width / 2, z0 + height / 2],
        ]
    )


def targets(n=N_TARGET, span=(0.5, 40.0), scale=CELL_RADIUS):
    """Return targets on a spiral spanning near to far in section radii."""
    offset = np.geomspace(span[0], span[1], n) * scale
    angle = np.linspace(0.0, 8.0 * np.pi, n)
    return R0 + offset * np.cos(angle), Z0 + offset * np.sin(angle)


def _time(fn, *args, **kwargs):
    """Return (wall seconds, result) for a single cold call."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - start, out


def _point(tr, tz, *, section):
    from nova.biot.greens import greens_bz_br, greens_psi

    centre = np.asarray(section).mean(axis=0)

    def run():
        psi = greens_psi(tr, tz, centre[0], centre[1])
        bz, br = greens_bz_br(tr, tz, centre[0], centre[1])
        return psi, br, bz

    return _time(run)


def _cylinder(tr, tz):
    from nova.biot.greens import cylinder_greens

    return _time(cylinder_greens, tr, tz, R0, Z0, *RECT)


def _hybrid(tr, tz):
    from nova.biot.greens import hybrid_greens

    return _time(hybrid_greens, tr, tz, R0, Z0, *RECT)


def _polygon(tr, tz, section, n_panels, n_nodes, block):
    from nova.biot.polygon import polygon_greens

    return _time(
        polygon_greens,
        tr,
        tz,
        section,
        n_panels=n_panels,
        n_nodes=n_nodes,
        block=block,
    )


def _polygon_complex_step(tr, tz, section, n_panels, n_nodes):
    """The complex-step curl the closed-form gradient replaced.

    Kept as a measurable baseline so the before/after is reproducible from the
    same script rather than from a remembered number.
    """
    from nova.biot import polygon

    v = np.asarray(section, dtype=np.float64)
    r = tr.ravel()[:, None]
    z = tz.ravel()[:, None]
    sign, area = polygon._orientation(v)
    phi, wts = polygon._phi_rule(n_panels, n_nodes)
    cosp, sinp = np.cos(phi), np.sin(phi)
    rule = (v, cosp, sinp, np.sin(2.0 * phi), wts * cosp, sign, area)

    def run():
        psi_r = polygon._psi_hat(r + 1j * CSTEP, z, *rule)
        dpsi_dz = polygon._psi_hat(r, z + 1j * CSTEP, *rule).imag / CSTEP
        two_pi_r = 2.0 * np.pi * r[:, 0]
        return psi_r.real, -dpsi_dz / two_pi_r, psi_r.imag / CSTEP / two_pi_r

    return _time(run)


def _rules():
    return {
        f"polygon_hex_{p}x{n}": (
            lambda tr, tz, p=p, n=n: _polygon(tr, tz, hexagon(), p, n, 64)
        )
        for p, n in [
            (16, 48),
            (8, 24),
            (8, 16),
            (4, 16),
            (4, 8),
            (2, 12),
            (2, 8),
            (1, 8),
            (1, 4),
        ]
    }


def _blocks():
    return {
        f"polygon_hex_block_{b}": (
            lambda tr, tz, b=b: _polygon(tr, tz, hexagon(), 16, 48, b)
        )
        for b in (8, 16, 32, 64, 128, 256, None)
    }


VARIANTS: dict[str, object] = {
    "point": lambda tr, tz: _point(tr, tz, section=hexagon()),
    "cylinder_rect": _cylinder,
    "hybrid_rect": _hybrid,
    "polygon_rect_16x48": lambda tr, tz: _polygon(tr, tz, rectangle(), 16, 48, 64),
    "polygon_hex_complex_step_16x48": (
        lambda tr, tz: _polygon_complex_step(tr, tz, hexagon(), 16, 48)
    ),
    "polygon_rect_complex_step_16x48": (
        lambda tr, tz: _polygon_complex_step(tr, tz, rectangle(), 16, 48)
    ),
    **_rules(),
    **_blocks(),
}


def main(name: str) -> None:
    tr, tz = targets()
    seconds, (psi, br, bz) = VARIANTS[name](tr, tz)  # type: ignore[operator]
    record = {
        "variant": name,
        "pairs": int(tr.size),
        "seconds": seconds,
        "us_per_pair": 1e6 * seconds / tr.size,
        "psi": np.asarray(psi).tolist(),
        "br": np.asarray(br).tolist(),
        "bz": np.asarray(bz).tolist(),
        "radii": (np.hypot(tr - R0, tz - Z0) / CELL_RADIUS).tolist(),
    }
    print(json.dumps(record))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        print("\n".join(VARIANTS))
    else:
        main(sys.argv[1])
