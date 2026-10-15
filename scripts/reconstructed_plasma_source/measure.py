"""Measure a prescribed MAST plasma against its reconstruction's total flux."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.interpolate import RectBivariateSpline

from nova.biot.greens import hybrid_greens
from nova.imas.mast_vacuum_cohort import SHOT_STORE, read_shot_waveforms
from nova.imas.plasma_source import CoilCurrentSeries, PlasmaSource
from nova.imas.prescribed_plasma import plasma_flux_at_points
from nova.media.poloidal import draw_boundary, draw_flux_contours, draw_nulls, draw_wall
from nova.media.sources.mast_efit import read_pulse

SHOT = 21978
TIME = 0.2
FLUX_FRACTION_LIMIT = 0.02
FIGURE = Path("docs/figures/reconstructed-plasma-source/mast-linked-flux.svg")


def mast_source(shot: int = SHOT, time: float = TIME) -> tuple[PlasmaSource, object]:
    """Pair Nova's EFIT frame with independently acquired coil-current traces."""
    pulse = read_pulse(shot, times=[time], require_plasma=1e5)
    waveforms = read_shot_waveforms(shot)
    names = tuple(sorted(waveforms.drives))
    values = np.column_stack([waveforms.drives[name] for name in names])
    coils = CoilCurrentSeries(np.asarray(waveforms.time), names, values)
    return (
        PlasmaSource.from_mast(
            pulse.frames, coils, provenance=f"MAST {shot} efm and measured coil drives"
        ),
        pulse,
    )


def measure_linkage(
    source: PlasmaSource, pulse: object, *, store: Path = SHOT_STORE
) -> dict:
    """Compare plasma plus EFIT's own conductor flux to its stored psi map.

    A common offset is fixed at the magnetic axis because the flux gauge is
    arbitrary. The source carries separately measured coil currents; EFIT's
    fitted conductor currents are used only to separate its own total psi map
    into plasma and conductor contributions for this comparison.
    """
    frame = pulse.frames[0]
    current = source.prescribe(0)
    group = zarr.open_group(str(store / f"{SHOT}.zarr"), mode="r")["efm"]
    row = int(np.argmin(abs(np.asarray(group["time"]) - frame.time)))
    filament_r = np.asarray(group["fcoil_r"], float)
    filament_z = np.asarray(group["fcoil_z"], float)
    owner = np.asarray(group["fcoil_circ"], int) - 1
    turns = np.asarray(group["fcoil_turns"], float)
    share = np.asarray(group["fcoil_xmult"], float)
    fitted_current = np.asarray(group["fcoil_c"][row], float)
    coil_points = np.array(
        [
            (
                np.average(
                    filament_r[owner == circuit], weights=abs(share[owner == circuit])
                ),
                np.average(
                    filament_z[owner == circuit], weights=abs(share[owner == circuit])
                ),
            )
            for circuit in range(fitted_current.size)
        ]
    )
    loop_points = np.c_[
        np.asarray(group["silop_r"], float),
        np.asarray(group["silop_z"], float),
    ]
    points = np.vstack((frame.magnetic_axis[None, :], coil_points, loop_points))
    labels = np.array(
        ["axis"]
        + [f"coil-{index}" for index in range(len(coil_points))]
        + [f"loop-{index}" for index in range(len(loop_points))]
    )
    inside = (
        (points[:, 0] >= frame.radius[0])
        & (points[:, 0] <= frame.radius[-1])
        & (points[:, 1] >= frame.height[0])
        & (points[:, 1] <= frame.height[-1])
    )
    points, labels = points[inside], labels[inside]
    if labels[0] != "axis" or not np.any(np.char.startswith(labels, "coil-")):
        raise ValueError(
            "the reconstruction has no axis or coil targets in its psi map"
        )
    psi = RectBivariateSpline(frame.radius, frame.height, frame.flux.T).ev(
        points[:, 0], points[:, 1]
    )
    plasma = plasma_flux_at_points(current, points[:, 0], points[:, 1])
    conductor = np.zeros(len(points))
    width = np.asarray(group["fcoil_width"], float)
    height = np.asarray(group["fcoil_height"], float)
    for index in range(len(filament_r)):
        response = hybrid_greens(
            points[:, 0],
            points[:, 1],
            float(filament_r[index]),
            float(filament_z[index]),
            float(width[index]),
            float(height[index]),
        )[0]
        conductor += (
            fitted_current[owner[index]] * turns[index] * share[index] * response
        )
    if not np.any(plasma) or not np.any(conductor) or not np.ptp(psi):
        raise ValueError("flux comparison controls are empty")
    gauge = float(psi[0] - plasma[0] - conductor[0])
    residual = plasma + conductor + gauge - psi
    span = abs(frame.flux_boundary - frame.flux_axis)
    limit = FLUX_FRACTION_LIMIT * span
    result = {
        "shot": SHOT,
        "time_s": frame.time,
        "index": row,
        "plasma_current_a": current.plasma_current,
        "current_scale": current.scale_factor,
        "boundary_source": current.boundary_source,
        "equilibrium_rows": len(pulse.frames),
        "coil_current_rows": len(source.coils.time),
        "coil_current_count": len(source.coils.names),
        "coil_current_sample_a": source.coil_currents(0).tolist(),
        "flux_span_wb": span,
        "limit_fraction": FLUX_FRACTION_LIMIT,
        "limit_wb": limit,
        "gauge_wb": gauge,
        "targets": {},
    }
    for kind in ("coil", "loop"):
        picked = np.char.startswith(labels, f"{kind}-")
        errors = abs(residual[picked])
        result["targets"][kind] = {
            "count": int(picked.sum()),
            "median_abs_wb": float(np.median(errors)) if errors.size else None,
            "max_abs_wb": float(np.max(errors)) if errors.size else None,
        }
    result["plot"] = {
        "points": points,
        "labels": labels,
        "residual": residual,
        "psi": psi,
        "plasma": plasma,
        "conductor": conductor,
    }
    return result


def draw_measurement(result: dict, pulse: object, path: Path = FIGURE) -> None:
    """Draw the checked targets beside the gauge-corrected flux residual."""
    frame = pulse.frames[0]
    plot = result["plot"]
    points, labels, residual = (
        plot["points"],
        plot["labels"],
        plot["residual"],
    )
    fig, (scene, errors) = plt.subplots(1, 2, figsize=(14, 5))
    levels = frame.flux_axis + np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * (
        frame.flux_boundary - frame.flux_axis
    )
    draw_flux_contours(
        scene, frame.radius, frame.height, frame.flux, np.sort(levels), color="#555555"
    )
    draw_wall(scene, units=pulse.geometry.wall_units)
    draw_boundary(scene, frame.boundary[:, 0], frame.boundary[:, 1])
    draw_nulls(
        scene,
        frame.magnetic_axis,
        frame.x_points,
        contain=pulse.geometry.wall_units,
    )
    for kind, color, marker in (
        ("coil", "#315f78", "o"),
        ("loop", "#a06635", "x"),
    ):
        picked = np.char.startswith(labels, f"{kind}-")
        scene.scatter(
            points[picked, 0], points[picked, 1], s=12, color=color, marker=marker
        )
        errors.plot(
            np.arange(picked.sum()),
            abs(residual[picked]),
            marker=marker,
            markersize=3,
            linestyle="none",
            color=color,
        )
        if picked.any():
            errors.text(
                picked.sum() - 1,
                abs(residual[picked])[-1],
                kind,
                color=color,
                va="bottom",
            )
    errors.axhline(result["limit_wb"], color="#666666", linewidth=1.2, linestyle=":")
    errors.text(0, result["limit_wb"], "2% of psi span", color="#666666", va="bottom")
    errors.set_xlabel("target index")
    errors.set_ylabel("absolute flux residual [Wb]")
    errors.spines[["top", "right"]].set_visible(False)
    errors.grid(False)
    scene.set_aspect("equal")
    scene.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    source, pulse = mast_source()
    result = measure_linkage(source, pulse)
    draw_measurement(result, pulse)
    report = {key: value for key, value in result.items() if key != "plot"}
    print("LINKAGE " + json.dumps(report, sort_keys=True))
    for kind in ("coil", "loop"):
        target = report["targets"][kind]
        if target["count"] == 0 or target["max_abs_wb"] > report["limit_wb"]:
            raise AssertionError(f"{kind} linked flux exceeds declared tolerance")


if __name__ == "__main__":
    main()
