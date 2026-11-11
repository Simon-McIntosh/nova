"""Diagnose saved total-flux maps without evaluating a forward map.

Coordinates are (R, Z), flux arrays are (radius, height), and offsets are
solved minus reference. A translation is the displacement APPLIED to the solved
field: aligned(R,Z) = solved(R-dR,Z-dZ). Only a constant gauge is removed;
no flux amplitude, coil current, profile or boundary condition is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import contourpy
import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares, root

HERE = Path(__file__).resolve().parent


def spline(radius, height, flux):
    """Interpolate one finite, radius-major map with a cubic tensor spline."""
    radius, height, flux = map(np.asarray, (radius, height, flux))
    if flux.shape != (radius.size, height.size) or not np.isfinite(flux).all():
        raise ValueError("flux must be finite and shaped (radius, height)")
    return RectBivariateSpline(radius, height, flux, s=0)


def polygon_mask(radius, height, boundary):
    """Select the labelled plasma interior on the saved node lattice."""
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    return (
        PolygonPath(boundary)
        .contains_points(np.column_stack((rr.ravel(), zz.ravel())))
        .reshape(rr.shape)
    )


def fit_translation(radius, height, solved, reference, mask):
    """Minimise gauge-free flux RMS on a fixed reference-plasma region.

    Bounds keep EVERY compared point inside the saved solved map. The mask
    never shrinks as a function of trial translation, so a fit cannot improve
    its score by dropping difficult points. Multiple starts detect local minima.
    """
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    points = np.column_stack((rr[mask], zz[mask]))
    if len(points) < 12:
        raise ValueError("translation fit needs at least twelve plasma nodes")
    field = spline(radius, height, solved)
    target = np.asarray(reference)[mask]
    lower = points.max(axis=0) - [radius[-1], height[-1]]
    upper = points.min(axis=0) - [radius[0], height[0]]
    if np.any(lower >= 0) or np.any(upper <= 0):
        raise ValueError("plasma region needs an interpolation margin")

    def residual(displacement):
        query = points - displacement
        difference = field.ev(query[:, 0], query[:, 1]) - target
        return difference - difference.mean()

    def jacobian(displacement):
        query = points - displacement
        columns = np.column_stack(
            (-field.ev(*query.T, dx=1), -field.ev(*query.T, dy=1))
        )
        return columns - columns.mean(axis=0)

    candidates = []
    for fraction in (np.zeros(2), np.array([0.35, 0.35]), np.array([-0.35, -0.35])):
        start = np.where(fraction >= 0, fraction * upper, -fraction * lower)
        candidates.append(
            least_squares(
                residual,
                start,
                jac=jacobian,
                bounds=(lower, upper),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )
        )
    optimization = min(candidates, key=lambda value: np.linalg.norm(value.fun))
    if not optimization.success:
        raise RuntimeError("rigid-translation fit did not converge")
    translation = optimization.x
    before = float(np.sqrt(np.mean(residual(np.zeros(2)) ** 2)))
    after = float(np.sqrt(np.mean(residual(translation) ** 2)))
    query = points - translation
    gauge = float(np.mean(field.ev(*query.T) - target))
    return dict(
        translation_m=translation.tolist(),
        gauge_wb=gauge,
        gauge_before_wb=float(np.mean(solved[mask] - reference[mask])),
        rms_before_wb=before,
        rms_after_wb=after,
        squared_mismatch_removed=1 - (after / before) ** 2 if before else 0,
        node_count=len(points),
        bounds_m=[lower.tolist(), upper.tolist()],
        bound_active=bool(np.any(np.abs(optimization.active_mask))),
        optimality=float(optimization.optimality),
    )


def classify_offsets(axis_offset, saddle_offset, fit):
    """Call an offset rigid only when field and BOTH landmarks support it.

    The numerical rule is >=50% squared flux mismatch removed, <=50% of the
    larger landmark offset left in their difference, and <=50% of the original
    saddle offset left after translation. 'local' means non-rigid by this rule;
    it does not assert that an arbitrary shape error is confined to the saddle.
    """
    axis_offset, saddle_offset = map(np.asarray, (axis_offset, saddle_offset))
    coherent = np.linalg.norm(saddle_offset - axis_offset) <= 0.5 * max(
        np.linalg.norm(axis_offset), np.linalg.norm(saddle_offset), 1e-12
    )
    remaining = saddle_offset + fit["translation_m"]
    corrected = np.linalg.norm(remaining) <= 0.5 * max(
        np.linalg.norm(saddle_offset), 1e-12
    )
    rigid = fit["squared_mismatch_removed"] >= 0.5 and coherent and corrected
    return "rigid" if rigid else "local"


def stationary(field, seed, saddle=False):
    """Polish a null of this interpolant and require its Hessian signature."""

    def gradient(point):
        return np.array([field.ev(*point, dx=1), field.ev(*point, dy=1)], float)

    result = root(gradient, seed, tol=1e-10)
    point = result.x
    hessian = np.array(
        [
            [field.ev(*point, dx=2), field.ev(*point, dx=1, dy=1)],
            [field.ev(*point, dx=1, dy=1), field.ev(*point, dy=2)],
        ],
        float,
    )
    if np.linalg.norm(gradient(point)) > 1e-7 or (np.linalg.det(hessian) < 0) != saddle:
        raise ValueError(f"unqualified {'saddle' if saddle else 'axis'} at {point}")
    return point, hessian


def boundary_curve(radius, height, flux, axis, level):
    """Read the 99.9% closed surface enclosing this field's axis.

    The finite marching grid cannot resolve the exact tangent saddle crossing.
    The boundary-gap receipt therefore declares this small inward flux offset;
    the null offset itself is always read from the continuous interpolant."""
    field = spline(radius, height, flux)
    axis_value = float(field.ev(*axis))
    interior_level = level + 1e-3 * (axis_value - level)
    dense_r = np.linspace(radius[0], radius[-1], 401)
    dense_z = np.linspace(height[0], height[-1], 601)
    contours = contourpy.contour_generator(
        x=dense_r, y=dense_z, z=field(dense_r, dense_z).T
    ).lines(interior_level)
    loops = [
        line
        for line in contours
        if len(line) >= 4
        and np.linalg.norm(line[0] - line[-1]) < 1e-8
        and PolygonPath(line).contains_point(axis)
    ]
    if not loops:
        raise ValueError("no closed axis-enclosing boundary at the saved boundary flux")
    return max(loops, key=len)


def horizontal_intersections(boundary, height):
    """Intersect ordered boundary segments with one common horizontal plane."""
    first, second = boundary, np.roll(boundary, -1, axis=0)
    valid = (first[:, 1] > height) != (second[:, 1] > height)
    a, b = first[valid], second[valid]
    if len(a) < 2:
        raise ValueError("boundary does not cross the comparison midplane twice")
    values = a[:, 0] + (height - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return float(values.min()), float(values.max())


def current_and_force(radius, height, flux, mask):
    """Ampere current and toroidal-current Lorentz force from the map itself.

    This is integral j_phi x B_p, not J x B - grad(p). The radial entry is
    the toroidally integrated radial scalar (hoop loading); its Cartesian
    vector cancels on integrating azimuth. No external field is substituted.
    """
    import jax.numpy as jnp
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.map_extraction import apply_delta_star
    from nova.equilibrium.convention import TOTAL_FLUX_FACTOR

    receipt = apply_delta_star(radius, height, flux)
    mesh = FluxLattice(radius, height)
    radial, vertical = [
        np.asarray(v).reshape(flux.shape)
        for v in mesh.gradient(jnp.asarray(flux).ravel())
    ]
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    selected = mask & receipt.valid
    density = np.where(selected, receipt.toroidal_current_density, 0)
    area = receipt.radial_step * receipt.vertical_step
    current = density * area
    total = float(current.sum())
    if not np.isfinite(total) or abs(total) < 1:
        raise ValueError("no measurable plasma current; absence is not a centroid")
    br = -vertical / (TOTAL_FLUX_FACTOR * rr)
    bz = radial / (TOTAL_FLUX_FACTOR * rr)
    force_r = float(np.sum(2 * np.pi * rr * current * bz))
    force_z = float(np.sum(-2 * np.pi * rr * current * br))
    return dict(
        total_current_a=total,
        centroid_rz_m=[
            float((current * rr).sum() / total),
            float((current * zz).sum() / total),
        ],
        force_rz_n=[force_r, force_z],
        nodes=int(selected.sum()),
        negative_current_fraction=float(-current[current < 0].sum() / abs(total)),
    )


def read_case(entry):
    """Verify a banked copy against its input hash before opening it."""
    path = Path(entry["files"]["terminal.npz"]["copy"])
    expected = entry["files"]["terminal.npz"]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"input hash differs: {path}")
    with np.load(path) as archive:
        data = {key: archive[key] for key in archive.files}
    receipt_path = Path(entry["files"]["receipt.json"]["copy"])
    if (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        != entry["files"]["receipt.json"]["sha256"]
    ):
        raise ValueError(f"receipt hash differs: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt["storage_order"] != "radius_height":
        raise ValueError("only declared radius-height storage is admitted")
    return data, receipt


def analyze(entry):
    """Decompose a saved field, recording limitations of the current read."""
    data, receipt = read_case(entry)
    radius, height = data["radius"], data["height"]
    reference, solved = data["label"], data["flux"]
    ref_field, sol_field = [spline(radius, height, v) for v in (reference, solved)]
    ref_axis, _ = stationary(ref_field, data["reference_axis"])
    ref_saddle, ref_hessian = stationary(
        ref_field, receipt["efit_x_point_rz_m"], saddle=True
    )
    axis, _ = stationary(sol_field, data["axis"][:2])
    saddle, _ = stationary(sol_field, data["continued_saddle"][:2], saddle=True)
    ref_mask = polygon_mask(radius, height, data["label_boundary"])
    fit = fit_translation(radius, height, solved, reference, ref_mask)
    axis_offset, saddle_offset = axis - ref_axis, saddle - ref_saddle
    raw_axis_offset = data["axis"][:2] - data["reference_axis"]
    ref_boundary = data["label_boundary"]
    sol_boundary = boundary_curve(
        radius, height, solved, axis, float(data["boundary"][2])
    )
    ref_in, ref_out = horizontal_intersections(
        ref_boundary, float(data["reference_axis"][1])
    )
    sol_in, sol_out = horizontal_intersections(
        sol_boundary, float(data["reference_axis"][1])
    )
    gradient_error = np.array(
        [
            sol_field.ev(*ref_saddle, dx=1) - ref_field.ev(*ref_saddle, dx=1),
            sol_field.ev(*ref_saddle, dy=1) - ref_field.ev(*ref_saddle, dy=1),
        ]
    )
    predicted = -np.linalg.solve(ref_hessian, gradient_error)
    current = current_and_force(
        radius, height, solved, np.asarray(data["connected"], bool)
    )
    reference_current = current_and_force(radius, height, reference, ref_mask)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    difference = solved - reference
    difference -= difference[ref_mask].mean()
    inner = ref_mask & (np.hypot(rr - ref_saddle[0], zz - ref_saddle[1]) > 0.3)
    near = ref_mask & ~inner
    return dict(
        arm=entry["arm"],
        case=entry["case"],
        converged=receipt["converged"],
        residual=receipt["residual"],
        terminal_reason=receipt["terminal_reason"],
        grid_spacing_cm=[100 * float(np.median(np.diff(a))) for a in (radius, height)],
        axis_offset_cm=(100 * axis_offset).tolist(),
        raw_stored_axis_offset_cm=(100 * raw_axis_offset).tolist(),
        xpoint_offset_cm=(100 * saddle_offset).tolist(),
        rigid_shift_residual_cm=(100 * (saddle_offset - axis_offset)).tolist(),
        xpoint_after_translation_cm=(
            100 * (saddle_offset + fit["translation_m"])
        ).tolist(),
        axis_rz_m=axis.tolist(),
        saddle_rz_m=saddle.tolist(),
        reference_axis_rz_m=ref_axis.tolist(),
        reference_saddle_rz_m=ref_saddle.tolist(),
        fit=fit,
        current=current,
        reference_current=reference_current,
        boundary_level_fraction=0.999,
        boundary_gap_cm=dict(
            inboard=100 * (sol_in - ref_in),
            outboard=100 * (sol_out - ref_out),
            top=100 * (sol_boundary[:, 1].max() - ref_boundary[:, 1].max()),
            xpoint=float(100 * np.linalg.norm(saddle_offset)),
        ),
        saddle_gradient_error_wb_per_m=gradient_error.tolist(),
        reference_saddle_hessian_wb_per_m2=ref_hessian.tolist(),
        linear_saddle_prediction_cm=(100 * predicted).tolist(),
        inner_rms_error_wb=float(np.sqrt(np.mean(difference[inner] ** 2))),
        near_saddle_rms_error_wb=float(np.sqrt(np.mean(difference[near] ** 2)))
        if near.any()
        else None,
        verdict=classify_offsets(axis_offset, saddle_offset, fit),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    import jax
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    rows = []
    for entry in json.loads(arguments.inputs.read_text()):
        row = analyze(entry)
        rows.append(row)
        arguments.output.write_text(
            json.dumps(dict(rows=rows, forward_solves=0), indent=2) + "\n"
        )
        print(json.dumps(row), flush=True)
    print(f"MEASUREMENT: {len(rows)} saved states; zero forward solves", flush=True)


if __name__ == "__main__":
    main()
