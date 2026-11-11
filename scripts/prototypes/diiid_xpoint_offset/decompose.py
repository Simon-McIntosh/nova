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
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator

from scripts.diiid_map_fidelity.measure import polish
from scripts.prototypes.diiid_axis_identification.census import (
    _critical_points,
    _boundary_winding,
)

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


def classify_offsets(saddle_offset, fit):
    """Require both whole-field and lower-saddle agreement for rigid motion.

    A translation must remove at least half the squared flux mismatch AND
    at least half the lower-X distance. Neither condition uses an axis seed.
    """
    saddle_offset = np.asarray(saddle_offset)
    remaining = saddle_offset + fit["translation_m"]
    corrected = np.linalg.norm(remaining) <= 0.5 * max(
        np.linalg.norm(saddle_offset), 1e-12
    )
    return "rigid" if fit["squared_mismatch_removed"] >= 0.5 and corrected else "local"


def critical_census(radius, height, flux, mask, boundary=None):
    """Census candidates through their owner, polish through the null owner.

    Two refinements must agree on every polished critical point, and the
    gradient boundary winding must agree with their signed index sum. Hessian
    eigenvalues distinguish maxima from minima: positive determinant alone
    only qualifies an extremum, not a magnetic axis.
    """
    field = spline(radius, height, flux)
    membership = RegularGridInterpolator(
        (radius, height),
        np.asarray(mask, float),
        method="nearest",
        bounds_error=False,
        fill_value=0,
    )
    previous = None
    histories = []
    for refine in (16, 32):
        fine_r = np.linspace(radius[0], radius[-1], (len(radius) - 1) * refine + 1)
        fine_z = np.linspace(height[0], height[-1], (len(height) - 1) * refine + 1)
        fine = field(fine_r, fine_z)
        candidates = _critical_points(fine, fine_r, fine_z)
        points = []
        for candidate in candidates:
            guess = [candidate["r"], candidate["z"]]
            null = polish(flux, radius, height, guess, False)
            if not null["valid"]:
                null = polish(flux, radius, height, guess, True)
            if not null["valid"]:
                continue
            point = np.asarray(null["rz_m"])
            if any(
                np.linalg.norm(point - np.asarray(p["rz_m"])) < 1e-6 for p in points
            ):
                continue
            hessian = np.array(
                [
                    [field.ev(*point, dx=2), field.ev(*point, dx=1, dy=1)],
                    [field.ev(*point, dx=1, dy=1), field.ev(*point, dy=2)],
                ],
                float,
            )
            eigenvalues = np.linalg.eigvalsh(hessian)
            kind = (
                "maximum"
                if eigenvalues[-1] < 0
                else "minimum"
                if eigenvalues[0] > 0
                else "saddle"
            )
            inside = (
                bool(PolygonPath(boundary).contains_point(point))
                if boundary is not None
                else bool(membership(point[None])[0] > 0.5)
            )
            points.append(
                dict(
                    **null,
                    kind=kind,
                    in_plasma=inside,
                    flux_wb=float(field.ev(*point)),
                    hessian_eigenvalues_wb_per_m2=eigenvalues.tolist(),
                )
            )
        points.sort(key=lambda point: tuple(point["rz_m"]))
        winding = float(_boundary_winding(fine, fine_r, fine_z))
        index_sum = sum(-1 if p["kind"] == "saddle" else 1 for p in points)
        coverage = round(winding) == index_sum
        histories.append(
            dict(
                refine=refine,
                points=len(points),
                winding=winding,
                index_sum=index_sum,
                coverage_ok=coverage,
            )
        )
        if previous is not None:
            stable = len(points) == len(previous) and all(
                first["kind"] == second["kind"]
                and np.linalg.norm(np.asarray(first["rz_m"]) - second["rz_m"]) < 1e-5
                for first, second in zip(previous, points)
            )
            if not stable or not coverage:
                raise ValueError(f"critical census not closed: {histories}")
        previous = points
    maxima = [p for p in points if p["kind"] == "maximum" and p["in_plasma"]]
    minima = [p for p in points if p["kind"] == "minimum" and p["in_plasma"]]
    if not maxima:
        raise ValueError("census found no in-plasma maximum; no axis can be reported")
    return dict(
        points=points,
        refinements=histories,
        in_plasma_maxima=maxima,
        maxima_count=len(maxima),
        in_plasma_minima=minima,
        minima_count=len(minima),
        maxima_flux_spread_mwb=float(1000 * np.ptp([p["flux_wb"] for p in maxima])),
        max_flux_wb=max(p["flux_wb"] for p in maxima),
        axis_determined=len(maxima) == 1 and len(minima) == 0,
    )


def core_shape(radius, height, flux, census, current):
    """Axis-free core flatness and native Ampere-density contrast.

    The vertical cut has a fixed 60 cm length, centred on the current centroid
    and at the mean radial coordinate of ALL maxima. The central and shoulder
    current samples use the same radius and native stencil, never a selected
    maximum's basin.
    """
    from nova.equilibrium.map_extraction import apply_delta_star
    from scipy.constants import mu_0

    field = spline(radius, height, flux)
    core_r = float(np.mean([p["rz_m"][0] for p in census["in_plasma_maxima"]]))
    centre_z = current["centroid_rz_m"][1]
    cut_z = np.linspace(centre_z - 0.3, centre_z + 0.3, 201)
    values = field.ev(np.full_like(cut_z, core_r), cut_z)
    density = apply_delta_star(radius, height, flux)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    radial = np.abs(rr - core_r) <= 1.6 * density.radial_step
    central = radial & (np.abs(zz - centre_z) <= 0.1) & density.valid
    shoulder = (
        radial
        & (np.abs(zz - centre_z) >= 0.15)
        & (np.abs(zz - centre_z) <= 0.35)
        & density.valid
    )
    central_j = float(np.median(density.toroidal_current_density[central]))
    shoulder_j = float(np.median(density.toroidal_current_density[shoulder]))
    # Compare the continuous and native-stencil density reads at the core.
    middle_derivatives = (
        float(field.ev(core_r, centre_z, dx=2)),
        float(field.ev(core_r, centre_z, dx=1)),
        float(field.ev(core_r, centre_z, dy=2)),
    )
    continuous_j = -(
        middle_derivatives[0] - middle_derivatives[1] / core_r + middle_derivatives[2]
    ) / (2 * np.pi * mu_0 * core_r)
    return dict(
        cut_radius_m=core_r,
        cut_centre_z_m=centre_z,
        cut_half_height_m=0.3,
        vertical_flux_range_mwb=float(1000 * np.ptp(values)),
        vertical_flux_interdecile_mwb=float(
            1000 * np.diff(np.quantile(values, [0.05, 0.95]))[0]
        ),
        central_current_density_a_per_m2=central_j,
        shoulder_current_density_a_per_m2=shoulder_j,
        central_to_shoulder_density=central_j / shoulder_j if shoulder_j else None,
        continuous_middle_current_density_a_per_m2=float(continuous_j),
        central_nodes=int(central.sum()),
        shoulder_nodes=int(shoulder.sum()),
    )


def boundary_curve(radius, height, flux, anchor, level, peak_flux=None):
    """Read the 99.9% closed surface enclosing a known plasma interior point.

    The finite marching grid cannot resolve the exact tangent saddle crossing.
    The boundary-gap receipt therefore declares this small inward flux offset;
    the null offset itself is always read from the continuous interpolant."""
    field = spline(radius, height, flux)
    axis_value = float(field.ev(*anchor)) if peak_flux is None else peak_flux
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
        and PolygonPath(line).contains_point(anchor)
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
    receipt = entry["observation"]
    if receipt["source_receipt_sha256"] != entry["files"]["receipt.json"]["sha256"]:
        raise ValueError("portable observation does not bind to its source receipt")
    if receipt["storage_order"] != "radius_height":
        raise ValueError("only declared radius-height storage is admitted")
    return data, receipt


def analyze(entry):
    """Decompose a saved field, recording limitations of the current read."""
    data, receipt = read_case(entry)
    radius, height = data["radius"], data["height"]
    reference, solved = data["label"], data["flux"]
    ref_field, sol_field = [spline(radius, height, v) for v in (reference, solved)]
    ref_mask = polygon_mask(radius, height, data["label_boundary"])
    solved_mask = np.asarray(data["connected"], bool)
    solved_core = critical_census(radius, height, solved, solved_mask)
    reference_core = critical_census(
        radius, height, reference, ref_mask, data["label_boundary"]
    )
    ref_saddle_read = polish(
        reference, radius, height, receipt["efit_x_point_rz_m"], True
    )
    saddle_read = polish(solved, radius, height, data["continued_saddle"][:2], True)
    if not ref_saddle_read["valid"] or not saddle_read["valid"]:
        raise ValueError("lower saddle failed its owner's qualification")
    ref_saddle = np.asarray(ref_saddle_read["rz_m"])
    saddle = np.asarray(saddle_read["rz_m"])
    ref_hessian = np.array(
        [
            [ref_field.ev(*ref_saddle, dx=2), ref_field.ev(*ref_saddle, dx=1, dy=1)],
            [ref_field.ev(*ref_saddle, dx=1, dy=1), ref_field.ev(*ref_saddle, dy=2)],
        ],
        float,
    )
    axis = (
        np.asarray(solved_core["in_plasma_maxima"][0]["rz_m"])
        if solved_core["axis_determined"]
        else None
    )
    ref_axis = (
        np.asarray(reference_core["in_plasma_maxima"][0]["rz_m"])
        if reference_core["axis_determined"]
        else None
    )
    axis_offset = axis - ref_axis if axis is not None and ref_axis is not None else None
    saddle_offset = saddle - ref_saddle
    fit = fit_translation(radius, height, solved, reference, ref_mask)
    current = current_and_force(radius, height, solved, solved_mask)
    reference_current = current_and_force(radius, height, reference, ref_mask)
    solved_shape = core_shape(radius, height, solved, solved_core, current)
    reference_shape = core_shape(
        radius, height, reference, reference_core, reference_current
    )
    ref_boundary = data["label_boundary"]
    sol_boundary = boundary_curve(
        radius,
        height,
        solved,
        np.asarray(current["centroid_rz_m"]),
        float(data["boundary"][2]),
        solved_core["max_flux_wb"],
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
        axis_offset_cm=(100 * axis_offset).tolist()
        if axis_offset is not None
        else None,
        solved_core=solved_core,
        reference_core=reference_core,
        solved_core_shape=solved_shape,
        reference_core_shape=reference_shape,
        core_flatness_ratio=solved_shape["vertical_flux_interdecile_mwb"]
        / reference_shape["vertical_flux_interdecile_mwb"],
        xpoint_offset_cm=(100 * saddle_offset).tolist(),
        rigid_shift_residual_cm=(100 * (saddle_offset - axis_offset)).tolist()
        if axis_offset is not None
        else None,
        xpoint_after_translation_cm=(
            100 * (saddle_offset + fit["translation_m"])
        ).tolist(),
        axis_rz_m=axis.tolist() if axis is not None else None,
        saddle_rz_m=saddle.tolist(),
        reference_axis_rz_m=ref_axis.tolist() if ref_axis is not None else None,
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
        verdict=classify_offsets(saddle_offset, fit),
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
            json.dumps(
                dict(
                    rows=rows,
                    forward_solves=0,
                    source_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(row), flush=True)
    print(f"MEASUREMENT: {len(rows)} saved states; zero forward solves", flush=True)


if __name__ == "__main__":
    main()
