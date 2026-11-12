"""Attribute saved-field saddle displacements with a frozen Green operator.

All fields are Nova total poloidal flux in Wb, radius-major. Differences and
displacements are forward minus EFIT. The linear response is -H^-1 grad(dPhi)
at the EFIT saddle. It describes a field perturbation, not a relaxed nonlinear
equilibrium. Plasma redistribution and representation closure are retained
explicitly rather than assigned to a prescribed input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root


def field(radius, height, flux):
    """Return the declared cubic saved-map interpolant."""
    value = np.asarray(flux, float)
    if value.shape != (len(radius), len(height)) or not np.isfinite(value).all():
        raise ValueError("field must be finite and radius-major")
    return RectBivariateSpline(radius, height, value, s=0)


def gradient(interpolant, point):
    return np.array([interpolant.ev(*point, dx=1), interpolant.ev(*point, dy=1)], float)


def saddle(radius, height, flux, guess):
    """Refuse an unqualified stationary point rather than returning its seed."""
    interpolant = field(radius, height, flux)
    result = root(lambda point: gradient(interpolant, point), guess, tol=1e-11)
    point = result.x
    hessian = np.array(
        [
            [interpolant.ev(*point, dx=2), interpolant.ev(*point, dx=1, dy=1)],
            [interpolant.ev(*point, dx=1, dy=1), interpolant.ev(*point, dy=2)],
        ],
        float,
    )
    if (
        np.linalg.norm(result.fun) > 1e-8
        or np.linalg.det(hessian) >= 0
        or not radius[0] < point[0] < radius[-1]
        or not height[0] < point[1] < height[-1]
    ):
        raise ValueError("saved field has no qualified saddle near the supplied seed")
    return point, hessian


def attribute_fields(radius, height, reference, point, perturbations):
    """Measure independent signed field sensitivities; never fit the offset."""
    point, hessian = saddle(radius, height, reference, point)
    if np.linalg.cond(hessian) > 1e10:
        raise ValueError("saddle Hessian is too ill-conditioned for attribution")
    gradients = {
        name: gradient(field(radius, height, value), point)
        for name, value in perturbations.items()
    }
    shifts = {
        name: -np.linalg.solve(hessian, value) for name, value in gradients.items()
    }
    return dict(
        reference_saddle_m=point,
        hessian_wb_per_m2=hessian,
        hessian_condition=float(np.linalg.cond(hessian)),
        gradient_wb_per_m=gradients,
        shifts_m=shifts,
        total_shift_m=sum(shifts.values(), start=np.zeros(2)),
        dominant=max(shifts, key=lambda name: np.linalg.norm(shifts[name])),
    )


def fit_conductors(response, difference, selected, rcond=1e-8):
    """Gauge-free, column-scaled SVD fit with a declared rank cutoff.

    The currents are diagnostic equivalent ampere-turns, never measured currents
    or a product correction. A uniform gauge is unidentifiable and is removed
    before the fit. The remainder is passive or unassigned, not proof of a vessel
    current. Column scaling makes the cutoff independent of response magnitude.
    """
    matrix = np.asarray(response)[selected]
    target = np.asarray(difference)[selected]
    means = matrix.mean(axis=0)
    matrix = matrix - means
    target = target - target.mean()
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale == 0):
        raise ValueError("a conductor response has no observable spatial variation")
    design = matrix / scale
    coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=rcond)
    currents = coefficients / scale
    fitted = (response - means) @ currents
    remainder = difference - np.mean(difference[selected]) - fitted
    return dict(
        equivalent_current_a_turn=currents,
        fitted_flux_wb=fitted,
        residual_flux_wb=remainder,
        rank=int(rank),
        columns=design.shape[1],
        rcond=rcond,
        singular_values=singular,
        scaled_condition=float(singular[0] / singular[-1]),
        retained_condition=float(singular[0] / singular[rank - 1]),
        fitted_squared_fraction=float(
            1
            - np.dot(remainder[selected], remainder[selected]) / np.dot(target, target)
        ),
    )


def operator_decomposition(
    reference,
    solved,
    vacuum,
    reference_plasma,
    source_reference,
    source_solved,
    target,
    reference_current,
    shadow,
    tolerance=1e-8,
):
    """Keep held field values separate from the physical source image.

    The physical image must close on every unshadowed node before a remainder
    can be named a shadow contribution. This positive closure check prevents
    an operator reconstruction error being assigned to a boundary input.
    """
    boundary = solved - vacuum - source_solved
    if np.max(np.abs(boundary[~shadow]), initial=0) > tolerance:
        raise ValueError("physical operator does not close on unshadowed nodes")
    return dict(
        profiles=source_reference * reference_current / target - reference_plasma,
        total_current=source_reference * (1 - reference_current / target),
        external=vacuum + reference_plasma - reference,
        plasma_response=source_solved - source_reference,
        boundary_shadow=boundary,
    )


def split_operator(folder, frame):
    """Combine native EFIT taring with the actual saved-state source images."""
    data = dict(np.load(folder / "native" / f"fields-{frame}.npz"))
    result = json.loads((folder / "native" / f"result-{frame}.json").read_text())
    count = data["reference"].size
    shape = data["reference"].shape
    saved = dict(np.load(Path(result["folder"]) / "terminal.npz"))
    seed = np.load(Path(result["folder"]) / "request-seed.npz")["state"]
    reference = dict(np.load(folder / f"operator-{frame}-reference.npz"))
    solved = dict(np.load(folder / f"operator-{frame}-solved.npz"))
    shadow = np.asarray(saved["shadow"][:count], bool).reshape(shape)
    vacuum = solved["external"][:count].reshape(shape)
    np.testing.assert_allclose(vacuum, data["vacuum"], atol=3e-14, rtol=0)
    parts = operator_decomposition(
        data["reference"],
        data["solved"],
        vacuum,
        data["reference_plasma"],
        reference["source_image"][:count].reshape(shape),
        solved["source_image"][:count].reshape(shape),
        result["current"]["prescribed_a"],
        result["current"]["reference_tare_a"],
        shadow,
    )
    np.testing.assert_allclose(
        sum(parts.values()), data["solved"] - data["reference"], atol=1e-12, rtol=0
    )
    raw = vacuum + solved["source_image"][:count].reshape(shape)
    raw_point, _ = saddle(
        data["radius"], data["height"], raw, result["solved_saddle_m"]
    )
    result["stencil_decomposition_control"] = result["attribution"]
    result["attribution"] = attribute_fields(
        data["radius"],
        data["height"],
        data["reference"],
        result["reference_saddle_m"],
        parts,
    )
    result["physical_image_saddle_m"] = raw_point
    result["physical_image_offset_cm"] = 100 * (
        raw_point - result["reference_saddle_m"]
    )
    result["physical_image_distance_cm"] = float(
        np.linalg.norm(result["physical_image_offset_cm"])
    )
    result["shadow_nodes"] = np.argwhere(shadow)
    result["unshadowed_closure_sup_wb"] = float(
        np.max(np.abs(parts["boundary_shadow"][~shadow]))
    )
    result["shadow_sup_wb"] = float(np.max(np.abs(parts["boundary_shadow"][shadow])))
    result["shadow_vs_request_seed_sup_wb"] = float(
        np.max(
            np.abs(
                saved["state"][:count][shadow.ravel()] - seed[:count][shadow.ravel()]
            )
        )
    )
    result["current"]["operator_solved_total_a"] = float(solved["source_current"].sum())
    rr, zz = np.meshgrid(data["radius"], data["height"], indexing="ij")
    physical_current = solved["source_current"]
    nonzero = physical_current > 0
    result["current"]["operator_solved"] = dict(
        total_a=float(physical_current.sum()),
        centroid_rz_m=[
            float(physical_current @ rr.ravel() / physical_current.sum()),
            float(physical_current @ zz.ravel() / physical_current.sum()),
        ],
        peaking=float(
            physical_current[nonzero].max() / physical_current[nonzero].mean()
        ),
    )
    result["held_node_values"] = [
        dict(
            index=index.tolist(),
            rz_m=[data["radius"][index[0]], data["height"][index[1]]],
            published_wb=float(data["solved"][tuple(index)]),
            source_image_wb=float(raw[tuple(index)]),
            request_seed_wb=float(seed[:count].reshape(shape)[tuple(index)]),
        )
        for index in np.argwhere(shadow)
    ]
    data.update(parts)
    data.update(physical_image=raw, shadow=shadow)
    write(folder / "native" / f"result-{frame}.json", result)
    np.savez_compressed(folder / "native" / f"fields-{frame}.npz", **data)
    print(
        "OPERATOR_ATTRIBUTION",
        frame,
        json.dumps(
            portable(
                dict(
                    physical_miss_cm=result["physical_image_distance_cm"],
                    published_miss_cm=result["measured_distance_cm"],
                    unshadowed_closure_wb=result["unshadowed_closure_sup_wb"],
                )
            )
        ),
        flush=True,
    )
    return result


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def write(path, value):
    path.write_text(json.dumps(portable(value), indent=2, allow_nan=False) + "\n")


def plasma_mask(radius, height, boundary):
    from matplotlib.path import Path as Polygon

    rr, zz = np.meshgrid(radius, height, indexing="ij")
    return (
        Polygon(boundary)
        .contains_points(np.c_[rr.ravel(), zz.ravel()])
        .reshape(rr.shape)
    )


def moments(radius, height, density, selected):
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    area = np.diff(radius).mean() * np.diff(height).mean()
    current = np.where(selected, density, 0) * area
    total = float(current.sum())
    if abs(total) < 1:
        raise ValueError("no measurable plasma current")
    return dict(
        total_a=total,
        centroid_rz_m=[
            float((current * rr).sum() / total),
            float((current * zz).sum() / total),
        ],
        peaking=float(np.max(density[selected]) / np.mean(density[selected])),
        negative_fraction=float(-current[current < 0].sum() / abs(total)),
    )


def profile_read(radius, height, flux, boundary, axis, boundary_flux):
    from nova.equilibrium.map_extraction import extract_flux_functions

    interpolant = field(radius, height, flux)
    axis_flux = float(interpolant.ev(*axis))
    coordinate = (flux - axis_flux) / (boundary_flux - axis_flux)
    result = extract_flux_functions(
        radius,
        height,
        flux,
        coordinate,
        plasma_mask=plasma_mask(radius, height, boundary),
        min_samples=6,
    )
    reliable = result.reliable & np.isfinite(result.p_prime + result.ff_prime)
    knots = result.psi_norm[reliable]
    return result, dict(
        psi_norm=result.psi_norm,
        p_prime=result.p_prime,
        ff_prime=result.ff_prime,
        reliable=reliable,
        first_knot=float(knots[0]),
        last_knot=float(knots[-1]),
        gaps=[
            list(pair)
            for pair in zip(knots[:-1], knots[1:])
            if pair[1] - pair[0] > 0.051
        ],
        condition=result.condition_number,
        projection_rms=result.projection_rms,
        p_prime_uncertainty=result.p_prime_uncertainty,
        ff_prime_uncertainty=result.ff_prime_uncertainty,
    )


def local_read(radius, height, flux, point, gauge=0):
    interpolant = field(radius, height, flux)
    grad = gradient(interpolant, point)
    return dict(
        flux_wb=float(interpolant.ev(*point) - gauge),
        br_t=float(-grad[1] / (2 * np.pi * point[0])),
        bz_t=float(grad[0] / (2 * np.pi * point[0])),
    )


def analyze(frame, folder, archive, output):
    """Tare the label using the benchmark's finite-cell Green responses."""
    import jax
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64
    from benchmarks import diiid_forward_gs_match as bench
    from benchmarks.diiid_corpus_conventions import corpus_flux_to_nova_total
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.map_extraction import apply_delta_star
    from nova.equilibrium.solve_request import SampledFluxFunction
    from nova.imas.diiid_current import circuit_current_map
    from nova.imas.diiid_description import (
        CIRCUIT_DRIVEN_CONDUCTORS,
        active_coil_response_from_imas,
    )
    from nova.imas.diiid_current import (
        DEFAULT_ACTIVE_COIL_ENTRY,
        DEFAULT_ACTIVE_COIL_DD_VERSION,
    )

    def refuse(*args, **kwargs):
        raise AssertionError(
            "forward solves are forbidden during saved-field attribution"
        )

    ForwardProfile.solve = refuse
    data = dict(np.load(folder / "terminal.npz"))
    receipt = json.loads((folder / "receipt.json").read_text())
    core = json.loads((folder / "core-analysis.json").read_text())
    sources = json.loads((folder / "source-inputs.json").read_text())
    radius, height = data["radius"], data["height"]
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    points = np.c_[rr.ravel(), zz.ravel()]
    row = bench._read(
        archive,
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
        + ("efit_li",),
    )
    row["_source_path"] = str(archive)
    fine_r, fine_z = bench.canonical_axes(row)
    fine_label = corpus_flux_to_nova_total(np.asarray(row["efit_psirz"][frame])).T
    np.testing.assert_allclose(fine_label[::2, ::2], data["label"], rtol=0, atol=1e-13)
    machine = bench.dataset_machine_description(row, source_row=str(archive))
    names, response = bench.vacuum_response(machine.physical, radius, height)
    response = np.stack([plane.T.ravel() for plane in response], axis=1)
    time = float(row["efit_times"][frame])
    currents = bench._source_current(row, machine.physical, names, time)
    shipped = dict(zip(names, currents, strict=True))
    extra_names, extra_response, extra_receipt = active_coil_response_from_imas(
        DEFAULT_ACTIVE_COIL_ENTRY,
        DEFAULT_ACTIVE_COIL_DD_VERSION,
        CIRCUIT_DRIVEN_CONDUCTORS,
        points[:, 0],
        points[:, 1],
    )
    extra_map = circuit_current_map(shipped)
    extra_currents = np.array([extra_map[name] for name in extra_names])
    extra_vacuum = extra_response.T @ extra_currents
    vacuum = (response @ currents + extra_vacuum).reshape(rr.shape)
    print("VACUUM", frame, len(names), len(extra_names), flush=True)
    green_path = output / f"green-{frame}.npz"
    if green_path.exists():
        green = np.load(green_path)["green"]
    else:
        green = bench._green_block(
            points, points, np.diff(radius).mean(), np.diff(height).mean()
        )
        np.savez_compressed(green_path, green=green)
    print("GREEN", frame, green.shape, flush=True)
    reference, solved = data["label"], data["flux"]
    ref_point, ref_hessian = saddle(
        radius, height, reference, receipt["efit_x_point_rz_m"]
    )
    sol_point, _ = saddle(
        radius, height, solved, receipt["terminal_topology"]["lower_saddle_rz_m"]
    )
    measured = sol_point - ref_point
    print(
        "REPRODUCTION",
        frame,
        "offset_cm",
        (100 * measured).tolist(),
        "distance_cm",
        100 * np.linalg.norm(measured),
        flush=True,
    )
    np.testing.assert_allclose(
        np.linalg.norm(measured),
        receipt["terminal_topology"]["lower_saddle_distance_m"],
        atol=1e-9,
        rtol=0,
    )
    ref_mask = plasma_mask(radius, height, data["label_boundary"])
    sol_mask = np.asarray(data["connected"], bool)
    ref_read = apply_delta_star(radius, height, reference)
    sol_read = apply_delta_star(radius, height, solved)
    ref_density = np.where(
        ref_mask & ref_read.valid, ref_read.toroidal_current_density, 0
    )
    sol_density = np.where(
        sol_mask & sol_read.valid, sol_read.toroidal_current_density, 0
    )
    area = np.diff(radius).mean() * np.diff(height).mean()
    ref_current = moments(radius, height, ref_density, ref_mask & ref_read.valid)
    sol_current = moments(radius, height, sol_density, sol_mask & sol_read.valid)
    fine_mask = plasma_mask(fine_r, fine_z, data["label_boundary"])
    fine_read = apply_delta_star(fine_r, fine_z, fine_label)
    fine_density = np.where(
        fine_mask & fine_read.valid, fine_read.toroidal_current_density, 0
    )
    fine_current = moments(fine_r, fine_z, fine_density, fine_mask & fine_read.valid)
    ref_plasma = (green @ (ref_density.ravel() * area)).reshape(rr.shape)
    implied_external = reference - ref_plasma
    external_difference = vacuum - implied_external
    external_gauge = float(np.mean(external_difference[ref_mask]))
    target = core["target_current_a"]
    ref_interpolant = field(radius, height, reference)
    ref_axis = data["reference_axis"]
    ref_boundary_flux = float(np.median(ref_interpolant.ev(*data["label_boundary"].T)))
    fine_interpolant = field(fine_r, fine_z, fine_label)
    fine_boundary_flux = float(
        np.median(fine_interpolant.ev(*data["label_boundary"].T))
    )
    ref_extraction, ref_profiles = profile_read(
        fine_r, fine_z, fine_label, data["label_boundary"], ref_axis, fine_boundary_flux
    )
    # Read the same saved source table and its actual core/edge evaluation rule.
    closures = {
        name: SampledFluxFunction(item["coordinate"], item["values"])
        for name, item in sources.items()
    }
    ref_norm = (reference - float(ref_interpolant.ev(*ref_axis))) / (
        ref_boundary_flux - float(ref_interpolant.ev(*ref_axis))
    )
    pressure = np.asarray(closures["p_prime"](ref_norm))
    diamagnetic = np.asarray(closures["ff_prime"](ref_norm))
    source_density = np.where(
        ref_mask, -2 * np.pi * (rr * pressure + diamagnetic / (mu_0 * rr)), 0
    )
    source_total = source_density.sum() * area
    normalized_source = source_density * ref_current["total_a"] / source_total
    profile_density = normalized_source - ref_density
    current_density = normalized_source * (target / ref_current["total_a"] - 1)
    profile_flux = (green @ (profile_density.ravel() * area)).reshape(rr.shape)
    current_flux = (green @ (current_density.ravel() * area)).reshape(rr.shape)
    response_flux = solved - vacuum - ref_plasma - profile_flux - current_flux
    perturbations = dict(
        profiles=profile_flux,
        total_current=current_flux,
        external=external_difference,
        redistribution_closure=response_flux,
        boundary_gauge=np.full(
            rr.shape, float(data["boundary"][2]) - ref_boundary_flux
        ),
    )
    attribution = attribute_fields(radius, height, reference, ref_point, perturbations)
    np.testing.assert_allclose(
        sum(perturbations[name] for name in perturbations if name != "boundary_gauge"),
        solved - reference,
        atol=1e-12,
        rtol=0,
    )
    fit_mask = plasma_mask(radius, height, data["limiter"]) & ref_read.valid
    fit = fit_conductors(response, external_difference.ravel(), fit_mask.ravel())
    fitted = fit.pop("fitted_flux_wb").reshape(rr.shape)
    residual = fit.pop("residual_flux_wb").reshape(rr.shape)
    split = attribute_fields(
        radius,
        height,
        reference,
        ref_point,
        dict(conductors=fitted, passive_unassigned=residual),
    )
    # The fitted columns must see a known current error before absence is read.
    injected = response[:, names.index("F1A")] * 1000
    control = fit_conductors(response, injected, fit_mask.ravel())
    assert np.linalg.norm(
        control["residual_flux_wb"][fit_mask.ravel()]
    ) < 1e-6 * np.linalg.norm(injected[fit_mask.ravel()])
    sol_boundary_flux = float(data["boundary"][2])
    from scripts.prototypes.diiid_xpoint_offset.decompose import boundary_curve

    sol_boundary = boundary_curve(
        radius,
        height,
        solved,
        np.asarray(sol_current["centroid_rz_m"]),
        sol_boundary_flux,
        core["census"]["max_flux_wb"],
    )
    _, sol_profiles = profile_read(
        radius,
        height,
        solved,
        sol_boundary,
        core["census"]["in_plasma_maxima"][0]["rz_m"],
        sol_boundary_flux,
    )
    input_differences = {}
    for name in ("p_prime", "ff_prime"):
        reliable = ref_profiles["reliable"]
        nodes = ref_profiles["psi_norm"][reliable]
        expected = ref_profiles[name][reliable]
        input_differences[name] = float(
            np.max(
                np.abs(
                    np.interp(
                        nodes, sources[name]["coordinate"], sources[name]["values"]
                    )
                    - expected
                )
            )
        )
    around = [
        ref_point + [dr, dz] for dr in (-0.05, 0, 0.05) for dz in (-0.05, 0, 0.05)
    ]
    near_read = [
        dict(
            rz_m=point,
            **local_read(radius, height, external_difference, point, external_gauge),
        )
        for point in around
    ]
    wall = data["limiter"]
    contact = {}
    for name, flux_map, boundary_flux in [
        ("efit", reference, ref_boundary_flux),
        ("solved", solved, sol_boundary_flux),
    ]:
        wall_values = field(radius, height, flux_map).ev(*wall.T)
        index = int(np.argmax(wall_values))
        contact[name] = dict(
            max_wall_flux_wb=float(wall_values[index]),
            rz_m=wall[index],
            boundary_flux_wb=boundary_flux,
            wall_to_boundary_wb=float(wall_values[index] - boundary_flux),
            active_limiter=bool(wall_values[index] >= boundary_flux),
        )
    result = dict(
        frame=frame,
        folder=str(folder),
        archive=str(archive),
        time_ms=time,
        converged=receipt["converged"],
        residual=receipt["residual"],
        measured_offset_cm=100 * measured,
        measured_distance_cm=100 * np.linalg.norm(measured),
        reference_saddle_m=ref_point,
        solved_saddle_m=sol_point,
        attribution=attribution,
        external_split=split,
        conductor_fit=fit,
        conductor_names=names,
        conductor_current_a_turn=currents,
        conductor_input_bindings=[
            dict(
                name=name,
                input_column=conductor.input_column,
                applied_multiplier=float(conductor.turns.applied_multiplier),
            )
            for name in names
            for conductor in machine.physical.conductors
            if conductor.name == name
        ],
        extra_names=extra_names,
        extra_current_a_turn=extra_currents,
        extra_response_receipt=extra_receipt,
        input_profiles=sources,
        reference_profiles=ref_profiles,
        solved_profiles=sol_profiles,
        input_profile_max_absolute_difference=input_differences,
        current=dict(
            reference_native=fine_current,
            reference_operator_grid=ref_current,
            solved_operator_grid=sol_current,
            prescribed_a=target,
            prescribed_source_unscaled_on_efit_a=source_total,
            normalization_at_terminal=core["amplitude"],
            archive_internal_inductance=float(row["efit_li"][frame]),
        ),
        boundary=contact,
        external_at_xpoint=local_read(
            radius, height, external_difference, ref_point, external_gauge
        ),
        external_neighborhood=near_read,
        external_gauge_removed_wb=external_gauge,
        fit_positive_control_residual_fraction=float(
            np.linalg.norm(control["residual_flux_wb"][fit_mask.ravel()])
            / np.linalg.norm(injected[fit_mask.ravel()])
        ),
        direct_correction_residual_cm=100
        * (measured - attribution["shifts_m"]["external"]),
        linearization_error_cm=100 * (measured - attribution["total_shift_m"]),
        forward_solves=0,
    )
    arrays = dict(
        radius=radius,
        height=height,
        reference=reference,
        solved=solved,
        vacuum=vacuum,
        reference_plasma=ref_plasma,
        implied_external=implied_external,
        external_difference=external_difference,
        fitted_external=fitted,
        unassigned_external=residual,
        extra_vacuum=extra_vacuum.reshape(rr.shape),
        limiter=wall,
        label_boundary=data["label_boundary"],
        reference_axis=ref_axis,
        solved_axis=core["census"]["in_plasma_maxima"][0]["rz_m"],
        **perturbations,
    )
    np.savez_compressed(output / f"fields-{frame}.npz", **arrays)
    write(output / f"result-{frame}.json", result)
    print(
        "RESULT",
        frame,
        json.dumps(
            portable(
                dict(
                    offset_cm=100 * measured,
                    shifts_cm={
                        name: 100 * value
                        for name, value in attribution["shifts_m"].items()
                    },
                    fit=fit,
                )
            )
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument(
        "--complete",
        action="store_true",
        help="also recover the actual source operator and native EFIT tare",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in json.loads(args.input_audit.read_text())["rows"]:
        for entry in item["files"]:
            with Path(entry["path"]).open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
                )
        folder = Path(item["files"][0]["path"]).parent
        archive = Path(item["files"][-1]["path"])
        rows.append(analyze(item["frame"], folder, archive, args.output))
        write(args.output / "results.json", dict(rows=rows, forward_solves=0))
    if args.complete:
        from scripts.prototypes.diiid_xpoint_attribution.operator_probe import probe
        from scripts.prototypes.diiid_xpoint_attribution.native_reference import refine
        from scripts.prototypes.diiid_xpoint_attribution.sensitivity import (
            analyze as integrate,
        )

        probe(args.output)
        completed_rows = []
        for frame in (137, 146):
            refine(args.output, frame)
            result = split_operator(args.output, frame)
            integrate(args.output / "native", frame)
            completed_rows.append(result)
        write(
            args.output / "native/results.json",
            dict(rows=completed_rows, forward_solves=0),
        )
    print("MEASUREMENT: 2 converged saved states; zero forward solves", flush=True)


if __name__ == "__main__":
    main()
