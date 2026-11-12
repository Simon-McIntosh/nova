"""Tare EFIT on its native grid and retain the strided-grid control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import mu_0

from scripts.prototypes.diiid_xpoint_attribution.attribute import (
    attribute_fields,
    field,
    fit_conductors,
    local_read,
    plasma_mask,
    write,
)


def refine(folder, frame):
    from benchmarks import diiid_forward_gs_match as bench
    from benchmarks.diiid_corpus_conventions import corpus_flux_to_nova_total
    from nova.equilibrium.map_extraction import apply_delta_star
    from nova.equilibrium.solve_request import SampledFluxFunction
    from shapely.geometry import LineString
    from scripts.prototypes.diiid_xpoint_offset.decompose import boundary_curve

    data = dict(np.load(folder / f"fields-{frame}.npz"))
    result = json.loads((folder / f"result-{frame}.json").read_text())
    row = bench._read(
        Path(result["archive"]), bench._LABEL_COLUMNS + bench._GEOMETRY_COLUMNS
    )
    row["_source_path"] = result["archive"]
    radius, height = data["radius"], data["height"]
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    points = np.c_[rr.ravel(), zz.ravel()]
    fine_r, fine_z = bench.canonical_axes(row)
    fine_flux = corpus_flux_to_nova_total(np.asarray(row["efit_psirz"][frame])).T
    fine_rr, fine_zz = np.meshgrid(fine_r, fine_z, indexing="ij")
    read = apply_delta_star(fine_r, fine_z, fine_flux)
    selected = plasma_mask(fine_r, fine_z, data["label_boundary"]) & read.valid
    density = read.toroidal_current_density[selected]
    source_points = np.c_[fine_rr[selected], fine_zz[selected]]
    fine_area = np.diff(fine_r).mean() * np.diff(fine_z).mean()
    green_path = folder / f"green-native-{frame}.npz"
    if green_path.exists():
        green = np.load(green_path)["green"]
    else:
        green = bench._green_block(
            points, source_points, np.diff(fine_r).mean(), np.diff(fine_z).mean()
        )
        np.savez_compressed(green_path, green=green)
    print("NATIVE_GREEN", frame, green.shape, flush=True)
    reference_plasma = (green @ (density * fine_area)).reshape(rr.shape)
    implied = data["reference"] - reference_plasma
    external = data["vacuum"] - implied
    reference_interpolant = field(fine_r, fine_z, fine_flux)
    axis_flux = float(reference_interpolant.ev(*data["reference_axis"]))
    edge_flux = float(np.median(reference_interpolant.ev(*data["label_boundary"].T)))
    norm = (fine_flux[selected] - axis_flux) / (edge_flux - axis_flux)
    profiles = {
        name: SampledFluxFunction(item["coordinate"], item["values"])
        for name, item in result["input_profiles"].items()
    }
    j_source = (
        -2
        * np.pi
        * (
            fine_rr[selected] * np.asarray(profiles["p_prime"](norm))
            + np.asarray(profiles["ff_prime"](norm)) / (mu_0 * fine_rr[selected])
        )
    )
    total = float(density.sum() * fine_area)
    target = result["current"]["prescribed_a"]
    normalized = j_source * total / (j_source.sum() * fine_area)
    profile_flux = (green @ ((normalized - density) * fine_area)).reshape(rr.shape)
    current_flux = (green @ (normalized * (target / total - 1) * fine_area)).reshape(
        rr.shape
    )
    redistribution = (
        data["solved"] - data["reference"] - external - profile_flux - current_flux
    )
    perturbations = dict(
        profiles=profile_flux,
        total_current=current_flux,
        external=external,
        redistribution_closure=redistribution,
        boundary_gauge=data["boundary_gauge"],
    )
    attribution = attribute_fields(
        radius, height, data["reference"], result["reference_saddle_m"], perturbations
    )
    # The nineteen physical columns are unchanged by the plasma tare resolution.
    machine = bench.dataset_machine_description(row, source_row=result["archive"])
    names, response = bench.vacuum_response(machine.physical, radius, height)
    response = np.stack([plane.T.ravel() for plane in response], axis=1)
    mask = plasma_mask(radius, height, data["limiter"])
    mask[:2] = False
    mask[-2:] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    fit = fit_conductors(response, external.ravel(), mask.ravel())
    fitted = fit.pop("fitted_flux_wb").reshape(rr.shape)
    residual = fit.pop("residual_flux_wb").reshape(rr.shape)
    split = attribute_fields(
        radius,
        height,
        data["reference"],
        result["reference_saddle_m"],
        dict(conductors=fitted, passive_unassigned=residual),
    )
    unit_sensitivity = attribute_fields(
        radius,
        height,
        data["reference"],
        result["reference_saddle_m"],
        {name: response[:, i].reshape(rr.shape) for i, name in enumerate(names)},
    )
    result["conductor_sensitivity_cm_per_ka_turn"] = {
        name: 100000 * value for name, value in unit_sensitivity["shifts_m"].items()
    }
    result["vacuum_at_xpoint"] = local_read(
        radius, height, data["vacuum"], result["reference_saddle_m"]
    )
    result["implied_external_at_xpoint"] = local_read(
        radius, height, implied, result["reference_saddle_m"]
    )
    gauge = float(external[plasma_mask(radius, height, data["label_boundary"])].mean())
    point = np.asarray(result["reference_saddle_m"])
    offsets = [point + [dr, dz] for dr in (-0.05, 0, 0.05) for dz in (-0.05, 0, 0.05)]
    result["strided_reference_control"] = dict(
        attribution=result["attribution"],
        external_at_xpoint=result["external_at_xpoint"],
        conductor_fit=result["conductor_fit"],
    )
    result.update(
        attribution=attribution,
        external_split=split,
        conductor_fit=fit,
        external_at_xpoint=local_read(radius, height, external, point, gauge),
        external_neighborhood=[
            dict(rz_m=p, **local_read(radius, height, external, p, gauge))
            for p in offsets
        ],
        external_gauge_removed_wb=gauge,
        direct_correction_residual_cm=np.asarray(result["measured_offset_cm"])
        - 100 * attribution["shifts_m"]["external"],
        linearization_error_cm=np.asarray(result["measured_offset_cm"])
        - 100 * attribution["total_shift_m"],
        reference_tare_grid=[len(fine_r), len(fine_z)],
        reference_tare_cells=int(selected.sum()),
    )
    result["current"]["reference_tare_a"] = total
    result["current"]["source_unscaled_native_a"] = float(j_source.sum() * fine_area)
    wall_curve = LineString(np.vstack((data["limiter"], data["limiter"][0])))
    sol_boundary = boundary_curve(
        radius,
        height,
        data["solved"],
        data["solved_axis"],
        float(field(radius, height, data["solved"]).ev(*result["solved_saddle_m"])),
    )
    for name, boundary in [("efit", data["label_boundary"]), ("solved", sol_boundary)]:
        result["boundary"][name]["unqualified_wall_maximum_is_contact"] = False
        result["boundary"][name].pop("active_limiter", None)
        result["boundary"][name]["closed_boundary_to_wall_m"] = float(
            LineString(np.vstack((boundary, boundary[0]))).distance(wall_curve)
        )
        result["boundary"][name]["boundary_class"] = "diverted"
        result["boundary"][name]["governing_contact_m"] = None
    data.update(
        reference_plasma=reference_plasma,
        implied_external=implied,
        external_difference=external,
        fitted_external=fitted,
        unassigned_external=residual,
        **perturbations,
    )
    destination = folder / "native"
    destination.mkdir(exist_ok=True)
    write(destination / f"result-{frame}.json", result)
    np.savez_compressed(destination / f"fields-{frame}.npz", **data)
    if not (destination / f"green-{frame}.npz").exists():
        (destination / f"green-{frame}.npz").symlink_to(
            (folder / f"green-{frame}.npz").resolve()
        )
    print(
        "NATIVE_RESULT",
        frame,
        json.dumps(
            dict(
                shifts_cm={
                    name: (100 * value).tolist()
                    for name, value in attribution["shifts_m"].items()
                },
                external=result["external_at_xpoint"],
                fit_condition=fit["scaled_condition"],
            )
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    args = parser.parse_args()
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    rows = [refine(args.measurement, frame) for frame in (137, 146)]
    write(args.measurement / "native/results.json", dict(rows=rows, forward_solves=0))


if __name__ == "__main__":
    main()
