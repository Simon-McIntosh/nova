"""Check local saddle linearization and frozen-source equilibrium response.

The homotopy is the straight line between saved fields, with no forward solve.
Its integrated local Hessian responses are path-dependent diagnostic shares.
The separate source tangent uses fixed node cells, sampled profiles, a common
connected support envelope and moving flux anchors. Its representation error
is reported as closure, never silently assigned to an input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import make_interp_spline
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root

from scripts.prototypes.diiid_xpoint_attribution.attribute import (
    attribute_fields,
    field,
    gradient,
    plasma_mask,
    saddle,
    write,
)


def path_response(data, result, count):
    radius, height = data["radius"], data["height"]
    reference, solved = data["reference"], data["solved"]
    names = ["profiles", "total_current", "external"] + (
        ["plasma_response", "boundary_shadow"]
        if "boundary_shadow" in data
        else ["redistribution_closure"]
    )
    fields = {name: field(radius, height, data[name]) for name in names}
    nodes, weights = np.polynomial.legendre.leggauss(count)
    nodes, weights = (nodes + 1) / 2, weights / 2
    shares = {name: np.zeros(2) for name in names}
    point = result["reference_saddle_m"]
    trajectory = []
    for fraction, weight in zip(nodes, weights, strict=True):
        flux = reference + fraction * (solved - reference)
        point, hessian = saddle(radius, height, flux, point)
        for name, interpolant in fields.items():
            shares[name] += weight * -np.linalg.solve(
                hessian, gradient(interpolant, point)
            )
        trajectory.append(
            dict(fraction=fraction, saddle_m=point, condition=np.linalg.cond(hessian))
        )
    return dict(shifts_m=shares, total_m=sum(shares.values()), trajectory=trajectory)


def interpolation_weights(radius, height, point):
    radial = make_interp_spline(radius, np.eye(len(radius)), k=3)(point[0])
    vertical = make_interp_spline(height, np.eye(len(height)), k=3)(point[1])
    return np.outer(radial, vertical).ravel()


def source_tangent(data, result, green, fraction):
    from nova.equilibrium.solve_request import SampledFluxFunction

    radius, height = data["radius"], data["height"]
    flux = data["reference"] + fraction * (data["solved"] - data["reference"])
    interpolant = field(radius, height, flux)
    axis_guess = (1 - fraction) * data["reference_axis"] + fraction * data[
        "solved_axis"
    ]
    axis_read = root(lambda point: gradient(interpolant, point), axis_guess, tol=1e-11)
    if np.linalg.norm(axis_read.fun) > 1e-8:
        raise ValueError("homotopy axis failed its stationary-point read")
    axis = axis_read.x
    xpoint, _ = saddle(radius, height, flux, result["reference_saddle_m"])
    axis_weight = interpolation_weights(radius, height, axis)
    edge_weight = interpolation_weights(radius, height, xpoint)
    axis_flux, edge_flux = axis_weight @ flux.ravel(), edge_weight @ flux.ravel()
    span = edge_flux - axis_flux
    coordinate = (flux.ravel() - axis_flux) / span
    rr = np.broadcast_to(radius[:, None], flux.shape).ravel()
    source = {
        name: SampledFluxFunction(item["coordinate"], item["values"])
        for name, item in result["input_profiles"].items()
    }
    envelope = plasma_mask(radius, height, data["label_boundary"])
    # The solved closed surface and the EFIT polygon define a common envelope.
    from scripts.prototypes.diiid_xpoint_offset.decompose import boundary_curve

    sol_boundary = boundary_curve(
        radius,
        height,
        data["solved"],
        data["solved_axis"],
        float(field(radius, height, data["solved"]).ev(*result["solved_saddle_m"])),
    )
    envelope |= plasma_mask(radius, height, sol_boundary)
    selected = envelope.ravel()

    def density(norm, hold_edge=False):
        if hold_edge:
            profiles = {
                name: np.interp(norm, item["coordinate"], item["values"])
                for name, item in result["input_profiles"].items()
            }
        else:
            profiles = {
                name: np.asarray(function(norm)) for name, function in source.items()
            }
        values = (
            -2 * np.pi * (rr * profiles["p_prime"] + profiles["ff_prime"] / (mu_0 * rr))
        )
        return np.where(selected & (norm < 1), values, 0)

    values = density(coordinate)
    derivative = (density(coordinate + 1e-5) - density(coordinate - 1e-5)) / 2e-5
    alternate = (density(coordinate + 3e-6) - density(coordinate - 3e-6)) / 6e-6
    derivative_difference = np.linalg.norm(derivative - alternate) / max(
        np.linalg.norm(derivative), 1e-30
    )
    norm_derivative = (
        np.eye(flux.size)
        - axis_weight[None, :]
        - coordinate[:, None] * (edge_weight - axis_weight)[None, :]
    ) / span
    source_derivative = derivative[:, None] * norm_derivative
    area = np.diff(radius).mean() * np.diff(height).mean()
    target = result["current"]["prescribed_a"]
    amplitude = target / (values.sum() * area)
    normalized_derivative = amplitude * (
        source_derivative
        - values[:, None] * source_derivative.sum(axis=0)[None, :] / values.sum()
    )
    tangent = green @ (area * normalized_derivative)
    image = green @ (amplitude * values * area)
    held = density(coordinate, True)
    held_image = green @ (target * held / held.sum())
    return (
        tangent,
        image,
        dict(
            derivative_relative_check=derivative_difference,
            axis_m=axis,
            saddle_m=xpoint,
            held_edge_forcing_wb=held_image - image,
        ),
    )


def analyze(folder, frame):
    data = dict(np.load(folder / f"fields-{frame}.npz"))
    result = json.loads((folder / f"result-{frame}.json").read_text())
    radius, height = data["radius"], data["height"]
    paths = [path_response(data, result, count) for count in (16, 32)]
    delta = np.asarray(result["measured_offset_cm"]) / 100
    path = paths[-1]
    path["quadrature_difference_m"] = {
        name: np.asarray(value) - paths[0]["shifts_m"][name]
        for name, value in path["shifts_m"].items()
    }
    path["closure_m"] = path["total_m"] - delta
    green = np.load(folder / f"green-{frame}.npz")["green"]
    nodes, weights = np.polynomial.legendre.leggauss(12)
    mean_tangent = np.zeros_like(green)
    derivative_checks = []
    for node, weight in zip((nodes + 1) / 2, weights / 2, strict=True):
        tangent, _, diagnostic = source_tangent(data, result, green, float(node))
        mean_tangent += weight * tangent
        derivative_checks.append(diagnostic["derivative_relative_check"])
    _, initial, _ = source_tangent(data, result, green, 0)
    _, terminal, terminal_diagnostic = source_tangent(data, result, green, 1)
    matrix = np.eye(len(green)) - mean_tangent
    factor = lu_factor(matrix)
    flux_delta = (data["solved"] - data["reference"]).ravel()
    tangent_error = (terminal - initial) - mean_tangent @ flux_delta
    ref_total = result["current"]["reference_operator_grid"]["total_a"]
    target = result["current"]["prescribed_a"]
    ref_plasma = data["reference_plasma"].ravel()
    forcings = dict(
        profiles=initial * ref_total / target - ref_plasma,
        total_current=initial * (1 - ref_total / target),
        external=data["external"].ravel(),
    )
    response_fields = {
        name: lu_solve(factor, value).reshape(data["reference"].shape)
        for name, value in forcings.items()
    }
    response_fields["model_closure"] = (data["solved"] - data["reference"]) - sum(
        response_fields.values()
    )
    response = attribute_fields(
        radius, height, data["reference"], result["reference_saddle_m"], response_fields
    )
    held_correction = lu_solve(
        factor, terminal_diagnostic["held_edge_forcing_wb"]
    ).reshape(data["reference"].shape)
    held_shift = attribute_fields(
        radius,
        height,
        data["solved"],
        result["solved_saddle_m"],
        dict(edge_hold=held_correction),
    )
    sensitivity = dict(
        path=path,
        source_response=response,
        source_tangent_condition=float(np.linalg.cond(matrix)),
        source_tangent_secant_relative_error=float(
            np.linalg.norm(tangent_error) / np.linalg.norm(terminal - initial)
        ),
        source_derivative_checks=derivative_checks,
        source_model_terminal_flux_rms_wb=float(
            np.sqrt(
                np.mean(
                    (data["solved"].ravel() - data["vacuum"].ravel() - terminal) ** 2
                )
            )
        ),
        held_edge_correction=held_shift,
        held_edge_predicted_residual_cm=100 * (delta + held_shift["total_shift_m"]),
    )
    np.savez_compressed(
        folder / f"sensitivity-fields-{frame}.npz",
        **response_fields,
        held_edge_correction=held_correction,
    )
    write(folder / f"sensitivity-{frame}.json", sensitivity)
    print(
        "SENSITIVITY",
        frame,
        json.dumps(
            dict(
                path_shares_cm={
                    name: (100 * value).tolist()
                    for name, value in path["shifts_m"].items()
                },
                path_closure_cm=(100 * path["closure_m"]).tolist(),
                source_shares_cm={
                    name: (100 * value).tolist()
                    for name, value in response["shifts_m"].items()
                },
                source_condition=sensitivity["source_tangent_condition"],
                secant_error=sensitivity["source_tangent_secant_relative_error"],
                held_edge_residual_cm=sensitivity[
                    "held_edge_predicted_residual_cm"
                ].tolist(),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    args = parser.parse_args()
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    for frame in (137, 146):
        analyze(args.measurement, frame)


if __name__ == "__main__":
    main()
