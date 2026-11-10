"""Evaluate reference-state map fidelity and reference-seeded Newton solves.

The map and solver are the registered benchmark's, without product changes.
The raw residual includes the additive flux gauge; the centred companion removes
one median constant only for diagnosis. Force is cell-current quadrature of the
field imaged by the map, with a reference-gradient companion. Its scale is
mu_0 I_p**2 / 2, the characteristic thin-ring hoop scale without the geometry
logarithm. This is a force scale, not a calculated net radial force.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root

from benchmarks import diiid_forward_gs_match as bench
from nova.equilibrium.conservation import poloidal_field
from nova.equilibrium.fixed_point import _relative_residual
from nova.equilibrium.map_extraction import apply_delta_star
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes
from scripts.diiid_map_fidelity.audit import continue_saddle

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def prepare_diiid(frame):
    selected = bench.select_frames(
        sorted(bench.DEFAULT_DATA.glob("*.parquet")),
        bench.EXECUTION_FRAME_COUNT,
        bench.polarity_population(),
    )
    item = next(item for item in selected if item.frame == frame)
    row = bench._read(
        item.path,
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS,
    )
    row["_source_path"] = str(item.path)
    built = bench._build_profile(row, frame)
    machine = bench.dataset_machine_description(row, source_row=str(item.path))
    time_ms = float(row["efit_times"][frame])
    shipped = bench.shipped_current_at(
        row, machine.physical, bench.POLOIDAL_CONDUCTORS, time_ms
    )
    adapter = bench.complete_profile_current_adapter(
        built.profile,
        shipped_names=bench.POLOIDAL_CONDUCTORS,
        shipped_current_a=shipped,
        use_circuit=True,
    )
    n = int(row["efit_lcfs_n"][frame])
    contour = np.c_[row["efit_lcfs_r"][frame][:n], row["efit_lcfs_z"][frame][:n]]
    return dict(
        profile=adapter.profile,
        seed=built.seed,
        label=built.label,
        wall=built.wall,
        contour=contour,
        axis=np.array([row["efit_r_axis"][frame], row["efit_z_axis"][frame]]),
        saddle=contour[np.argmin(contour[:, 1])],
        current=np.asarray(adapter.resolution.current(())),
        target=bench._target_current(row, time_ms),
        row=row,
        frame=frame,
        identity=f"diiid-{item.path.stem}-{frame}",
        surface_receipt=built.surface_receipt,
    )


def prepare_mast():
    from benchmarks import efit_forward_parity_slice as mast

    case, context = mast._mast_case_from_selection(
        mast.SHOT_STORE, {"shot": 21978, "slice_index": 35}, None, grid_points=33
    )
    case, profile, _ = mast._passive_inclusive_case(case, context)
    group = context["group"]
    contour = np.asarray(case["boundary"])
    return dict(
        profile=profile,
        seed=np.asarray(case["state"]),
        label=np.asarray(context["reference_flux"]),
        wall=np.c_[np.asarray(group["limiterr"]), np.asarray(group["limiterz"])],
        contour=contour,
        axis=np.asarray(case["axis"]),
        saddle=contour[np.argmin(contour[:, 1])],
        current=np.asarray(profile.operator.external_current),
        target=abs(case["reference"]["plasma_current_a"]),
        identity="mast-21978-35",
        surface_receipt={},
    )


def defect(mapped, seed):
    difference = np.asarray(mapped) - np.asarray(seed)
    scale = float(np.max(np.abs(mapped)))
    gauge = float(np.median(difference))
    value = float(np.max(np.abs(difference)) / scale)
    np.testing.assert_allclose(
        value, float(_relative_residual(mapped, seed)), rtol=2e-15
    )
    return dict(
        relative=value,
        max_abs_wb=float(np.max(np.abs(difference))),
        denominator_wb=scale,
        median_gauge_wb=gauge,
        centred_relative=float(np.max(np.abs(difference - gauge)) / scale),
    )


def polish(flux, radius, height, guess, saddle):
    spline = RectBivariateSpline(radius, height, flux)
    result = root(lambda x: [spline.ev(*x, dx=1), spline.ev(*x, dy=1)], guess)
    point = result.x
    determinant = float(
        spline.ev(*point, dx=2) * spline.ev(*point, dy=2)
        - spline.ev(*point, dx=1, dy=1) ** 2
    )
    valid = bool(
        result.success
        and np.linalg.norm(result.fun) < 1e-7
        and radius[0] <= point[0] <= radius[-1]
        and height[0] <= point[1] <= height[-1]
        and (determinant < 0 if saddle else determinant > 0)
    )
    return dict(
        rz_m=point.tolist(),
        valid=valid,
        gradient_norm=float(np.linalg.norm(result.fun)),
        hessian_determinant=determinant,
    )


def map_measure(prepared, output):
    profile = prepared["profile"]
    op = profile.operator
    seed = jnp.asarray(prepared["seed"])
    radius, height = (
        np.asarray(profile.lattice.radius),
        np.asarray(profile.lattice.height),
    )
    count = profile.lattice.node_count
    label = np.asarray(seed[:count]).reshape(profile.lattice.shape)
    external = op.external(jnp.asarray(prepared["current"]))
    mapped_fn = jax.jit(
        profile.flux_map(
            jnp.asarray(prepared["current"]), target_current=prepared["target"]
        )
    )
    integrals = jax.jit(
        lambda state: profile._integral_state(state, target_current=prepared["target"])
    )
    rows = []
    arrays = dict(
        radius=radius,
        height=height,
        label=label,
        wall=prepared["wall"],
        label_contour=prepared["contour"],
        seed=np.asarray(seed),
    )
    spline = RectBivariateSpline(radius, height, label)
    for dz in (0.0, -0.02, 0.02):
        state = seed
        if dz:
            grid_r, grid_z = np.meshgrid(radius, height, indexing="ij")
            wall = np.asarray(op.wall.coordinate)
            shifted = np.r_[
                spline.ev(grid_r.ravel(), grid_z.ravel() - dz),
                spline.ev(wall[:, 0], wall[:, 1] - dz),
            ]
            state = jnp.asarray(shifted)
        mapped = mapped_fn(state)
        mapped.block_until_ready()
        moments, _, masks, topology, amplitude = jax.device_get(integrals(state))
        cell_current = np.asarray(moments.cell_current)
        radial, _ = poloidal_field(profile.lattice, mapped[:count])
        reference_br, _ = poloidal_field(profile.lattice, state[:count])
        factor = 2 * np.pi * np.asarray(profile.lattice.node_radius)
        force = float(-np.sum(cell_current * np.asarray(radial) * factor))
        reference_force = float(
            -np.sum(cell_current * np.asarray(reference_br) * factor)
        )
        scale = float(mu_0 * prepared["target"] ** 2 / 2)
        record = dict(
            dz_m=dz,
            **defect(mapped, state),
            force_n=force,
            reference_gradient_force_n=reference_force,
            hoop_scale_n=scale,
            force_over_hoop=force / scale,
            amplitude=float(amplitude),
            achieved_current_a=float(cell_current.sum()),
            axis_rz_m=np.asarray(topology.axis).tolist(),
            diverted=bool(topology.diverted),
            core_cells=int(np.sum(masks.core)),
        )
        rows.append(record)
        if dz == 0:
            difference = np.asarray(mapped[:count]) - label.ravel()
            worst = int(np.argmax(np.abs(difference)))
            record["maximum_grid_rz_m"] = [
                float(profile.lattice.node_radius[worst]),
                float(height[np.unravel_index(worst, label.shape)[1]]),
            ]
            arrays.update(
                mapped=np.asarray(mapped),
                defect=difference.reshape(label.shape),
                cell_current=cell_current,
            )
            unscaled = np.asarray(
                op.current_moment_image(
                    op.coupling_current_moments(
                        op.scaled_current_moments(moments, 1 / amplitude)
                    )
                )
            )
            plasma = np.asarray(mapped - external)
            arrays.update(
                plasma=plasma, external=np.asarray(external), unscaled_plasma=unscaled
            )
            record["current_target_effect"] = defect(external + unscaled, seed)
            record["target_scaling_delta_sup_wb"] = float(
                np.max(np.abs(plasma - unscaled))
            )
        write_json(output / "reference.json", clean(rows))
        print("REFERENCE", prepared["identity"], json.dumps(clean(record)), flush=True)
    arrays["reference_axis"] = polish(label, radius, height, prepared["axis"], False)[
        "rz_m"
    ]
    arrays["reference_saddle"] = polish(
        label, radius, height, prepared["saddle"], True
    )["rz_m"]
    np.savez_compressed(output / "fields.npz", **arrays)
    return rows, arrays


def decomposition(prepared, arrays, output):
    """Tare the label with its Delta-star current through the same Green image."""
    profile = prepared["profile"]
    radius, height = arrays["radius"], arrays["height"]
    label = arrays["label"]
    from matplotlib.path import Path as PolygonPath

    grid_r, grid_z = np.meshgrid(radius, height, indexing="ij")
    mask = PolygonPath(prepared["contour"]).contains_points(
        np.c_[grid_r.ravel(), grid_z.ravel()]
    )
    direct = apply_delta_star(radius, height, label)
    density = np.asarray(direct.toroidal_current_density).ravel()
    valid = np.asarray(direct.valid).ravel() & mask & np.isfinite(density)
    area = float(np.median(np.diff(radius)) * np.median(np.diff(height)))
    current = np.where(valid, density * area, 0.0)
    moments = CellCurrentMoments(
        jnp.asarray(current),
        jnp.zeros_like(jnp.asarray(current)),
        jnp.zeros_like(jnp.asarray(current)),
    )
    tare = np.asarray(
        profile.operator.current_moment_image(
            profile.operator.coupling_current_moments(moments)
        )
    )
    n = label.size
    plasma = arrays["plasma"][:n]
    external = arrays["external"][:n]
    reference_external = label.ravel() - tare[:n]
    pieces = {
        "plasma_against_label_tare": plasma - tare[:n],
        "external_against_label_remainder": external - reference_external,
        "target_scaling": plasma - arrays["unscaled_plasma"][:n],
    }
    scores = {}
    for name, values in pieces.items():
        centred = values - np.median(values)
        scores[name] = dict(
            sup_wb=float(np.max(np.abs(values))),
            centred_rms_wb=float(np.sqrt(np.mean(centred**2))),
        )
    op = profile.operator
    zero_current = np.zeros_like(prepared["current"])
    prescribed_only = np.asarray(op.external(jnp.asarray(zero_current)))
    coil = arrays["external"] - prescribed_only
    scores["coils_only"] = dict(
        sup_wb=float(np.max(np.abs(coil[:n]))), current_channels=len(zero_current)
    )
    scores["prescribed_circuits_only"] = dict(
        sup_wb=float(np.max(np.abs(prescribed_only[:n]))),
        present=op.prescribed_field is not None,
    )
    closure = (
        pieces["plasma_against_label_tare"] + pieces["external_against_label_remainder"]
    )
    np.testing.assert_allclose(closure, arrays["defect"].ravel(), atol=1e-13)
    record = dict(
        method=(
            "Delta-star current inside EFIT LCFS; "
            "node-area zeroth moments through map Green operator"
        ),
        tare_current_a=float(current.sum()),
        target_current_a=prepared["target"],
        components=scores,
        largest_centred_rms=max(
            ("plasma_against_label_tare", "external_against_label_remainder"),
            key=lambda key: scores[key]["centred_rms_wb"],
        ),
    )
    write_json(output / "decomposition.json", clean(record))
    return record


def solve(prepared, arrays, output):
    profile = prepared["profile"]
    request = bench._registered_solve_request(
        profile,
        jnp.asarray(prepared["seed"]),
        jnp.asarray(prepared["current"]),
        prepared["target"],
        carrier_identity=prepared["identity"] + ":reference",
    )
    request = replace(request, policy=replace(request.policy, warmup=0))
    write_json(
        output / "request.json",
        clean(
            request.policy.to_dict()
            if hasattr(request.policy, "to_dict")
            else vars(request.policy)
        ),
    )
    print("SOLVE", prepared["identity"], "warmup=0", flush=True)
    start = time.perf_counter()
    receipt = profile.solve(request)
    receipt.equilibrium.flux.block_until_ready()
    equilibrium = receipt.equilibrium
    state = np.asarray(equilibrium.flux)
    n = profile.lattice.node_count
    field = state[:n].reshape(profile.lattice.shape)
    axis = np.asarray(equilibrium.topology.axis)
    saddle = np.asarray(equilibrium.topology.x_point)
    reference_axis = np.asarray(arrays["reference_axis"])
    reference_saddle = np.asarray(arrays["reference_saddle"])
    branch = bool(equilibrium.topology.diverted)
    continued, connected, _ = continue_saddle(
        arrays["label"], field, arrays["radius"], arrays["height"], reference_saddle
    )
    same_branch = bool(
        branch and connected and np.linalg.norm(continued - saddle) < 1e-5
    )
    mapped = jax.jit(
        profile.flux_map(
            jnp.asarray(prepared["current"]), target_current=prepared["target"]
        )
    )(equilibrium.flux)
    record = dict(
        seconds=time.perf_counter() - start,
        residual=float(equilibrium.fixed_point.residual),
        direct_terminal_map=defect(mapped, equilibrium.flux),
        converged=bool(equilibrium.fixed_point.converged),
        qualified=bool(receipt.qualified),
        reason=int(equilibrium.fixed_point.termination_reason),
        topology="diverted" if branch else "limited",
        axis_rz_m=axis.tolist(),
        axis_delta_m=(axis - reference_axis).tolist(),
        axis_distance_m=float(np.linalg.norm(axis - reference_axis)),
        selected_saddle_rz_m=saddle.tolist(),
        admitted_same_branch=same_branch,
        xpoint_distance_m=float(np.linalg.norm(saddle - reference_saddle))
        if same_branch
        else None,
        attempted=int(receipt.attempted_newton_promotions),
        accepted=int(receipt.accepted_newton_promotions),
    )
    arrays.update(
        solved=field,
        solved_axis=axis,
        solved_saddle=saddle,
        solved_diverted=np.array(branch),
    )
    np.savez_compressed(output / "fields.npz", **arrays)
    write_json(output / "solve.json", clean(record))
    print("TERMINAL", prepared["identity"], json.dumps(clean(record)), flush=True)
    return record


def reproduce_prior(output):
    from benchmarks import diiid_root_existence as prior

    bank = json.loads((ROOT / prior.DEFAULT_OUTPUT / prior.RECEIPT_NAME).read_text())
    recorded = bank["result"]["frames"][0]
    row = prior._read(
        bench.DEFAULT_DATA / recorded["shot"],
        bench._LABEL_COLUMNS + bench._GEOMETRY_COLUMNS + bench._CURRENT_COLUMNS,
    )
    row["_source_path"] = str(bench.DEFAULT_DATA / recorded["shot"])
    radius, height = prior.canonical_axes(row)
    measured, _ = prior.evaluate_frame(
        row, recorded["frame"], prior.filament_response(radius, height)
    )
    fields = (
        "direct_operator_fixed_boundary_control",
        "fixed_boundary_profile",
        "free_boundary_profile",
    )
    comparison = {
        key: dict(
            recorded=recorded["residuals"][key]["fractional_rms"],
            measured=measured["residuals"][key]["fractional_rms"],
        )
        for key in fields
    }
    write_json(
        output / "prior-reproduction.json",
        clean(
            dict(shot=recorded["shot"], frame=recorded["frame"], comparison=comparison)
        ),
    )
    print("PRIOR", json.dumps(comparison), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_dtypes()
    assert jax.config.jax_enable_x64
    print(
        "IDENTITY",
        json.dumps(
            dict(
                revision=subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
                module=bench.__file__,
                cwd=str(Path.cwd().resolve()),
                devices=str(jax.devices()),
            )
        ),
        flush=True,
    )
    reproduce_prior(args.output)
    for frame in (None, 137, 146, 179):
        prepared = prepare_mast() if frame is None else prepare_diiid(frame)
        output = args.output / prepared["identity"]
        output.mkdir(exist_ok=True)
        write_json(
            output / "identity.json",
            clean(
                dict(
                    identity=prepared["identity"],
                    target_current_a=prepared["target"],
                    surface_receipt=prepared["surface_receipt"],
                )
            ),
        )
        rows, arrays = map_measure(prepared, output)
        components = decomposition(prepared, arrays, output)
        terminal = solve(prepared, arrays, output)
        write_json(
            output / "row.json",
            clean(dict(reference=rows, decomposition=components, terminal=terminal)),
        )
        jax.clear_caches()
    print("MEASUREMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
