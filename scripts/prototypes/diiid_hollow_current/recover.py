"""Recover original source inputs and evaluate saved fields without solving."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=lambda v: np.asarray(v).tolist()) + "\n"
    )


def native_table(function):
    """Read the actual interpolation nodes from either supported representation."""
    if hasattr(function, "coordinate"):
        return np.asarray(function.coordinate), np.asarray(function.values)
    leaves = dict(
        zip(
            function.__code__.co_freevars,
            (cell.cell_contents for cell in function.__closure__),
            strict=True,
        )
    )
    return np.asarray(leaves["grid"]), np.asarray(leaves["samples"])


class SolveGuard:
    """Count attempted forward solves and refuse them before execution."""

    def __init__(self):
        self.calls = 0

    def refuse(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("forward solve forbidden during source recovery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--product-tree", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.product_tree))
    spec = importlib.util.spec_from_file_location("hollow_trace", HERE / "trace.py")
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    import jax
    import jax.numpy as jnp
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64
    from nova.equilibrium.forward import ForwardProfile
    from scripts.prototypes.diiid_residual_plateau import measure as capture
    from scripts.prototypes.diiid_xpoint_offset.decompose import (
        spline,
        critical_census,
        polygon_mask,
        current_and_force,
    )
    from nova.equilibrium.map_extraction import apply_delta_star
    from scipy.interpolate import RegularGridInterpolator

    calls, paths, rows, closure_bank = (
        [],
        {HERE.parent / "diiid_xpoint_offset/results.json"},
        [],
        [],
    )
    audit_active = True

    def audit(event, values):
        if audit_active and event == "open" and isinstance(values[0], str):
            path = Path(values[0])
            if path.is_absolute() and (
                ".zarr/" in str(path)
                or path.suffix in (".parquet", ".nc", ".npz", ".h5")
            ):
                if not str(path).startswith(str(args.output)):
                    paths.add(path)

    sys.addaudithook(audit)

    guard = SolveGuard()

    def refuse(*a, **kw):
        calls.append("ForwardProfile.solve")
        write_json(args.output / "solve-calls.json", calls)
        return guard.refuse(*a, **kw)

    ForwardProfile.solve = refuse
    original_read = capture.measure.bench._read

    def read(path, *a, **kw):
        paths.add(Path(path))
        return original_read(path, *a, **kw)

    capture.measure.bench._read = read
    from benchmarks import efit_forward_parity_slice as mast

    mast_context = []
    original_mast = mast._mast_case_from_selection

    def mast_case(*a, **kw):
        case, context = original_mast(*a, **kw)
        mast_context.append(context)
        return case, context

    mast._mast_case_from_selection = mast_case
    source_revision = subprocess.check_output(
        ["git", "-C", str(args.product_tree), "rev-parse", "HEAD"], text=True
    ).strip()
    geometry_rows = json.loads(
        (HERE.parent / "diiid_xpoint_offset/results.json").read_text()
    )["rows"]
    for case, frame, arm in [
        ("frame-137-diverted", 137, "head"),
        ("frame-146-diverted", 146, "head"),
        ("mast-21978-35", 35, "base"),
    ]:
        print("PREPARE", case, flush=True)
        prepared = (
            capture.prepare_mast()
            if frame == 35
            else capture.measure._prepare_frame(
                frame, capture.measure.bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
            )
        )
        profile = prepared["profile"]
        op, core = profile.operator, profile.operator.source.core
        folder = args.bank_root / "titan-reader-repaired/measurement" / arm / case
        terminal = {
            key: value for key, value in np.load(folder / "terminal.npz").items()
        }
        radius, height = terminal["radius"], terminal["height"]
        np.testing.assert_array_equal(np.asarray(profile.lattice.radius), radius)
        np.testing.assert_array_equal(np.asarray(profile.lattice.height), height)
        np.testing.assert_allclose(
            prepared["label"], terminal["label"], rtol=0, atol=1e-13
        )
        native = {}
        for name in ("p_prime", "ff_prime"):
            nodes, values = native_table(getattr(core, name))
            native[name] = dict(coordinate=nodes, values=values)
        shape = trace.profile_shape(float(terminal["reference_axis"][0]), core)
        provenance = dict(
            case=case,
            arm=arm,
            product_revision=source_revision,
            target_current_a=float(prepared["target_current_a"]),
            source_path=prepared.get("row", {}).get("_source_path"),
            representation=type(core.p_prime).__name__,
            profiles=native,
            profile_shape=shape,
        )
        if frame != 35:
            import pyarrow.parquet as pq

            archive = Path(provenance["source_path"])
            paths.add(archive)
            names = pq.read_schema(archive).names
            with archive.open("rb") as stream:
                provenance["source_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
            provenance["archive_profile_columns"] = [
                name
                for name in names
                if any(
                    token in name.lower()
                    for token in ("pprime", "ffprime", "p_prime", "ff_prime")
                )
            ]
            provenance["archive_positive_control"] = [
                name
                for name in ("efit_psirz", "efit_r_axis", "efit_lcfs_r")
                if name in names
            ]
        else:
            context = mast_context[-1]
            group, index = context["group"], context["row"]
            provenance["source_path"] = str(mast.SHOT_STORE / "21978.zarr")
            provenance["archive_profile_comparison"] = {}
            for field, name in [("pprime", "p_prime"), ("ffprime", "ff_prime")]:
                raw = np.asarray(group[field][index], float)
                expected = -raw / (2 * np.pi)
                provenance["archive_profile_comparison"][name] = dict(
                    conversion=-1 / (2 * np.pi),
                    max_absolute_difference=float(
                        np.max(np.abs(native[name]["values"] - expected))
                    ),
                )
        closure_bank.append(provenance)
        write_json(args.output / "closures.json", closure_bank)
        print("CLOSURE", case, json.dumps(shape), flush=True)

        @jax.jit
        def operator_read(state):
            masks, topology, connected, admitted = op._fixed_design_read(state)
            moments = op.cell_current_moments(state)
            unscaled = jnp.sum(moments.cell_current)
            amplitude = op.current_normalisation_amplitude(
                prepared["target_current_a"], unscaled
            )
            return masks, topology, connected, admitted, unscaled, amplitude

        geometry = next(
            r for r in geometry_rows if r["case"] == case and r["arm"] == arm
        )
        ref_state = np.asarray(
            prepared["seed"] if frame == 35 else prepared["label_seed"]
        ).copy()
        count = terminal["flux"].size
        ref_state[:count] = terminal["label"].ravel()
        stages = [("reference", ref_state), ("terminal", terminal["state"])]
        inner_paths = sorted(
            folder.glob("inner-*.npz"), key=lambda p: int(p.stem.split("-")[-1])
        )
        with np.load(inner_paths[-1]) as last:
            scaled = last["state"][: terminal["state"].size]
        scale = float(np.dot(terminal["state"], scaled) / np.dot(scaled, scaled))
        np.testing.assert_allclose(
            scaled * scale, terminal["state"], rtol=1e-11, atol=1e-11
        )
        for i, path in enumerate(inner_paths):
            with np.load(path) as data:
                if i == 0:
                    stages.append(
                        (
                            "trip-0-initial",
                            data["initial"][: terminal["state"].size] * scale,
                        )
                    )
                stages.append(
                    (
                        f"trip-{i}-returned",
                        data["state"][: terminal["state"].size] * scale,
                    )
                )
        first_nonmonotonic = None
        state_rows = []
        for stage, state in stages:
            print("READ", case, stage, flush=True)
            masks, topology, connected, admitted, unscaled, amplitude = jax.device_get(
                operator_read(jnp.asarray(state))
            )
            flux = state[:count].reshape(terminal["flux"].shape)
            field = spline(radius, height, flux)
            if frame == 35:
                axis_flux, boundary_flux = (
                    float(op.declared_axis_flux),
                    float(op.declared_boundary_flux),
                )
                support = np.asarray(op.declared_support).reshape(flux.shape)
            else:
                axis_flux, boundary_flux = (
                    float(topology.axis_flux),
                    float(topology.boundary_flux),
                )
                support = np.asarray(masks.profile_participation).reshape(flux.shape)
            coordinate = (flux - axis_flux) / (boundary_flux - axis_flux)
            rr, zz = np.meshgrid(radius, height, indexing="ij")
            terms = trace.split_current(rr, coordinate, core, float(amplitude), support)
            map_density = apply_delta_star(
                radius, height, flux
            ).toroidal_current_density
            map_centre = np.asarray(
                geometry["reference_current" if stage == "reference" else "current"][
                    "centroid_rz_m"
                ]
            )
            core_r = float(
                geometry[
                    "reference_core_shape"
                    if stage == "reference"
                    else "solved_core_shape"
                ]["cut_radius_m"]
            )
            current = current_and_force(radius, height, flux, support)
            density_masks = (
                polygon_mask(radius, height, terminal["label_boundary"])
                if stage == "reference"
                else support
            )
            # The census is needed only until the first non-monotonic trip.
            census = None
            if stage.startswith("trip") and first_nonmonotonic is None:
                census = critical_census(
                    radius, height, flux, density_masks, terminal["label_boundary"]
                )
                if census["maxima_count"] > 1 or census["minima_count"] > 0:
                    first_nonmonotonic = stage
            cuts = {}
            membership = RegularGridInterpolator(
                (radius, height),
                support.astype(float),
                method="nearest",
                bounds_error=False,
                fill_value=0,
            )
            for cut in ("midplane", "core"):
                x = (
                    np.linspace(radius[0], radius[-1], 601)
                    if cut == "midplane"
                    else np.linspace(map_centre[1] - 0.45, map_centre[1] + 0.45, 601)
                )
                r = x if cut == "midplane" else np.full_like(x, core_r)
                z = (
                    np.full_like(x, float(terminal["reference_axis"][1]))
                    if cut == "midplane"
                    else x
                )
                norm = (field.ev(r, z) - axis_flux) / (boundary_flux - axis_flux)
                cut_terms = trace.split_current(
                    r, norm, core, float(amplitude), membership(np.c_[r, z]) > 0.5
                )
                cut_terms["map_density"] = spline(
                    radius[2:-2], height[2:-2], map_density[2:-2, 2:-2]
                ).ev(r, z)
                cuts[cut] = dict(x=x, **cut_terms)
            central = (np.abs(rr - core_r) <= 1.6 * np.diff(radius).mean()) & (
                np.abs(zz - map_centre[1]) <= 0.1
            )
            shoulder = (
                (np.abs(rr - core_r) <= 1.6 * np.diff(radius).mean())
                & (np.abs(zz - map_centre[1]) >= 0.15)
                & (np.abs(zz - map_centre[1]) <= 0.35)
            )
            ratio = float(
                np.median(terms["density"][central])
                / np.median(terms["density"][shoulder])
            )
            row = dict(
                stage=stage,
                map_current=current,
                axis_flux_wb=axis_flux,
                boundary_flux_wb=boundary_flux,
                axis_rz_m=np.asarray(topology.axis),
                axis_admitted=bool(admitted),
                unscaled_current_a=float(unscaled),
                amplitude=float(amplitude),
                source_central_to_shoulder=ratio,
                central_support_fraction=float(support[central].mean()),
                psi_norm_central_range=[
                    float(coordinate[central].min()),
                    float(coordinate[central].max()),
                ],
                census=census,
                cuts=cuts,
            )
            state_rows.append(row)
            np.savez_compressed(
                args.output / f"{case}-{stage}.npz",
                **{key: np.asarray(value) for key, value in terms.items()},
                flux=flux,
                map_density=map_density,
                radius=radius,
                height=height,
            )
            write_json(
                args.output / (case + ".json"),
                dict(
                    **provenance,
                    first_nonmonotonic=first_nonmonotonic,
                    flux_scale=scale,
                    states=state_rows,
                    solve_calls=len(calls),
                ),
            )
            print(
                "STATE",
                case,
                stage,
                "amplitude",
                float(amplitude),
                "source_ratio",
                ratio,
                "central_support",
                row["central_support_fraction"],
                "nonmonotonic",
                first_nonmonotonic,
                flush=True,
            )
        rows.append(
            dict(
                case=case,
                first_nonmonotonic=first_nonmonotonic,
                profile_shape=shape,
                states=len(state_rows),
            )
        )
    audit_active = False
    modules = {
        name: str(module.__file__)
        for name, module in sys.modules.items()
        if getattr(module, "__file__", None)
        and (
            name.startswith("nova.")
            or name.startswith("benchmarks.")
            or name.startswith("scripts.")
        )
    }
    paths.update(Path(path) for path in modules.values() if path.endswith(".py"))
    write_json(args.output / "imported-modules.json", modules)
    inventory = []
    for path in sorted(paths):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            inventory.append(
                dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
            )
    write_json(args.output / "reference-inputs.json", inventory)
    write_json(
        args.output / "summary.json",
        dict(rows=rows, solve_calls=len(calls), product_revision=source_revision),
    )
    assert len(calls) == 0
    print("SOLVE_CALLS=0", flush=True)


if __name__ == "__main__":
    main()
