"""Measure source continuation through the governed-wall branch cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "branch_measure", ROOT / "scripts/prototypes/diiid_diverted_branch/measure.py"
)
branch_measure = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = branch_measure
spec.loader.exec_module(branch_measure)


def load_trace(path):
    spec = importlib.util.spec_from_file_location("source_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_analysis(trace, context):
    original = branch_measure.enrich

    def enrich(output, profile, scale, record, data, branches, seeds):
        original(output, profile, scale, record, data, branches, seeds)
        import jax
        import jax.numpy as jnp
        import numpy as np
        from scipy.interpolate import RegularGridInterpolator

        from nova.equilibrium.map_extraction import apply_delta_star
        from scripts.prototypes.diiid_xpoint_offset import decompose as geometry

        op, core = profile.operator, profile.operator.source.core
        radius, height, flux = (data[key] for key in ("radius", "height", "flux"))
        support = np.asarray(data["connected"], bool)
        current = geometry.current_and_force(radius, height, flux, support)
        boundary = geometry.boundary_curve(
            radius,
            height,
            flux,
            np.asarray(current["centroid_rz_m"]),
            float(data["boundary"][2]),
            float(flux[support].max()),
        )
        census = geometry.critical_census(radius, height, flux, support, boundary)
        reference_mask = geometry.polygon_mask(radius, height, data["label_boundary"])
        reference_census = geometry.critical_census(
            radius, height, data["label"], reference_mask, data["label_boundary"]
        )
        shape = geometry.core_shape(radius, height, flux, census, current)
        _mask, topology, _connected, _admitted = jax.device_get(
            jax.jit(op._fixed_design_read)(jnp.asarray(data["state"]))
        )
        if getattr(op, "declared_support", None) is not None:
            axis_flux, boundary_flux = (
                float(op.declared_axis_flux),
                float(op.declared_boundary_flux),
            )
        else:
            axis_flux, boundary_flux = (
                float(topology.axis_flux),
                float(topology.boundary_flux),
            )
        rr, zz = np.meshgrid(radius, height, indexing="ij")
        norm = (flux - axis_flux) / (boundary_flux - axis_flux)
        moments = jax.jit(op.cell_current_moments)(jnp.asarray(data["state"]))
        unscaled = float(jnp.sum(moments.cell_current))
        amplitude = float(
            op.current_normalisation_amplitude(context["target_current"], unscaled)
        )
        source = trace.split_current(rr, norm, core, amplitude, support=support)
        central = (
            np.abs(rr - shape["cut_radius_m"]) <= 1.6 * np.diff(radius).mean()
        ) & (np.abs(zz - shape["cut_centre_z_m"]) <= 0.1)
        shoulder = (
            (np.abs(rr - shape["cut_radius_m"]) <= 1.6 * np.diff(radius).mean())
            & (np.abs(zz - shape["cut_centre_z_m"]) >= 0.15)
            & (np.abs(zz - shape["cut_centre_z_m"]) <= 0.35)
        )
        source_ratio = float(
            np.median(source["density"][central])
            / np.median(source["density"][shoulder])
        )
        x = np.linspace(radius[2], radius[-3], 601)
        z = np.full_like(x, data["reference_axis"][1])
        field = geometry.spline(radius, height, flux)
        cut_norm = (field.ev(x, z) - axis_flux) / (boundary_flux - axis_flux)
        membership = RegularGridInterpolator(
            (radius, height),
            support.astype(float),
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )
        cut_support = membership(np.c_[x, z]) > 0.5
        cut_source = trace.split_current(
            x, cut_norm, core, amplitude, support=cut_support
        )
        native = apply_delta_star(radius, height, flux).toroidal_current_density
        label_native = apply_delta_star(
            radius, height, data["label"]
        ).toroidal_current_density
        cut = dict(
            radius=x.tolist(),
            source=trace.arrays_to_lists(cut_source),
            map_density=geometry.spline(radius[2:-2], height[2:-2], native[2:-2, 2:-2])
            .ev(x, z)
            .tolist(),
            efit_support=(
                RegularGridInterpolator(
                    (radius, height),
                    reference_mask.astype(float),
                    method="nearest",
                    bounds_error=False,
                    fill_value=0,
                )(np.c_[x, z])
                > 0.5
            ).tolist(),
            efit_density=geometry.spline(
                radius[2:-2], height[2:-2], label_native[2:-2, 2:-2]
            )
            .ev(x, z)
            .tolist(),
        )
        summary = dict(
            target_current_a=context["target_current"],
            amplitude=amplitude,
            source_inputs=context["source_inputs"],
            negative_source_evaluations=context["negative_evaluations"],
            terminal_negative_nodes=dict(
                count=int(np.sum((norm < 0) & support)),
                index=np.argwhere((norm < 0) & support).tolist(),
                rz_m=np.column_stack(
                    (rr[(norm < 0) & support], zz[(norm < 0) & support])
                ).tolist(),
                psi_norm=norm[(norm < 0) & support].tolist(),
            ),
            maxima_count=census["maxima_count"],
            minima_count=census["minima_count"],
            census=census,
            reference_census=reference_census,
            source_centre_to_shoulder=source_ratio,
            map_core_shape=shape,
            closure_shape=trace.profile_shape(float(data["reference_axis"][0]), core),
            midplane=cut,
            profiles={
                name: dict(
                    coordinate=np.asarray(getattr(core, name).coordinate).tolist(),
                    values=np.asarray(getattr(core, name).values).tolist(),
                )
                for name in ("p_prime", "ff_prime")
                if hasattr(getattr(core, name), "coordinate")
            },
        )
        (output / "core-analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            "CORE_SUMMARY "
            + json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "maxima_count",
                        "minima_count",
                        "source_centre_to_shoulder",
                        "closure_shape",
                    )
                }
            ),
            flush=True,
        )

    branch_measure.enrich = enrich


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.jax.config import configure_dtypes
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.solve_request import ForwardSolveRequest, SampledFluxFunction

    configure_dtypes()
    assert jax.config.jax_enable_x64
    context = dict(source_inputs={}, negative_evaluations=[])
    solve = ForwardProfile.solve

    def measured(profile, request=None, *a, **kw):
        if isinstance(request, ForwardSolveRequest):
            context["target_current"] = float(request.target_current)
            np.savez_compressed(
                args.output / "request-seed.npz",
                state=np.asarray(request.seed_policy.state),
            )
            core = profile.operator.source.core
            context["source_inputs"] = {
                name: dict(
                    coordinate=np.asarray(getattr(core, name).coordinate).tolist(),
                    values=np.asarray(getattr(core, name).values).tolist(),
                )
                for name in ("p_prime", "ff_prime")
            }
            (args.output / "source-inputs.json").write_text(
                json.dumps(context["source_inputs"], indent=2) + "\n"
            )
        return solve(profile, request, *a, **kw)

    ForwardProfile.solve = measured
    if args.branch == "mast":
        from benchmarks import efit_forward_parity_slice as mast

        build = mast._mast_case_from_selection

        def sampled_case(*a, **kw):
            kw["flux_function_factory"] = SampledFluxFunction
            return build(*a, **kw)

        mast._mast_case_from_selection = sampled_case
        moments = mast.DeclaredAnchorOperator.cell_current_moments

        def record_negative(coordinate, positions, selected):
            negative = np.asarray(selected, bool) & (np.asarray(coordinate) < 0)
            if negative.any():
                context["negative_evaluations"].append(
                    dict(
                        count=int(negative.sum()),
                        rz_m=np.asarray(positions)[negative].tolist(),
                        psi_norm=np.asarray(coordinate)[negative].tolist(),
                    )
                )
                (args.output / "negative-source-nodes.json").write_text(
                    json.dumps(context["negative_evaluations"], indent=2) + "\n"
                )

        def observed_moments(operator, state, requested_class=None):
            result = moments(operator, state, requested_class)
            coordinate = (
                jnp.asarray(state)[: operator.grid.node_number]
                - operator.declared_axis_flux
            ) / (operator.declared_boundary_flux - operator.declared_axis_flux)
            jax.debug.callback(
                record_negative,
                coordinate,
                operator.grid.coordinate,
                operator.declared_support,
                ordered=True,
            )
            return result

        mast.DeclaredAnchorOperator.cell_current_moments = observed_moments
    install_analysis(load_trace(args.trace_helper), context)
    branch_measure.row(args)


if __name__ == "__main__":
    main()
