"""Read physical shadow closure beside the established core-hold cohort."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from provenance import load_trace

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "core_measure", ROOT / "scripts/prototypes/diiid_core_hold/measure.py"
)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
core.load_trace = load_trace
install_core_analysis = core.install_analysis


def install_analysis(trace, context):
    """Retain the cohort's census and add an independent physical image read."""
    install_core_analysis(trace, context)
    original = core.branch_measure.enrich

    def enrich(output, profile, scale, record, data, branches, seeds):
        original(output, profile, scale, record, data, branches, seeds)
        import jax
        import jax.numpy as jnp
        import numpy as np

        op = profile.operator

        @jax.jit
        def read(state, target):
            moments = op.cell_current_moments(state)
            amplitude = op.current_normalisation_amplitude(
                target, jnp.sum(moments.cell_current)
            )
            physical = op.external() + op.current_moment_image(
                op.scaled_current_moments(moments, amplitude)
            )
            return physical, op.residual_shadow_mask(state)

        data = dict(np.load(output / "terminal.npz"))
        state = data["state"]
        physical, shadow = jax.device_get(
            read(jnp.asarray(state), jnp.asarray(context["target_current"]))
        )
        physical, shadow = np.asarray(physical), np.asarray(shadow, bool)
        np.testing.assert_array_equal(shadow, data["shadow"])
        difference = state - physical
        seed = np.load(output / "request-seed.npz")["state"]
        analysis = json.loads((output / "core-analysis.json").read_text())
        receipt = json.loads((output / "receipt.json").read_text())
        grid_count = data["flux"].size
        coordinates = np.vstack((op.grid.coordinate, op.wall.coordinate))
        selected = np.flatnonzero(shadow)
        summary = dict(
            revision=receipt["base_revision"],
            module=receipt["module"],
            cwd=receipt["cwd"],
            residual=receipt["residual"],
            converged=receipt["converged"],
            termination_reason=receipt.get("termination_reason"),
            maxima_count=analysis["maxima_count"],
            shadow_node_count=int(shadow.sum()),
            shadow_grid_node_count=int(shadow[:grid_count].sum()),
            shadow_image_sup_wb=float(np.max(abs(difference[shadow]), initial=0)),
            unshadowed_image_sup_wb=float(np.max(abs(difference[~shadow]), initial=0)),
            physical_image_relative_sup=float(
                np.max(abs(difference)) / max(np.max(abs(physical)), 1e-30)
            ),
            shadow_seed_change_sup_wb=float(
                np.max(abs((state - seed)[shadow]), initial=0)
            ),
            shadow_indices=selected.tolist(),
            shadow_rz_m=coordinates[selected].tolist(),
            lower_x_distance_m=receipt["terminal_topology"]["lower_saddle_distance_m"],
            lower_x_rz_m=receipt["terminal_topology"]["lower_saddle_rz_m"],
            efit_x_rz_m=receipt["efit_x_point_rz_m"],
        )
        solve = json.loads((output / "solve-receipt.json").read_text())
        summary["termination_reason"] = solve["terminal_reason"]
        data.update(physical_image=physical)
        np.savez_compressed(output / "terminal.npz", **data)
        (output / "shadow-image.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SHADOW_IMAGE " + json.dumps(summary), flush=True)

    core.branch_measure.enrich = enrich


if __name__ == "__main__":
    core.install_analysis = install_analysis
    core.main()
