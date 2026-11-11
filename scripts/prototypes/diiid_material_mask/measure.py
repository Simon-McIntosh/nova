"""Measure both cold branches with the probe's governed physical wall."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))
import measure_branches as probe  # noqa: E402
from boundary_competition import Grid, _wall_distance  # noqa: E402
from nova.equilibrium.connectivity_boundary import host_boundary_read  # noqa: E402
from nova.media.poloidal import inside_wall_units  # noqa: E402


def run_frame(frame, root, expansion):
    prepared = probe._prepare_frame(frame, expansion)
    profile = prepared["profile"]
    limiter, provenance = probe.bench._physical_wall_ring(
        probe.bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
        probe.bench.DEFAULT_MACHINE_ARTIFACT_DIGEST,
    )
    lattice = profile.lattice
    material = np.asarray(profile.operator.inside_material)
    root.mkdir(parents=True, exist_ok=True)
    for branch in ("limited", "diverted"):
        kind = probe.TopologyClass[branch.upper()]
        stem = f"frame-{frame}-{branch}"
        record = {
            "frame": frame,
            "branch": branch,
            "surface": "physical" if expansion is None else "pseudo",
            "pseudo_wall_expansion": expansion,
            "material_cells": int(material.sum()),
            "wall_provenance": provenance,
            "storage_order": "radius_height",
        }
        try:
            seed = prepared["cold"].branches.flux[int(kind)]
            receipt, seconds = probe._solve(
                profile,
                seed,
                prepared["current"],
                prepared["target_current_a"],
                f"diiid-material-mask:{frame}:{branch}",
            )
            state = np.asarray(receipt.equilibrium.flux)
            masks, topology, connected, admitted = profile.operator._fixed_design_read(
                jnp.asarray(state[: profile.operator.physical_node_number])
            )
            field = state[: lattice.node_count].reshape(lattice.shape)
            axis = np.asarray(topology.axis)
            try:
                boundary = host_boundary_read(
                    field.T,
                    Grid(lattice.radius, lattice.height, limiter),
                    tuple(axis),
                )
            except (ValueError, RuntimeError) as error:
                boundary = None
                record["host_boundary_refusal"] = str(error)
            try:
                surface, _ = probe.bench._assembled_boundary_geometry(
                    field,
                    lattice.radius,
                    lattice.height,
                    float(topology.boundary_flux),
                    axis,
                )
            except (ValueError, RuntimeError) as error:
                surface = np.empty((0, 2))
                record["surface_refusal"] = str(error)
            gap = _wall_distance(surface, limiter)[0] if len(surface) else np.array([])
            record.update(
                status="solved",
                seconds=seconds,
                support_cells=int(np.asarray(connected).sum()),
                support_outside_limiter=int(
                    np.sum(
                        np.asarray(connected)
                        & ~probe.bench.inside_polygon(
                            lattice.coordinate[:, 0],
                            lattice.coordinate[:, 1],
                            limiter[:, 0],
                            limiter[:, 1],
                        )
                    )
                ),
                axis_rz_m=axis.tolist(),
                axis_inside_limiter=bool(inside_wall_units(axis[None], limiter)[0]),
                axis_admitted=bool(admitted),
                residual=float(receipt.equilibrium.fixed_point.residual),
                converged=bool(receipt.equilibrium.fixed_point.converged),
                qualified=bool(receipt.qualified),
                termination_reason=int(receipt.termination_reason),
                residual_history=probe._finite_list(receipt.residual_history),
                changed_mask_entries=probe._int_list(receipt.changed_mask_entries),
                host_boundary_read_class=(
                    "refused"
                    if boundary is None
                    else "diverted"
                    if boundary.is_diverted
                    else "limited"
                ),
                host_boundary_found=bool(boundary is not None and boundary.found),
                host_boundary_class_margin=(
                    None if boundary is None else float(boundary.class_margin)
                ),
                wall_gap_m=float(np.min(gap)) if gap.size else None,
                x_point_rz_m=np.asarray(topology.x_point).tolist(),
            )
            np.savez_compressed(
                root / (stem + ".npz"),
                state=state,
                flux=field,
                radius=lattice.radius,
                height=lattice.height,
                connected=np.asarray(connected),
                inside_material=material,
                limiter=limiter,
                label=prepared["label"],
                reference_axis=prepared["axis"],
                reference_saddle=prepared["saddle"],
                reference_boundary=prepared["contour"],
                axis=axis,
                x_point=np.asarray(topology.x_point),
                boundary_flux=float(topology.boundary_flux),
                axis_flux=float(topology.axis_flux),
                surface=surface,
            )
        except Exception as error:
            record.update(
                status="refused",
                error=type(error).__name__ + ": " + str(error),
                traceback=traceback.format_exc(),
            )
        (root / (stem + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--pseudo-wall-expansion", type=float, default=None)
    args = parser.parse_args()
    if args.frame is not None:
        run_frame(args.frame, args.output, args.pseudo_wall_expansion)
        return
    for frame in (179, 137, 146):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--frame",
            str(frame),
            "--output",
            str(args.output),
        ]
        if args.pseudo_wall_expansion is not None:
            command.extend(["--pseudo-wall-expansion", str(args.pseudo_wall_expansion)])
        result = subprocess.run(command, check=False)
        print(f"FRAME={frame} EXIT={result.returncode}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
