"""Dump the production support read's host locals after a registered solve."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.equilibrium.topology import Topology
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))
import measure_branches as measure  # noqa: E402


def dump_read(operator, state):
    """Run the actual read body on the host and capture its final locals.

    Only the outer jit is unwrapped. Subordinate product kernels still execute
    normally; no boundary, mask or flood expression is copied into this probe.
    The operator makes its original-material read followed by its rescue read.
    """
    original = Topology.read_qualification
    body = original.__wrapped__
    captures = []

    def traced(*args, **kwargs):
        local = {}

        def capture(frame, event, arg):
            if frame.f_code is body.__code__ and event == "return":
                local.update(frame.f_locals)
            return capture

        previous = sys.gettrace()
        sys.settrace(capture)
        try:
            result = body(*args, **kwargs)
            result.connected.block_until_ready()
        finally:
            sys.settrace(previous)
        captures.append(local)
        return result

    Topology.read_qualification = traced
    try:
        result = operator._fixed_design_read(jnp.asarray(state))
    finally:
        Topology.read_qualification = original
    assert len(captures) == 2
    return captures, result


def run(frame, branch, trips, root):
    configure_dtypes()
    assert jax.config.jax_enable_x64
    prepared = measure._prepare_frame(
        frame, float(measure.bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION)
    )
    profile = prepared["profile"]
    kind = measure.TopologyClass[branch.upper()]
    seed = prepared["cold"].branches.flux[int(kind)]
    receipt, seconds = measure._solve(
        profile,
        seed,
        prepared["current"],
        prepared["target_current_a"],
        f"support-trace:{frame}:{branch}:{trips}",
        active_set_steps=trips,
    )
    state = np.asarray(receipt.equilibrium.flux)
    captures, result = dump_read(profile.operator, state)
    local = captures[-1]
    lattice = profile.lattice
    field = state[: lattice.node_count].reshape(lattice.shape)
    surface = fit_tensor_spline(
        jnp.asarray(lattice.radius), jnp.asarray(lattice.height), jnp.asarray(field.T)
    )
    axis = np.asarray(local["data_o"][:2])
    polished = polish_stationary_points(
        surface, jnp.asarray(axis[None]), jnp.ones(1, bool)
    )
    shift = float(np.linalg.norm(np.asarray(polished["position_rz"])[0] - axis))
    spacing = min(lattice.radial_step, lattice.vertical_step)
    assert bool(polished["converged"][0]) and shift < spacing, (shift, spacing)
    coordinate = np.asarray(local["self"].connectivity_coordinate)
    owner = int(np.argmin(np.sum((coordinate - axis) ** 2, axis=1)))
    arrays = {
        name: np.asarray(local[name])
        for name in (
            "comparison_flux",
            "closed",
            "connected",
            "inside_material",
            "data_o",
            "data_x",
            "data_b",
            "data_w",
        )
    }
    material = arrays["inside_material"].astype(bool)
    closed = arrays["closed"].astype(bool)
    arrays.update(
        flux=field,
        state=state,
        radius=lattice.radius,
        height=lattice.height,
        coordinate=coordinate,
        shared_edges=np.asarray(local["self"].connectivity_shared_edges),
        rings=np.asarray(local["self"].connectivity_rings),
        label=prepared["label"],
        reference_axis=prepared["axis"],
        reference_saddle=prepared["saddle"],
        limiter=measure.bench._physical_wall_ring(
            measure.bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
            measure.bench.DEFAULT_MACHINE_ARTIFACT_DIGEST,
        )[0],
    )
    counts = {
        "material": int(material.sum()),
        "closed_flux_test": int(closed.sum()),
        "material_and_closed": int((material & closed).sum()),
        "axis_connected": int(arrays["connected"].sum()),
    }
    record = {
        "frame": frame,
        "branch": branch,
        "trips": trips,
        "counts": counts,
        "storage_order": "radius_height",
        "axis_polish_shift_m": shift,
        "lattice_spacing_m": spacing,
        "axis_cell": owner,
        "axis_flux_wb": float(local["data_o"][2]),
        "boundary_flux_wb": float(local["data_b"][2]),
        "polarity": int(profile.operator.polarity),
        "comparison_flux_at_axis_cell_wb": float(arrays["comparison_flux"][owner]),
        "axis_cell_material": bool(material[owner]),
        "axis_cell_closed": bool(closed[owner]),
        "boundary_uncertainty_wb": float(local["boundary_uncertainty"]),
        "boundary_is_xpoint": bool(local["boundary_is_xpoint"]),
        "axis_admitted": bool(result[3]),
        "original_material_counts": {
            "material": int(np.asarray(captures[0]["inside_material"]).sum()),
            "connected": int(np.asarray(captures[0]["connected"]).sum()),
        },
        "mask_change_history": measure._int_list(receipt.changed_mask_entries),
        "attempted_newton_promotions": int(receipt.attempted_newton_promotions),
        "accepted_newton_promotions": int(receipt.accepted_newton_promotions),
        "residual": float(receipt.equilibrium.fixed_point.residual),
        "converged": bool(receipt.equilibrium.fixed_point.converged),
        "seconds": seconds,
        "module": inspect.getfile(Topology),
        "read_line": inspect.getsourcelines(Topology.read_qualification)[1],
        "psi_mask_line": inspect.getsourcelines(Topology.psi_mask)[1],
        "axis_component_line": inspect.getsourcelines(Topology.axis_component)[1],
        "state_node_count": state.size,
        "lattice_node_count": lattice.node_count,
    }
    stem = f"frame-{frame}-{branch}-trip-{trips}"
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / f"{stem}.npz", **arrays)
    (root / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    assert counts["axis_connected"] > 0, (
        "support really empties; inspect captured operands"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--branch", choices=("limited", "diverted"))
    parser.add_argument("--trips", type=int)
    args = parser.parse_args()
    if args.frame is not None:
        run(args.frame, args.branch, args.trips, args.output)
    else:
        for frame, branch, trips in (
            (179, "limited", 1),
            (146, "diverted", 3),
            (137, "diverted", 1),
        ):
            command = [
                sys.executable,
                __file__,
                "--output",
                str(args.output),
                "--frame",
                str(frame),
                "--branch",
                branch,
                "--trips",
                str(trips),
            ]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
