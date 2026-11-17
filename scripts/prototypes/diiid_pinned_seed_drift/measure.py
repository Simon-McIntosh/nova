"""Read the pinned frame through the benchmark's profile and current owners."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes


def identity(value):
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def main():
    configure_dtypes()
    from benchmarks import diiid_current_pinned_forward as pinned

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=102)
    parser.add_argument("--revision")
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--undo-average", action="store_true")
    parser.add_argument("--point-current", action="store_true")
    parser.add_argument("--compare-integration", action="store_true")
    parser.add_argument("--source-read-only", action="store_true")
    parser.add_argument("--normalisation-reference", type=Path)
    parser.add_argument("--hold-profile-ends", action="store_true")
    arguments = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    revision = (
        arguments.revision
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    print(f"REVISION={revision} MODULE={pinned.__file__} CWD={Path.cwd()}", flush=True)
    print(f"DEVICES={jax.devices()} X64={jax.config.jax_enable_x64}", flush=True)
    declared = next(
        item
        for item in pinned.REPRESENTATIVE_COHORT
        if item["frame"] == arguments.frame
    )
    path = pinned.DEFAULT_DATA / declared["shot"]
    columns = tuple(
        dict.fromkeys(
            (
                *pinned._LABEL_COLUMNS,
                *pinned._CURRENT_COLUMNS,
                *pinned._GEOMETRY_COLUMNS,
                *pinned.PLASMA_CURRENT_COLUMNS,
            )
        )
    )
    row = pinned._read(path, columns)
    row["_source_path"] = str(path)
    started = time.monotonic()
    profile, seed, label, wall, reliable, statement = pinned.build_profile(
        row, arguments.frame, pinned.PSEUDO_WALL_EXPANSION
    )
    print(f"BUILT seconds={time.monotonic() - started:.3f}", flush=True)
    if arguments.hold_profile_ends:
        from benchmarks.diiid_forward_gs_match import _profile_function

        source = profile.operator.source
        functions = {
            key: _profile_function(
                getattr(source.core, key).coordinate,
                getattr(source.core, key).values,
            )
            for key in ("p_prime", "ff_prime")
        }
        source = replace(source, core=replace(source.core, **functions))
        profile = replace(profile, operator=replace(profile.operator, source=source))
    time_ms = float(row["efit_times"][arguments.frame])
    if arguments.source_read_only:
        shipped = current = profile.operator.external_current
        authority = {"adapter_applied": False}
    else:
        profile, shipped, current, authority = pinned.description_driven_currents(
            row, profile, time_ms
        )
    if arguments.point_current:
        profile = replace(
            profile, operator=replace(profile.operator, use_linear_moments=False)
        )
    if arguments.undo_average:
        profile = replace(
            profile, operator=replace(profile.operator, cell_average_stencil=None)
        )
    target = pinned._target_current(row, time_ms)
    moments = profile.operator.cell_current(seed, pinned.TopologyClass.DIVERTED)
    unscaled = float(np.sum(np.asarray(moments)))
    amplitude = float(
        profile.operator.current_normalisation_amplitude(target, unscaled)
    )
    masks, topology = profile.operator.read(seed, pinned.TopologyClass.DIVERTED)
    samples = jnp.linspace(0.0, 1.0, 65, dtype=jnp.float64)
    result = {
        "revision": revision,
        "module": pinned.__file__,
        "cwd": str(Path.cwd()),
        "shot": declared["shot"],
        "frame": arguments.frame,
        "time_ms": time_ms,
        "unscaled_seed_plasma_current_a": unscaled,
        "registered_seed_current_a": declared["unscaled_source_ip_a"],
        "probe_seed_current_a": 1119994.165,
        "seed_amplitude": amplitude,
        "target_current_a": target,
        "current_authority": authority,
        "seed_identity": identity(seed),
        "label_identity": identity(label),
        "wall_identity": identity(wall),
        "cell_current_identity": identity(moments),
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "use_linear_moments": bool(profile.operator.use_linear_moments),
        "undo_average": arguments.undo_average,
        "point_current": arguments.point_current,
        "source_read_only": arguments.source_read_only,
        "hold_profile_ends": arguments.hold_profile_ends,
        "topology": {
            key: np.asarray(getattr(topology, key)).tolist()
            for key in ("axis", "axis_flux", "boundary_flux", "x_point", "diverted")
        },
        "normalised_flux_identity": identity(masks.psi_norm),
        "core_cells": int(np.count_nonzero(np.asarray(masks.core))),
        "source_samples": {
            key: np.asarray(
                getattr(profile.operator.source.core, key)(samples)
            ).tolist()
            for key in ("p_prime", "ff_prime")
        },
        "reliable_surfaces": reliable,
        "wall_statement": statement,
        "relative_registration_error": unscaled / declared["unscaled_source_ip_a"]
        - 1.0,
        "elapsed_seconds": time.monotonic() - started,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"CURRENT_A={unscaled:.12f} OUTPUT={arguments.output}", flush=True)
    if arguments.normalisation_reference:
        reference = json.loads(arguments.normalisation_reference.read_text())
        old = reference["topology"]
        old_axis = old["axis_flux"]
        old_boundary = old["boundary_flux"]
        old_span = old_boundary - old_axis
        operator = profile.operator
        physical = jnp.asarray(seed)
        masks, topology, _, _ = operator._fixed_design_read(
            physical, pinned.TopologyClass.DIVERTED
        )
        old_masks = masks._replace(
            psi_norm=(physical[: operator.grid.node_number] - old_axis) / old_span
        )
        old_topology = topology._replace(axis_flux=old_axis, boundary_flux=old_boundary)
        sample_flux = (operator.sample_node_flux(physical) - old_axis) / old_span
        support = operator._profile_support(
            old_masks, old_topology, physical, sample_flux
        )
        old_booked = operator._partitioned_current_moments(
            (old_masks, old_topology, sample_flux, support)
        )
        old_point = operator._point_current_moments(old_masks)
        normalisation = {
            "reference_revision": reference["revision"],
            "head_revision": revision,
            "booked_current_a": unscaled,
            "old_normalisation_booked_current_a": float(
                np.sum(old_booked.cell_current)
            ),
            "old_normalisation_point_current_a": float(np.sum(old_point.cell_current)),
            "old_axis_flux": old_axis,
            "old_boundary_flux": old_boundary,
            "head_topology": result["topology"],
            "old_normalisation_booked_current": np.asarray(
                old_booked.cell_current
            ).tolist(),
            "old_normalisation_point_current": np.asarray(
                old_point.cell_current
            ).tolist(),
        }
        normalisation_path = arguments.output.with_name("normalisation-comparison.json")
        normalisation_path.write_text(
            json.dumps(normalisation, indent=2, sort_keys=True) + "\n"
        )
        print(
            "NORMALISATION_COMPARISON "
            + json.dumps(
                {
                    key: value
                    for key, value in normalisation.items()
                    if key.endswith("_a")
                }
            ),
            flush=True,
        )
    if arguments.compare_integration:
        point = replace(profile.operator, use_linear_moments=False)
        averaged = np.asarray(point.cell_current(seed, pinned.TopologyClass.DIVERTED))
        centroid = np.asarray(
            replace(point, cell_average_stencil=None).cell_current(
                seed, pinned.TopologyClass.DIVERTED
            )
        )
        comparison = {
            "booked_current_a": unscaled,
            "point_averaged_current_a": float(np.sum(averaged)),
            "point_centroid_current_a": float(np.sum(centroid)),
            "averaged_current": averaged.tolist(),
            "centroid_current": centroid.tolist(),
            "registered_seed_current_a": declared["unscaled_source_ip_a"],
            "current_authority": authority,
            "revision": revision,
        }
        comparison_path = arguments.output.with_name("integration-comparison.json")
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )
        print(
            "INTEGRATION_COMPARISON "
            + json.dumps(
                {key: value for key, value in comparison.items() if key.endswith("_a")}
            ),
            flush=True,
        )
    if arguments.measure_only:
        return
    # Reuse the constructed inputs so the benchmark's own qualification executes.
    pinned.build_profile = lambda *args, **kwargs: (
        profile,
        seed,
        label,
        wall,
        reliable,
        statement,
    )
    pinned.description_driven_currents = lambda *args: (
        profile,
        shipped,
        current,
        authority,
    )
    pinned.solve_frame(
        row, pinned.FrameInput(declared["shot"], arguments.frame), declared
    )


if __name__ == "__main__":
    main()
