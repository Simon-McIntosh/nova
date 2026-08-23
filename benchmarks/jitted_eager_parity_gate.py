"""Measure eager-versus-jitted parity on the frozen MAST held-out cases.

The tolerance receipt is written before any comparison is evaluated.  Each case
uses the production moment-seed image and exact-tangent profile solve at the
current source tree, with the six case identities inherited from the immutable
``spine_bench`` shot set and its banked reference slice per shot.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _stored_lcfs,
    build_profile,
    select_slices_by_shot,
)
from nova.equilibrium.forward import _lattice_cells
from nova.equilibrium.stencil_mesh import (
    CellCurrentMoments,
    MomentGeometry,
    StencilMesh,
)
from nova.geometry.hexstencil import hex_stencil
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


OUTPUT = Path("docs/figures/mast-catalog-gpu-solve/jitted-eager-parity-gate.json")
SPINE_BENCH = Path("/home/ITER/mcintos/Code/imas-ambix/imas_ambix/spine_bench")
SHOTSET_MODULE = SPINE_BENCH / "shots.py"
SHOTSET_VERSION = "v0-mast-heldout-6"
REGISTERED_TOLERANCE = 1.0e-10
SOLVE_OPTIONS = {
    "route": "newton_krylov",
    "newton_steps": 12,
    "gmres_iterations": 12,
    "warmup": 0,
}
REPAIRED_EQUILIBRIUM_COMMITS = (
    "6143221d7a2dc5c3ad2a74b822bf1a026c05eed3",
    "06b09f5bfdcf184a69066e829a6e7ee16fdd3a2b",
    "257af9dcbe5dd19313708f862a8bb3a383f917b2",
    "7e33c4d000131ce0bb76b9b3c7ad2b71ce668d36",
)


def _git(*arguments: str) -> str:
    """Return one git fact required to identify the measured tree."""

    return subprocess.check_output(
        ["git", *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _digest(path: Path) -> str:
    """Return a content digest for one held-out-set authority file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    """Return a stable UTC timestamp."""

    return datetime.now(UTC).isoformat()


def _write(path: Path, receipt: dict[str, Any]) -> None:
    """Write one human-readable JSON receipt at its registered path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n")


def _registration() -> dict[str, Any]:
    """Return the comparison bound before any case is measured."""

    return {
        "registered_before_scoring": True,
        "absolute_tolerance": REGISTERED_TOLERANCE,
        "relative_tolerance": REGISTERED_TOLERANCE,
        "criterion": (
            "numpy.allclose(jitted, eager, atol=1e-10, rtol=1e-10, "
            "equal_nan=True) for floating quantities; exact equality for "
            "integer and boolean quantities"
        ),
        "derivation": (
            "The live plan pre-registers 1e-10 from Nova's existing jitted "
            "operator/topology batched-versus-per-slice parity floor. The same "
            "dimensionless floor is retained here before inspecting any result; "
            "it is not fitted to this run."
        ),
    }


def _tree_identity() -> dict[str, Any]:
    """Return the exact committed source tree and repaired changes it contains."""

    return {
        "commit_sha": _git("rev-parse", "HEAD"),
        "tree_sha": _git("rev-parse", "HEAD^{tree}"),
        "repaired_equilibrium_commits": list(REPAIRED_EQUILIBRIUM_COMMITS),
        "repaired_commits_are_ancestors": {
            commit: subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, "HEAD"], check=False
            ).returncode
            == 0
            for commit in REPAIRED_EQUILIBRIUM_COMMITS
        },
    }


def _named_tree(value: Any) -> Any:
    """Convert named tuples to dictionaries so receipt paths retain field names."""

    if hasattr(value, "_asdict"):
        return {name: _named_tree(item) for name, item in value._asdict().items()}
    if isinstance(value, Mapping):
        return {str(name): _named_tree(item) for name, item in value.items()}
    if isinstance(value, tuple | list):
        return [_named_tree(item) for item in value]
    return value


def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a named result tree without discarding unavailable-looking leaves."""

    if isinstance(value, Mapping):
        output = {}
        for name, item in value.items():
            child = f"{prefix}.{name}" if prefix else str(name)
            output.update(_leaves(item, child))
        return output
    if isinstance(value, list):
        output = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            output.update(_leaves(item, child))
        return output
    return {prefix: value}


def _difference(eager: Any, jitted: Any) -> dict[str, Any]:
    """Score one numeric leaf against the pre-registered tolerance."""

    left = np.asarray(eager)
    right = np.asarray(jitted)
    if left.shape != right.shape:
        return {
            "shape": list(left.shape),
            "jitted_shape": list(right.shape),
            "maximum_absolute_difference": None,
            "maximum_relative_difference": None,
            "passes": False,
            "criterion": "same shape",
        }
    exact = left.dtype.kind in "biu" or right.dtype.kind in "biu"
    if exact:
        equal = np.array_equal(left, right)
        absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
        maximum_absolute = float(np.max(absolute)) if absolute.size else 0.0
        maximum_relative = 0.0 if equal else float("inf")
        passes = bool(equal)
        criterion = "exact equality"
    else:
        finite = np.isfinite(left) & np.isfinite(right)
        nan_pattern_equal = np.array_equal(np.isnan(left), np.isnan(right))
        if np.any(finite):
            absolute = np.abs(right[finite] - left[finite])
            maximum_absolute = float(np.max(absolute))
            reference_scale = max(
                float(np.max(np.abs(left[finite]))), np.finfo(float).tiny
            )
            maximum_relative = maximum_absolute / reference_scale
        else:
            maximum_absolute = 0.0
            maximum_relative = 0.0
        passes = bool(
            nan_pattern_equal
            and np.allclose(
                right,
                left,
                atol=REGISTERED_TOLERANCE,
                rtol=REGISTERED_TOLERANCE,
                equal_nan=True,
            )
        )
        criterion = "atol=1e-10 plus rtol=1e-10"
    return {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "passes": passes,
        "criterion": criterion,
    }


def _compare_trees(eager: Any, jitted: Any) -> dict[str, dict[str, Any]]:
    """Compare every named leaf and reject any structural omission."""

    eager_leaves = _leaves(_named_tree(eager))
    jitted_leaves = _leaves(_named_tree(jitted))
    if eager_leaves.keys() != jitted_leaves.keys():
        missing = sorted(eager_leaves.keys() - jitted_leaves.keys())
        extra = sorted(jitted_leaves.keys() - eager_leaves.keys())
        raise RuntimeError(f"result tree differs: missing={missing}, extra={extra}")
    return {
        name: _difference(eager_leaves[name], jitted_leaves[name])
        for name in eager_leaves
    }


def _merge_quantity_rows(
    aggregate: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> None:
    """Accumulate worst differences and conjunctive verdicts across cases."""

    for name, row in rows.items():
        if name not in aggregate:
            aggregate[name] = {**row, "case_count": 1}
            continue
        target = aggregate[name]
        target["case_count"] += 1
        target["passes"] = bool(target["passes"] and row["passes"])
        for metric in (
            "maximum_absolute_difference",
            "maximum_relative_difference",
        ):
            values = (target[metric], row[metric])
            target[metric] = (
                max(values) if all(value is not None for value in values) else None
            )


def _moment_image(profile, cell_current):
    """Return the traced portion of the production moment-seed construction."""

    cell_current = jnp.asarray(cell_current, dtype=jnp.float64)
    zero = jnp.zeros_like(cell_current)
    moments = CellCurrentMoments(cell_current, zero, zero)
    coefficients = profile.operator.coupling_current_moments(moments)
    flux = profile.operator.external() + profile.operator.current_moment_image(
        coefficients
    )
    return {"flux": flux, "current_moment_coefficients": coefficients}


def _with_moment_geometry(profile):
    """Complete the legacy frozen-case builder with production moment geometry."""

    if profile.operator.moment_geometry is not None:
        return profile
    lattice = profile.lattice
    mesh = StencilMesh(
        coordinate=lattice.coordinate,
        stencil=hex_stencil(lattice.shape),
        area=lattice.cell_area,
    )
    operator = replace(
        profile.operator,
        moment_geometry=MomentGeometry.from_cells(mesh, _lattice_cells(lattice)),
        prescribed_current_field=profile.operator.prescribed_field,
    )
    return replace(profile, operator=operator)


def _case_rows(store: Path) -> list[tuple[int, int, dict[str, Any]]]:
    """Return one immutable reference slice for every frozen held-out shot."""

    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    rows = [
        (int(row["shot"]), int(row["slice_index"]), row)
        for row, _qualification in selected
    ]
    if len(rows) != 6 or len({shot for shot, _index, _row in rows}) != 6:
        raise RuntimeError("the frozen held-out cohort must contain six distinct shots")
    missing = [
        shot for shot, _index, _row in rows if not (store / f"{shot}.zarr").is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"held-out shot stores are absent: {missing}")
    return rows


def measure(output: Path, store: Path) -> dict[str, Any]:
    """Register the tolerance, then measure every eager and jitted quantity."""

    configure_dtypes()
    registered_utc = _utc_now()
    receipt: dict[str, Any] = {
        "schema": "nova-jitted-eager-parity/1.0",
        "status": "tolerance_registered",
        "registered_utc": registered_utc,
        "tolerance_registration": _registration(),
        "source_tree": _tree_identity(),
        "held_out_set": {
            "root": str(SPINE_BENCH),
            "shotset_module": str(SHOTSET_MODULE),
            "shotset_module_sha256": _digest(SHOTSET_MODULE),
            "version": SHOTSET_VERSION,
        },
    }
    _write(output, receipt)

    moment_quantities: dict[str, dict[str, Any]] = {}
    profile_quantities: dict[str, dict[str, Any]] = {}
    cases = []
    for shot, slice_index, _row in _case_rows(store):
        group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
        profile, _reference_seed, _reference, _provenance = build_profile(
            group, shot, slice_index, "fcoil_c"
        )
        profile = _with_moment_geometry(profile)
        boundary = _stored_lcfs(group, slice_index)
        target_current = abs(float(group["plasma_current_c"][slice_index]))
        seed = profile.moment_seed(boundary, target_current)

        eager_moment = _moment_image(profile, seed.cell_current)
        jitted_moment = jax.jit(lambda current: _moment_image(profile, current))(
            seed.cell_current
        )
        jax.block_until_ready(jitted_moment)
        moment_rows = _compare_trees(eager_moment, jitted_moment)
        _merge_quantity_rows(moment_quantities, moment_rows)

        def solve(state):
            return profile.solve(state, target_current=target_current, **SOLVE_OPTIONS)

        eager_profile = solve(seed.flux)
        jitted_profile = jax.jit(solve)(seed.flux)
        jax.block_until_ready(jitted_profile)
        profile_rows = _compare_trees(eager_profile, jitted_profile)
        _merge_quantity_rows(profile_quantities, profile_rows)
        cases.append(
            {
                "shot": shot,
                "slice_index": slice_index,
                "time_s": float(group["time"][slice_index]),
                "moment_seed_supported_cells": int(seed.supported_cells),
                "moment_seed_passes": bool(
                    all(row["passes"] for row in moment_rows.values())
                ),
                "profile_solve_passes": bool(
                    all(row["passes"] for row in profile_rows.values())
                ),
                "failed_moment_quantities": sorted(
                    name for name, row in moment_rows.items() if not row["passes"]
                ),
                "failed_profile_quantities": sorted(
                    name for name, row in profile_rows.items() if not row["passes"]
                ),
            }
        )

    all_pass = all(
        row["passes"]
        for row in (*moment_quantities.values(), *profile_quantities.values())
    )
    receipt.update(
        {
            "status": "passed" if all_pass else "failed",
            "completed_utc": _utc_now(),
            "backend": {
                "platform": jax.default_backend(),
                "device": jax.devices()[0].device_kind,
                "jax_version": jax.__version__,
                "precision": "float64",
            },
            "held_out_set": {
                **receipt["held_out_set"],
                "case_count": len(cases),
                "cases": cases,
                "case_selection": (
                    "one pre-existing qualified reference slice per frozen shot, "
                    "selected by select_slices_by_shot from the committed "
                    "decomposition bank"
                ),
            },
            "solve_route": {
                "entry_point": "ForwardProfile.solve",
                "options": SOLVE_OPTIONS,
                "seed_entry_point": "ForwardProfile.moment_seed",
            },
            "comparisons": {
                "moment_seed_traced_image": {
                    "passes": all(row["passes"] for row in moment_quantities.values()),
                    "quantities": moment_quantities,
                },
                "profile_solve": {
                    "passes": all(row["passes"] for row in profile_quantities.values()),
                    "quantities": profile_quantities,
                },
            },
            "uncomparable_quantities": [
                {
                    "stage": "moment_seed",
                    "quantities": [
                        "boundary support selection",
                        "predicted current and centroid",
                        "seed radius",
                        "supported cell count",
                    ],
                    "reason": (
                        "ForwardProfile.moment_seed deliberately performs polygon, "
                        "limiter, and constrained-disc construction through host "
                        "NumPy geometry. Those values are common immutable inputs "
                        "to both paths and have no traced production counterpart. "
                        "The JAX production portion beginning at CellCurrentMoments "
                        "is compared in full; the host-only fields are named here "
                        "rather than omitted."
                    ),
                }
            ],
            "overall_passes": all_pass,
        }
    )
    _write(output, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    """Return the benchmark command-line interface."""

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=OUTPUT)
    result.add_argument("--store", type=Path, default=SHOT_STORE)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = measure(arguments.output, arguments.store)
    print(
        json.dumps(
            {
                "status": result["status"],
                "case_count": result["held_out_set"]["case_count"],
                "overall_passes": result["overall_passes"],
            },
            indent=2,
        )
    )
