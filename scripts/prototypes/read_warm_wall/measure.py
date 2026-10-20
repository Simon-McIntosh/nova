"""Profile compiled topology read stages after warm-up and device synchronization."""

# Precision precedes array-valued imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import contour_tree, topology
from nova.equilibrium.solve_request import TopologyPolicy
from tests.equilibrium import test_topology_read as oracle_fixture

assert jax.config.jax_enable_x64 is True


def _root() -> Path:
    """Resolve the tree under measurement from the environment, at call time."""

    return Path(os.environ["NOVA_WORKTREE"]).resolve()


_INSTALLED_BUILDER = str(Path(contour_tree.__file__).resolve())


def _install_base_module(path: str) -> None:
    """Point every builder call at a module loaded from ``path``.

    ``topology.read`` rebinds ``build_contour_tree`` from the module at call
    time and the standalone contour stage reads it as a module attribute, so
    rebinding ``contour_tree.build_contour_tree`` moves both arms of the
    comparison onto the other implementation.
    """

    global _INSTALLED_BUILDER
    spec = importlib.util.spec_from_file_location("contour_tree_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["contour_tree_base"] = module
    spec.loader.exec_module(module)
    contour_tree.build_contour_tree = module.build_contour_tree
    _INSTALLED_BUILDER = str(Path(path).resolve())


def _timed_stage(function, *args):
    started = perf_counter()
    executable = jax.jit(function).lower(*args).compile()
    compile_wall = perf_counter() - started
    result = executable(*args)
    jax.block_until_ready(result)
    started = perf_counter()
    result = executable(*args)
    jax.block_until_ready(result)
    warm_wall = perf_counter() - started
    return result, warm_wall, compile_wall


def _contour_stage(centre_flux, geometry, sigma):
    mesh = geometry.contour_mesh
    vertex_flux = jnp.sum(
        centre_flux[geometry.contour_sources] * geometry.contour_weights, axis=1
    )
    return contour_tree.build_contour_tree(
        vertex_flux,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        sigma,
    )


def measure(kind: str, cells: int, label: str) -> dict:
    _oracle, field, wall, _axis, saddle = oracle_fixture._analytic_inputs(kind)
    geometry = oracle_fixture._realised_hex_geometry(wall, cells)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0,
        normal_form_pitch_floor=1.5,
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    effective_policy = topology._read_policy(geometry, policy)
    result, total_wall, total_compile = _timed_stage(
        topology.read, field, geometry, convention, policy
    )
    sampled, sampled_wall, sampled_compile = _timed_stage(
        topology._sampled_flux, field, geometry
    )
    coefficient, centre_flux = sampled
    tree, tree_wall, tree_compile = _timed_stage(
        _contour_stage, centre_flux, geometry, convention.sigma
    )
    census, census_wall, census_compile = _timed_stage(
        topology._null_census, field, geometry, coefficient, effective_policy, tree
    )
    nulls, live, required, census_resolved = census
    finished, finish_wall, finish_compile = _timed_stage(
        topology._read_at_nulls,
        field,
        geometry,
        convention,
        effective_policy,
        coefficient,
        nulls,
        live,
        required,
        census_resolved,
        tree,
    )
    if not np.array_equal(
        np.asarray(finished.fragment_component),
        np.asarray(result.fragment_component),
    ) or not np.array_equal(np.asarray(finished.valid), np.asarray(result.valid)):
        raise AssertionError("split read differs from the whole read")
    support, support_wall, support_compile = _timed_stage(
        topology._support_at_level,
        geometry,
        coefficient,
        convention.sigma,
        result.boundary_flux,
        result.axis,
        result.saddle_form,
        result.boundary_class == int(topology.TopologyClass.DIVERTED),
        effective_policy,
    )
    labels, label_wall, label_compile = _timed_stage(
        topology._connected_fragment_labels,
        support[0],
        support[1],
        geometry,
        effective_policy.edge_tolerance,
    )
    if not np.array_equal(np.asarray(labels[0]), np.asarray(result.fragment_component)):
        raise AssertionError("standalone component labels differ from the read")
    stage_seconds = {
        "sampled field": sampled_wall,
        "contour tree": tree_wall,
        "null census": census_wall,
        "support selection": finish_wall,
    }
    return {
        "revision": subprocess.check_output(
            ["git", "-C", str(_root()), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cwd": str(Path.cwd().resolve()),
        "module": str(Path(topology.__file__).resolve()),
        "builder_module": _INSTALLED_BUILDER,
        "label": label,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "device": str(jax.devices()[0]),
        "case": kind,
        "cells": cells,
        "realised_cells": int(geometry.centre.shape[0]),
        "valid": bool(result.valid),
        "reason": int(result.reason),
        "qualified": bool(result.qualified),
        "compile_seconds": total_compile,
        "stage_compile_seconds": {
            "sampled field": sampled_compile,
            "contour tree": tree_compile,
            "null census": census_compile,
            "support selection": finish_compile,
            "terminal support": support_compile,
            "component labels": label_compile,
        },
        "warm_total_seconds": total_wall,
        "stage_seconds": stage_seconds,
        "stage_sum_seconds": sum(stage_seconds.values()),
        "substage_seconds": {
            "terminal support": support_wall,
            "component labels": label_wall,
        },
        "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="after")
    parser.add_argument("--base-module", default=None)
    args = parser.parse_args()
    if args.base_module is not None:
        _install_base_module(args.base_module)
    row = measure(args.kind, args.cells, args.label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print("PROFILE_ROW " + json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
