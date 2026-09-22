"""Compare polygon integration and certificate lowering in one CPU allocation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
BASE = "c054dc50fe02add3b03555a99ce66a099485dc72"
MODULE_PATH = "nova/equilibrium/clip_quadrature.py"
TESTS = (
    "tests/test_exact_clip_moments.py",
    "tests/test_exact_clip_closed_form_budget.py",
)
MUTATION = "restore the Gauss antiderivative path"


def revision():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def source(path):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BASE}:{path}"], text=True
    )


def baseline_module():
    module = types.ModuleType("nova.equilibrium.clip_quadrature")
    module.__file__ = str(ROOT / MODULE_PATH)
    exec(compile(source(MODULE_PATH), module.__file__, "exec"), module.__dict__)
    return module


def install_baseline():
    import nova.equilibrium.clip_quadrature as current

    baseline = baseline_module()
    # Replace only the two functions this experiment varies. Existing imports
    # of public reducers still resolve these globals from their own module.
    current._sampled_arc_polynomial_moments = baseline._sampled_arc_polynomial_moments
    current._monomial_antiderivative = baseline._monomial_antiderivative
    current.cut_cell_moment_evaluation_bound = baseline.cut_cell_moment_evaluation_bound


def persist(name, value):
    path = OUTPUT / name
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(value, allow_nan=False), flush=True)


def suite(arm):
    import pytest

    if arm == "baseline":
        install_baseline()
        with tempfile.TemporaryDirectory(
            prefix="baseline-tests-", dir=OUTPUT
        ) as directory:
            paths = []
            for path in TESTS:
                destination = Path(directory) / Path(path).name
                destination.write_text(source(path))
                paths.append(str(destination))
            return pytest.main(["-q", "-p", "no:cacheprovider", *paths])
    return pytest.main(["-q", "-p", "no:cacheprovider", *TESTS])


def executable(arm):
    import jax

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    if arm in ("baseline", "negative"):
        install_baseline()
    from benchmarks.program_scope_census import _build_profile, _parse_instruction
    from nova.equilibrium.forward_operator import set_support_clip_mode
    from nova.equilibrium import clip_quadrature

    set_support_clip_mode("exact")
    profile, operator, request, seed, target = _build_profile(
        "weak-rotation-reactor-static", 110
    )
    program = profile._accelerated_history_program(
        "newton_krylov",
        requested_class=None,
        target_current=target,
        **request.policy.kernel_options(),
    )
    result = {
        "revision": revision(),
        "base_revision": BASE,
        "arm": arm,
        "tree": str(ROOT),
        "job": os.environ["SLURM_JOB_ID"],
        "node": os.environ.get("SLURMD_NODENAME"),
        "requested_cells": 110,
        "realised_cells": int(operator.grid.node_number),
        "mode": "exact",
        "evaluations_per_cut_cell": clip_quadrature.cut_cell_moment_evaluation_bound(),
        "completed": False,
    }
    started = time.perf_counter()
    lowered = program.lower(seed, operator.external(None, None))
    result["lower_seconds"] = time.perf_counter() - started
    text = lowered.as_text(debug_info=True)
    result["lowered_antiderivative_present"] = "_monomial_antiderivative" in text
    result["lowered_closed_form_present"] = "_straight_edge_monomial_moments" in text
    persist(f"{arm}-program.json", result)
    started = time.perf_counter()
    compiled = lowered.compile()
    result["compile_seconds"] = time.perf_counter() - started
    serialized = compiled.runtime_executable().serialize()
    result["serialized_bytes"] = len(serialized)
    result["serialized_sha256"] = hashlib.sha256(serialized).hexdigest()
    hlo = compiled.as_text()
    counts = Counter()
    for line in hlo.splitlines():
        parsed = _parse_instruction(line)
        if parsed is not None:
            counts[parsed["opcode"]] += 1
    assert counts["multiply"] > 0, "opcode instrument missed known arithmetic"
    result["top_five_hlo_ops"] = counts.most_common(5)
    result["hlo_instructions"] = sum(counts.values())
    result["hlo_antiderivative_occurrences"] = hlo.count("_monomial_antiderivative")
    result["hlo_closed_form_occurrences"] = hlo.count("_straight_edge_monomial_moments")
    result["hlo_bernstein_binom_occurrences"] = hlo.count("Bernstein.binom")
    result["completed"] = True
    result["below_byte_ceiling"] = len(serialized) < 100_000_000
    persist(f"{arm}-program.json", result)


def child(arguments, name):
    command = [sys.executable, str(Path(__file__).resolve()), *arguments]
    with (OUTPUT / f"{name}.log").open("w") as log:
        if name.startswith("negative"):
            log.write(MUTATION + "\n")
        log.write(f"revision={revision()} tree={ROOT} command={command!r}\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    (OUTPUT / f"{name}.exit").write_text(str(result.returncode) + "\n")
    print(f"{name}: exit {result.returncode}", flush=True)
    return result.returncode


def row(case, cells):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    from benchmarks.exact_clip_moment_floor import _build
    from nova.equilibrium import clip_quadrature as candidate
    from nova.equilibrium.stencil_mesh import CellCurrentMoments

    baseline = baseline_module()
    operator, support, field, capacity, span = _build(case, -int(cells))
    selected = np.asarray(support.included) & np.asarray(support.boundary)
    cells_index = np.flatnonzero(selected)
    assert len(cells_index) > 0, "row must contain known cut cells"
    vertices = support.support_vertices[cells_index]
    count = support.vertex_count[cells_index]
    points, flux, _, _, centre, scale = candidate._density_sample_field(
        field, cells_index
    )
    density = operator.source.core.current_density(points[..., 0], flux)
    coefficients = candidate._density_coefficients(density)
    local = (vertices - centre[:, None, :]) / scale[:, None, :]
    shifts = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    powers = tuple((p, total - p) for total in range(7) for p in range(total + 1))

    def closed_form(local_vertices, live_count, density_coefficients):
        table = candidate._straight_edge_monomial_moments(
            local_vertices, live_count, max_degree=6
        )
        return jnp.stack(
            [
                jnp.sum(
                    density_coefficients
                    * table[
                        :,
                        jnp.asarray(
                            [
                                powers.index((p + a, q + b))
                                for p, q in candidate._DENSITY_POWERS
                            ]
                        ),
                    ],
                    axis=1,
                )
                for a, b in shifts
            ]
        )

    def gauss(local_vertices, live_count, density_coefficients):
        slot = jnp.arange(local_vertices.shape[1])
        valid = slot[None] < live_count[:, None]
        following_slot = jnp.where(
            slot[None] + 1 < live_count[:, None], slot[None] + 1, 0
        )
        following = jnp.take_along_axis(
            local_vertices, following_slot[..., None], axis=1
        )
        delta = following - local_vertices
        cross = (
            local_vertices[..., 0] * following[..., 1]
            - following[..., 0] * local_vertices[..., 1]
        )
        orientation = jnp.where(
            jnp.sum(jnp.where(valid, cross, 0), axis=1) < 0, -1.0, 1.0
        )
        samples = (
            local_vertices[:, :, None]
            + jnp.asarray(baseline._ARC_EDGE_NODE)[None, None, :, None]
            * delta[:, :, None]
        )
        weights = jnp.where(
            valid[:, :, None],
            jnp.asarray(baseline._ARC_EDGE_WEIGHT)[None, None] * delta[..., 1, None],
            0.0,
        )
        return jnp.stack(
            [
                orientation
                * jnp.sum(
                    weights
                    * baseline._monomial_antiderivative(
                        samples,
                        density_coefficients[:, None, None],
                        baseline._DENSITY_POWERS,
                        a,
                        b,
                    ),
                    axis=(1, 2),
                )
                for a, b in shifts
            ]
        )

    before = np.asarray(jax.jit(gauss)(local, count, coefficients))
    after = np.asarray(jax.jit(closed_form)(local, count, coefficients))

    def physical(values):
        offset = np.asarray(centre - support.centroids[cells_index])
        sx, sy = np.asarray(scale).T
        dx, dy = offset.T
        zero, x, y, xx, xy, yy = values
        return (
            sx
            * sy
            * np.stack(
                (
                    zero,
                    sx * x + dx * zero,
                    sy * y + dy * zero,
                    sx * sx * xx + 2 * dx * sx * x + dx * dx * zero,
                    sx * sy * xy + dx * sy * y + dy * sx * x + dx * dy * zero,
                    sy * sy * yy + 2 * dy * sy * y + dy * dy * zero,
                )
            )
        )

    before, after = physical(before), physical(after)
    relative = np.linalg.norm(after - before, axis=1) / np.maximum(
        np.linalg.norm(before, axis=1), np.finfo(float).tiny
    )

    def production(module):
        return jax.jit(
            lambda carried_support, carried_field: (
                module.clipped_support_current_moments(
                    carried_support,
                    carried_support.included,
                    carried_field,
                    operator.source.core,
                    cut_cell_capacity=capacity,
                    boundary_reduction=True,
                )
            )
        )(support, field)

    before_current = production(baseline)
    after_current = production(candidate)

    def image(moments):
        return np.asarray(
            operator.current_moment_image(
                operator.coupling_current_moments(CellCurrentMoments(*moments))
            )
        )

    image_before, image_after = image(before_current), image(after_current)
    image_delta = float(np.max(np.abs(image_after - image_before)) / abs(span))
    refused_count = support.refused_cells()
    refusal_message = None
    if refused_count:
        try:
            support.assert_no_refusal()
        except ValueError as error:
            refusal_message = str(error)
        assert refusal_message, "capacity refusal must be observable"
    result = {
        "revision": revision(),
        "base_revision": BASE,
        "case": case,
        "requested_cells": int(cells),
        "realised_cells": len(support.vertex_count),
        "cut_cells": len(cells_index),
        "refused_cell_count": refused_count,
        "refusal_message": refusal_message,
        "moment_relative_l2": dict(
            zip(
                (
                    "current",
                    "radial",
                    "vertical",
                    "radial_squared",
                    "radial_vertical",
                    "vertical_squared",
                ),
                relative.tolist(),
                strict=True,
            )
        ),
        "frozen_image_sup_over_span": image_delta,
        "image_floor": 1.52e-15,
        "moment_tolerance": 1e-12,
        "passed": bool(
            np.all(relative <= 1e-12) and image_delta <= 1.52e-15 and refused_count == 0
        ),
        "job": os.environ["SLURM_JOB_ID"],
        "backend": jax.default_backend(),
        "evaluations_before": baseline.cut_cell_moment_evaluation_bound(),
        "evaluations_after": candidate.cut_cell_moment_evaluation_bound(),
    }
    np.savez(
        OUTPUT / f"{case}-{cells}.npz",
        before=before,
        after=after,
        cut_cells=cells_index,
        image_before=image_before,
        image_after=image_after,
    )
    persist(f"{case}-{cells}.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("baseline", "candidate"))
    parser.add_argument("--program", choices=("baseline", "candidate", "negative"))
    parser.add_argument("--row", nargs=2)
    args = parser.parse_args()
    if args.suite:
        return suite(args.suite)
    if args.program:
        executable(args.program)
        return 0
    if args.row:
        row(*args.row)
        return 0
    assert os.environ.get("SLURM_JOB_ID"), "run the controller on all_debug"
    child(["--suite", "baseline"], "baseline-suite")
    if child(["--program", "baseline"], "baseline-program"):
        return 1
    deadline = time.monotonic() + 1200
    while not (OUTPUT / "candidate-ready.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("candidate not committed within twenty minutes")
        time.sleep(5)
    child(["--suite", "candidate"], "candidate-suite")
    child(["--program", "candidate"], "candidate-program")
    child(["--program", "negative"], "negative-program")
    for case in (
        "weak-rotation-reactor-static",
        "moderate-rotation-conventional-static",
        "strong-rotation-compact-static",
        "diverted-single-null",
    ):
        for cells in (110, 300):
            child(["--row", case, str(cells)], f"{case}-{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
