"""Audit the derivative ceiling and reproduce the native read's cold cost."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

# Cache selection precedes importing JAX or any module that constructs arrays.
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"revision={revision} tree={root} command=" + json.dumps(sys.argv), flush=True
    )
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import topology
    from scripts.prototypes.order2.read import make_read
    from tests.equilibrium import test_topology_read as fixtures

    assert jax.config.jax_enable_x64 is True
    assert not jax.config.jax_enable_compilation_cache
    assert jax.default_backend() == "gpu"
    print(f"MEASUREMENT_MODULE={topology.__file__}", flush=True)
    print(f"MEASUREMENT_CWD={Path.cwd().resolve()}", flush=True)
    print(f"DEVICE={jax.devices()[0].device_kind}", flush=True)
    receipt = {
        "revision": revision,
        "job_id": os.environ["SLURM_JOB_ID"],
        "log_path": os.environ["MEASUREMENT_LOG"],
        "module": topology.__file__,
        "cwd": str(Path.cwd().resolve()),
        "cache_enabled": bool(jax.config.jax_enable_compilation_cache),
        "device": jax.devices()[0].device_kind,
    }

    def checkpoint(phase):
        receipt["phase"] = phase
        receipt["peak_host_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print("MEASUREMENT " + json.dumps(receipt), flush=True)

    oracle, total, wall, axis, _ = fixtures._analytic_inputs("limited")
    geometry = fixtures._realised_hex_geometry(wall, 132)
    field = fixtures._kernel_backed_field("limited", oracle, total, geometry)
    assert isinstance(field, topology.TotalField)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = fixtures.TopologyPolicy()
    operands = field, geometry, convention, policy
    receipt["cells"] = len(geometry.centre)
    receipt["kernel_edges"] = field.coupling.edge.shape[0]
    receipt["pitch_m"] = float(np.median(geometry.pitch))
    receipt["spacing_m"] = 0.01 * receipt["pitch_m"]
    checkpoint("input-ready")

    # Exact higher derivatives are a positive control for the graph instrument.
    point = jnp.asarray(axis)
    pitch = jnp.asarray(receipt["pitch_m"])
    stencil_graph = jax.make_jaxpr(topology._curvature_derivatives)(field, point, pitch)

    def exact_curvature(operand, target):
        def hessian(p):
            return operand.evaluate(p).hessian

        return jax.jacfwd(hessian)(target), jax.jacfwd(jax.jacfwd(hessian))(target)

    exact_graph = jax.make_jaxpr(exact_curvature)(field, point)
    receipt["stencil_curvature_graph"] = fixtures._jaxpr_equation_counts(stencil_graph)
    receipt["exact_curvature_graph"] = fixtures._jaxpr_equation_counts(exact_graph)
    assert (
        receipt["exact_curvature_graph"]["expanded_equations"]
        > receipt["stencil_curvature_graph"]["expanded_equations"]
    )
    checkpoint("derivative-ceiling-audited")
    del exact_graph, stencil_graph
    jax.clear_caches()

    start = time.perf_counter()
    graph = jax.make_jaxpr(topology.read)(*operands)
    receipt["trace_seconds"] = time.perf_counter() - start
    receipt["current_graph"] = fixtures._jaxpr_equation_counts(graph)
    variant_graph = jax.make_jaxpr(make_read())(*operands)
    receipt["prototype_graph"] = fixtures._jaxpr_equation_counts(variant_graph)
    receipt["default_graph_identical"] = str(graph) == str(variant_graph)
    assert receipt["default_graph_identical"], (
        "default prototype must preserve the read"
    )
    # The recorded read is already the finite-Hessian-stencil graph.
    receipt["recorded_expanded_equations"] = 827553
    receipt["recorded_cold_compile_seconds"] = 771.9303107708693
    receipt["baseline_graph_reproduced"] = (
        receipt["current_graph"]["expanded_equations"] == 827553
    )
    checkpoint("traced")
    if not receipt["baseline_graph_reproduced"]:
        raise AssertionError("recorded graph differs; do not measure a variant")
    del graph, variant_graph
    start = time.perf_counter()
    lowered = jax.jit(topology.read).lower(*operands)
    receipt["lower_seconds"] = time.perf_counter() - start
    checkpoint("lowered")
    start = time.perf_counter()
    executable = lowered.compile()
    receipt["compile_seconds"] = time.perf_counter() - start
    receipt["cold_compile_wall_seconds"] = (
        receipt["lower_seconds"] + receipt["compile_seconds"]
    )
    receipt["executable_bytes"] = len(executable.runtime_executable().serialize())
    receipt["compile_clause_met"] = receipt["cold_compile_wall_seconds"] <= 60.0
    checkpoint("compiled")
    result = executable(*operands)
    jax.block_until_ready(result)
    receipt["valid"] = bool(result.valid)
    receipt["qualified"] = bool(result.qualified)
    receipt["reason"] = int(result.reason)
    receipt["axis_m"] = np.asarray(result.axis).tolist()
    receipt["analytic_axis_m"] = np.asarray(axis).tolist()
    receipt["axis_error_m"] = float(np.linalg.norm(np.asarray(result.axis) - axis))
    receipt["boundary_class"] = int(result.boundary_class)
    receipt["admitted_x_points"] = int(np.count_nonzero(result.x_point_valid))
    assert result.valid and result.qualified
    assert receipt["axis_error_m"] < receipt["pitch_m"]
    checkpoint("executed")
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
