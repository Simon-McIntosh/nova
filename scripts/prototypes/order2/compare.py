"""Measure isolated native and batched point-kernel reads."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("native", "batched"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"revision={revision} tree={root} command="
        + json.dumps([sys.executable, *sys.argv]),
        flush=True,
    )
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import topology
    from scripts.prototypes.order2.read import batched_field, make_read
    from scripts.prototypes.order2.attribution import attribute_graph
    from tests.equilibrium import test_topology_read as fixtures

    assert jax.config.jax_enable_x64 is True
    assert not jax.config.jax_enable_compilation_cache
    assert jax.default_backend() == "gpu"
    print(f"MEASUREMENT_MODULE={topology.__file__}", flush=True)
    print(f"MEASUREMENT_CWD={Path.cwd().resolve()}", flush=True)
    output = args.directory / f"{args.arm}-{args.cells}.json"
    receipt = dict(
        revision=revision,
        job_id=os.environ["SLURM_JOB_ID"],
        log_path=os.environ["MEASUREMENT_LOG"],
        arm=args.arm,
        cells=args.cells,
        device=jax.devices()[0].device_kind,
        cache_enabled=False,
        measurement_module=topology.__file__,
        measurement_cwd=str(Path.cwd().resolve()),
    )

    def checkpoint(phase):
        receipt["phase"] = phase
        receipt["peak_host_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        print("ROW " + json.dumps(receipt), flush=True)

    oracle, total, wall, axis, _ = fixtures._analytic_inputs("limited")
    geometry = fixtures._realised_hex_geometry(wall, args.cells)
    field = fixtures._kernel_backed_field("limited", oracle, total, geometry)
    if args.arm == "batched":
        field = batched_field(field)
    read = topology.read if args.arm == "native" else make_read()
    operands = (
        field,
        geometry,
        topology.TopologyConvention.from_cocos(17, 1.0),
        fixtures.TopologyPolicy(),
    )
    receipt["realised_cells"] = len(geometry.centre)
    receipt["pitch_m"] = float(np.median(geometry.pitch))
    checkpoint("input-ready")
    start = time.perf_counter()
    graph = jax.make_jaxpr(read)(*operands)
    receipt["trace_seconds"] = time.perf_counter() - start
    receipt.update(fixtures._jaxpr_equation_counts(graph))
    if args.arm == "native" and args.cells == 132:
        assert receipt["expanded_equations"] == 827553
    attribution = attribute_graph(graph)
    assert attribution["expanded_equations"] == receipt["expanded_equations"]
    attribution_path = args.directory / f"{args.arm}-{args.cells}-attribution.json"
    attribution_path.write_text(json.dumps(attribution, indent=2) + "\n")
    receipt["attribution_path"] = str(attribution_path)
    receipt["packed_kernel_invocations"] = attribution["packed_kernel_invocations"]
    receipt["hessian_invocations"] = attribution["hessian_invocations"]
    if args.arm == "native" and args.cells == 132:
        assert receipt["packed_kernel_invocations"] == 10
        assert receipt["hessian_invocations"] == 6
    stencil = jax.make_jaxpr(topology._curvature_derivatives)(
        field, jnp.asarray(axis), jnp.asarray(receipt["pitch_m"])
    )
    receipt["stencil_equations"] = fixtures._jaxpr_equation_counts(stencil)
    checkpoint("traced")
    del graph, stencil
    assert not jax.config.jax_enable_compilation_cache
    start = time.perf_counter()
    lowered = jax.jit(read).lower(*operands)
    receipt["lower_seconds"] = time.perf_counter() - start
    checkpoint("lowered")
    start = time.perf_counter()
    executable = lowered.compile()
    receipt["compile_seconds"] = time.perf_counter() - start
    receipt["cold_compile_wall_seconds"] = (
        receipt["lower_seconds"] + receipt["compile_seconds"]
    )
    receipt["executable_bytes"] = len(executable.runtime_executable().serialize())
    receipt["compile_peak_host_rss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    checkpoint("compiled")
    result = executable(*operands)
    jax.block_until_ready(result)
    assert result.valid and result.qualified
    arrays = {
        name: np.asarray(getattr(result, name))
        for name in (
            "axis",
            "boundary",
            "boundary_flux",
            "boundary_class",
            "x_points",
            "x_point_valid",
        )
    }
    arrays.update(
        {
            f"saddle_{name}": np.asarray(getattr(result.saddle_form, name))
            for name in ("position", "direction", "curvature", "cubic")
        }
    )
    # The limited read has no saddle: compare a nonzero off-axis jet as well.
    if args.cells == 132:
        point = jnp.asarray(axis) + jnp.asarray((0.2, 0.1)) * receipt["pitch_m"]
        jet = jax.jit(lambda f, p: f.evaluate(p))(field, point)
        jax.block_until_ready(jet)
        arrays.update(
            {
                f"point_{name}": np.asarray(getattr(jet, name))
                for name in ("value", "gradient", "hessian")
            }
        )
        assert np.linalg.norm(arrays["point_hessian"]) > 0
    arrays_path = args.directory / f"{args.arm}-{args.cells}-outputs.npz"
    np.savez(arrays_path, **arrays)
    receipt["output_path"] = str(arrays_path)
    receipt["valid"] = bool(result.valid)
    receipt["qualified"] = bool(result.qualified)
    receipt["boundary_class"] = int(result.boundary_class)
    if args.arm == "batched":
        native = np.load(args.directory / f"native-{args.cells}-outputs.npz")
        differences = {}
        for name, value in arrays.items():
            reference = native[name]
            if value.dtype.kind in "biu":
                np.testing.assert_array_equal(value, reference)
                differences[name] = 0.0
            else:
                np.testing.assert_allclose(value, reference, rtol=1e-12, atol=0)
                scale = float(np.max(np.abs(reference))) if reference.size else 0.0
                error = (
                    float(np.max(np.abs(value - reference))) if reference.size else 0.0
                )
                differences[name] = error / scale if scale else error
        receipt["maximum_relative_differences"] = differences
        receipt["identity_passed"] = True
    checkpoint("executed")
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
