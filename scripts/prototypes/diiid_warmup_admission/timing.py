"""Record actual backend compilation separately from instrumented solve wall."""

from __future__ import annotations

import json
from time import perf_counter


class CompilationTiming:
    """Time backend compile/load calls, including programs not persisted in cache."""

    def __init__(self):
        from jax._src import compiler

        self.rows = []
        original = compiler.backend_compile_and_load

        def measured(*args, **kwargs):
            module = args[1] if len(args) > 1 else kwargs["module"]
            name = str(module.operation.attributes["sym_name"])
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                row = dict(program=name, compile_load_seconds=perf_counter() - started)
                self.rows.append(row)
                print("BACKEND_COMPILE " + json.dumps(row), flush=True)

        compiler.backend_compile_and_load = measured

    def control(self):
        import jax
        import jax.numpy as jnp

        jax.jit(lambda x: jnp.sin(x) + 0.137)(jnp.ones(7)).block_until_ready()
        assert self.rows and sum(r["compile_load_seconds"] for r in self.rows) > 0
        print("COMPILE_CONTROL=" + json.dumps(self.rows), flush=True)
        self.rows.clear()

    def install_solve(self):
        import jax

        from nova.equilibrium.forward import ForwardProfile
        from nova.equilibrium.solve_request import ForwardSolveRequest

        original = ForwardProfile.solve
        self.solves = []

        def measured(profile, request=None, *args, **kwargs):
            if not isinstance(request, ForwardSolveRequest):
                return original(profile, request, *args, **kwargs)
            jax.effects_barrier()
            start_row = len(self.rows)
            started = perf_counter()
            result = original(profile, request, *args, **kwargs)
            jax.block_until_ready(result.equilibrium.flux)
            jax.effects_barrier()
            wall = perf_counter() - started
            compile_seconds = sum(
                row["compile_load_seconds"] for row in self.rows[start_row:]
            )
            row = dict(
                solve_wall_seconds=wall,
                cold_compile_load_seconds=compile_seconds,
                execute_and_host_seconds=wall - compile_seconds,
                backend=jax.default_backend(),
                device=str(jax.devices()[0]),
                device_kind=jax.devices()[0].device_kind,
                timing_contract="execute_and_host excludes backend compile/load; "
                "includes tracing, dispatch, synchronization and operand callbacks",
            )
            self.solves.append(row)
            print("SOLVE_TIMING " + json.dumps(row), flush=True)
            return result

        ForwardProfile.solve = measured
