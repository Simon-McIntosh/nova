# RED: current `origin/main` rejects `ForwardTopologyState` at portfolio batch boundaries

Measured commit: `0747683489a8ba637731b4b4873572f62990477e`, equal to freshly fetched `origin/main` before execution. The checkout was clean and source and tests were unmodified.

## Verbatim branch failure

The named portfolio test was collected, but did not pass. It failed at the earlier `jax.jit(solve_portfolio)(seeds)` call on line 782, before the test could reach the outer `jax.jit(jax.vmap(solve_portfolio))` call on line 809 or any assertion on `ensemble.branches`:

```text
_________ test_the_topology_portfolio_matches_per_branch_solves_under_jit_and_vmap ___

axis_data = AxisData(name=<object object at 0x7fe0b6f1afa0>, size=2, spmd_name=None, explicit_mesh_axis=None)
src = None, dst = 0
x = ForwardTopologyState(axis=VmapTracer(aval=float64[2], batched=float64[2,2]), axis_flux=VmapTracer(aval=float64[], batched=float64[2]), boundary=VmapTracer(aval=float64[2], batched=float64[2,2]), boundary_flux=VmapTracer(aval=float64[], batched=float64[2]), x_point=VmapTracer(aval=float64[2], batched=float64[2,2]), x_point_flux=VmapTracer(aval=float64[], batched=float64[2]), wall_point=VmapTracer(aval=float64[2], batched=float64[2,2]), wall_point_flux=VmapTracer(aval=float64[], batched=float64[2]), diverted=VmapTracer(aval=bool[], batched=bool[2]))
sum_match = False

    def matchaxis(axis_data, src, dst, x, sum_match=False):
      try:
>       _ = core.typeof(x)
            ^^^^^^^^^^^^^^

../../../../nova/.venv/lib/python3.14/site-packages/jax/_src/interpreters/batching.py:723:

    def typeof(x: Any) -> Any:
      """Return the JAX type (i.e. :class:`AbstractValue`) of the input.

      Raises a ``TypeError`` if ``x`` is not a valid JAX type.
      """
      typ = type(x)
      if (aval_fn := pytype_aval_mappings.get(typ)):  # fast path
        return aval_fn(x)
      for t in typ.__mro__[1:]:
        if (aval_fn := pytype_aval_mappings.get(t)):
          return aval_fn(x)
      if getattr(x, '__jax_array__', None) is not None:
        raise ValueError(
            'Triggering __jax_array__() during abstractification is no longer'
            ' supported. To avoid this error, either explicitly convert your object'
            ' using jax.numpy.array(), or register your object as a pytree.'
        )
>     raise TypeError(f"Argument '{x}' of type '{typ}' is not a valid JAX type")
E     TypeError: Argument 'ForwardTopologyState(axis=VmapTracer(aval=float64[2], batched=float64[2,2]), axis_flux=VmapTracer(aval=float64[], batched=float64[2]), boundary=VmapTracer(aval=float64[2], batched=float64[2,2]), boundary_flux=VmapTracer(aval=float64[], batched=float64[2]), x_point=VmapTracer(aval=float64[2], batched=float64[2,2]), x_point_flux=VmapTracer(aval=float64[], batched=float64[2]), wall_point=VmapTracer(aval=float64[2], batched=float64[2,2]), wall_point_flux=VmapTracer(aval=float64[], batched=float64[2]), diverted=VmapTracer(aval=bool[], batched=bool[2]))' of type '<class 'nova.equilibrium.forward_operator.ForwardTopologyState'>' is not a valid JAX type

../../../../nova/.venv/lib/python3.14/site-packages/jax/_src/core.py:1975: TypeError

The above exception was the direct cause of the following exception:

    def test_the_topology_portfolio_matches_per_branch_solves_under_jit_and_vmap(
        machine,
    ):
        """The branch axis and an outer ensemble axis share the same solve path."""

        profile, seed, _vacuum = machine
        seeds = jnp.stack((seed, seed))

        def solve_portfolio(state):
            return profile.solve_portfolio(
                state,
                route="picard",
                evaluations=1,
                tolerance=np.inf,
            )

>       portfolio = jax.jit(solve_portfolio)(seeds)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_equilibrium_forward_solve.py:782:
tests/test_equilibrium_forward_solve.py:775: in solve_portfolio
    return profile.solve_portfolio(
nova/equilibrium/forward.py:1402: in solve_portfolio
    branches = jax.vmap(
E   TypeError: Output from batched function ForwardTopologyState(axis=VmapTracer(aval=float64[2], batched=float64[2,2]), axis_flux=VmapTracer(aval=float64[], batched=float64[2]), boundary=VmapTracer(aval=float64[2], batched=float64[2,2]), boundary_flux=VmapTracer(aval=float64[], batched=float64[2]), x_point=VmapTracer(aval=float64[2], batched=float64[2,2]), x_point_flux=VmapTracer(aval=float64[], batched=float64[2]), wall_point=VmapTracer(aval=float64[2], batched=float64[2,2]), wall_point_flux=VmapTracer(aval=float64[], batched=float64[2]), diverted=VmapTracer(aval=bool[], batched=bool[2])) with type <class 'nova.equilibrium.forward_operator.ForwardTopologyState'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
```

The same file also failed `test_the_batched_ensemble_solve_matches_the_per_slice_solve` at `forward.py:1444`, with the identical JAX-type exception from the `solve_batch` inner `vmap`. The complete unabridged failure blocks are retained in the test log listed below.

The production transport test reaches the same failure without an outer `jit`:

```text
__ test_equilibrium_sweep_consumes_interpolated_sources_and_returns_receipts ___

>       sweep = equilibrium_sweep(
            profile,
            converged.flux,
            source_waveform,
            coarse_time,
            source_from_sample,
            route="anderson",
            solve_options={"evaluations": EVALUATIONS},
        )

tests/test_transport_coupled_window.py:276:
nova/transport/coupled_window.py:1636: in equilibrium_sweep
    portfolio = sampled_profile.solve_portfolio(
nova/equilibrium/forward.py:1402: in solve_portfolio
    branches = jax.vmap(
E   TypeError: Output from batched function ForwardTopologyState(axis=VmapTracer(aval=float64[2], batched=float64[2,2]), axis_flux=VmapTracer(aval=float64[], batched=float64[2]), boundary=VmapTracer(aval=float64[2], batched=float64[2,2]), boundary_flux=VmapTracer(aval=float64[], batched=float64[2]), x_point=VmapTracer(aval=float64[2], batched=float64[2,2]), x_point_flux=VmapTracer(aval=float64[], batched=float64[2]), wall_point=VmapTracer(aval=float64[2], batched=float64[2,2]), wall_point_flux=VmapTracer(aval=float64[], batched=float64[2]), diverted=VmapTracer(aval=bool[], batched=bool[2])) with type <class 'nova.equilibrium.forward_operator.ForwardTopologyState'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
```

## Exact commands and counts

All authoritative counts below came from separate fresh CPU-pinned processes. CPU pinning was necessary because the first unpinned process selected the login GPU, exhausted device memory in fixture setup, and produced 26 environmental setup errors before the discriminating test could run.

```text
JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider -m "slow or not slow" tests/test_equilibrium_forward_solve.py
collected 28; passed 26; failed 2; skipped 0; errors 0; 1 warning; 57.68 s

JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider -m "slow or not slow" tests/test_transport_coupled_window.py
collected 12; passed 11; failed 1; skipped 0; errors 0; 34.22 s

JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider -m "slow or not slow" tests/test_two_class_recovery.py
collected 5; passed 0; failed 5; skipped 0; errors 0; 273.58 s
```

The five recovery failures are independent banked-result assertions, not JAX container errors:

```text
test_cold_limited_branch_recovers_each_banked_root[coarse]
>       assert bool(branch.converged)
E       assert False
E        +  where False = bool(Array(False, dtype=bool))

test_cold_limited_branch_recovers_each_banked_root[fine]
>       assert bool(branch.converged)
E       assert False
E        +  where False = bool(Array(False, dtype=bool))

test_cold_seed_receipts_keep_one_fixed_branch_axis_under_vmap
>       np.testing.assert_array_equal(
            np.asarray(portfolios.branches.achieved_class[0]),
            (int(TopologyClass.LIMITED), int(TopologyClass.LIMITED)),
        )
E       AssertionError: Arrays are not equal
E       Mismatched elements: 1 / 2 (50%)
E       ACTUAL: array([0, 1], dtype=int8)
E       DESIRED: array([0, 0])

test_diverted_near_basin_perturbation_ladder_recovers_banked_root
>       assert np.all(np.asarray(receipt.passed))
E       assert np.False_
E        +  where np.False_ = <function all at 0x7f45a0305f70>(array([ True,  True, False]))

test_banked_diverted_state_is_a_machine_precision_pinned_root
>       assert measured == banked
E       AssertionError: assert {'schema': 'n...ue, ...}, ...} == {'composition...receipt', ...}
E         Omitting 5 identical items, use -vv to show
E         Differing items:
E         {'composition': {'external_field': {'sha256': 'd6941b63cd30c1a60b31cd18bb3f473e27c500295fa0155251583ae6c23c69e6', 'maximum_absolute_flux_wb': ...}, 'closure_absolute_residual_wb': 4.440892098500626e-16}} != {'composition': {'closure_absolute_residual_wb': 4.440892098500626e-16, 'external_field': {'maximum_absolute_flux_wb': ...}}}
```

The complete assertion contexts and tracebacks for all five are in the recovery log.

## Discriminating code-path answer

`ForwardFluxOperator.read()` at `forward_operator.py:470-484` always constructs a `ForwardTopologyState` and stores the deferred margin as a Python closure: `lambda: self._connectivity_class_margin(physical, topology)`. The callable is therefore present whether or not `.class_margin` is consumed.

`ForwardProfile.solve_portfolio()` at `forward.py:1361-1416` places `_branch_receipt()` inside its own branch-axis `jax.vmap`. `_branch_receipt()` returns a `ForwardEquilibrium`; `_receipt()` stores the result of `operator.read()` in `ForwardEquilibrium.topology`. Consequently `ForwardTopologyState` is constructed inside the branch transform and becomes part of that transform's output. JAX must flatten the output at the `vmap` boundary and rejects the unregistered frozen dataclass. Margin consumption is not required. The production `equilibrium_sweep()` call at `coupled_window.py:1636` therefore fails even though `equilibrium_sweep()` itself is eager: `solve_portfolio()` supplies the inner transform.

`ForwardProfile.solve_batch()` at `forward.py:1418-1449` has the same shape: `_solve_accelerated()` is mapped by an inner ensemble-axis `jax.vmap`, its `ForwardEquilibrium` output carries `ForwardTopologyState`, and the second forward-solve failure occurs at that boundary.

By contrast, `ForwardFluxOperator.topology_margin()` at `forward_operator.py:454-468` does not construct `ForwardTopologyState`. It obtains the original `TopologyState` from `_fixed_design_topology.read()` and immediately calls `_connectivity_class_margin()`. This is the actual deferred-margin consumption route. Direct eager `operator.read()` constructs `ForwardTopologyState` outside a caller transform; it remains usable until the state itself must cross a JAX transform boundary.

The assumption that the callable is only a problem when the margin is consumed is therefore false for the observed regression: merely returning the callable-bearing state from `vmap` is enough. The existing forward portfolio test already covers the failing construction/output path and should remain the regression gate. It does not cover deferred margin consumption, because neither the test nor `solve_portfolio()` accesses `.class_margin`; a separate focused transformed-margin test is needed for that path, preferably around `jax.jit(operator.topology_margin)(state)` and, if the public `read()` contract is expected to support it, `jax.jit(lambda state: operator.read(state)[1].class_margin)(state)`. The existing test cannot reach its line-809 outer-ensemble assertion until the inner branch-axis output is made traceable.

The recovery file provides an important negative discriminator: its `jax.jit(jax.vmap(solve))` call at line 207 completed and reached later physics assertions, so it did not reproduce the JAX container exception in that fixture/route. It nevertheless does not consume the deferred margin and cannot qualify that path; its five failures are separate stale/banked expectation failures.

## Logs

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T091038465601-primary-branch-vmap-status/logs/test_equilibrium_forward_solve_cpu.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T091038465601-primary-branch-vmap-status/logs/test_transport_coupled_window_cpu.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T091038465601-primary-branch-vmap-status/logs/test_two_class_recovery_cpu.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T091038465601-primary-branch-vmap-status/logs/test_equilibrium_forward_solve.log` — retained environmental GPU-OOM attempt: collected 28, passed 2, errors 26, failed 0, skipped 0.
