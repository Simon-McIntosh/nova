node: primary-branch-vmap-status
status: complete
commits: 5db3c436
changed_paths: docs/figures/discrete-operator-analytic-error/primary-branch-vmap-status.md; /home/ITER/mcintos/Code/nova/docs/state/nova/manifests/primary-branch-vmap-status.md
tests: |
  RED PRIMARY BRANCH at freshly fetched origin/main 0747683489a8ba637731b4b4873572f62990477e, with a clean and unmodified checkout.

  Exact command:
  JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider -m "slow or not slow" tests/test_equilibrium_forward_solve.py
  Verbatim verdict: collected 28; 26 passed, 2 failed, 0 skipped, 0 errors; 1 warning; 57.68 s.

  Both failing tracebacks are preserved verbatim in test_equilibrium_forward_solve_cpu.log. The discriminating failure is:

  > portfolio = jax.jit(solve_portfolio)(seeds)
  tests/test_equilibrium_forward_solve.py:782
  tests/test_equilibrium_forward_solve.py:775: in solve_portfolio
      return profile.solve_portfolio(
  nova/equilibrium/forward.py:1402: in solve_portfolio
      branches = jax.vmap(
  E TypeError: Output from batched function ForwardTopologyState(axis=VmapTracer(aval=float64[2], batched=float64[2,2]), axis_flux=VmapTracer(aval=float64[], batched=float64[2]), boundary=VmapTracer(aval=float64[2], batched=float64[2,2]), boundary_flux=VmapTracer(aval=float64[], batched=float64[2]), x_point=VmapTracer(aval=float64[2], batched=float64[2,2]), x_point_flux=VmapTracer(aval=float64[], batched=float64[2]), wall_point=VmapTracer(aval=float64[2], batched=float64[2,2]), wall_point_flux=VmapTracer(aval=float64[], batched=float64[2]), diverted=VmapTracer(aval=bool[], batched=bool[2])) with type <class 'nova.equilibrium.forward_operator.ForwardTopologyState'> is not a valid JAX type
  E --------------------
  E For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

  The same file also failed test_the_batched_ensemble_solve_matches_the_per_slice_solve at forward.py:1444 with the identical exception from solve_batch's inner vmap. The named portfolio test containing line 809 was collected, but it failed at line 782 before reaching jax.jit(jax.vmap(solve_portfolio)) on line 809 or its ensemble.branches assertions. It therefore did not pass.

  Exact command:
  JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider -m "slow or not slow" tests/test_transport_coupled_window.py
  Verbatim verdict: collected 12; 11 passed, 1 failed, 0 skipped, 0 errors; 34.22 s.

  Full assertion and traceback are preserved verbatim in test_transport_coupled_window_cpu.log. The failure is:

  > sweep = equilibrium_sweep(
  tests/test_transport_coupled_window.py:276
  nova/transport/coupled_window.py:1636: in equilibrium_sweep
      portfolio = sampled_profile.solve_portfolio(
  nova/equilibrium/forward.py:1402: in solve_portfolio
      branches = jax.vmap(
  E TypeError: Output from batched function ForwardTopologyState(...) with type <class 'nova.equilibrium.forward_operator.ForwardTopologyState'> is not a valid JAX type

  Exact command:
  JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider -m "slow or not slow" tests/test_two_class_recovery.py
  Verbatim verdict: collected 5; 0 passed, 5 failed, 0 skipped, 0 errors; 273.58 s.

  All five assertion tracebacks are preserved verbatim in test_two_class_recovery_cpu.log. Their exact assertion headlines are:

  > assert bool(branch.converged)
  E assert False
  test_cold_limited_branch_recovers_each_banked_root[coarse]

  > assert bool(branch.converged)
  E assert False
  test_cold_limited_branch_recovers_each_banked_root[fine]

  > np.testing.assert_array_equal(np.asarray(portfolios.branches.achieved_class[0]), (int(TopologyClass.LIMITED), int(TopologyClass.LIMITED)))
  E AssertionError: Arrays are not equal; ACTUAL array([0, 1], dtype=int8), DESIRED array([0, 0])
  test_cold_seed_receipts_keep_one_fixed_branch_axis_under_vmap

  > assert np.all(np.asarray(receipt.passed))
  E assert np.False_; array([True, True, False])
  test_diverted_near_basin_perturbation_ladder_recovers_banked_root

  > assert measured == banked
  E AssertionError: measured and banked composition differ
  test_banked_diverted_state_is_a_machine_precision_pinned_root

  The first unpinned forward-solve process selected CUDA and exhausted device memory during module-fixture setup: collected 28, 2 passed, 0 failed, 0 skipped, 26 errors in 14.58 s. It is retained only as environmental evidence and was superseded by the authoritative fresh CPU process above.
test_logs: /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T091038465601-primary-branch-vmap-status/logs/test_equilibrium_forward_solve_cpu.log; /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T091038465601-primary-branch-vmap-status/logs/test_transport_coupled_window_cpu.log; /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T091038465601-primary-branch-vmap-status/logs/test_two_class_recovery_cpu.log; /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T091038465601-primary-branch-vmap-status/logs/test_equilibrium_forward_solve.log
artifacts: docs/figures/discrete-operator-analytic-error/primary-branch-vmap-status.md — current origin/main is red: forward solve 26 pass/2 fail, transport 11 pass/1 fail, recovery 0 pass/5 fail; primary regression is unregistered ForwardTopologyState crossing inner vmap output.
evidence_inputs: |
  ForwardFluxOperator.read at forward_operator.py:470-484 constructs ForwardTopologyState with a Python lambda in _class_margin_read on every call, whether or not class_margin is consumed.
  ForwardProfile.solve_portfolio at forward.py:1361-1416 places _branch_receipt inside an inner branch-axis jax.vmap. The returned ForwardEquilibrium carries ForwardTopologyState in its topology field, so JAX must flatten it as transformed output and rejects the unregistered dataclass. Margin consumption is not required.
  ForwardProfile.solve_batch at forward.py:1418-1449 has the same failure shape through its own inner ensemble-axis jax.vmap.
  equilibrium_sweep at coupled_window.py:1636 is not itself jitted or vmapped; it still fails because solve_portfolio supplies the inner vmap.
  ForwardFluxOperator.topology_margin at forward_operator.py:454-468 does not construct ForwardTopologyState. It obtains the original NamedTuple TopologyState and directly consumes _connectivity_class_margin. Direct eager operator.read constructs ForwardTopologyState outside a transform.
  Therefore the presumption that the callable matters only when margin is consumed is false for the observed regression. The existing forward portfolio test already covers the failing transformed-construction/output path. It does not consume the deferred margin, so a different focused transformed-margin test is needed for that path, around jax.jit(operator.topology_margin)(state) and, if transformed public read is a supported contract, jax.jit(lambda state: operator.read(state)[1].class_margin)(state).
  The existing portfolio test cannot reach line 809's outer jit(vmap) assertion until the earlier inner branch-vmap output at line 782 is traceable.
  The recovery file's jax.jit(jax.vmap(solve)) call completed and reached later physics assertions; it did not reproduce the container exception in that fixture/route. It never consumes class_margin and does not qualify the deferred-margin path. Its five failures are separate banked-result regressions.
follow_ons: Repair is fenced to the sibling owner. Keep the existing portfolio test as the container regression gate; add a distinct transformed topology-margin consumption gate in the repair scope. Independently triage the five stale/banked recovery assertions; this node made no source or test changes.
blockers: none
