# Primary branch green check

## Verdict

**RED. The batched topology receipt repair clears the topology-container failures, but all five independent recovery assertion failures SURVIVE it.**

The three files were run from a clean, detached worktree at
`c701600431f9a7f4d5d06b885e0213161e329de2`, the merge commit titled
`Merge the batched topology receipt repair`. Immediately before measurement,
`git fetch origin main` established that this commit was current `origin/main`.
During the runs, `origin/main` advanced to
`1efe00ea81d0003fe71a485517ca418829a6e711`; that intervening commit changes
only plan, evidence, crew-state, and manifest documents. It changes no source,
test, configuration, lock, or banked recovery artifact used by these checks, so
the measured code and test inputs are identical to current `origin/main`.

Every file ran in its own fresh process with `JAX_PLATFORMS=cpu` explicitly
pinned and pytest's full slow/fast marker expression enabled. No source, test,
configuration, or banked-result file was modified.

## Exact commands and results

1. Command:

   ```bash
   JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m "slow or not slow" tests/test_equilibrium_forward_solve.py > /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_equilibrium_forward_solve.log 2>&1
   ```

   Result: **28 passed, 0 failed, 0 skipped, 0 errors** in 88.25 s. This file
   previously measured 26 passed and 2 failed, so both container-path failures
   clear under the repair.

2. Command:

   ```bash
   JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m "slow or not slow" tests/test_transport_coupled_window.py > /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_transport_coupled_window.log 2>&1
   ```

   Result: **12 passed, 0 failed, 0 skipped, 0 errors** in 31.33 s. This file
   previously measured 11 passed and 1 failed, so the production coupled-window
   container failure clears under the repair.

3. Command:

   ```bash
   JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m "slow or not slow" tests/test_two_class_recovery.py > /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_two_class_recovery.log 2>&1
   ```

   Result: **0 passed, 5 failed, 0 skipped, 0 errors** in 254.91 s. This is
   unchanged from the earlier 0-passed/5-failed measurement. The failures are
   assertion failures on banked recovery behavior, not the repaired
   `ForwardTopologyState` JAX-container `TypeError`.

## Deciding recovery result

The recovery failures **SURVIVE** the repair:

- both limited banked roots remain topology-consistent and classified limited,
  but `converged` is false, with terminal residuals about 1.29980506 and
  1.30949372;
- the vmapped branch receipt still reports achieved classes `[0, 1]` where the
  bank expects `[0, 0]`;
- the perturbation ladder still reports `receipt.passed == [True, True, False]`;
- the freshly measured pinned-root receipt still differs from the banked
  composition.

Therefore the topology-container regression is repaired, but the primary
branch is not green on the three-file gate. The surviving recovery set is a
second independent regression requiring attribution before any repair.

## Complete surviving assertion tracebacks

The following is the complete pytest failure section, copied verbatim from the
single recovery-process log:

```text
=================================== FAILURES ===================================
__________ test_cold_limited_branch_recovers_each_banked_root[coarse] __________

resolution = 'coarse'

    @pytest.mark.slow
    @pytest.mark.parametrize("resolution", ("coarse", "fine"))
    def test_cold_limited_branch_recovers_each_banked_root(resolution):
        configure_dtypes()
        profile, seeds, receipt, root = _limited_problem(resolution)
        banked_seed = np.asarray(root["seed_state"])
        np.testing.assert_array_equal(np.asarray(seeds.branches.flux[0]), banked_seed)
        assert int(seeds.branches.construction[0]) == int(
            ColdSeedConstruction.CURRENT_CENTROID_DISC
        )
        assert not bool(seeds.branches.stored_flux_samples_used[0])
        assert bool(seeds.branches.anchor_available[0])
        assert not bool(seeds.branches.anchor_available[1])

        portfolio = _solve(profile, seeds.branches.flux)
        limited = int(TopologyClass.LIMITED)
        branch = jax.tree.map(lambda value: value[limited], portfolio.branches)
        assert branch.equilibrium.flux.shape == banked_seed.shape
        assert int(branch.requested_class) == limited
        assert int(branch.achieved_class) == limited
        assert bool(branch.topology_consistent)
>       assert bool(branch.converged)
E       assert False
E        +  where False = bool(Array(False, dtype=bool))
E        +    where Array(False, dtype=bool) = ForwardBranchReceipt(equilibrium=ForwardEquilibrium(flux=Array([-0.48907374, -0.57302252, -0.50351777, ..., -0.6152480...idual=Array(1.29980506, dtype=float64), iterations=Array(10, dtype=int32), topology_consistent=Array(True, dtype=bool)).converged

tests/test_two_class_recovery.py:179: AssertionError
___________ test_cold_limited_branch_recovers_each_banked_root[fine] ___________

resolution = 'fine'

    @pytest.mark.slow
    @pytest.mark.parametrize("resolution", ("coarse", "fine"))
    def test_cold_limited_branch_recovers_each_banked_root(resolution):
        configure_dtypes()
        profile, seeds, receipt, root = _limited_problem(resolution)
        banked_seed = np.asarray(root["seed_state"])
        np.testing.assert_array_equal(np.asarray(seeds.branches.flux[0]), banked_seed)
        assert int(seeds.branches.construction[0]) == int(
            ColdSeedConstruction.CURRENT_CENTROID_DISC
        )
        assert not bool(seeds.branches.stored_flux_samples_used[0])
        assert bool(seeds.branches.anchor_available[0])
        assert not bool(seeds.branches.anchor_available[1])

        portfolio = _solve(profile, seeds.branches.flux)
        limited = int(TopologyClass.LIMITED)
        branch = jax.tree.map(lambda value: value[limited], portfolio.branches)
        assert branch.equilibrium.flux.shape == banked_seed.shape
        assert int(branch.requested_class) == limited
        assert int(branch.achieved_class) == limited
        assert bool(branch.topology_consistent)
>       assert bool(branch.converged)
E       assert False
E        +  where False = bool(Array(False, dtype=bool))
E        +    where Array(False, dtype=bool) = ForwardBranchReceipt(equilibrium=ForwardEquilibrium(flux=Array([-0.58633076, -0.51825696, -0.53748996, ..., -0.6099637...idual=Array(1.30949372, dtype=float64), iterations=Array(10, dtype=int32), topology_consistent=Array(True, dtype=bool)).converged

tests/test_two_class_recovery.py:179: AssertionError
________ test_cold_seed_receipts_keep_one_fixed_branch_axis_under_vmap _________

    def test_cold_seed_receipts_keep_one_fixed_branch_axis_under_vmap():
        configure_dtypes()
        profile, seeds, _fixture, state, geometry = _diverted_problem()

        def solve(states):
            return profile.solve_portfolio(
                states,
                route="newton_krylov",
                tolerance=RECOVERY_CRITERION,
                warmup=0,
                gmres_iterations=KRYLOV_ITERATIONS,
            )

        batch = jnp.stack((seeds.branches.flux, seeds.branches.flux))
        portfolios = jax.jit(jax.vmap(solve))(batch)
        assert seeds.branches.flux.shape == (2, state.size)
        assert seeds.branches.anchor.shape == (2, 2)
        assert int(seeds.branches.construction[1]) == int(
            ColdSeedConstruction.AXIS_SADDLE_GEOMETRY
        )
        np.testing.assert_array_equal(seeds.branches.declared_axis[1], geometry.axis)
        np.testing.assert_array_equal(seeds.branches.declared_boundary[1], geometry.saddle)
        assert not bool(seeds.branches.stored_flux_samples_used[1])
        assert (
            np.linalg.norm(np.asarray(seeds.branches.anchor[1]) - geometry.saddle) < 1.0e-2
        )
        assert portfolios.branches.equilibrium.flux.shape == (2, 2, state.size)
        np.testing.assert_array_equal(
            np.asarray(portfolios.branches.requested_class[0]),
            (int(TopologyClass.LIMITED), int(TopologyClass.DIVERTED)),
        )
>       np.testing.assert_array_equal(
            np.asarray(portfolios.branches.achieved_class[0]),
            (int(TopologyClass.LIMITED), int(TopologyClass.LIMITED)),
        )
E       AssertionError:
E       Arrays are not equal
E
E       Mismatched elements: 1 / 2 (50%)
E       Mismatch at index:
E        [1]: 1 (ACTUAL), 0 (DESIRED)
E       Max absolute difference among violations: 1
E       Max relative difference among violations: inf
E        ACTUAL: array([0, 1], dtype=int8)
E        DESIRED: array([0, 0])

tests/test_two_class_recovery.py:224: AssertionError
______ test_diverted_near_basin_perturbation_ladder_recovers_banked_root _______

    @pytest.mark.slow
    def test_diverted_near_basin_perturbation_ladder_recovers_banked_root():
        configure_dtypes()
        profile, seeds, _fixture, state, _geometry = _diverted_problem()
        diverted = int(TopologyClass.DIVERTED)
        cold_diverted = np.asarray(seeds.branches.flux[diverted])
        direction = cold_diverted - state
        policy = PerturbedSeedPolicy()

        references = jnp.stack((jnp.asarray(state), jnp.asarray(state)))
        directions = jnp.stack((jnp.asarray(direction), jnp.asarray(direction)))
        receipts = jax.jit(
            jax.vmap(
                lambda reference, perturbation: profile.solve_diverted_perturbations(
                    reference,
                    perturbation,
                    policy,
                )
            )
        )(references, directions)
        receipt = jax.tree.map(lambda value: value[0], receipts)

        assert _digest(state) == DIVERTED_STATE_DIGEST
        np.testing.assert_array_equal(
            np.asarray(receipt.relative_amplitude),
            np.asarray(policy.relative_amplitudes),
        )
        actual_amplitude = np.max(
            np.abs(np.asarray(receipt.seed_flux) - state), axis=1
        ) / float(receipt.reference_flux_span)
        np.testing.assert_allclose(actual_amplitude, policy.relative_amplitudes)
        np.testing.assert_array_equal(
            np.asarray(receipt.rungs.requested_class),
            np.full(len(policy.relative_amplitudes), diverted),
        )
>       assert np.all(np.asarray(receipt.passed))
E       assert np.False_
E        +  where np.False_ = <function all at 0x7fce84014170>(array([ True,  True, False]))
E        +    where <function all at 0x7fce84014170> = np.all
E        +    and   array([ True,  True, False]) = <built-in function asarray>(Array([ True,  True, False], dtype=bool))
E        +      where <built-in function asarray> = np.asarray
E        +      and   Array([ True,  True, False], dtype=bool) = ForwardPerturbedSeedReceipt(relative_amplitude=Array([0.001, 0.01 , 0.05 ], dtype=float64), reference_flux_span=Array(... dtype=float64), passed=Array([ True,  True, False], dtype=bool), largest_passing_amplitude=Array(0.01, dtype=float64)).passed

tests/test_two_class_recovery.py:284: AssertionError
________ test_banked_diverted_state_is_a_machine_precision_pinned_root _________

    @pytest.mark.slow
    def test_banked_diverted_state_is_a_machine_precision_pinned_root():
        banked = json.loads(ROOT_RECEIPT_PATH.read_text(encoding="utf-8"))
        measured = qualify(write=False)
>       assert measured == banked
E       AssertionError: assert {'schema': 'n...ue, ...}, ...} == {'composition...receipt', ...}
E
E         Omitting 5 identical items, use -vv to show
E         Differing items:
E         {'composition': {'external_field': {'sha256': 'd6941b63cd30c1a60b31cd18bb3f473e27c500295fa0155251583ae6c23c69e6', 'max..._boundary_m': [1.2043590455560238, -0.43060062587647197], ...}, 'closure_absolute_residual_wb': 4.440892098500626e-16}} != {'composition': {'closure_absolute_residual_wb': 4.440892098500626e-16, 'external_field': {'maximum_absolute_flux_wb':...m_absolute_flux_wb': 0.14172801675859822, 'p_prime_pa_per_wb': -16125.767218728875, 'repeat_difference_wb': 0.0, ...}}}
E         Use -v to get more diff

tests/test_two_class_recovery.py:320: AssertionError
```

## Logs

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_equilibrium_forward_solve.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_transport_coupled_window.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_two_class_recovery.log`

## Follow-on boundary

Attribution and repair of the five recovery regressions are outside this test
node's write scope. The next action is an independent recovery-regression triage
against the surviving assertion set, preserving the banked machine-precision
pinned-root result until the composition difference is explained.
