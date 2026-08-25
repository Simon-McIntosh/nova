# Post-cutover forward-solve verification

## Verdict

**AT BASELINE.** The saddle-aware partition cutover left the four named
forward-solve test surfaces at their expected state. Three files are fully
green. The recovery file retains exactly the two deliberately visible
cold-start basin failures, at the exact expected residuals, and its other
three tests pass. No source, test, or banked artifact was changed.

Each file ran once in its own fresh process with `JAX_PLATFORMS=cpu`, the
shared Nova environment, the worktree code first on `PYTHONPATH`, and
`-m 'slow or not slow'`.

## Quantitative comparison

| File | Expected baseline | Observed | Baseline verdict | Runtime |
| --- | --- | --- | --- | --- |
| `tests/test_equilibrium_forward_solve.py` | 28 passed, 0 failed | 28 passed, 0 failed, 0 skipped, 0 errors | exact | 666.70 s |
| `tests/test_transport_coupled_window.py` | 12 passed, 0 failed | 12 passed, 0 failed, 0 skipped, 0 errors | exact | 417.50 s |
| `tests/test_two_class_recovery.py` | 3 passed, 2 failed at residuals 1.29980506 and 1.30949372 | 3 passed, 2 failed, 0 skipped, 0 errors at residuals 1.29980506 and 1.30949372 | exact; the two visible failures remain by design | 883.01 s |
| `tests/test_observable_acceptance.py` | 12 passed, 0 failed | 12 passed, 0 failed, 0 skipped, 0 errors | exact | 60.10 s |

The forward-solve file also emitted one SciPy runtime warning, outside the
requested pass/fail/skip/error counts:

```text
tests/test_equilibrium_forward_solve.py::test_the_host_root_find_holds_the_equilibrium_it_is_seeded_on
  /home/ITER/mcintos/Code/nova/.venv/lib/python3.14/site-packages/scipy/optimize/_nonlin.py:376: RuntimeWarning: invalid value encountered in scalar divide
    and dx_norm/self.x_rtol <= x_norm))
```

## Recovery failures and class stability

The recovery file's two failures are the baseline cold-start pair. Both fail
only at `assert bool(branch.converged)`, after the test has already asserted
that the requested and achieved classes are `LIMITED` and that topology is
consistent. The verbatim failure evidence is:

```text
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

tests/test_two_class_recovery.py:180: AssertionError
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

tests/test_two_class_recovery.py:180: AssertionError
```

**No achieved topology class moved.** Runtime assertions establish the
following unchanged classifications:

- Both cold limited branches achieved `LIMITED` and remained topology-consistent;
  only their convergence flags failed at the two baseline residuals.
- The vectorized cold diverted branch achieved `DIVERTED` and remained
  topology-consistent.
- Every banked rung in the diverted perturbation ladder achieved `DIVERTED`
  and remained topology-consistent.
- The pinned diverted state achieved `DIVERTED` and remained
  topology-consistent.

## Commands and logs

The command template used independently for each file was:

```bash
JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" \
  uv run --no-sync pytest -m 'slow or not slow' tests/<file>.py
```

Complete logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T180434049467-post-cutover-solve-verification/logs/test_equilibrium_forward_solve.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T180434049467-post-cutover-solve-verification/logs/test_transport_coupled_window.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T180434049467-post-cutover-solve-verification/logs/test_two_class_recovery.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T180434049467-post-cutover-solve-verification/logs/test_observable_acceptance.log`
