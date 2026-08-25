node: primary-branch-green-check
status: complete
commits:
  - 9a9f031a9cb61585213987f76ca86820ddd067f6
changed_paths:
  - docs/figures/discrete-operator-analytic-error/primary-branch-green-check.md
tests:
  - 'JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m "slow or not slow" tests/test_equilibrium_forward_solve.py > /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_equilibrium_forward_solve.log 2>&1 — 28 passed, 0 failed, 0 skipped, 0 errors in 88.25 s'
  - 'JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m "slow or not slow" tests/test_transport_coupled_window.py > /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_transport_coupled_window.log 2>&1 — 12 passed, 0 failed, 0 skipped, 0 errors in 31.33 s'
  - 'JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m "slow or not slow" tests/test_two_class_recovery.py > /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_two_class_recovery.log 2>&1 — 0 passed, 5 failed, 0 skipped, 0 errors in 254.91 s'
test_logs:
  - /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_equilibrium_forward_solve.log
  - /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_transport_coupled_window.log
  - /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_two_class_recovery.log
artifacts:
  - 'docs/figures/discrete-operator-analytic-error/primary-branch-green-check.md — RED: topology-container files clear at 28/28 and 12/12, recovery remains 0/5 with complete assertion tracebacks'
evidence_inputs:
  - 'Measured from a clean detached worktree at c701600431f9a7f4d5d06b885e0213161e329de2, freshly fetched as origin/main immediately before testing; this is the merge of the batched topology receipt repair.'
  - 'origin/main advanced during testing to 1efe00ea81d0003fe71a485517ca418829a6e711, but the intervening commit modifies only plans, evidence, crew state, and a manifest; no measured source, test, config, lock, or banked recovery artifact differs.'
  - 'The former ForwardTopologyState transformed-output regression clears: tests/test_equilibrium_forward_solve.py improved from 26 passed/2 failed to 28 passed/0 failed, and tests/test_transport_coupled_window.py improved from 11 passed/1 failed to 12 passed/0 failed.'
  - 'The deciding recovery failures SURVIVE unchanged at 0 passed/5 failed: converged remains false for coarse and fine limited roots; achieved_class remains [0,1] against [0,0]; receipt.passed remains [True,True,False]; measured composition still differs from the banked receipt.'
  - 'All recovery failures are AssertionError outcomes, not the repaired JAX container TypeError. This establishes a second independent regression and leaves the three-file primary-branch gate RED.'
  - 'Every file ran in its own fresh process with JAX_PLATFORMS=cpu and -m "slow or not slow"; no source, test, configuration, or banked-result file was changed.'
follow_ons:
  - 'Independently attribute the five surviving recovery regressions before repair; preserve the banked machine-precision pinned-root result until the measured-versus-banked composition delta is explained.'
blockers: none
