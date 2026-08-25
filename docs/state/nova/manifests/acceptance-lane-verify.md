node: acceptance-lane-verify
status: complete
commits: 2b77347c
changed_paths: docs/figures/roundoff-scale-acceptance-bounds/acceptance-lane-verify.md
tests: |
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_observable_acceptance.py -m "slow or not slow"` — 12 passed, 0 failed, 0 skipped, 0 errors in 7.88 s.
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_equilibrium_stencil_mesh.py -m "slow or not slow"` — 19 passed, 0 failed, 0 skipped, 0 errors in 70.01 s.
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_observable_reduction_parity.py -m "slow or not slow"` — 1 passed, 0 failed, 0 skipped, 0 errors in 13.00 s.
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_equilibrium_forward.py -m "slow or not slow"` — 35 passed, 0 failed, 0 skipped, 0 errors in 23.74 s.
  Total: 67 passed, 0 failed, 0 skipped, 0 errors across four fresh pytest processes. The grep discovery command was `rg -n --glob 'tests/**/*.py' 'observable_acceptance|conservation_ledger' tests`; it found exactly these four files.
test_logs: |
  /tmp/nova-acceptance-lane-verify.eCJLSX/test_observable_acceptance.log
  /tmp/nova-acceptance-lane-verify.eCJLSX/test_equilibrium_stencil_mesh.log
  /tmp/nova-acceptance-lane-verify.eCJLSX/test_observable_reduction_parity.log
  /tmp/nova-acceptance-lane-verify.eCJLSX/test_equilibrium_forward.log
artifacts: docs/figures/roundoff-scale-acceptance-bounds/acceptance-lane-verify.md — full surface green at 67 passed; frozen receipt asserts 69 of 69 observables pass at both batch sizes 1 and 4.
evidence_inputs: |
  Source commit a59932ae is present in this worktree and the whole importer-derived acceptance/conservation surface is green.
  The 69-of-69 cohort rescore reproduces at the committed-receipt validation level for both frozen widths: batch 1 maximum divergence_j difference 1.1641532182693481e-10 and batch 4 maximum 1.4551915228366852e-10, each below 1.0536712127723509e-8. The receipt retains the banked 67-of-69 baseline. This lane did not recompute the production cohort from source data.
  `test_the_divergence_floor_is_truncation_and_falls_with_the_mesh` passes. Its truncation/refinement floor is populated only from `conservation_ledger(mesh, ...)` on the non-commuting StencilMesh route, with required refinement ratios greater than 3.0 for divergence_b and 1.5 for divergence_j. The separate `conservation_ledger(lattice, ...)` assertions require the commuting FluxLattice route below 1.0e-12. The name describes the ring-mesh assertion but is route-underqualified and must not be read as describing production FluxLattice cancellation.
  No failures, assertions, or tracebacks exist to report.
follow_ons: none
blockers: none
