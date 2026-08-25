# Acceptance and conservation lane verification

## Verdict

Commit `a59932ae` leaves the grep-derived acceptance and conservation test
surface green: **67 passed, 0 failed, 0 skipped, 0 errors** across four fresh
pytest processes. Every process explicitly selected both marker classes with
`-m "slow or not slow"`.

The frozen-cohort receipt validation reproduces **69 of 69 registered
observables passing at both registered batch sizes, 1 and 4**. The committed
receipt retains the banked 67-of-69 baseline and records maximum
`divergence_j` acceptance differences of `1.1641532182693481e-10` at batch size
1 and `1.4551915228366852e-10` at batch size 4 against the replacement absolute
bound `1.0536712127723509e-8`. This is a repository receipt-validation result;
this lane did not recompute the frozen production cohort from its source data.

## Surface discovery

The required search was:

```text
rg -n --glob 'tests/**/*.py' 'observable_acceptance|conservation_ledger' tests
```

It found the two named files and two additional importers:

- `tests/test_observable_acceptance.py`
- `tests/test_equilibrium_stencil_mesh.py`
- `tests/test_observable_reduction_parity.py`
- `tests/test_equilibrium_forward.py`

## Test results

Every command ran once, with complete output redirected to the corresponding
log before any selective reading.

| Test file | Exact command | Passed | Failed | Skipped | Errors | Duration |
|---|---|---:|---:|---:|---:|---:|
| `tests/test_observable_acceptance.py` | `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_observable_acceptance.py -m "slow or not slow"` | 12 | 0 | 0 | 0 | 7.88 s |
| `tests/test_equilibrium_stencil_mesh.py` | `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_equilibrium_stencil_mesh.py -m "slow or not slow"` | 19 | 0 | 0 | 0 | 70.01 s |
| `tests/test_observable_reduction_parity.py` | `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_observable_reduction_parity.py -m "slow or not slow"` | 1 | 0 | 0 | 0 | 13.00 s |
| `tests/test_equilibrium_forward.py` | `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_equilibrium_forward.py -m "slow or not slow"` | 35 | 0 | 0 | 0 | 23.74 s |

Logs:

- `/tmp/nova-acceptance-lane-verify.eCJLSX/test_observable_acceptance.log`
- `/tmp/nova-acceptance-lane-verify.eCJLSX/test_equilibrium_stencil_mesh.log`
- `/tmp/nova-acceptance-lane-verify.eCJLSX/test_observable_reduction_parity.log`
- `/tmp/nova-acceptance-lane-verify.eCJLSX/test_equilibrium_forward.log`

No failure assertion or traceback exists to report.

## Mesh-route interpretation

`test_the_divergence_floor_is_truncation_and_falls_with_the_mesh` passed as
part of the 19-test stencil-mesh process. It explicitly exercises both routes:

- `raster = conservation_ledger(lattice, ...)` uses the commuting
  `FluxLattice` route and is required to keep both relative divergence values
  below `1.0e-12`.
- `ring = conservation_ledger(mesh, ...)` uses the non-commuting
  `StencilMesh` route. Only the ring values populate `floor`, and the test
  requires refinement ratios greater than `3.0` for `divergence_b` and `1.5`
  for `divergence_j` from 45 to 91 nodes.

The test name therefore still describes the ring-mesh quantity it actually
checks for truncation and refinement, but it is route-underqualified: it must
not be read as describing the production `FluxLattice` route, whose commuting
derivatives cancel to a floating-point floor. This is consistent with the
landed attribution rather than a contradiction.

## Evidence for writeback

- Complete grep-derived surface: 4 files, 67 passed, 0 failed, 0 skipped,
  0 errors.
- Frozen receipt: 69 of 69 at batch sizes 1 and 4; banked baseline 67 of 69.
- Replacement `divergence_j` bound: `1.0536712127723509e-8`.
- Largest frozen acceptance difference: `1.4551915228366852e-10`, giving
  `72.40773439350247` times slack in the committed receipt.
- Failability witness remains banked: a one-part-per-million radial-current
  scale mismatch produces `7.568358362676842e-8`, or `7.182846291077339`
  times the bound, and fails acceptance.
- No production or test defect was found, and no source repair is indicated by
  this lane.
