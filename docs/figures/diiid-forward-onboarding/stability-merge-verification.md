# Integrated forward-solve suite verification

## Verdict

The combined equilibrium changes are **green** on the integrated code and test
tree: **61 passed, 0 failed** across six independently captured suites. Every
suite retained its isolated-node pass count. No test that passed in isolation
failed after integration, and no pass/fail-count divergence identifies a merge
interaction.

## Integrated revision under test

The verification worktree was pinned at
`ed01c71d129911118ca3c9282cc5e53079e4208c`. The primary branch had advanced to
`ad289d9e07aa6ceefd937e2e76405c76c84d8920` only to register this verification
member; `git diff --name-only HEAD main -- nova tests` was empty. The product and
test tree exercised here is therefore byte-identical to current integrated
`main` for all paths under `nova/` and `tests/`.

The three separately integrated equilibrium changes present in the tested
ancestry are:

- `045ab9e97dd1dbf19e83a6a05e6972317f4757c8` — **Merge the exact tangent
  primitive repair**, integrating implementation commit
  `06b09f5bfdcf184a69066e829a6e7ee16fdd3a2b`.
- `2f2575dcbca8f6daaea44e6c4d0fa8ab7061a368` — **Merge the bumped global
  plasma default and its live GPU demonstration**, integrating implementation
  commit `2e5a7ba62318e769c117efe8ce6d6d1d9d1cf29a`.
- `b03f8ddb7f3307de0128e3b74b051b5a828e1f77` — **Merge topology-qualified
  trial admission**, integrating implementation commit
  `257af9dcbe5dd19313708f862a8bb3a383f917b2`.

All three merge commits were verified as ancestors of the tested revision.

## Captured suite results

Each file was run once in its own complete capture with the repository's shared
environment and the worktree code on `PYTHONPATH`:

```text
JAX_PLATFORMS=cpu \
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv \
PYTHONPATH="$PWD" \
uv run --no-sync pytest -p no:cacheprovider <suite>
```

| Suite | Integrated result | Isolated-node result | Delta | Merge-interaction verdict |
|---|---:|---:|---:|---|
| `tests/test_fixed_point.py` | 15 passed, 0 failed | 15 passed, 0 failed | 0 | none |
| `tests/test_krylov_action_qualification.py` | 4 passed, 0 failed | 4 passed, 0 failed | 0 | none |
| `tests/test_topology_qualified_admission.py` | 4 passed, 0 failed | 4 passed, 0 failed | 0 | none |
| `tests/test_forward_operator_tangent.py` | 1 passed, 0 failed | 1 passed, 0 failed | 0 | none |
| `tests/test_equilibrium_forward_solve.py` | 28 passed, 0 failed | 28 passed, 0 failed | 0 | none |
| `tests/test_equilibrium_forward_constrained.py` | 9 passed, 0 failed | 9 passed, 0 failed | 0 | none |
| **Total** | **61 passed, 0 failed** | **61 passed, 0 failed** | **0** | **none** |

The five isolation counts supplied for this node—15, 4, 4, 1, and 28—match the
first five rows exactly. The constrained suite's separately banked global-mesh
isolation receipt reports the sixth baseline as 9/9; that count is also
preserved here.

Isolation evidence came from these worker receipts:

- Krylov qualification:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T140348838236-krylov-action-qualification/manifest.md`
- Exact tangent repair:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T140338153895-nonfinite-jvp-primitive/manifest.md`
- Topology-qualified admission:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T142733951322-topology-qualified-admission/manifest.md`
- Global plasma-default integration, including constrained 9/9 and forward
  solve 28/28:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T121523253511-global-dplasma-gpu-rebuild/manifest.md`

## Captures

- `tests/test_fixed_point.py`:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T152244411612-stability-merge-verification/logs/test-fixed-point.log`
- `tests/test_krylov_action_qualification.py`:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T152244411612-stability-merge-verification/logs/test-krylov-action-qualification.log`
- `tests/test_topology_qualified_admission.py`:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T152244411612-stability-merge-verification/logs/test-topology-qualified-admission.log`
- `tests/test_forward_operator_tangent.py`:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T152244411612-stability-merge-verification/logs/test-forward-operator-tangent.log`
- `tests/test_equilibrium_forward_solve.py`:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T152244411612-stability-merge-verification/logs/test-equilibrium-forward-solve.log`
- `tests/test_equilibrium_forward_constrained.py`:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T152244411612-stability-merge-verification/logs/test-equilibrium-forward-constrained.log`

Each capture ends with `EXIT_MARKER=0`.

## Retained qualifications

The forward-solve suite emitted the same SciPy invalid-value runtime warning in
the host root-find test that its isolated-node receipts already retained. The
constrained suite emitted one JAX configuration warning: an explicitly
requested float64 scalar was truncated to float32 because this capture did not
set `JAX_ENABLE_X64`. Both suites completed with their full isolated pass counts;
neither warning hid a failure or a changed count. The constrained isolation
receipt records its pass count but does not state a warning count, so warning
parity is not claimed.

## Combination statement

The exact tangent repair, denser global plasma default, Krylov-action
qualification, and topology-qualified trial admission coexist without a
detected regression in the contracted forward-solve surface. The combination
is green for these six suites. The set of tests that pass alone but fail
integrated is **empty**.
