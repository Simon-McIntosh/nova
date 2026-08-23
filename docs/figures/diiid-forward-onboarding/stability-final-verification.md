# Stability final verification

## Verdict

**GREEN.** The completed forward-stability combination is green at final integrated tree
`7321e26f199937edca86ddcb91e454af3f80f2b7`. Seven required suites were run once each
against that exact commit: **68 passed, 0 failed**. Every suite reproduced its isolated
pass baseline exactly, so there is **no pass/fail-count divergence and no merge
interaction to report**.

This is a tree-pinned result, not a branch-relative claim. The worktree was detached at
the commit above when all captures ran; the commit was also the then-current `main` and
`origin/main` tip.

## Stability merge ancestry

The pinned tree contains every merge in the stability wave, verified with
`git merge-base --is-ancestor <merge> 7321e26f199937edca86ddcb91e454af3f80f2b7`:

| Merge commit | Integrated result |
|---|---|
| `83dc40fbb1bff3c37ce97730b9eaa9bba0344117` | Forward-solve stability reuse map |
| `931efcb75ac82fa944b11d81b7165f6c8f7e2924` | Krylov-action qualification |
| `045ab9e97dd1dbf19e83a6a05e6972317f4757c8` | Exact tangent primitive repair |
| `b03f8ddb7f3307de0128e3b74b051b5a828e1f77` | Topology-qualified trial admission |
| `06c22a50f9ab48783415cdc1b5f1b6d435531b27` | Advisory amplification observation |
| `8062944fd4d65262f13e90d57cd1905c6fdf51cc` | Integrated stability-suite verification |
| `7321e26f199937edca86ddcb91e454af3f80f2b7` | Topology-qualified mesh-convergence measurement |

The corresponding implementation/evidence commits are also ancestors of the pinned
tree: `2275c1993f8fce285411ca48c518dbb34de8ebdc`,
`6143221d7a2dc5c3ad2a74b822bf1a026c05eed3`,
`06b09f5bfdcf184a69066e829a6e7ee16fdd3a2b`,
`257af9dcbe5dd19313708f862a8bb3a383f917b2`,
`7e33c4d000131ce0bb76b9b3c7ad2b71ce668d36`,
`b188a77811a2b1b5d5308f3f7ab0bd286929cc8d`, and
`121ee8054341008156ba3852c80a4ae4d27b5c17`.

## One-capture suite results

Each invocation used the repository's single shared environment without syncing it:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv \
PYTHONPATH=<pinned-worktree> \
uv run --no-sync pytest -p no:cacheprovider <suite>
```

| Suite | Isolated baseline | Final-tree result | Delta | Log |
|---|---:|---:|---:|---|
| `tests/test_fixed_point.py` | 15 pass, 0 fail | 15 pass, 0 fail | 0 | `test_fixed_point.log` |
| `tests/test_krylov_action_qualification.py` | 4 pass, 0 fail | 4 pass, 0 fail | 0 | `test_krylov_action_qualification.log` |
| `tests/test_topology_qualified_admission.py` | 4 pass, 0 fail | 4 pass, 0 fail | 0 | `test_topology_qualified_admission.log` |
| `tests/test_amplification_observation.py` | 7 pass, 0 fail | 7 pass, 0 fail | 0 | `test_amplification_observation.log` |
| `tests/test_forward_operator_tangent.py` | 1 pass, 0 fail | 1 pass, 0 fail | 0 | `test_forward_operator_tangent.log` |
| `tests/test_equilibrium_forward_solve.py` | 28 pass, 0 fail | 28 pass, 0 fail | 0 | `test_equilibrium_forward_solve.log` |
| `tests/test_equilibrium_forward_constrained.py` | 9 pass, 0 fail | 9 pass, 0 fail | 0 | `test_equilibrium_forward_constrained.log` |

The durable logs are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T153702300654-stability-final-verification/`.

## Coverage-gap closure

The prior integration verification did not cover both edge suites after the advisory
observation landed. That gap is closed here:

- `tests/test_forward_operator_tangent.py`: **1 passed, 0 failed** with advisory
  observation implementation commit `7e33c4d000131ce0bb76b9b3c7ad2b71ce668d36`
  present in the tested tree.
- `tests/test_equilibrium_forward_constrained.py`: **9 passed, 0 failed** with the same
  advisory observation commit present in the tested tree.

The combination of exact tangents, qualified Krylov actions, topology-qualified
admission, and advisory-only amplification observation is therefore **plainly green at
the final integrated commit**.

## Non-failing qualifications

Two suites each emitted one warning without changing their pass/fail baseline:

- `tests/test_equilibrium_forward_solve.py`: SciPy reported an invalid scalar division
  inside its host nonlinear solver convergence check; all 28 tests passed.
- `tests/test_equilibrium_forward_constrained.py`: JAX reported a requested `float64`
  scalar being truncated to `float32` because x64 was not enabled for that test; all 9
  tests passed.

These are retained as qualifications, not described as failures or merge interactions.
No source file or test file was edited; this report is the only repository write.
