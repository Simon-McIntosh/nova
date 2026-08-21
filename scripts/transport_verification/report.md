# Transport suite verification

## Verdict

PASS. The full transport lane, including tests marked `slow`, completed once
with exit code 0: **62 passed, 0 failed, 0 skipped, 0 deselected**. Pytest
reported 157.85 s for the session; the outer command took 209 s including
`uv` startup and process shutdown.

The previously deselected TORAX cross-check
`test_solver_formulation_matches_torax_on_the_shared_circular_case` was
selected and **passed**. Its JUnit testcase duration was 1.101 s (reported as
1.10 s for the call in the pytest duration table).

There were no failures to classify as either product defects or environment
artifacts.

## Tested source

- Worktree commit: `88e3a1faaffad43e9fa751359a5a88290cdfa75d`
- Current `main` and `origin/main` at evidence extraction:
  `038d119649defb2e0a1fe7e5f23433fc378055fa`
- The worktree commit is an ancestor of current `main`. The intervening merge
  adds only density-forcing measurement artifacts. A path-scoped comparison
  found no differences in `pyproject.toml`, `uv.lock`, `nova/transport`,
  `nova/equilibrium`, or any of the ten transport test modules, so the tested
  transport source and tests are byte-identical to current merged `main`.

## Command

The command was executed once from the detached verification worktree. Output
was redirected to the full log and was not rerun for formatting.

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD \
  uv run --no-sync pytest -m 'slow or not slow' \
  tests/test_current_diffusion.py \
  tests/test_transport_environment.py \
  tests/test_transport_geometry.py \
  tests/test_transport_cocos.py \
  tests/test_forward_transport.py \
  tests/test_transport_evolved_state.py \
  tests/test_transport_ensemble.py \
  tests/test_transport_coupled_window.py \
  tests/test_transport_window_gradients.py \
  tests/test_transport_public_surface.py \
  --durations=0 --durations-min=0 \
  --junitxml=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T153931549081-transport-suite-verification/pytest-junit.xml
```

## Module results and durations

Module durations below are the cumulative setup, call, and teardown times of
that module's JUnit testcases from the single combined run. They intentionally
do not pretend to be standalone wall times. The cumulative testcase time is
112.903 s; the JUnit session time is 157.844 s, leaving 44.941 s of collection
and session-level overhead not assignable to one module.

| Module | Passed | Failed | Skipped | Duration (s) |
|---|---:|---:|---:|---:|
| `tests/test_current_diffusion.py` | 29 | 0 | 0 | 27.349 |
| `tests/test_transport_environment.py` | 1 | 0 | 0 | 5.703 |
| `tests/test_transport_geometry.py` | 3 | 0 | 0 | 28.552 |
| `tests/test_transport_cocos.py` | 6 | 0 | 0 | 1.079 |
| `tests/test_forward_transport.py` | 4 | 0 | 0 | 14.242 |
| `tests/test_transport_evolved_state.py` | 4 | 0 | 0 | 8.825 |
| `tests/test_transport_ensemble.py` | 6 | 0 | 0 | 18.025 |
| `tests/test_transport_coupled_window.py` | 6 | 0 | 0 | 5.805 |
| `tests/test_transport_window_gradients.py` | 1 | 0 | 0 | 3.323 |
| `tests/test_transport_public_surface.py` | 2 | 0 | 0 | 0.000 |
| **Total** | **62** | **0** | **0** | **112.903** |

The marker expression selected the full collected surface, so the deselected
count was 0.

## Non-failing diagnostics

Pytest reported one third-party `DeprecationWarning`: `jaxopt` announces that
it is no longer maintained during the shared-process environment test. After
the green summary, the numerical library also emitted six `DLASCL` parameter
diagnostic lines. Neither produced a failure, skip, deselection, or nonzero
exit, but both remain visible in the retained full log.

## Retained evidence

- Full pytest output:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T153931549081-transport-suite-verification/pytest-full.log`
- JUnit record used for counts, per-module timing, and the slow-test verdict:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T153931549081-transport-suite-verification/pytest-junit.xml`
- Outer command timing and exit status:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T153931549081-transport-suite-verification/pytest-meta.txt`
