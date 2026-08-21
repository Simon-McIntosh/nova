NEEDS-HELP: Batch B did not produce a terminal pytest summary before the 60-minute node fence, so the required two-batch green count is incomplete.

tried: Ran each requested batch exactly once in a separate fresh pytest process at worktree HEAD `2a80b890bf3cd869ae96123c0b1265b276e794bb`, using `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest`. Batch A completed green. Batch B collected 32 tests, completed one test, then remained active until it was terminated at the node fence.

options: Redispatch Batch B alone with a larger time fence; split the eight Batch B files into additional fresh-process lanes and amend the live evidence contract first; or investigate why this fresh process still spent more than 43 minutes after its first completed test.

leaning: Redispatch Batch B alone with a larger time fence, because its single captured run showed no assertion failure and changing the declared two-batch evidence contract requires plan authority.

cost-if-wrong: If the long runtime is a merge-integration defect rather than an execution-environment effect, a longer rerun will consume the extra allocation without producing the missing terminal count and the stalled test will still require isolation and repair.

# Integrated retirement lineage verification

## Identity and method

- Worktree HEAD: `2a80b890bf3cd869ae96123c0b1265b276e794bb`
- Commit subject: `Merge the transport-kernel retirement and its acquittal lineage`
- Each batch was invoked exactly once, in its own fresh pytest process.
- Long output was redirected directly to one log per invocation; neither batch was rerun for formatting.
- Failure classification: **environment artefact**. No pytest assertion failed. Batch B was stopped by the worker's hard time fence after making insufficient progress to emit a terminal summary; exit status 143 records that termination.

## Batch A — completed green

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_flux_surface_extraction.py tests/test_transport_geometry_reference.py tests/test_current_diffusion.py tests/test_tensor_spline.py
```

Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T155910201039-retirement-batch-verification/batch-a.log`

Terminal pytest count, quoted verbatim:

```text
====== 46 passed, 1 deselected, 1 xfailed, 1 warning in 921.12s (0:15:21) ======
```

- Wall time: **931.61 seconds**.
- Exit status: **0**.
- Selection: 48 collected, 1 deselected, 47 selected.

## Batch B — incomplete at the time fence

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_transport_geometry.py tests/test_transport_cocos.py tests/test_forward_transport.py tests/test_transport_evolved_state.py tests/test_transport_ensemble.py tests/test_transport_coupled_window.py tests/test_transport_window_gradients.py tests/test_transport_public_surface.py
```

Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T155910201039-retirement-batch-verification/batch-b.log`

No terminal `passed` / `failed` / `xfailed` / `skipped` / `deselected` pytest line was emitted, so there is no valid terminal count to quote. The complete captured progress was:

```text
collected 32 items

tests/test_transport_geometry.py .
```

- Wall time before termination: **2597.48 seconds**.
- Exit status: **143** after explicit termination at the hard 60-minute node fence.
- Completed progress visible in the log: **1 of 32 collected tests**.
- Assertion failures: **none emitted**.

## Quantitative verdict

- Batch A: **GREEN — 46 passed, 1 xfailed, 1 deselected**.
- Batch B: **NO VERDICT — no terminal pytest count; 1 of 32 tests visibly completed before the time fence**.
- Integrated two-batch lineage: **BLOCKED**. The requested combined green evidence cannot be claimed until Batch B completes once and emits its terminal count.

# Transport retirement verification

## Eight-file fresh-process completion

The remaining retirement-verification lane completed at worktree HEAD
`204a13622b745f56aa2fef434ddf3a7da9029eb0`. Each test file ran exactly once
in its own fresh pytest process. Every invocation used the repository's shared
environment and the worktree source:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest <file>
```

Each process was captured to its own named log through an outer wall-time
measurement and a 360-second safety guard. No file was rerun for output
formatting.

### `tests/test_transport_geometry.py`

- Invocation: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_transport_geometry.py`
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T171331934464-batch-b-per-file-verification/logs/test_transport_geometry.log`
- Terminal pytest line: `=================== 3 passed, 1 warning in 151.79s (0:02:31) ===================`
- Outer wall time: **162.26 seconds**; process exit: **0**.

### `tests/test_transport_cocos.py`

- Invocation: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_transport_cocos.py`
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T171331934464-batch-b-per-file-verification/logs/test_transport_cocos.log`
- Terminal pytest line: `======================== 6 passed, 1 warning in 11.90s =========================`
- Outer wall time: **17.54 seconds**; process exit: **0**.

### `tests/test_forward_transport.py`

- Invocation: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_forward_transport.py`
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T171331934464-batch-b-per-file-verification/logs/test_forward_transport.log`
- Terminal pytest line: `======================== 4 passed, 1 warning in 28.54s =========================`
- Outer wall time: **42.53 seconds**; process exit: **0**.

### `tests/test_transport_evolved_state.py`

- Invocation: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_transport_evolved_state.py`
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T171331934464-batch-b-per-file-verification/logs/test_transport_evolved_state.log`
- Terminal pytest line: `======================== 4 passed, 1 warning in 22.57s =========================`
- Outer wall time: **27.45 seconds**; process exit: **0**.

### `tests/test_transport_ensemble.py`

- Invocation: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_transport_ensemble.py`
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T171331934464-batch-b-per-file-verification/logs/test_transport_ensemble.log`
- Terminal pytest line: `======================== 6 passed, 1 warning in 36.42s =========================`
- Outer wall time: **42.16 seconds**; process exit: **0**.

### `tests/test_transport_coupled_window.py`

- Invocation: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_transport_coupled_window.py`
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T171331934464-batch-b-per-file-verification/logs/test_transport_coupled_window.log`
- Terminal pytest line: `============================== 6 passed in 14.25s ==============================`
- Outer wall time: **19.12 seconds**; process exit: **0**.

### `tests/test_transport_window_gradients.py`

- Invocation: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_transport_window_gradients.py`
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T171331934464-batch-b-per-file-verification/logs/test_transport_window_gradients.log`
- Terminal pytest line: `======================== 1 passed, 1 warning in 15.47s =========================`
- Outer wall time: **20.23 seconds**; process exit: **0**.

### `tests/test_transport_public_surface.py`

- Invocation: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/test_transport_public_surface.py`
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T171331934464-batch-b-per-file-verification/logs/test_transport_public_surface.log`
- Terminal pytest line: `============================== 2 passed in 5.57s ===============================`
- Outer wall time: **9.77 seconds**; process exit: **0**.

## Combined verdict

All eight files are green: **32 passed, 0 failed, 0 xfailed, 0 skipped, and
0 deselected**. The sum of pytest-reported durations is **286.51 seconds**;
the sum of independently measured outer wall times is **341.06 seconds**.

No failure required classification as a migration defect, merge-integration
defect, or environment artefact. The per-file fresh-process evidence closes
the remaining retirement-verification lane.
