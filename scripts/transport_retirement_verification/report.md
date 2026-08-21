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
