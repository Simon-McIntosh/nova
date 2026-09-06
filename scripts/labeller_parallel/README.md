# Parallel corpus labeller

`scheduler.py` keeps `batch_per_device * device_count` slots resident. Each
slot stays with one shot and carries the terminal state forward as the next
slice's warm state. When the shot ends, the slot is refilled from the ranked
decoder corpus. Device results cross to the host once per step and frame
assembly is submitted to a process pool while the next device step runs.

The current `ArrayBatchEngine` is an explicit array-contract stub. It validates
the production contract (`state[batch,1126]`, decisions and centroid vectors,
plus arbitrary labelled fields with a leading batch axis) and can be replaced
by the batched engine without changing scheduling or persistence. Shot output
uses the sequential writer's exact helpers and names: `<shot>.nc`,
`<shot>.npz`, and `<shot>.manifest.json`. The session is built from
`SteeringFrame` values by `_write_session_file`; conditioning diagnostics are
written by the shard helper currently named `_write_companion`. The presence
of the session and manifest makes that shot resumable and skipped on restart.

Run the bounded evidence smoke with:

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" \
  uv run --no-sync python scripts/labeller_parallel/smoke.py \
  --output docs/figures/playable-forward-solve/labeller-parallel/smoke
```

The smoke selects the first sixteen shots from the real ranked decoder corpus,
writes a sequential reference, runs one- and three-device scheduler arms, and
requires every per-slice manifest record and session variable to be identical.
