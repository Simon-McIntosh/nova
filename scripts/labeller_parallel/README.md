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
uses the sequential writer's names: `<shot>.nc`, `<shot>.manifest.json`, plus a
digest `<shot>.receipt.json`; the presence of all three makes that shot
resumable and skipped on restart.

Run the bounded evidence smoke with:

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" \
  uv run --no-sync python scripts/labeller_parallel/smoke.py \
  --output docs/figures/playable-forward-solve/labeller-parallel/smoke
```

The smoke selects the first sixteen shots from the real ranked decoder corpus,
writes a sequential reference, runs one- and three-device scheduler arms, and
requires every per-slice manifest record and session variable to be identical.
