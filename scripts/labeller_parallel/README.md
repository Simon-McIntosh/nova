# Parallel corpus labeller

`scheduler.py` keeps `batch_per_device * device_count` slots resident. Each
slot stays with one shot and carries the terminal state forward as the next
slice's warm state. When the shot ends, the slot is refilled from the ranked
decoder corpus. Device results cross to the host once per step and frame
assembly is submitted to a process pool while the next device step runs.

`SequentialCompiledEngine` is the interim production engine behind the array
contract. Each active slot runs the compiled reduced-Newton slice program on
its assigned JAX device and owns separate free and centroid-conditioned
programs. Its result arrays keep the replacement boundary fixed while carrying
the sequential writer's guard, centroid, convergence and termination readings
beside them.

Host workers construct `FluxSurfaceGeometry` and call `assemble_frame` with the
EFM flux functions, applied currents and branch reference. Shot output then
uses the card-job writer helpers and names: `<shot>.nc`, `<shot>.npz`, and
`<shot>.manifest.json`. A slot does not advance until its frame is assembled,
so an assembly failure resets the warm state exactly as the sequential route
does. The presence of the session and complete manifest makes a shot resumable.

Run the bounded evidence smoke with:

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" \
  uv run --no-sync python scripts/labeller_parallel/smoke.py \
  --output docs/figures/playable-forward-solve/labeller-parallel \
  --devices 1 --batch-per-device 1 --host-workers 4 --max-slices 1 \
  --run-reference --condition-on-guard-failure --replace
```

The smoke selects the first sixteen shots from the ranked decoder corpus and
invokes `scripts/labeller_batch/shard.py` as an independent sequential
reference. The identity arm uses four admitted slices per shot. Its receipt
reports difference counts by manifest key, companion NPZ array and session
variable. Independent wall-clock values are checked for the same typed shape;
all scientific values are compared exactly. The parallel manifest declares its
additional per-slice requested topology class and its process-start source-tree
identities.

The one-card identity allocation is launched with:

```bash
scripts/labeller_parallel/run.sh --submit \
  --output-root docs/figures/playable-forward-solve/labeller-parallel \
  --batch-per-device 1
```

The launcher uses one H200, four host CPUs, 128 GiB and a one-hour bound. It
reuses the completed independent reference, then runs the one-device compiled
scheduler and comparison. A clean initial run adds `--run-reference` to the
driver command. The three-device throughput arm is a separate follow-on.
