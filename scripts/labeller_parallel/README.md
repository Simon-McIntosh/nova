# Parallel corpus labeller

`scheduler.py` keeps `batch_per_device * device_count` slots resident. Each
slot stays with one shot and carries the terminal state forward as the next
slice's warm state. When the shot ends, the slot is refilled from the ranked
decoder corpus. Device results cross to the host once per step and frame
assembly is submitted to a process pool while the next device step runs.

`HostRouteEngine` is the production default behind the array contract. Each
active slot calls the same host-loop free reduced solve as the sequential shard,
then calls the same constrained solve when the free result raised, did not
converge, or missed the centroid branch guard. Slots own separate free and
conditioned programs and are driven concurrently across visible devices.
`SequentialCompiledEngine` remains selectable with `--engine compiled` for the
batched-engine receipt once that route can reuse its program across shots.

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
  --engine host --devices 1 --batch-per-device 1 --host-workers 4 \
  --max-slices 4 --condition-on-guard-failure --replace
```

The smoke selects the first sixteen shots from the ranked decoder corpus and
compares against the existing independent `scripts/labeller_batch/shard.py`
reference. Pass `--run-reference` only when deliberately replacing that
reference. The identity arm uses four admitted slices per shot. Its receipt
reports difference counts by manifest key, companion NPZ array and session
variable. Independent wall-clock values are checked for the same typed shape;
discrete scientific values compare exactly and floats use the tolerance stated
below. The parallel manifest declares its additional per-slice requested
topology class and its process-start source-tree identities.

Submit one resumable production job with:

```bash
scripts/labeller_parallel/run.sh --submit \
  --output-root /work/projects/imas_gpu/sophelio/labeller_sessions/76906a29
```

The default launcher uses one H200, four host CPUs, 128 GiB and a 24-hour bound.
It walks the full ranked corpus, skips shots whose session and manifest already
exist, uses the host route, and writes its log under the output root's `logs/`
directory. `--devices` scales CPUs and memory at four CPUs and 128 GiB per
device; `--batch-per-device` changes the number of resident shot slots.

The one-card identity smoke agreed exactly on every discrete field for all 64
slices and kept every float within `rtol=1e-12`, `atol=1e-14` for 63 of 64.
Shot 24751 row 82 is recorded as the single marginal slice: the compiled and
host routes followed slightly different terminal paths after warm-state
propagation while retaining the same decisions. The receipt lists every field
and its maximum difference. It excludes the compiled Newton-step counter until
its source-level tuple binding is corrected.
