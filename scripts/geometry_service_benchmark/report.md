# Extraction-service assembly benchmark

## Production result

The production configuration is the real `iterhybrid_cocos17.eqdsk` ITER map
at 129x129, 24 radial cells, and **28 surface bins**. At that configuration,
the current extraction service is not practical on CPU for a coupled window:
the single-map JIT-warm median is **179.334230161 s (n=3)**, well above ten
seconds by a factor of 17.9, and an eight-map `vmap` is **779.241352104 s total
or 97.405169013 s per map (n=1)**. The corresponding H200 medians are
**1.567966604 s for one map (n=7)** and **5.925816003 s total or 0.740727000 s
per map for a vmapped batch of eight (n=7)**. The H200 is therefore decisive
for this assembly route: it is 114.4 times faster for the single-map warm
executable and 131.5 times faster for batch-eight throughput.

Cold lowering and compilation are separated from every execution number.
On CPU, the production single-map executable compiles in **7.108359312 s
(n=1)** but runs in **179.334230161 s median (n=3)**, so execution costs 25.2
times compilation. Batch eight compiles in **8.823600586 s (n=1)** and runs in
**779.241352104 s median (n=1)**, an 88.3-to-one execution/compile ratio. This
confirms, rather than refutes, the unusual inversion seen in the 96-bin probe:
CPU optimization must first reduce work performed by the assembled executable,
not its compilation latency.

The split reverses on the H200. Single-map compilation is **28.373160664 s
(n=1)** against **1.567966604 s warm execution median (n=7)**, and batch-eight
compilation is **29.611152896 s (n=1)** against **5.925816003 s warm execution
median (n=7)**. Compilation is respectively 18.1 and 5.0 times the warm
execution, so a production GPU process should retain and reuse its compiled
executable. Runtime work still matters, but avoiding repeat compilation is the
first lifecycle optimization on that device.

## Complete production timings

All times are wall-clock seconds. “Per map” for a batch is the measured batch
wall time divided by eight; it is an amortized throughput value, not a timing
distribution over individual maps. A one-observation median is reported as
`n=1` and has no repetition-based uncertainty estimate.

### Current extraction service

| Device | Execution | Batch | Cold compile median total / per map | Compile repetitions | JIT-warm median total / per map | Warm repetitions |
|---|---|---:|---:|---:|---:|---:|
| H200 | `jit` | 1 | 28.373160664 / 28.373160664 | 1 | 1.567966604 / 1.567966604 | 7 |
| H200 | `jit_vmap` | 8 | 29.611152896 / 3.701394112 | 1 | 5.925816003 / 0.740727000 | 7 |
| CPU | `jit` | 1 | 7.108359312 / 7.108359312 | 1 | 179.334230161 / 179.334230161 | 3 |
| CPU | `jit_vmap` | 8 | 8.823600586 / 1.102950073 | 1 | 779.241352104 / 97.405169013 | 1 |

CPU batching improves the per-map warm median by 1.84 times, from
**179.334230161 s (n=3)** to **97.405169013 s (n=1)**, but it does not make the
route viable on CPU. H200 batching improves the per-map warm median by 2.12
times, from **1.567966604 s (n=7)** to **0.740727000 s (n=7)**.

### Retired kernel and host contour references

| Route | Device | Execution | Batch | Cold compile or first-execution median total / per map | Repetitions | Warm median total / per map | Warm repetitions |
|---|---|---|---:|---:|---:|---:|---:|
| Retired Gaussian-shell kernel | CPU | `jit` | 1 | 1.150820550 / 1.150820550 | 1 | 0.008356422 / 0.008356422 | 7 |
| Retired Gaussian-shell kernel | CPU | `jit_vmap` | 8 | 1.565439987 / 0.195679998 | 1 | 0.065053952 / 0.008131744 | 7 |
| Host contour | CPU | host call | 1 | 0.408564260 / 0.408564260 | 1 | 0.299431051 / 0.299431051 | 7 |
| Host contour | CPU | Python loop | 8 | 2.423263120 / 0.302907890 | 1 | 2.375443946 / 0.296930493 | 7 |

The host route has no JIT compilation phase, so its first-execution medians
replace the cold-compile column. Its batch result is eight serial Python calls,
not `vmap`. The 28-bin setting applies to the current and retired traced
kernels; the host algorithm has no surface-bin parameter and used its native
129 contour surfaces, 256 angles, and the same 25-point radial output.

These references quantify assembly cost, not semantic equivalence. On CPU, the
current service warm median is 21,460.6 times the retired kernel for one map and
11,978.4 times it for batch eight. It is 598.9 times the host contour route for
one map and 328.0 times it for the eight-call batch. The retired kernel did not
assemble the current higher-order production record, and the host contour path
is a referee rather than the equilibrium authority; neither lower number is a
drop-in replacement verdict.

## Coupled-window consequence

The strongly coupled case reached 124 iterations, and extraction is called once
per sample per iteration. Holding the production single-map CPU median fixed,
124 calls imply **22,237.444540 s, or 6.177 h, of extraction per sample**, based
on the **179.334230161 s median (n=3)** and excluding every other operation in
the window. The comparable H200 single-map projection is **194.427859 s, or
3.240 min per sample**, based on the **1.567966604 s median (n=7)**. If samples
can be supplied in batches of eight, the H200 per-map throughput projects to
**91.850148 s, or 1.531 min per sample**, based on the **0.740727000 s median
(n=7)**; the whole eight-sample batch would spend **734.801184 s, or 12.247
min**, in extraction across 124 calls. These are linear cost projections from
assembly medians, not end-to-end window measurements.

## Separate 96-bin diagnostic

The cancelled diagnostic used **96 surface bins**, not the production setting,
and must not be compared as a headline value. On the same 129x129 map with
seven allocated CPUs, lowering plus compilation took **7.038763228 s (n=1,
median of one)** and one post-compile execution took **607.032261677 s (n=1,
median of one)**. It established configuration cost and motivated completing
the contracted run at 28 bins. No 96-bin batch or GPU claim is made, and these
observations are intentionally absent from the production TSV.

## Method and provenance

- SLURM job `1252999` completed successfully in 28 minutes 45 seconds on
  partition `betelgeuse`, reservation `gpu_0003_grpA`, with one NVIDIA H200 NVL,
  seven allocated Intel Xeon 6530P CPU cores, 64 GiB requested memory, and a
  55-minute limit. Every requested lane ran; measurement subprocesses `.1`
  through `.8` completed with exit code zero, and none is unavailable. An
  auxiliary GPU-telemetry subprocess `.0` was cancelled when the allocation tore
  down; it was not a benchmark lane.
- The batch script exports `TMPDIR=/tmp`. The site prologue also recorded
  `Unable to create TMPDIR [/run/user/39486]: Permission denied` followed by
  `Setting TMPDIR to /tmp`, confirming the compute-node fallback.
- Removing the redundant inner `--gpus-per-task=1` let the one-task job step
  inherit the H200 already owned by its enclosing allocation. The preceding
  rejected launch had reported `srun: fatal: gpus-per-task is mutually exclusive
  with tres-per-task`; the successful job needed no alternate TRES request.
- Every route and batch size ran in a fresh process. CPU processes fixed
  `JAX_PLATFORMS=cpu`; GPU processes fixed `JAX_PLATFORMS=cuda`; the persistent
  JAX compilation cache was disabled. Each timed call blocks every returned
  array before stopping the wall clock.
- Batch inputs are eight distinct maps, scaled by `1 + 1e-5*i`; axis and
  boundary flux anchors are scaled consistently. Every first service result
  was checked for a valid geometry record.
- The current service was measured from checkout revision
  `984fcef41594b548ade43ef4627f7d31effd7910`. That revision changes only the
  benchmark scaffold relative to product parent
  `936053e23b78b2afea9e23958605d9b431706710`.
- The retired Gaussian-shell code was imported without switching the worktree:
  `git archive c51c09fc~1 nova` resolved to
  `a0798fe28a342dd8fb922ef1dae2ddc917db4e3c` and was placed in a temporary
  import overlay. It received the same map, radial cells, 28 bins, and batch
  construction as the current service.
- The host comparator calls
  `FluxSurfaceGeometry.from_flux_map` in
  `nova/equilibrium/flux_surface_geometry.py` on CPU. The full min/max values,
  device metadata, source revisions, status, and job identity are retained in
  `results.tsv`.

The benchmark is rerunnable from the repository root with:

```bash
sbatch scripts/geometry_service_benchmark/bench.sbatch
```

The batch job reinitializes `results.tsv`, writes its full output to the named
crew-run log, and exits nonzero if any requested lane fails.
