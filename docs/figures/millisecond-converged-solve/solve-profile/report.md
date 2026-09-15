# Solve-profile: kernel count, launch latency and program size of one whole-cell solve

Case `weak-rotation-reactor-static`, whole-cell mode (`set_support_clip_mode("chord")`),
at 300 and 1000 cells on the reserved H200 (betelgeuse, `gpu_0003_grpA`).
Each cell count: one warm solve (compiles and runs the production program once),
then the same already-compiled program re-executes inside `jax.profiler.trace`.
Receipt: `docs/figures/millisecond-converged-solve/solve-profile/solve-profile.json`
(per-cell parts `solve-profile-300.json`, `solve-profile-1000.json`), with a
cumulative-kernel-time SVG per cell count (`solve-profile-300.svg`,
`solve-profile-1000.svg`).

## Kernel count per solve

| metric | 300 cells | 1000 cells |
|---|---|---|
| requested cells / mesh nodes | 300 / 342 | 1000 / 1072 |
| device kernels executed (per solve) | 2,057,009 | ≥ 2,066,825 (floor) |
| distinct kernel programs | 1,444 | ≥ 890 (floor) |
| summed kernel wall (device) | 3.358 s | ≥ 4.451 s (floor) |
| serial launch gap (device idle between kernels) | 4.127 s | ≥ 3.689 s (floor) |
| traced solve wall (annotation window) | 9.976 s | 65.817 s (exact) |
| launch-gap share of wall | 41.4% | ≥ 5.6% (floor) |
| kernel-wall share of wall | 33.7% | ≥ 6.8% (floor) |
| kernel wall inside scan/while bodies | 99.96% | 99.99% |

The 300-cell trace is authoritative (CUPTI drop counter = 1, negligible; the
perfetto export is not event-capped). At 1000 cells the CUPTI activity buffer
drop counter reached 50,001 log lines × 10⁴ occurrences each (≥ 5×10⁸ dropped
event occurrences, logged during the traced solve) and the export hits the
perfetto 5,000,000-X-event cap exactly, so every device-side figure at 1000
cells is a **lower bound**; only the 65.8 s annotation wall is exact. The
launch counts land within 0.5% of each other across a 3.3× mesh because the
same Newton/Krylov iteration structure dominates at both cell counts.

## Launch-latency floor vs the millisecond target

Measured serial gap between consecutive device kernels is **2.006 µs/kernel at
300 cells** (4.127 s / 2,057,009) and **1.785 µs/kernel at 1000 cells** (3.689 s /
2,066,825, recorded set) — both well below the 7.6 µs/kernel scan-iteration
reference floor. Per-kernel launch latency is not the binding constraint. What
binds is the **count**: ~2.06 million launched kernels per solve, so the pure
serial launch overhead is ~4 s per solve at either cell count, four orders of
magnitude above a 1 ms whole-solve target. Reaching 1 ms requires collapsing
the launch count (fused/hoisted scan bodies, cuGraph reduction already visible
as `command_buffer::execute` events on the host), not faster launches.

## Dominant kernels (top ten by summed device time, HLO op names)

300 cells (of 3.04 s kernel wall, independent parse):
`loop_add_fusion_38` 9.8%, `input_scatter_fusion_95` 8.2%, `input_reduce_fusion_64`
7.8%, `loop_select_fusion_165` 6.6%, `input_scatter_fusion_416` 5.7%,
`loop_select_fusion_166` 5.5%, `input_reduce_fusion_65` 5.3%,
`input_concatenate_fusion_94` 5.2%, `input_concatenate_fusion_383` 4.9%,
`loop_select_fusion_302` 4.8%.

1000 cells (of ≥ 4.10 s kernel wall, floor): `input_transpose_fusion_11` 18.6%,
`wrapped_scatter` 12.8%, `loop_add_fusion_39` 9.6%, `input_reduce_fusion_63` 7.0%,
`input_scatter_fusion_23` 6.2%, `input_transpose_fusion_49` 6.1%,
`input_concatenate_fusion_23` 5.2%, `loop_select_fusion_152` 4.7%,
`loop_reduce_fusion_45` 4.5%, `input_reduce_fusion_64` 4.4%.

## Trips / Newton: scanned, not unrolled

The pre-lowering StableHLO census is a single rolled loop: 180,931 ops at both
cell counts, `loop_body_op_share` 1.0, `loop_instruction_count` 187 — the whole
solve lives inside one 187-op while body. Doubling the Newton budget (10 → 20
steps) leaves the StableHLO op count flat at both cell counts (178,192 vs
180,931, −1.5%), so the extra trips are extra iterations of the same body, not
unrolled code. The backend-lowered compiled HLO at 300 cells reports 423,284
ops with loop share 0.0: XLA fusion flattens the loop into the launch pipeline
(2.3× the StableHLO count), which is where the 2×10⁶ launch count comes from.

## Program size and compile time

| metric | 300 cells | 1000 cells |
|---|---|---|
| compiled-HLO instruction count (`as_text`) | 423,284 | unavailable |
| StableHLO op count (fallback census) | 180,931 | 180,931 |
| serialized executable bytes | 455,261,717 (434 MiB) | unavailable |
| generated code bytes (`size_of_generated_code_in_bytes`) | 468,703,656 (447 MiB) | 3,782,311,952 (3.52 GiB) |
| cold compile (inside first solve) | ~119 s of 129.3 s warm | ~131 s (2m11s alarm) of 595.3 s warm |
| re-`compile_seconds` (in-process XLA cache hit) | 8.3e-5 s | 5.6e-5 s |

At 1000 cells the compiled HLO module text (~3.9 GB) and the GpuExecutableProto
(~4 GB) both exceed the 2 GiB protobuf transport limit, so `as_text()` and
`runner.serialize()` report `None` with their `*_unavailable` reason strings;
`size_of_generated_code_in_bytes()` is a direct size query and still reports.
The persistent compilation cache cannot store the >2 GiB executable, so every
cold 1000-cell run recompiles (~2m11s). `compile_seconds` is an in-process
cache hit because the program was already compiled by the preceding warm solve;
the cold compile is inside `warm_solve_seconds`.

## Caveats

- 1000-cell device-side figures (launches, programs, kernel wall, gap) are
  lower bounds: CUPTI dropped activity buffers and the export caps at 5M events.
- The 65.8 s 1000-cell annotation wall with only ~8 s of recorded device work
  means host dispatch of a 3.5 GiB program plus dropped device events dominate
  that wall; the split is not separately attributable from this trace.
- `compiled.as_text()` and `runner.serialize()` are unavailable at 1000 cells by
  design (2 GiB protobuf limit — guarded, reason recorded, not a crash).
