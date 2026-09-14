# Scanned outer trip loop against the production fori loop

A whole-cell solve whose outer trip loop drives the production per-trip body
under `jax.lax.scan`, compared arm-to-arm with the production
`jax.lax.fori_loop` outer loop, at 300 cells (weak-rotation-reactor-static,
clip mode "chord", `active_set_settled`, 3 trips).  Both arms run the same
verbatim per-trip body, the same carry, the same seed; the scan arm differs
from production only in the single coordination line
(`fixed_point.py:3837` — `jax.lax.scan(lambda carry, index: (outer_body(index,
carry), ()), outer, jnp.arange(1, active_set_steps))`).  Measured on CPU
(`all_debug`, job 1270960) and one H200 (job 1270961).

## Both platforms, arm by arm

| metric | CPU production | CPU scanned | CUDA production | CUDA scanned |
|---|---|---|---|---|
| compiled instructions | 657,378 | 657,390 | 448,355 | 448,367 |
| temporaries (`temp_size_in_bytes`) | 253.523 MiB | 253.523 MiB | 190.965 MiB | 190.966 MiB |
| optimised HLO bytes | 174.1 MB | 197.6 MB | 138.5 MB | 137.6 MB |
| on-disk executable | 286.843 MiB | 286.902 MiB | 374.433 MiB | 374.413 MiB |
| execution wall (min of repeats) | 77.92 s | 78.17 s | 4.905 s | 4.899 s |

Instruction and HLO counts are per-compiled-program reads (XLA HLO
instruction count + `memory_analysis`); executable size is the serialized
compiled executable's on-disk artifact from the persistent compilation cache.

| metric | CPU both arms | CUDA both arms |
|---|---|---|
| terminal residual |  6.195329e-03 | 6.195329e-03 |
| converged flag | False | False |
| termination reason | active_set_settled | active_set_settled |
| active-set iterations | 3 | 3 |
| testing axis error | 0.1160 m | 0.1160 m |
| terminal state bits identical | True | True |
| terminal state sup difference | 0.0 | 0.0 |
| per-trip residual sup difference (finite slots) | 0.0 | 0.0 |

## The only measured compiled difference

The scanned arm is the production program with its outer coordination line
replaced; every compiled-program statistic that differs does so by a rounding
of the same order:

| delta (scanned − production) | CPU | CUDA |
|---|---|---|
| instructions | +12 | +12 |
| temporaries | +128 B | +512 B |
| HLO bytes | +23.5 MB | −0.9 MB |
| on-disk executable | +62,383 B | −21,328 B |

The +12 instructions are the scan coordination (an explicit arange-index loop
slot) over the fori_loop's runtime-bounded while.  The HLO and executable
deltas are compiler-layout noise at the 0.05-13% level on 138-197 MB HLO /
287-374 MiB artifacts, opposite in sign across the two backends.

## Per-trip residual history

Both arms settle identically: trip 1 residual 6.195396e-03, settling to
6.195329e-03 at trip 2 and holding through trip 3 (paywalled below the
convergence tolerance, hence `active_set_settled` unconverged).  The finite
residual slots are bit-identical between arms on each platform
(`per_trip_residual_sup_difference` = 0.0 on both).

![per-trip residual, CPU vs scanned](/nova/figures/millisecond-converged-solve/scan-prototype/per-trip-residual-cpu.svg)

![per-trip residual, CUDA vs scanned](/nova/figures/millisecond-converged-solve/scan-prototype/per-trip-residual-cuda.svg)

## Can the outer loop be scanned without a production change, and what does it buy?

**Yes — the production outer loop is already scanable with no source change.**
The production trip loop is a pure traced function of its carry
(`fori_loop` + cond-skip, host callbacks disabled by default,
`stream_active_set=False`), so the only edit ever required was in the
benchmark's copy: replacing the coordination line with `lax.scan` over an
explicit trip index.  The transcription is verbatim everywhere else,
verified by whitespace-normalised inspect-diff (one semantic line changed).

**What it buys is architectural, not a measured speedup at 300 cells.**  At
this size the loop is 3 trips and body-dominated, so the measured differences
are a rounding error: +12 compiled instructions (0.0005% of the CPU HLO /
0.0002% of the CUDA HLO), ≤512 B of change in ~0.25 GiB of temporaries, and
no wall-clock effect (CUDA 4.905 s vs 4.899 s min-of-repeats; CPU +0.25 s,
body-driven).  What the scan actually changes is the loop's shape: the trip
bound becomes a static `arange` rather than a runtime-evaluated condition, so
the loop has fixed memory and no dynamic bounds for the allocator and the
scheduler to negotiate; the scan body becomes a first-class traced value the
XLA pipeline can segment, pipeline, or parallelise over trips, and the loop
can be embedded or vmapped where the while-loop cannot.  None of that is
exercised at 3 trips, which is why the honest reading is "no regression now;
a structural option for larger or fixed-shape trip counts", not a speedup.

## Verification

- Whitespace-normalised lineage diff of `scanned_active_set_newton_krylov`
  against production `_active_set_newton_krylov`: exactly the coordination
  line differs.
- 110-cell smoke before the 300-cell lanes: scanned arm lowered inline
  (655,660 instructions, 0.1243 GiB temporaries, residual 5.22e-18,
  converged), proving the shadow map functions are threaded through the scan
  body.
- Both platforms, both arms: same seed, same terminal state bit-for-bit;
  per-trip residual histories bit-identical on the finite slots; axis error
  equal to full precision (0.116027 m on CPU, 0.116027 m on CUDA).
- Wall times are min-of-repeats on device; compile wall is one-off and not
  reported (cold persistent-cache compiles dominated `warm_execution`).

Evidence receipts: `docs/figures/millisecond-converged-solve/scan-prototype/parts/`
(cpu/cuda × production/scanned census + execution, cpu/cuda comparison, cpu/cuda figure).
