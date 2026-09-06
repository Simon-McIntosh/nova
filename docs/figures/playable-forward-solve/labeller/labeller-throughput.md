# Forward labeller throughput on the H200 — f-pfs-012

_Node pfs-labeller-throughput. Numbers from the completed H200 measurement,
job 1263148 on `98dci4-gpu-0003` (partition `betelgeuse`, reservation
`gpu_0003_grpA`), source revision `14c353c4` plus the working-tree seed
fallback._

## What was measured

Driver `benchmarks/forward_labeller_throughput.py`, one MAST shot (22086): the
57 consecutive efm slices with finite EFIT centre heights and fitted currents
(rows 1..57, 5 ms cadence, t = 0.020..0.300 s) stepped in time order through
the constrained reduced-Newton route on one reserved H200, warm-started slice
to slice from the previous equilibrium and, in the conditioned arm, the
previous compensating unknown. One operator built from the calibrated keyframe
(22086/43) through the passive-inclusive frozen-six response carrier; each
slice supplies its own fitted circuit currents (`efm/fcoil_c`) as the
prescribed current and, in the conditioned arm, its own EFIT centre height
(`efm/current_centrd_z`, decision `labeller-vertical-constraint`).

Two arms: **free** (no constraint row) and **conditioned** (vertical
current-centroid row pinned to that slice's EFIT centre height).  Per slice
the receipt records trips, Newton steps, wall, converged and qualified flags,
achieved centroid against the target, the compensating current and the
conditioning flag with its target source, persisted as each slice lands; the
compile cost and cache-hit count per program; and a vmap probe over 8 and 16
slices of the fused trip.

## Slice costs and rates

| arm | solved | converged | qualified | median as-built wall/slice | median warm wall/slice | slices/s (as-built) | slices/GPU-hour (as-built) | warm slices/s | warm slices/GPU-hour |
|---|---|---|---|---|---|---|---|---|---|
| free | 56/57 (98.2%) | 49 (87.5% of solved) | — | 4.88 s | 48.6 ms | 0.212 | 764.5 | 12.18 | 43 835 |
| conditioned | 47/57 (82.5%) | 38 (80.9% of solved) | 38/38 (100% of converged) | 17.33 s | 50.3 ms | 0.064 | 231.6 | 13.85 | 49 851 |

Unsolved rows: free row 2; conditioned rows 1, 12, 48, 49, 52–57 (the ramp-up
and ramp-down frames, where the topology class or the pinned-row derivation
fails the read; the free arm also drops row 2 only). The warm wall is billed
as trips × the arm's median warm trip, so it is what the same loop costs once
the programs are hot and currents and targets are traced arguments instead of
trace constants.

## Compile cost per program

As merged, the currents, the reduced coordinates and the constraint target
enter the compiled kernels as trace constants, so every slice's solve is a
distinct program and the first trip pays a compile or persistent-cache load:
56 programs compiled in the free arm, 47 in the conditioned arm, at a median
as-built wall of 4.9 s (free) and 17.3 s (conditioned) per slice.  The
per-slice as-built wall is therefore compile-dominated by one to two orders of
magnitude over the warm steady state (48–50 ms per warm trip).

## Batched-entry (vmap) probe

The fused trip-boundary kernel was vmapped over 8 and 16 states of one program
(the keyframe, slice 43). vmap does **not** help: per element it is 1.49× (batch
8) and 1.57× (batch 16) slower than the same calls in a serial loop (0.257 vs
0.173 s and 0.067 vs 0.043 s per slice), because the fused close is a
write-then-read topology pass whose per-element work and memory dominate and
do not amortise.  A genuine batched entry over distinct-current slices does not
exist in the route: external currents, reduced coordinates and the constraint
target are trace constants, so distinct slices are distinct programs, and the
Newton ladder is a host loop with data-dependent backtracking.  The lever that
changes the rate is therefore the traced-argument change (one program per
shot), not batching.

## Extrapolation to the 1,341,435-slice census

| scenario | hours on 1 H200 | hours on 8 H200 |
|---|---|---|
| free, as-built (current route) | 1 755 | 219 |
| free, warm steady state (traced arguments) | 30.6 | 3.8 |
| conditioned, as-built (current route) | 5 792 | 724 |
| conditioned, warm steady state (traced arguments) | 26.9 | 3.4 |

## Can the rate feed training online?

The decoder consumer runs at 8 frames/s per process (125 ms per label).  The
warm steady-state label latency is 82 ms (free) and 72 ms (conditioned) —
comparable to the demand, so one warm program could in principle keep a single
8 fps process fed, with little margin and none for decode, transfer and host
overhead.  The as-built rate with per-slice compiles (0.21 and 0.06 slices/s)
cannot feed online at all.  **Answer in numbers:** warm ~12–14 slices/s per GPU
versus 8 frames/s demand; as-built 0.06–0.21 slices/s.  The labels are produced
**once, ahead of training**, written per shot to file in the decoder session
shape (§8C), and the solve rate bounds how long the corpus takes to build (30 h
on one GPU at the warm steady state), not how fast training reads it.

## Two records carried beside the numbers

1. **Conditioning flag and target source.** Every conditioned slice is flagged
   `conditioned=True, target_source=efm/current_centrd_z` alongside the arm; the
   free arm carries `conditioned=False`. Neither imas-ambix guard sees a number
   handed over as data; the flag is the removal mechanism (`which labels carry
   the pin is a flag read, not a corpus audit`).
2. **One-scalar-per-slice leak caveat.** Every conditioned label inherits one
   reconstruction scalar per slice on the axis position. Any held-out result
   scored by the EFIT referee on a corpus built with these labels carries a
   one-scalar-per-slice leak on exactly that number (the interior quantity
   whose skill claims have been the weakest and most contested) — a demo
   figure quoted as skill would be flattered on it. Applies to conditioned
   slices only; the free arm is clean.

The height pin is authorised **for the demo only** and is removed when a
Thomson forward model can supply the plasma height instead of the
reconstruction (the removal sentence is written at the row application in the
driver, where a reader touching the pin can tell whether the condition is met).

## Evidence

- Receipt: `docs/figures/playable-forward-solve/labeller/forward-labeller-throughput.json`
- Figure: `docs/figures/playable-forward-solve/labeller/forward-labeller-throughput.png`
- Harness: `docs/figures/playable-forward-solve/labeller/run_labeller_h200.sh`
- Human receipt: `docs/figures/playable-forward-solve/labeller/labeller-throughput.md`
- Job log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260905T211013371009-pfs-labeller-throughput/`
- Source revision: `14c353c4`

## Data constraint

`shot 22086` holds 60 reconstructed slices; rows 0, 58, 59 lack finite EFIT
centre-height/current references, so the measured chain is 57 consecutive
slices, not the 100+ the node brief quoted (the store's largest single-shot
slice count is 75). The rate and extrapolation are per-slice figures
unaffected by chain length.
