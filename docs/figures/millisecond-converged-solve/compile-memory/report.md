# Whole-cell forward-solve compile: host memory, wall time, executable size

One H200 job (1270971), source bbc5c744, three rungs of the weak restricted
case (`weak-rotation-reactor-static`), each a fresh child process that compiles
--- never executes --- the accelerated history program `profile.solve(request)`
would run.  Cold is a first compile with an empty persistent cache; warm is a
second fresh process for the same rung, the cache already holding the first
process's entry.  Host peak is `ru_maxrss` (this process) and the allocation's
cgroup high-water; both allocations were `--mem=250G`.

## Tables

### Cold compile

| cells | construct (s) | lower (s) | backend (s) | total (s) | peak RSS (GiB) | exec bytes | exec proto (GiB) if >2 GiB | generated code (GiB) | stablehlo instrs | optimized instrs |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 17.5 | 10.5 | 467.3 | 535.2 | 48.84 | 461,724,765 | — | 0.44 | 254,873 | 448,360 |
| 1000 | 18.4 | 10.6 | 479.7 | 548.0 | 62.79 | serialization failed | 3.74 | 3.52 | 254,873 | — |
| 2500 | 27.3 | 10.6 | 562.8 | 711.2 | 136.43 | serialization failed | 19.2 | 19.0 | 254,873 | — |

### Warm (persistent-cache retrieval)

| cells | construct (s) | lower (s) | backend (s) | total (s) | peak RSS (GiB) | retrieved? |
|---|---|---|---|---|---|---|
| 300 | 4.4 | 10.5 | 74.4 | 114.0 | 10.04 | yes — backend 467.3 s → 74.4 s |
| 1000 | 6.2 | 10.8 | 480.5 | 536.6 | 62.60 | no — cache put failed, recompiled |
| 2500 | 15.1 | 10.7 | 566.3 | 701.1 | 136.32 | no — cache put failed, recompiled |

cgroup high-water tracks RSS within 0.4-0.8 GiB at every cold rung
(49.25 / 63.25 / 137.02 GiB); the warm 300-cell process peaks at 10.04 GiB RSS
because it never allocates the Ptxas/relink workspace.

## Findings

**The 2500-cell compile needs ~137 GiB of host memory, and peak scales much
more gently than the executable it produces.**  Cold peak RSS rises
48.8 → 62.8 → 136.4 GiB across 300/1000/2500 cells.  That is sub-linear to
~linear growth (piecewise exponent ≈ 0.21 from 300→1000, ≈ 0.85 from
1000→2500): the ~49 GiB floor at 300 cells is the compiler working set that
barely moves with the mesh, after which host peak grows nearly as cells do.
At 2500 cells that is 55% of the 250 GiB allocation — the large rung is a
memory-bound compile, not a wall-bound one.

**Lowering is a fixed ~10.5 s at every rung; all growth is in the backend.**
The stablehlo module contains 254,873 instructions at every rung — instruction
count is shape-independent, exactly as the census-kernel measurement found for
its kernel.  The program is mesh-generic: cells enter as traced shapes, so
code differs only in array sizes.  Backend compile (Ptxas over the expanded
shapes) rises 467 → 480 → 563 s, a much weaker dependence than the executable
size.

**The serialized executable grows ~cells^1.8 and crosses the 2 GiB protobuf
limit between 300 and 1000 cells.**  Serialized size at 300 is 461.7 MB
(440 MiB); at 1000 the GpuExecutableProto already reaches 3.74 GiB and at 2500
19.2 GiB, both rejected by `RESOURCE_EXHAUSTED` (XLA's 2 GiB message cap), so
`executable_bytes` is unavailable there.  The compiler's own allocation census
agrees: generated code 0.44 → 3.52 → 19.0 GiB (exponent ≈ 1.8), and its peak
buffer allocation 0.64 → 4.07 → 20.1 GiB.  `serialize()` wall time itself is
small even where it fails (20-44 s).

**Persistent-cache retrieval scales with the executable — and stops working
above the 2 GiB serialization ceiling.**  At 300 cells the 461.7 MB executable
is stored (392 MB on disk under `~/.cache/nova/compile-host-memory`), and the
warm process retrieves it: backend stage drops 467.3 → 74.4 s (6.3×) and total
535.2 → 114.0 s (4.7×).  At 1000 and 2500 cells the executable can never reach
the cache — the cache put path serializes the executable (as does every XLA
path), so a >2 GiB proto is refused *before any entry is written* — and warm
recompiles at full cost (480.5 / 566.3 s ≈ cold).  The measured fact is that
retrieval serves an executable that fits the limit in ~1/6 of a cold compile
with a tenth of the RSS, and that the whole-cell solve is past the limit
between 300 and 1000 cells, so 1000/2500 rungs pay cold cost every time.

Figure: server path `/nova/figures/millisecond-converged-solve/compile-memory/compile-memory.svg` (host peak and executable/code size against cells, 2 GiB ceiling marked).
