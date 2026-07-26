# Benchmark baselines

Machine: 98dci4-srv-1006 (Xeon Gold 6442Y, login node under shared load) ·
python 3.14 · asv 0.6.6 (`asv run --quick -E existing`) · recorded 2026-07-24.
Timings are single-round `--quick` figures intended as order-of-magnitude
references; re-run on a quiet node before reading small deltas as regressions.

## biotoperate (asv)

| benchmark | time |
|---|---|
| PlasmaEvaluate.time_flux_function | 109 ms |
| PlasmaEvaluate.time_flux_function_ev_only | 109 ms |
| PlasmaEvaluate.time_radial_field | 117 ms |
| PlasmaEvaluate.time_field_magnitude (`bp`) | 107 ms |
| PlasmaOperate.time_load_operators | 4.4 ms |
| PlasmaOperate.time_solve | 963 ms |
| PlasmaTurns.time_update_turns (svd_rank 10/75/200/500/-1) | 229 / 229 / 209 / 290 / 205 ms |

## framespace (asv)

| benchmark | time |
|---|---|
| GetCurrent.time_getattr | 40.4 µs |
| GetCurrent.time_getattr_subspace | 8.8 µs |
| GetCurrent.time_getitem | 35.6 µs |
| GetCurrent.time_loc | 43.1 µs |
| SetCurrent.time_metaframe_data | 2.3 µs |
| SetCurrent.time_subspace | 7.5 µs |

## framesetloc (asv, indexer sweep, µs)

Get item: loc 31.1 / sloc 6.1 / aloc 25.5 / saloc 3.1 (Ic); nturn loc 7.3 / aloc 3.8; x loc 8.0.
Get subitem: loc 44.8 / sloc 11.2 / aloc 29.3 / saloc 6.0 (Ic); nturn loc 31.0 / aloc 7.3; x loc 37.3.
Set item: sloc 28.6 / saloc 18.0 (Ic); nturn loc 24.4 / aloc 20.3; x loc 26.5.
Set subitem: sloc 13.6 / saloc 5.5 (Ic); nturn loc 50.6 / aloc 18.9; x loc 49.8.

## Standalone scripts

| script | result |
|---|---|
| coil_construction.py — construct n=100/500/1000/5000 | 2.12 / 3.29 / 4.73 / 16.65 ms |
| coil_construction.py — read `loc[:, "Ic"]` / `.Ic` (n=5000) | 0.009 / 0.008 ms |
| io_ingest.py — AL full-get vs warm zarr re-read | 981.6 ms vs 6.9 ms (**142×**) |

## zeta quadrature — fixed-node rules vs uniform midpoint

Recorded 2026-07-25, same machine and python as above. **Methodology:** every
figure is a FRESH python process, first run, one variant per process — repeating
a solve in-process warms allocators and reports roughly 40% fast. Three
processes per variant; the spread below is the observed range, not an estimate.
The `midpoint` column drives `zeta_midpoint`, the uniform rule retained as the
equivalence-test reference, in place of the production rules, so the pair
brackets the change on one build.

Production rules: Gauss–Legendre 48 nodes where `|gamma| >= 0.2 r`, tanh-sinh
177 nodes below that. The uniform rule used ~500 panels per unit `alpha`
(785 at `alpha = pi/2`).

| measurement | midpoint | fixed-node | speedup |
|---|---|---|---|
| bow point-cloud solve, 10 segments × 20k targets | 9.36–9.37 s | 4.04–4.33 s | **2.3×** |
| — zeta share of that solve | 5.63–5.64 s (60%) | 0.41–0.42 s (10%) | **13.6×** |
| zeta kernel, 200k elements (half inside the near-plane band) | 1.66–1.68 s | 0.53–0.55 s | **3.1×** |

Accuracy over the same change: worst relative error against a converged
reference falls from 3.4e-3 (in the plane of the source corner) / 8.7e-7 (far
field) to 5.5e-14 everywhere — see `tests/test_biotzeta.py` for the scan.

The zeta block size matters more than the node count: sizing a block's
temporaries to stay in cache is worth 4× over multi-megabyte blocks, and past a
few megabytes allocator churn costs another order of magnitude (2.44 s at 2**22
doubles, 8.4 s at 2**19, 0.53 s at 2**16 on the 200k-element kernel).

## coupling kernel cost per pair — compute node, fresh process, median of 3

Recorded 2026-07-25 (sun_debug node; login-node timings showed 5× spread and
were discarded). Drivers: `benchmarks/kernel_cost.py` / `kernel_cost_table.py`;
accuracy columns and the near/far + order-vs-distance evidence are archived in
`docs/archive/biot-operator-assembly-s3-landed.html`.

| method | µs/pair | × point |
|---|---|---|
| point filament | 0.42 | 1 |
| hybrid rectangle (switch=3) | 33.9 | 81 |
| cylinder rectangle | 61.0 | 145 |
| bow corner zeta (pre fixed-node zeta) | 148.0 | 352 |
| polygon hex 16×48, closed-form gradient + block 16 | 857.8 | 2039 |
| polygon hex 16×48, complex-step unblocked (retired) | 4335.9 | 10307 |
| polygon hex CLOSED FORM, 128 residual nodes, by corner | 171.4 | 408 |
| polygon hex closed form, PACKED driver (no shortcuts, traceable) | 230.8 | 550 |
| polygon hex closed form, by edge limit (superseded) | 333.9 | 795 |
| faithful Part V full-turn floor (special functions + g_p only) | 12.5–16.1 | ~30–38 |

The closed form (`nova/biot/polygonanalytic.py`, driver
`benchmarks/analytic_cost_floor.py`) returns the flux and BOTH field components
for the same cost — they share every reduction — and its cost depends on the
section's shape, because the evaluation is organised by CORNER and a corner is
shared only where both its edges are live. Same job, one idle core, fresh process
per variant: hexagon 333.9 → 176.6 µs/pair (1.89×), thin plate 220.7 → 119.3
(1.85×), rectangle 109.4 → 113.2 (0.97× — it drops two horizontal edges, so no
corner is shared and there is nothing to win). Floor 100.1 µs/pair, of which
73.3 is the one graded residual quadrature that survives per edge limit — the
other is a function of the corner alone and cancels around a closed chain of
edges. Peak allocation 12.1 kB/pair (was 10.6: up to three corner parts are live
at once).

Re-measured 2026-07-26 after the complete elliptic integrals moved onto one
complement-native descent (`nova/biot/completeelliptic.py`): the closed form's
third-kind integrals fell 12.70 → 5.06 µs/pair and its floor 100.1 → 92.9, with
every one of the eighteen recorded accuracy entries UNCHANGED. Assembled hexagon
176.6 → 171.4. The graded residual quadrature is untouched at 73.2 and is now 79%
of the floor. Complete-integral routes, 4096 elements, one core, fresh process,
median of 3:

| route | µs/element | × Cephes |
|---|---|---|
| `ellipk(m) + ellipe(m)` | 0.0308 | 1.00 |
| `ellipkm1(k'²) + ellipe(m)` | 0.0312 | 1.01 |
| Carlson `R_F + 2 R_G` (K and E) | 0.1495 | 4.9 |
| descent, K and E from one sweep | 0.1687 | 5.5 |
| Carlson `R_F + (n/3) R_J` (third kind) | 0.3928 | 12.8 |
| descent, third kind | 0.1175 | 3.8 |

So a complement-native FIRST kind is free on the host (`ellipkm1`, which the
point kernels take), and the descent's THIRD kind — which no Cephes route offers
and which the polygon reduction needs several of per corner — is 3.3× cheaper
than Carlson's as well as being the only traceable one.

The PACKED driver (`packed_analytic_greens`) is the same reduction with the host
driver's three value-dependent shortcuts replaced by arithmetic so that it
traces: 230.8 vs 171.4 µs/pair at one array width, i.e. the shortcuts are worth
1.35× on the host.

Tiled assembly (`nova/biot/tiledassembly.py`): ~48 B/pair (vs 267 B/pair
through the ≤500-source chunking), 16-core scaling 8.78×; measured 16-core
rate 107 µs/pair ⇒ projected 2000-cell exact-everywhere polygon build 7.1 min.

## tiled backend — one JAX trace on CPU and GPU vs the process pool

Recorded 2026-07-25. Driver: `benchmarks/tiled_backend.py`. Same operator
throughout: 320 hex cells × their own centres = 102,400 pairs, 64 tiles of
40×40, exact-everywhere 16×48 rule, block 16; median of 3 fresh processes.
Parity numpy↔jax: worst absolute deviation 2.6e-17 (CPU) / 5.8e-17 (GPU)
through the assembled zarr stores; one compilation per build.

CPU node (sun_debug, 16 cores):

| variant | seconds | µs/pair | compile s |
|---|---|---|---|
| numpy 1 core | 95.49 | 932.6 | — |
| numpy 8 cores | 16.44 | 160.6 | — |
| numpy 16 cores | 11.54 | 112.7 | — |
| jax scan (16 cores, 1 process) | 19.82 | 193.6 | 1.57 |
| jax vmap (16 cores, 1 process) | 8.07 | 78.8 | 1.25 |

GPU node (betelgeuse, H200 NVL):

| variant | seconds | µs/pair | compile s |
|---|---|---|---|
| numpy 8 cores (same node) | 9.36 | 91.4 | — |
| jax vmap | 0.76 | 7.4 | 1.54 |
| jax scan | 1.11 | 10.8 | 2.12 |

Tile-size sweep (H200, vmap, 160 cells): 400 pairs/tile → 15.68 µs/pair /
131 MB device high-water; 1,600 → 7.18 / 864 MB; 6,400 → 2.83 / 1.43 GB
(1.3% of the 112.6 GB card). `TilePlan.peak_bytes` does NOT bound a batched
tile — budget models for device batching must be calibrated from these
measured high-water marks.

Projected 2000-cell exact-everywhere build (4e6 pairs): 7.5 min at 16 numpy
cores; 5.3 min jax vmap on the same 16 cores in one process; ~30 s on one
H200 at 40×40 tiles; ~11 s at 80×80.

### the CLOSED FORM as a tile kernel

Recorded 2026-07-26, same driver with `--variants closed-*`. 64 hex cells ×
their own centres = 4,096 pairs, 4 tiles of 32×32, block 16, one fresh process
per variant. `tile_evaluator(kernel="closed")` traces
`packed_analytic_greens`; parity against the same driver on numpy 2.5e-12
(CPU) / 3.2e-11 (GPU, through the store), one compilation per build.

| variant | µs/pair | compile s | where |
|---|---|---|---|
| closed-host (host driver, per section) | 840.5 | — | 64 targets/call — Python overhead per call, not comparable with the 171.4 above |
| closed-numpy (packed driver over tiles) | 272.4 | — | 1,024 pairs/call |
| **closed-vmap, one H200** | **5.5** | 101.3 | against the 16×48 quadrature's 5.3 on the same card |
| closed-scan, one H200 | 65.0 | 91.3 | scan loses on the device here too |

The closed form on one H200 costs what the 16×48 quadrature costs, for one to
two orders more accuracy — so on a device the exact kernel is free relative to
the rule it replaces. But **the compile is not a footnote to that, it is half the
build**, and it is where the traced closed form has to be judged.

    device / tile            compile s   note
    H200, 32×32                  101.3   167,730 HLO ops (six-edge sections)
    L4, 3×3                      136.1   the same graph, a smaller tile
    16 CPU cores, 32×32         >3600    killed by the queue limit, never finished
    16 CPU cores, 4×4 (16 pairs)>3300    never finished either -- XLA:CPU raises its
                                         own slow_operation_alarm and keeps going

So the traced closed form is a DEVICE-ONLY path: XLA:CPU lowers the whole tile to
one LLVM function and the optimiser does not get through it at any tile size worth
building. `polygon_analytic_greens` remains the host route, and the premise of one
traced code path serving both devices — which holds for the quadrature kernel at a
1.3 s compile — does not hold for this kernel.

The graph is EXACTLY LINEAR in the corner count — 27,947 HLO operations per edge,
measured at 83,889 / 111,836 / 167,730 / 195,677 for three, four, six and seven
edges — against the quadrature kernel's 3,272 for the same block, and compile
time tracks the count at roughly half a millisecond per operation on a device.
The recursions everyone reaches for first are only 31% of it (`harmonic_moments`
3,580 ops per corner, the pole families 454 each, the descent 402, each seed
290); the other 69% is the harmonic-series coefficient algebra, which is not a
recursion. Rolling every recursion into a `scan` is therefore worth about 1.45×.

What that means for choosing a route, at 5.5 µs/pair on the H200 against the host
closed form's 171.4 on one core and a projected 19.5 on a 16-core pool:

    total pairs in the build     host 1 core   host 16 cores   H200 + compile
    20,000                            3.4 s          0.4 s          101 s
    400,000                          68.6 s          7.8 s          103 s
    4,000,000                        11.4 min        78 s           123 s
    25,000,000                       71 min          8.1 min        4.0 min

The compile equals the device kernel at 18.4M pairs and the device only beats the
16-core pool above 7.2M. So the device is the right route for ONE all-to-all
machine matrix and the wrong one for a per-block build — unless the compile is
amortised, which is what the next section measures.

### what the compile is paid PER

Recorded 2026-07-26. Driver: `benchmarks/tiled_backend.py --variants
*-positions,*-cache`, one fresh process per measurement, cache off except where
it is the subject. Two things changed and no arithmetic did: `tile_evaluator`
memoises on `(plan, batched, kernel)`, so one tile shape has one executable per
process; and `compilation_cache` points JAX's persistent cache at
`NOVA_COMPILATION_CACHE` (default `~/.cache/nova/kernels`, 2 GiB LRU, `off` to
refuse it).

A **geometry scan** — the same 64 hex cells at four positions 13 mm apart in R,
each position a whole build through `assemble` into its own store — on one H200
at 4 tiles of 32×32, median of three processes:

| kernel | first position | each later position | compilations |
|---|---|---|---|
| closed | 101.65 s | 0.027 s | 1 |
| quadrature | 2.20 s | 0.050 s | 1 |
| quadrature, 16 CPU cores (320 cells, 40×40) | 9.44 s | 8.20 s | 1 |

The first position pays the compile and no later one pays anything: moving a
section changes argument VALUES, and geometry is an argument to the tile kernel
rather than a constant of it, so a scan cannot force a retrace. **A closed-form
pack swept through eight positions costs 102 s instead of 813 s.** The 8.20 s
per CPU position is the 8.14 s a standalone build of the same operator takes, so
reuse costs nothing per build; the 27 ms on the device is 6.7 µs/pair against
the kernel's own 5.5, the difference being one zarr store created per position.

Across a **process boundary**, the same build twice with one on-disk cache:

| kernel / device | cold compile | warm compile | cache written |
|---|---|---|---|
| closed, H200 | 101.89 s | 8.45 s | 2.8 MB |
| quadrature, H200 | 1.36 s | 0.23 s | 0.17 MB |
| quadrature, 16 CPU cores | 1.33 s | 0.57 s | 0.17 MB |

Three executables are stored per build and all three are hits in the second
process, which reports 91.2 s of compile saved on JAX's own counter. The
remaining 8.45 s is the part a cache cannot skip — tracing the reduction and
lowering it to HLO, which is what the cache key is computed FROM. So the
persistent cache removes 92 % of a cold closed-form compile and the warm
evaluator removes the rest.

Steady state is unchanged, measured the same way as the table above:
closed-vmap 5.5 µs/pair on the H200 (spread 5.1–5.8 over three processes,
against 5.5 recorded on 26 July), jax-vmap 5.6 (spread 5.5–7.0, against 5.3),
and on 16 CPU cores jax-vmap 79.5 µs/pair against 78.8. Compile itself is also
unchanged where it is still paid cold: 101.3 s on the H200 against 101.3.

What that does to the route choice, against the same host figures:

    total pairs in the build     host 16 cores   H200 cold   warm cache   warm evaluator
    20,000                              0.4 s       101 s        8.5 s          0.1 s
    400,000                             7.8 s       104 s       10.6 s          2.2 s
    4,000,000                            78 s       124 s         30 s           22 s
    25,000,000                        8.1 min      4.0 min      2.4 min        2.3 min

**The build size at which the device beats a 16-core host pool falls from 7.2 M
pairs to 0.60 M with the cache warm, and to nothing at all with the evaluator
warm** — at which point the device is a flat 3.5× the pool at any size. A
per-block build is still the wrong shape for it (0.4 s against 8.5 s at 20,000
pairs, in a fresh process), but a session that builds repeatedly now pays the
compile once rather than once a build.

## three-band polygon-section coupling (opt-in: PolySection.configured(banded=True))

Recorded 2026-07-25 (sun_debug node; drivers were throwaway — acceptance
sweeps live in `tests/test_biotbandedcoupling.py`). Bands by distance to the
section contour: 16×48 rule inside 2.2 contour radii, 8×24 to the far seam,
moment-corrected filament beyond (second + third moments; the third-moment
term is a bit-exact no-op on symmetric sections). Far seam 6.8 radii for
section skew ≤ 1e-3, else 16.0.

Per-pair cost (hexagon / wall-clipped): near 871/1012 µs, mid 255/298 µs,
far filament 2.5/5.0 µs; whole 2339-target column 13.1/52.0 µs/pair vs
869/1008 exact-everywhere — **66× / 19× cheaper per column**. Banded node
count 1.20% / 4.28% of exact. Worst per-component error beyond the near band
≤ 1.7e-7 of local |B| (≤ 6.8e-9 on ψ contour maps); the near band is
bit-identical to the production 16×48 rule. Seam jumps ≤ 1.6e-7 of local.
