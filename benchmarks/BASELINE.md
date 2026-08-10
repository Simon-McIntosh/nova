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
| PlasmaOperate.time_fresh_process_reload | not measured in this historical run |
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

## bow grid solve — the genuine finite-section winding build

Recorded 2026-07-27 on sun_debug node 98dci4-clu-3141, one core, fresh process
per run, first run, `TMPDIR=/tmp`, `JAX_PLATFORMS=cpu`; driver
`benchmarks/bow_grid_solve.py`. The winding, the imports and the coilset are all
resolved before the timer, so the figure is `grid.solve` alone.

| measurement | wall clock |
|---|---|
| 10 bow segments × 2,052 grid targets (57 × 36), `grid.solve(2000, 0.5)` | 4.68 s (4.65–5.24) |

`winding.insert(..., filament=False)` is what labels the segments `bow`; the
default `filament=True` labels them `arc` and the finite-cross-section kernel is
never reached, which is why an earlier "10×2000 bow grid solve 5.9 s" entry could
not be reproduced. Reaching Bow through a grid additionally needed the incomplete
third kind's amplitude fold corrected — an amplitude one representable step below
a quarter turn, which is where a target on the arc's own end plane lands, used to
abort — see `tests/test_biotoperate.py`.

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

Re-measured 2026-07-28 on sun_debug node 98dci4-clu-3141, one core
(`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`), fresh process per variant, first
run, three processes per variant, `benchmarks/kernel_cost.py` unchanged — the same
512-target spiral and the same 0.1 × 0.08 m section at R = 6.2 m as the table
above, so the figures replace those rows directly. What this prices is the corner
antiderivative moving onto the complement-native complete integrals
(`nova/biot/completeelliptic.py`) in place of Carlson `elliprf`/`elliprj` for the
three ring poles per corner.

| method | µs/pair (median of 3) | spread | row it replaces |
|---|---|---|---|
| point filament | 0.38 | 0.38–0.39 | 0.42 |
| hybrid rectangle (switch=3) | 20.6 | 20.2–21.1 | 33.9 |
| cylinder rectangle | 30.6 | 30.4–32.6 | 61.0 |
| polygon rectangle 16×48, block 64 | 783.2 | 778.1–810.2 | not previously recorded |

**The rectangle kernel is 2.0× cheaper than the 61.0 on record, but only 1.12× of
that belongs to the complement-native descent.** Measured either side of that
change on the same node in the same session, `cylinder rectangle` goes 34.3 → 30.6
µs/pair and `hybrid rectangle` 21.6 → 20.6, while the point kernel and the 16×48
quadrature are unchanged (0.39 → 0.38, 811 → 783) — as they must be, since neither
reaches the corner antiderivative. The remaining 61.0 → 34.3 is older than this
measurement pair and is recorded as unattributed: nothing was measured between the
two, so assigning it would be a guess.

The 12.70 → 5.06 µs/pair third-kind figure quoted further down is the closed
form's own special-function cost inside `polygon_analytic_greens`, not a
rectangle-kernel figure — the axisymmetric rectangle kernel's per-pair cost is the
`cylinder rectangle` row. The two are easy to read as one number and are not
comparable: one is a special-function call per polygon corner, the other a whole
kernel call per source-target pair.

`benchmarks/analytic_cost_floor.py` carried the superseded 12.7 in its module
docstring, and its whole table has been re-measured on the same node and date as the
rows above — this module run as a script in three fresh processes, medians: third
kind 5.17 (5.06–5.20), moment stacks 19.09, graded residual 73.08, floor 93.05,
assembled psi 173.77, psi + field 173.68. That reproduces the 5.06 and 92.9 recorded
on 26 July, and additionally moves the two rows never re-measured with the descent:
moment stacks 26.8 → 19.09, which falls because it is built from the same first and
second kinds, and assembled psi 189.8 → 173.77. The graded residual is untouched at
73.08 against 73.3 and is now 79% of the floor.

Re-measured 2026-07-26 through the element's own dispatch, over a whole
2,339-target column instead of the 512-target spiral above (sun_debug node
98dci4-clu-3141, one core, fresh process per variant, median of 3, module import
resolved before the timer starts; driver `benchmarks/polygon_route_cost.py`).
`PolySection.closed_form` selects which exact kernel serves the lane, and it
composes with either binning, so there are four arrangements and not two.

| arrangement | hexagon µs/pair | wall-clipped µs/pair | on record |
|---|---|---|---|
| point filament | 0.2 | 0.2 | 0.42 (512-target spiral) |
| exact-everywhere quadrature | 849.7 | 1003.6 | 869 / 1008 |
| exact-everywhere CLOSED FORM | **162.9** | **196.3** | 171.4 (hexagon) |
| three-band, quadrature near | 13.6 | 52.9 | 13.1 / 52.0 |
| three-band, CLOSED FORM near | 31.5 | 72.6 | — |

Every recorded figure reproduces. **The closed form is 5.2× cheaper than the
boundary quadrature it replaces on the exact-everywhere lane (849.7 → 162.9),
and 2.3× DEARER on the three-band scheme's near band (13.6 → 31.5).** Those are
not in tension: what separates them is the number of pairs in one kernel call.

Cost against DISTANCE is flat for both — 256 pairs a ring, contour distance 0.2
to 29 section radii, the closed form a constant 0.34–0.35× the quadrature at
every one — so a per-pair rate may be quoted for a whole column with no distance
weighting, and the near band's disagreement is not about distance.

Cost against BATCH WIDTH is not flat. The quadrature builds one angular rule and
reuses it across the batch (1084 µs/pair at 8 pairs falling to 851 at 4096, a
factor of 1.3); the closed form holds up to three corner parts live at once and
falls by a factor of 38 over the same range (6532 → 170).

| pairs in one call | quadrature | closed form | quadrature / closed |
|---|---|---|---|
| 8 | 1084.5 | 6532.2 | 0.17× |
| 16 | 980.2 | 3291.0 | 0.30× |
| 64 | 883.7 | 873.3 | **1.01× — they cross here** |
| 256 | 859.9 | 301.5 | 2.85× |
| 1024 | 849.1 | 182.9 | 4.64× |
| 4096 | 851.4 | 170.3 | 5.00× |

**The two kernels cross at 64 pairs in one call.** An exact-everywhere column
hands the kernel all 2,339 pairs at once and the closed form wins five-fold; the
three-band scheme hands its near band 13, an order below the crossing, and there
the quadrature wins. Choosing the exact kernel is therefore not separable from
choosing the binning, which is the one thing the two flags were expected to be
orthogonal about.

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

The projection tables in this historical receipt predate the explicit product
adapter and are retained only to explain earlier route decisions. They are not
current executable-route measurements; current benchmark drivers emit completed
kernel, compile, transfer, reduction, store, and reload timings without scaling
them to unexecuted hardware or mesh sizes.

### a real plasma-grid build through each arrangement

Recorded 2026-07-26, same node and protocol. `CoilSet(dplasma=-500,
tplasma="hex")` on the tracked baseline's first wall (`ellip [4.2, -0.4, 1.25,
4.2]`, `turn="hex"`) meshes to **560 cells**, and the plasma subframe's `segment`
column is relabelled from `circle` to `polysection` to route the solve through
the polygon element — the frame tier needs no change, `Solve.generator` already
maps the label. 560 cells against their own centres is 313,600 pairs. The mesh,
the imports and the grid instance are all resolved before the timer, so the
figure is the operator build alone: kernel, per-column dispatch, composition and
tessellation.

| arrangement | build s | µs/pair | × point build |
|---|---|---|---|
| point filament (explicit reference route) | 1.16 (1.13–1.19) | 3.7 | 1 |
| exact-everywhere quadrature | 274.53 (273.17–275.14) | 875.4 | 237 |
| exact-everywhere CLOSED FORM | **54.46** (54.25–54.61) | 173.7 | **47** |
| three-band, quadrature near | 19.18 (19.13–19.26) | 61.1 | 17 |
| three-band, closed form near | 36.37 (35.91–36.72) | 116.0 | 31 |

The point build reproduces the tracked 963 ms at 1.16 s — the tracked figure
repeats the solve inside one warm process, and a fresh one costs 1.20× that. The
per-pair rates match the column figures above to within a few percent (875 vs
850, 174 vs 163, 61 vs 14 and 116 vs 32), and the two banded rows are the
exception because the build's cells are not the idealised section: see the band
populations below. **The closed form takes the exact-everywhere build from 4.6 min
to 54 s, a 5.0× saving on a real grid, and makes the three-band build 1.9×
dearer.**

### projected 2000-cell first build

2000 cells against their own centres is 4e6 pairs. One core is the measured
column rate; the host pool divides it by the measured 8.78× tiled scaling and is
therefore a **projection, not a measurement**; the device columns take the traced
kernels' recorded steady-state rates and compile costs and apply only to the two
arrangements that have a traced tile kernel — the banded arrangements bin pairs
into three shapes per section and no traced kernel does that.

| arrangement | µs/pair | 1 core | 16 cores | H200 cold | H200 warm cache |
|---|---|---|---|---|---|
| point filament | 0.2 | 0.8 s | 0.1 s | — | — |
| exact-everywhere quadrature | 849.7 | 56.6 min | 6.5 min | 0.4 min | 0.4 min |
| exact-everywhere CLOSED FORM | 162.9 | **10.9 min** | **1.2 min** | 2.1 min | 0.5 min |
| three-band, quadrature near | 13.6 | 0.9 min | 0.1 min | — | — |
| three-band, closed form near | 31.5 | 2.1 min | 0.2 min | — | — |

Against the ~20 min first-build budget: the exact-everywhere quadrature is the
one arrangement that **misses it on a single core** (56.6 min) and needs the pool
to make it (6.5 min, consistent with the 7.1 min projected from the measured
16-core tiled rate). Every other arrangement is inside the budget everywhere, and
**the closed form is the only exact lane that fits on one core** — 10.9 min, with
no parallel assembly and no device at all.

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
H200 at 40×40 tiles; ~11 s at 80×80. All four are the QUADRATURE kernel; the
same build through the host closed form is 10.9 min on one core and 1.2 min
projected on sixteen, so the exact lane no longer needs a device or a pool to
make the first-build budget — see the projection table above.

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

## three-band polygon-section coupling (opt-in: `PolySectionPolicy(arrangement="banded")`)

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
bit-identical to the explicit 16×48 quadrature rule. Seam jumps ≤ 1.6e-7 of local.

### band populations, and the closed form on the near band

Recorded 2026-07-26, same node and protocol as the per-pair section above;
driver `benchmarks/polygon_route_cost.py`. The populations are what turn a
per-band cost into a column rate, so a banded per-pair figure cannot be read
without them. The target cloud is the 2,339-centre hexagonal tiling a 2000-cell
plasma grid lays out; the third row is a real 560-cell grid, aggregated over
every one of its source columns.

| section | corners | skew | far seam | near | mid | far |
|---|---|---|---|---|---|---|
| hexagon | 6 | 1.9e-14 | 6.8 a | 0.56% | 2.57% | 96.88% |
| wall-clipped | 7 | 4.2e-03 | 16.0 a | 0.56% | 14.88% | 84.57% |
| real 560-cell grid | 3–12 (median 6) | 4.2e-13 | 6.8 a | 2.10% | 12.08% | 85.82% |

A real grid is not one repeated section: 420 of its 560 cells are regular
hexagons and the wall cuts the boundary ring into polygons of **three to twelve**
corners, down to slivers of 2.0 mm circumradius against an interior cell's 60.7
mm. 120 of the 560 carry enough skew to take the wide far seam, which is why its
mid band holds 12.1% of pairs where an idealised hexagon holds 2.6% — and the mid
rule is 268 µs/pair against the far filament's 0.7, so that fraction is most of
the column's cost. **A banded rate projected from the idealised hexagon is
optimistic for a real grid by about a factor of four** (13.6 µs/pair projected,
61.1 measured on the build).

Each band's treatment, on exactly the pairs the scheme routes to it
(hexagon / wall-clipped, of 2,339):

| band | pairs | µs/pair | seconds of the column |
|---|---|---|---|
| near, 16×48 quadrature | 13 / 13 | 984.2 / 1109.3 | 0.0128 / 0.0144 |
| near, CLOSED FORM | 13 / 13 | 4023.4 / 4664.3 | 0.0523 / 0.0606 |
| mid, 8×24 quadrature | 60 / 348 | 268.4 / 293.0 | 0.0161 / 0.1020 |
| far, moment filament | 2266 / 1978 | 0.7 / 1.8 | 0.0017 / 0.0036 |

**The closed form costs 4.1× MORE than the quadrature on the near band** — not
because of where the pairs are (cost is flat in distance for both) but because
there are only thirteen of them, an order below the 64-pair width at which the
two kernels cross. It is the right exact kernel for a lane that hands it a whole
column and the wrong one for a band that hands it thirteen pairs; served closed,
the whole column goes 13.6 → 31.5 µs/pair (hexagon) and 52.9 → 72.6
(wall-clipped). Serving the near band closed would need the near pairs of many
source columns batched into one call, which the per-column dispatch does not do.

## FrameSpace operator overhead by geometry and batch width

Recorded 2026-07-29 on one `sun_debug` node. Driver:
`benchmarks/frame_operator_overhead.py`; one source, geometry and clouds built
outside the timer, one warm call followed by the median of three. This reproduces
the protocol behind the earlier “frame assembly” number and makes its meaning
explicit: it times `Source`/`Target`, local/global transforms, xarray allocation,
solve composition and operator wrapping. It does **not** time path or section
construction.

| pairs | straight direct µs/pair | straight frame µs/pair | ratio | arc direct µs/pair | arc frame µs/pair | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 286.10 | 23517.30 | 82.20 | 133034.87 | 153616.83 | 1.15 |
| 8 | 39.27 | 2884.07 | 73.45 | 16515.51 | 19305.62 | 1.17 |
| 64 | 6.09 | 372.99 | 61.25 | 2260.10 | 2621.04 | 1.16 |
| 512 | 1.66 | 55.77 | 33.70 | 527.09 | 580.56 | 1.10 |
| 4096 | 1.95 | 50.47 | 25.86 | 404.40 | 413.69 | 1.02 |

The remembered “about 10× geometry assembly” was therefore two conflated
facts. The archived 512-pair prism result was about **31×**, and the current
matched run is **33.7×**; a warm profile of that solve attributes 23 ms to
FrameSpace metadata, 12 ms to xarray initialization, 5 ms to operator wrapping,
4 ms to matrix transforms and about 1 ms to the polygon kernel. But this is a
cheap straight kernel. On the finite arc, where arithmetic is actually
expensive, the same frame adds 10% at 512 pairs and 2% at 4096. The fix for
dense operator construction is a packed array boundary that bypasses xarray,
not a special geometry cache inside the interactive frame API.

## Finite-arc geometry bands and equivalent-filament placement

Recorded 2026-07-29 on the same node. Drivers:
`benchmarks/arc_operator_dispatch.py placement` and `wall --targets 4096`.
The band coordinate is distance to the finite swept section: poloidal contour
distance inside the angular span, and the hypotenuse of poloidal gap plus the
nearest-end chord outside it. The far seam is 32 bounding radii for the hexagon
and skewed trapezium; the elongated acceptance plate widens its own seam to
52.256 from section aspect.

The far model carries first, second and third area moments about its filament.
Across all three sections, poloidal and both off-end rays:

| placement | worst relative error beyond its seam |
|---|---:|
| area centroid + moments | 4.543e-08 |
| RMS radius + moments | 4.539e-08 |
| bare RMS radius | 1.804e-04 |

RMS plus moments is numerically indistinguishable from the centroid while adding
a non-zero first moment; bare RMS misses by four orders more. The centroid stays.
On the 4096-target three-dimensional column, 2225 pairs (54.32%) remain exact and
1871 take the filament: exact-everywhere 2.594 s, banded 1.378 s,
**1.883× faster**, worst row-scaled deviation 5.805e-08.

The exact thin-plate arc ceases to be a credible far-field reference at extreme
standoff because its large cancelling terms exhaust the row; the acceptance
sweep stops at 80 section radii, beyond every selected seam, rather than
mistaking reference cancellation for filament error.

## GPU geometry dispatch and pair-block sharding

Recorded 2026-07-29 on `98dci4-gpu-0003` (H200 NVL). Drivers:
`benchmarks/arc_operator_dispatch.py gpu` and
`benchmarks/gpu_pair_sharding.py`.

The finite-arc packed driver is traceable and agrees with its shortcut host
driver in NumPy and eager JAX, but XLA is the wrong implementation route for this
reduction: a **one-pair** graph took 695.94 s to compile and 18.82 ms/pair warm.
Removing broken-edge residual quadrature for a closed edge chain was immaterial
(720.74 s and 18.10 ms/pair), so that specialization was rejected. At a
4096-wide host call the exact arc is 404.4 µs/pair: even warm, the one-pair GPU
shape loses about 45×. The production optimization is the geometry band above,
not GPU compilation of the full five-row reduction.

The axisymmetric polygon ring is the opposite case: its fixed quadrature has
enough uniform work to shard. Geometry is replicated and pair blocks are divided
evenly with `pmap`.

| ring build, 320×320 | cold wall | warm wall | warm µs/pair |
|---|---:|---:|---:|
| 1 H200, 40×40 tiles | 2.302 s | 0.794 s | 7.755 |
| 4 H200, 40×40 tiles | 4.936 s | 0.518 s | 5.062 |
| 1 H200, 80×80 tiles (prior baseline) | — | — | 2.830 |
| 4 H200, 80×80 tiles | 3.425 s | 0.189 s | 1.850 |

Four devices give only **1.53×** steady-state speedup because one H200 already
fills a mapped tile; they also double cold dispatch at the smaller tile. A
complete eight-device run was unavailable because unrelated long-running services
occupied three cards, so no eight-device estimate is retained. The operational
rule is therefore:
use one large-tile H200 by default; shard only an already-large ring build whose
remaining wall justifies the extra devices. Straight prisms, filament arcs,
rectangular bows and cylinders remain host routes until they acquire a
fixed-shape packed driver and a measured crossover.
