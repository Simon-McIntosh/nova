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
| polygon hex CLOSED FORM, 128 residual nodes, by corner | 176.6 | 420 |
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
