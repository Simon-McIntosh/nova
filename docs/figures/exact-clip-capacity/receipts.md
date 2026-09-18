# Traced-polygon vertex capacity: receipts

Node `ecq-capacity-sizing`, run `r-20260917T100810663195`. Worktree head
`f7fefbd7` on base `b9ce8d62`. Every measurement below ran on `all_debug` with
`JAX_PLATFORMS=cpu`, the repository's root interpreter invoked directly with
`PYTHONPATH` naming the measured tree and no `uv` on the compute node.

## The capacity, and where the 3,840 came from

`traced_polygon_vertex_capacity(straight_vertex_capacity)` in
`nova/equilibrium/separatrix_clip.py` now returns the maximal live vertex count
of one traced clipped polygon as `_SPLINE_BOUNDARY_SEGMENTS +
straight_vertex_capacity`; the derivation is its docstring. A clipped polygon
is its traced level arc, which enters as the fixed 128 arc samples, joined to a
straight chain of cell-boundary edges drawn from the compact cell polygon that
carries at most `straight_vertex_capacity` vertices; the arc replaces one of
them, so the arc count plus the straight chain covers the layout.

The constant this replaces was produced in the same module, in `_traced_clip`'s
`curve_evaluator is not None` branch, as the product
`_SPLINE_BOUNDARY_SEGMENTS * chord_capacity` — 128 times the compact polygon's
maximal vertex count, i.e. 3,840 for the weak row's 30-vertex cells. That
product reserves the whole arc sample count for every straight slot, which a
layout spending one arc and a handful of straight edges never does. Every other
path keeps the pattern: the capacity is passed in, and `AtomicCellMesh`'s host
refusal (`clipped support exceeded its fixed mesh capacity`) is untouched.

## The realised layout

`layout-panel.json`, rendered by `render_layout_panel.py` in one `all_debug`
allocation (job `1273196`) from the same row build the identity lanes used
(`benchmarks.exact_clip_moment_floor._build`):

| requested cells | cut cells | included | realised max live vertices | derived capacity | superseded product |
|---|---|---|---|---|---|
| 110 | 135 | 121 | 134 | 158 (128 + 30) | 3,840 |
| 300 | 342 | 307 | 134 | 152 (128 + 24) | 3,072 |
| 1000 | 1072 | 954 | 134 | 150 (128 + 22) | 2,816 |

Every row realises the arc once and a 6-vertex straight chain, and no
production cell realises a second traced arc, so the one-arc derivation is the
realised layout on all three rows and the capacity sits above it (134 against
158 and 150) while the product it replaced sat 19 to 24 times above the layout.
`layout-panel.png` / `.svg` draw the 43 clipped supports of the weak 110 row,
with the four cells whose live count sets the realised maximum (87, 104, 108,
110) drawn heavy and labelled. The panel was rendered in the job named above;
the caption numbers are the receipt's, and the panel itself was not inspected
by eye by this worker.

## Bit identity against the pre-change tree

One `all_debug` allocation per row, each measuring one tree's snapshot and one
comparing the pair above. Moments `[3, cut cells]`, coupled current-moment
image `[state size]`. `identity-{110,300,1000}.json`:

| row | moments sha256 | image sha256 | max abs difference | passed |
|---|---|---|---|---|
| 110 | `0bfb489eb76aabdb486471f31515d748132c153120789dda4f8c5d4519f5b688` | `9fa2d10eb50c396f3000df4c9d01185707e5372e541bd5705b2f844083b38569` | 0.0 | true |
| 300 | `f5a920ecbb6d2079b6c469f946a489fc94e590712892f7e10943b9042736f748` | `bbed9f45861e683198a05291ce5d5c1c433c8ef2f770f50ba0183b43db7470a8` | 0.0 | true |
| 1000 | `8a4cc018bcb7bbe76ff76fc00b9e69550f4fedcdad38f8dda03cd1706e3de0e2` | `76952994eb103dd6667b9622f27527e465fb15fd74f7663c9410365482c50828` | 0.0 | true |

The included mask and the per-cell vertex count are equal bit-for-bit on all
three rows as well (`mismatch_count: 0` on every array), so the narrower
capacity moved no produced number and no cell's polygon.

### These identity receipts are compare-only reruns

`identity-{110,300,1000}.json` is **not** one job's before/after. It is a
comparison of two separately captured snapshots: `snap-<cells>-{base,current}.npz`
are each written by their own process with `PYTHONPATH` naming that revision's
tree (`benchmarks.exact_clip_identity.py`, module-level import selects the code
under measurement), and `compare-<cells>.log` is a third process that reads the
two snapshots and differs them last-bit. Three fresh processes per row, not one.

Two consequences of that shape, both visible in the receipt:

- The receipt's own `base_revision` and `current_revision` fields read the
  literal strings `"base"` and `"current"` — the role of each arm, not a
  revision. The revision each arm measured is not carried in the receipt and
  has to be taken from the run's tree, so a reader cannot check the receipt
  against a commit from the receipt alone.
- The per-row identity allocations that were meant to sit behind these
  receipts ended in a `TypeError` and produced no entry, so the `arrays` block
  holds only the compare-only result and no per-row allocation identity.

## Memory scaling before and after

`benchmarks/exact_clip_memory_scaling.py` unchanged, one `all_debug` allocation
`memory-{base,current}.json`, three rungs each, `--skip-hlo` on both sides (the
HLO text census is the multi-GiB export the script's own docstring separates
from the gate; the two sides are therefore symmetric).

| requested cells | realised | peak temporary before | after | argument / output bytes |
|---|---|---|---|---|
| 110 | 135 | 1.5705 GiB | 1.3672 GiB | 6176 / 6209 (unchanged) |
| 300 | 342 | 3.1609 GiB | 2.8049 GiB | 12232 / 12265 (unchanged) |
| 1000 | 1072 | 9.0065 GiB | 8.0192 GiB | 32168 / 32201 (unchanged) |

The 1000-cell peak the section asks for falls 9.0065 GiB to 8.0192 GiB
(-11.0 percent; 110 falls 12.9 percent, 300 11.3 percent), with the argument
and output sizes identical, so the whole delta is compiled-temporary from the
narrower traced support. The temporary's exponent against realised cell count moves
from 0.9165 to 0.9195 between the 300 and 1000 rungs and from 0.7525 to 0.7731
between the 110 and 300 rungs — the scaling itself is essentially unchanged by
the capacity. The instrument reports `executed: false` on both sides: it
compiles and reads `memory_analysis`, it never runs the program, so the peak
field is the compile-time peak temporary and no allocator peak exists on either
side. The `largest_array_intermediates` census is empty on both sides by that
same flag choice.

## Compile wall at 110 and 300 cells before and after

The section's named instrument for this row is `exact_clip_warm_cost.py
--worker production`, and it did not produce a receipt. Its four arms (base and
current, 110 and 300 cells, in the two allocations `1273143` and `1273144`)
exited 2 at argparse: its production worker requires `--cells`, `--output`,
`--cache-root` and `--dump-root`, and the lane script passed the first two
only. That defect is one line to fix, but the fix does not make the arm
runnable on this section's lane: the same worker raises in `_lane()` unless JAX
reports a GPU device whose kind contains `H200`, the partition is
`betelgeuse`, the reservation is `gpu_0003_grpA` and `SLURM_CPUS_PER_TASK` is
8, and the driver is outside this node's write scope. The compile wall is
therefore reported from the instrument that did run unchanged in the same two
allocations, whose rows time `jax.jit(solve_program).lower(seed).compile()` for
the same certificate solve at three cell counts:

| requested cells | before | after | delta |
|---|---|---|---|
| 110 | 848.188 s | 847.235 s | -0.11 percent |
| 300 | 965.073 s | 970.609 s | +0.57 percent |
| 1000 | 1210.914 s | 1212.967 s | +0.17 percent |

Both trees ran on `98dci4-clu-3141`, one allocation each, and the timer starts
before the certificate row build, so each figure is an upper bound on the
compile alone. The wall is flat within 0.6 percent at 110 and 300, as the
capacity is not the term that sets this compile.

## The refusal

`tests/test_equilibrium_separatrix_clip.py` runs green on `all_debug`
(`test-module-all_debug.log`: 25 passed in 63.37 s), including
`test_traced_polygon_capacity_is_the_arc_plus_the_straight_chain` and
`test_spline_clip_refuses_a_polygon_above_the_derived_capacity`, which presents
258 live vertices against a fixture capacity of 136 and observes included 0,
vertex count 0 and area 0.0.

## Artifacts

`identity-{110,300,1000}.json`, `memory-{base,current}.json`,
`layout-panel.{png,svg,json}`, `render_layout_panel.py`. Logs and the full
receipt set under
`/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/exact-capacity/`.