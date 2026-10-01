# Capacity refusal: receipts

Node `ecq-capacity-refusal-must-report`. Worktree base `cdc915ba`. Every
measurement below ran on `all_debug` with `JAX_PLATFORMS=cpu`, the repository's
root interpreter invoked directly with `PYTHONPATH` naming the measured tree,
no `uv` on the compute node. Job `1273409`, host `98dci4-clu-3141`.

## The refusal is counted, and fails closed

`nova/equilibrium/separatrix_clip.py` packs any clipped cell whose live vertex
count passes `traced_polygon_vertex_capacity` as an empty inclusion. Before this
change the count was not carried anywhere: a refused cell's geometry was zeroed
and its `included` flag cleared, so on the returned support it was
indistinguishable from a cell the clip legitimately placed outside the plasma
and left the support in silence.

`TracedClippedSupports` now carries `vertex_capacity` and
`refused_cell_count`, and `assert_no_refusal()` is the fail-closed path: a
caller that must not lose a cell gets `TracedCapacityRefusalError` naming the
count and the capacity instead of a zeroed row.

`tests/test_equilibrium_separatrix_clip.py` presents a two-arc layout (a saddle
level function on one rectangular cell), which realises 260 live vertices
against a 136-slot fixture capacity, and asserts that the refusal is
**counted** (one cell) and **raised**. The assertions that the cell carries
included 0, vertex count 0 and area 0.0 are kept beside them, because the point
of the change is that the zero alone was the silent case.

The 260 is measured, not read off the entry that refused. The fixture's saddle
level function leaves the level set on two runs, so the clip hands
`_pack_traced_vertices` an array named `_pack_traced_vertices.vertices` of full
shape `(1, 1024, 2)` — 8 chord slots × 128 arc samples — carrying **260** live
entries on its vertex axis. Raising the derived capacity to 4096 for one
diagnostic run admits the same cell with `vertex_count` 260 and area
0.7499999999999982; the head run refuses it with `refused_cell_count` 1,
`vertex_count` 0 and area 0.0. Measured by
`docs/figures/exact-clip-moment-quadrature/capacity/measure_live_vertices.py`
on the login node (`JAX_PLATFORMS=cpu`, root interpreter, exit 0); the full
record of every packer call is `live-vertex-fixture.json`. The earlier 258 was
read from the cell's zeroed `vertex_count`, which cannot exceed the bound it
failed to meet.

Module gate: `tests/test_equilibrium_separatrix_clip.py` 25 passed in 62.12 s,
one fresh `all_debug` process (`reports/nova/s19-local/exact-capacity-refusal/test-module.log`).

## The realised layout at 110 cells, four analytic rows

`realised_layout_census.py` in this directory builds each analytic case at 110
requested cells through `benchmarks.exact_clip_moment_floor._build` and reports
the largest live vertex count the clip realises. One job, rows visited
sequentially inside the allocation, receipt per row written as it lands
(`census-<case>.json`, `censused-rows.json`).

| case | realised cells | included | straight capacity | derived capacity | realised max live | refused cells |
|---|---|---|---|---|---|---|
| weak-rotation-reactor-static | 135 | 121 | 30 | 158 | 134 | 0 |
| moderate-rotation-conventional-static | 136 | 121 | 32 | 160 | 134 | 0 |
| strong-rotation-compact-static | 135 | 121 | 30 | 158 | 134 | 0 |
| diverted-single-null | 132 | 85 | 34 | 162 | 134 | **1** |

The three rotating rows realise the arc once and a short straight chain, with
24 to 28 vertices of headroom, and confirm the one-arc premise the derived
capacity rests on. The weak row reproduces the capacity-sizing receipt's
measured maximum of 134 exactly.

## The single-null row refuses one cell — the silent drop was live

`diverted-single-null` at 110 cells reports **one refused cell**: a production
analytic row does realise a layout wider than one arc plus its straight chain,
and before this change that cell was dropped from the support with no error.
This is the case the followup predicted and it is not hypothetical.

The cell's own live vertex count is now measured. Building the row twice in one
process — once at the derived bound, once with the capacity raised to 4096 for
the diagnostic run — the refused cell is **cell 9** (centroid
`(1.4180122291353958, -1.022978034837412)`, cell bounds
`[1.32404975578631, -1.1043519237601387, 1.5119747024844816, -0.9416041459146856]`):

| quantity | value |
|---|---|
| derived bound for this row | 162 = 128 arc samples + 34 straight slots |
| base run | refused cell count 1, cell 9 included 0, vertex count 0, area 0.0 |
| capacity-raised run | refused cell count 0, cell 9 admitted, vertex count **260**, area 0.007077604229482379 |
| 260 decomposed by position | **256 interior vertices** (level-arc samples) + **4 on the cell boundary** (straight chain), boundary indices `[0, 128, 129, 259]` |

So 260 = 2 × 128 + 4: the cell realises **two** traced level runs, each entering
as the fixed 128 arc samples, joined by a 4-vertex straight chain. The derived
bound is not merely tight by a few vertices — it is a bound for a layout with
**one** traced arc, and this cell's layout is not that one. The row's one-arc
premise, measured only on the three rotating rows, fails on the diverted
single-null row.

Deciding what to do about it (widen the bound to a two-arc layout, or refuse the
row's cells by design) is a change to the derived bound and belongs to the
coordinator, not to this record repair. What is settled here is the measurement
the decision needs: the refused cell carries 260 live vertices against 162, its
extra 98 vertices are a whole second arc plus the chain's final edges, and its
area is 0.007077604229482379.

Figure: `diverted-single-null-refused-cell.png` — the row's 132 atomic cells
with cell 9 outlined, beside cell 9's own admitted polygon at the raised
capacity, its 256 arc samples and 4 boundary vertices drawn in their own styles.
Driver: `measure_refused_cell.py`, receipt `refused-cell-diverted-single-null.json`,
both on the login node (`JAX_PLATFORMS=cpu`, root interpreter, exit 0).

Two consequences, both for the coordinator rather than this node:

- The row's *maximal live vertex count* cannot be read from the `included`
  column. A refused cell has its `vertex_count` zeroed before the maximum is
  taken, so the 134 above is the maximum over the cells that survived, and the
  row's true maximum is 260, measured above.
- Whether the single-null row *should* refuse that cell is a question about the
  derived bound, not about the refusal. The measurement answers it: the cell
  realises a second traced run, so its layout is not the one the bound was
  derived for.

## The largest-intermediate census on the 300-cell row

The capacity-sizing receipt's memory table was compiled with `--skip-hlo` on
both sides. That flag selects `_compile_memory_only`, which never serializes the
optimized HLO, so `largest_array_intermediates` came back **empty on both
sides** — a skipped measurement, not a measured absence. Dropping the flag takes
the `certificate._compile_solve_memory` path, which reads the optimized HLO text
and ranks its arrays. Both arms ran in **one** `all_debug` allocation (job
`1273437`, `JAX_PLATFORMS=cpu`, root interpreter invoked directly, `TMPDIR=/tmp`,
no `uv` on the compute node). Each tree ran a 20-cell control arm first, whose
ranked census was read as a positive control that the census path yields arrays
at all rather than an empty list (16 entries on both trees), and then its
300-cell arm. Both 300-cell arms exited 0 and the 59-minute job reports
`completed: true` with `LANE_DONE`.

The sizing node's two trees had been reclaimed, so each arm measured a tree
re-materialised from the object store into the reports fence with its own
`gitdir` into the shared object store; each receipt reports the revision it
actually measured (`census-hlo-<arm>.json` → `source_revision`), which is the
per-arm provenance the identity receipts lack: `b9ce8d62` for base,
`f7fefbd7` for current.

| arm | revision | realised cells | compile wall (s) | peak temporary | largest array of the optimized HLO | census entries | predicate entries |
|---|---|---|---|---|---|---|---|
| base | `b9ce8d62` | 342 | 870.2 | 3,394,009,232 B (3.1609 GiB) | `f64[24,12,342,129]` 101,647,872 B | 16 | 0 |
| current | `f7fefbd7` | 342 | 892.3 | 3,011,689,360 B (2.8049 GiB) | same shape, same 101,647,872 B | 16 | 0 |

Four findings.

**The instrument reproduces the superseded measurement to the byte.** Base
`3,394,009,232 B` and current `3,011,689,360 B` are the capacity-sizing
receipt's two 300-cell peak temporaries exactly, from a different job on a
different day, so the two trees measured here are the pair the memory table
compared and the −11.3 percent between them is not a lane artefact.

**The largest single intermediate does not shrink.** Both arms present the same
top-16 by size: ten arrays of `f64[24,12,342,129]` at 101,647,872 B (96.91 MiB),
then six of 59,294,592 B in permutations of `[24,7,44118]`, `[24,7,342,129]` and
`[24,342,129,7]`. So the narrower capacity is not a narrower largest array, and
the peak reduction has to be read as a change in how many large temporaries the
program admits rather than how big the biggest one is. Ten of the 96.91 MiB
arrays are 0.95 GiB of the 3.16 GiB peak on their own, so no single array is the
peak either.

**Where the count does move is the cell-major arc-mask predicate.** Counting the
full census (`<arm>-array-census.jsonl.gz`, ~237,000–240,000 instruction records
per arm) at every shape of 1 MiB or more: the predicate array of shape
`(342, 3072, 2)` appears 1136 times on base against 212 on current, and
`(342, 3072)` 618 against 348, while `(24, 342, 129)` is 636 on both sides and
`(12, 342, 129, 2)` is 330 on both. The realised cell axis is 342 and 3072 is
24 × 128, so these are the per-cell support masks over the arc samples; the
narrower capacity admits far fewer of them while leaving the largest f64
intermediates untouched.

**The cell-squared pairwise construction is absent, now measured.** No record in
either arm's census places the realised cell count in two shape axes (zero
records carry 342 twice), and neither arm's top 16 holds a predicate array at
all (`largest_predicate_intermediates: []`). The all-cell pairwise construction
the memory section's predicate-signature ranking exists to catch is therefore
not present at 300 cells on either tree, and that is now a statement about the
optimized HLO rather than about a field the capture flag left empty.

Two cautions on reading this table. Census records are HLO instruction records,
not live buffers: a shape's record count is evidence about the program's
structure, and the peak temporary is the measure of memory. And the current
side's optimized HLO text is 443,432,901 B against 442,572,016 B (+0.19 %) over
239,993 records against 236,771 (+1.4 %), so the narrower capacity compiles
slightly *more* HLO text while using clearly less temporary.

## Artifacts

The 110-cell layout census is `realised_layout_census.py`, `census-<case>.json`
(four) and `censused-rows.json` in this directory, built by
`reports/nova/s19-local/exact-capacity-refusal/{run.sh,slurm-1273409.log,test-module.log,census.log}`.

The 300-cell census of the section above lives in
`reports/nova/s19-local/exact-capacity-refusal/`: `hlo_census_lane.sh`,
`census-slurm-1273437.log`, `census-hlo-{base,current}.json`, `census-hlo-<arm>.log`,
`census-smoke-{base,current}.json` with their logs, and per-arm compiler
artifacts `census-hlo-artifacts-<arm>/exact-300-{optimised.hlo.txt,array-census.jsonl}.gz`
(~30 MB and ~2.6 MB each). The two measured trees are `tree-base` and
`tree-current`, each with a `gitdir-<arm>` holding only `HEAD` (the measured
revision), `objects/info/alternates` (the shared object store) and a minimal
config, so each arm's `rev-parse HEAD` answers with the revision that arm
measured rather than with the worktree's.