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
level function on one rectangular cell), which realises 258 live vertices
against a 136-slot fixture capacity, and asserts that the refusal is
**counted** (one cell) and **raised**. The assertions that the cell carries
included 0, vertex count 0 and area 0.0 are kept beside them, because the point
of the change is that the zero alone was the silent case.

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

Two consequences, both for the coordinator rather than this node:

- The row's *maximal live vertex count* cannot be read from the `included`
  column. A refused cell has its `vertex_count` zeroed before the maximum is
  taken, so the 134 above is the maximum over the cells that survived, and the
  row's true maximum is at least the capacity `162`. The instrument in this
  directory cannot see above the bound; measuring the refused cell's own live
  count needs the capacity raised for one diagnostic run.
- Whether the single-null row *should* refuse that cell is a question about the
  derived bound, not about the refusal: either a clipped polygon there really
  carries a second traced run and the capacity is too tight, or its live count
  is inflated by a layout the bound was not derived for. Both are decidable
  from the refused cell's polygon and neither is settled here.

## Artifacts

`realised_layout_census.py`, `census-<case>.json` (four), `censused-rows.json`.
Logs: `reports/nova/s19-local/exact-capacity-refusal/{run.sh,slurm-1273409.log,test-module.log,census.log}`.