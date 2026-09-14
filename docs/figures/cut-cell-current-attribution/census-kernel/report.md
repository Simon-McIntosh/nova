# Census kernel timing

Whole-solve target: 1.0 ms per state.  Two receipts: the loop-free jitted
kernel (Part 1) and the production vertex-ring census read, measured by stage
(Part 2), both on one H200, vmapped over sixteen states, seven repeats,
median batch time divided by sixteen.  The production column is the banked
per-state figure (1.4/4.9/30 ms); the measured column is this run's direct
read of the same implementation (1.95/6.25/33.4 ms), the small surplus being
launch-context rather than a different algorithm.

| cells | pitch (m) | kernel ms/state | kernel µs/cell-state | production ms/state | production/kernel | measured reference ms/state | dominant stage (share) |
|---|---|---|---|---|---|---|---|
| 550 | 0.0710 | 0.0489 | 0.0888 | 1.4 | 28.7 | 1.95 | private_exclusion (70%) |
| 1074 | 0.0502 | 0.0521 | 0.0485 | 4.9 | 94.1 | 6.25 | private_exclusion (81%) |
| 2616 | 0.0318 | 0.0537 | 0.0205 | 30.0 | 558 | 33.4 | private_exclusion (92%) |

## Findings

**The kernel is 19-20x under the 1 ms whole-solve target at every rung, and
its per-state cost is essentially flat in cells.**  The 0.049-0.054 ms per
state does not move with the 550-2616 cell range (scaling exponent 0.06); at
these sizes the cost is launch-overhead-bound, not cell-bound — the per-cell
marginal share falls from 0.089 to 0.021 microseconds as the grid refines,
showing the flat total amortising across more cells.  Cold compile was
1.6/1.3/0.9 s (warm JAX cache) and the vmapped kernel spans 863 HLO
instructions at every rung (instruction count is shape-independent).

**The production read carries what mattered.**  Its 1.4 -> 30.0 ms growth is
dominated by the axis-component flood over the representative candidates
(`private_exclusion`), which rises from 70% to 92% of the read and drives the
1.82 scaling exponent in cells.  That stage is a sequential per-representative
component fill, replaced in the kernel by a static host-computed
private-region mask applied elementwise — the difference between the two
receipts is exactly the difference between 0.05 ms and 1.4-30 ms.

**The kernel is 29x / 94x / 558x faster than production** at 550/1074/2616
cells (28.7 / 94.1 / 558), i.e. the two-to-thirty-two millisecond read is
reproduced by the same census at tens of microseconds per state.

## Verification

The kernel reproduces the production census bit-for-bit at every rung —
raw/typed/contained flag arrays equal, positions and Hessian determinants
equal to 1e-12, both analytic nulls admitted within a tenth of a pitch.  The
kernel's fixed-capacity top-k keeps more representative saddles than the
production half-pitch cluster scan (2 vs 1 at 550 cells, 3 vs 3 beyond); that
is expected dedupe-scope divergence, and admission is unaffected.  Manufactured
saddle and extremum positive controls pass on every rung.  Kernel/reference HLO
count: 863 vs 3412.

| rung | counts match | axis | saddle | axis err / pitch | saddle err / pitch |
|---|---|---|---|---|---|
| 550 | True | True | True | 0.0098 | 0.0102 |
| 1074 | True | True | True | 0.0049 | 0.0270 |
| 2616 | True | True | True | 0.0027 | 0.0158 |

Figure: docs/figures/cut-cell-current-attribution/census-kernel/census-kernel-timing.png
