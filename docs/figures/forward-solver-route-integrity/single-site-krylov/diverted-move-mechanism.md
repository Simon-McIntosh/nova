# The 0.216 move on the diverted rung: its mechanism

The diverted certificate rung (`diverted-single-null`, 300 cells) read a terminal
fixed-point residual of `0.279485232201572` under the single-site slice, and
`0.06325730373091486` under the exit loop. The difference is 0.216.
**The move is not caused by the Krylov stream.** It comes from the operator
modules merged into the single-site slice at `e62c8110e`. Under either Krylov
body, the two operator trees terminate 0.216 apart. Under either operator tree,
the two Krylov bodies terminate at most 3.07e-12 apart.

Receipt: `diverted-move-mechanism.json`. Figure: `diverted-move-trips.png` / `.svg`.
Runs: CPU `all_debug` jobs 1277103 (four processes: capacity scan and exit loop,
each plain and instrumented) and 1277109 (the two pre-merge arms). Driver:
`diverted_trace.py`, which runs `benchmarks.solovev_certificate._measure`, the
same call the reduced-Newton test makes. No solver code was changed in this step.

## The four runs

| Krylov body | Operator modules | Terminal residual |
|---|---|---|
| capacity scan (`fixed_point.py` at `e62c8110e`) | this tree (merged main) | `0.06325730373273203` |
| exit loop (`63a32bb4a`) | this tree (merged main) | `0.06325730373091486` |
| capacity scan (`8d02dd0f`, where slice 1 was measured) | `8d02dd0f` (pre-merge) | `0.279485232201572` |
| exit loop (`63a32bb4a`) | `8d02dd0f` (pre-merge) | `0.27948523220464483` |

The pre-merge scan arm reproduces the recorded slice-1 reading bitwise, so the
shared environment and the import hook are faithful. The ten modules served from
`8d02dd0f` are `nova/biot/null.py`, `nova/equilibrium/{clip_quadrature, fixed_point,
flux_surface_extraction, forward, forward_operator, separatrix_clip, topology}.py`,
`nova/geometry/section.py` and `nova/linalg/interpolant.py`. These are every `nova`
file that differs from this tree. The benchmark, the oracle fixture scripts and the
test conftest are identical between the two revisions. The pre-merge exit arm reads
`0.27948523220464483`, bitwise the value the pre-slice base read at `f803b640`.

- Krylov body, merged operator: 1.817e-12.
- Krylov body, pre-merge operator: 3.073e-12, the slice-1 re-base bound.
- Operator merge: 0.216 under either Krylov body.

## Per-trip live relative residual, merged operator

| Trip | Capacity scan | Exit loop | Absolute difference |
|---|---|---|---|
| 1 | 0.7373759191833157 | 0.7373759191847825 | 1.47e-12 |
| 2 | 0.0633636401761647 | 0.06336364017433883 | 1.83e-12 |
| 3 | 0.06325731112049199 | 0.06325731111867483 | 1.82e-12 |
| 4 | 0.0632573073074543 | 0.06325730730563711 | 1.82e-12 |
| 5 | 0.06325730540093524 | 0.06325730539911846 | 1.82e-12 |
| 6 | 0.06325730444767613 | 0.06325730444585953 | 1.82e-12 |
| 7 | 0.06325730397104673 | 0.06325730396922995 | 1.82e-12 |
| 8 | 0.06325730373273203 | 0.06325730373091486 | 1.82e-12 |
| 9 | 0.06325730373273203 | 0.06325730373091486 | 1.82e-12 |

## Per-trip live relative residual, pre-merge operator

| Trip | Capacity scan | Exit loop | Absolute difference |
|---|---|---|---|
| 1 | 0.2794919679426085 | 0.27949196794568415 | 3.08e-12 |
| 2 | 0.27948525564877524 | 0.27948525565184834 | 3.07e-12 |
| 3 | 0.27948524225037397 | 0.27948524225344734 | 3.07e-12 |
| 4 | 0.2794852355511726 | 0.2794852355542457 | 3.07e-12 |
| 5 | 0.279485232201572 | 0.27948523220464483 | 3.07e-12 |
| 6 | 0.279485232201572 | 0.27948523220464483 | 3.07e-12 |

## Where the two Krylov bodies first differ (merged operator)

- **First differing trip:** trip 1, with live residuals `0.7373759191833157` and
  `0.7373759191847825`. The mask difference (18 cells) and the damping flag are equal.
- **Inner iterations:** the receipt keeps ten globalisation records, and all of
  them sit at the terminal residual (0.0633). So they belong to the final trip and
  cannot place the first difference in time. The instrumented event sequence places
  it, below.
- **First differing operation:** the very first operator application of the first
  Krylov call (event 0). Its input is bitwise identical (norm 12.5506).
  Its output differs in 1148 of 1547
  elements, by at most 3.33e-16 absolute (3.0e-16
  relative to the largest element). This is the operator's own arithmetic rounding
  differently. Its body is compiled into a `cond` branch inside a `scan` in one program,
  and into a `while` body in the other, so XLA fuses and orders its reductions differently.
- **Semantics are unchanged:**
  - Both runs make 41 Krylov calls.
  - Every call applies the operator 64 times in both runs; the per-call application
    counts are equal element for element.
  - Every call's qualification code is equal.
  - The trip counts (9) and the mask differences are equal.
- **Growth:**
  - The first call's step differs by 2.56e-11
    (4.8e-11 relative). GMRES solves each call to an achieved
    reduction of about 1e-13, so the operator's rounding passes directly into the step.
  - The largest per-call step difference over the run is 3.87e-10.
  - The residual-vector difference entering later calls stays near 3e-12.
  - The trajectories do not separate: both converge to the same stagnation value
    and end 1.82e-12 apart.

## Classification

**The Krylov-body difference is floating-point reassociation, and it stays
bounded.** It enters at the operator's first application at the 3e-16 level,
with identical inputs, identical application counts, identical restart and exit
behaviour and an identical carry. It stays near 1e-12 in the per-trip residual on
both operator trees. It is not amplified into the 0.216.

**The 0.216 is a change of operator, not of solver.** The operator modules merged
from main at `e62c8110e` define a different forward map for this rung. Under
that map the non-converged rung stagnates at 0.0633 instead of 0.2795, from the
first trip onward: trip 1 reads 0.737 against 0.279. The slice-1 record compared
its stream against a base on the pre-merge operator, and the exit-loop record
compared against that slice-1 reading. The exit-loop gate run was the first to
measure this rung on the merged operator. So it attributed the operator's move
to the Krylov change.

Not instrumented: the per-application trace of the pre-merge pair. Its
difference (3.07e-12, from trip 1) has the same size as slice 1's recorded move
and is presumed to share the mechanism. The mechanism by which the merged
operator changes this rung's stagnation point is outside this step.
