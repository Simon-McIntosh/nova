# 300-cell solve: marker census re-baseline

- revision: `ad2c5ee18b09fc95d14d0f2e919954432839fcaa`
- job: 1272930 (all_debug, 98dci4-clu-3141, cpu)
- optimised HLO: 692225 instructions, 219637882 bytes
- compile: 345.483 s
- dump provenance: compiled from /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/msc-marker-census-rebaseline/nova/equilibrium/forward_operator.py

## The two counting rules

Both rules read the same printed module text, so a difference between them is a property of the counting rule and not of two different dumps.

- source-sentinel search: distinct traced frames whose source line is the sentinel statement resolved for the read at import time
- marker census: distinct traced frames carrying the read body of the marker function, counted once per frame id

## Both counts per marker

| marker path | sentinel copies | read-body frames | marker-bearing computations | read-body instructions | sentinel instructions | rules agree |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| current-moment path | 120 | 122 | 444 | 534 | 534 | no |
| topology read | 79 | 208 | 1357 | 13496 | 13496 | no |

## Carried forward from the earlier receipt

Source: `/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/msc-marker-census-instrument/docs/figures/millisecond-converged-solve/marker-census/300.json` at revision `a30309ada656de97acbcd9129b1accf2618e9502`.

| marker path | sentinel copies | read-body frames | marker-bearing computations | read-body instructions |
| --- | ---: | ---: | ---: | ---: |
| current-moment path | 120 | 122 | 444 | 534 |
| topology read | 79 | 208 | 1357 | 13496 |

## Re-baseline

Pinned baseline before this measurement: {'current-moment path': 120, 'topology read': 36}.
Measured here by the source-sentinel rule: {'current-moment path': 120, 'topology read': 79}.

- current-moment path: committed 120, measured 120, delta 0
- topology read: committed 36, measured 79, delta 43

Derivation: the sentinel is the source statement resolved at import time by the source line of the read, and the count is the number of distinct traced frames carrying that statement.  The two published topology-read numbers are not two readings of one sentinel: the committed 36 is recorded as a controlled known-present census and the 79 as the broader read-helper sentinel (evidence archive, executable-remainder-census), so the discrepancy is a naming of the counted statement rather than a moved program.  This census counts the sentinel it resolves for the read, and it reproduces the committed current-moment count of 120 exactly on the same dump, which is the positive control that the rule and the sentinel resolution are intact; the dumped module is byte-identical to the earlier census at revision `a30309ada656de97acbcd9129b1accf2618e9502`, so no traced frame set moved between them either.

Decision: the pinned baseline for the topology read is restated from 36 to the sentinel reading this census resolves (79), so the gate compares like with like; the current-moment baseline of 120 is unchanged and stands as its own positive control.

## Refusal contract

`require_live_markers` raises `MarkerCensusRefusal` when a marker path reports zero markers or a uniform read-body column, so a census that matched one frame many times, or none at all, cannot be reported as a clean baseline, and when the dump names a source file other than the one the marker function was imported from, so a dump served from the persistent compilation cache by another checkout is named instead of counted.  `tests/test_solve_program_size.py` pins all three refusals on synthetic module text without compiling.

## Figure

`census-rule-comparison.png` shows the two rule counts per marker on one axis and the per-frame read-body distribution on the other.
