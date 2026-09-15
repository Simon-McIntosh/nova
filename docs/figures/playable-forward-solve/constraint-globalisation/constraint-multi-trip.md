# Multi-cap constrained centroid command

Current-base reproduction: the earlier large command still used 5 nonzero capped applications and 9 recorded trips, ending at 4640.72928 A. Its recorded termination reason was `converged`.

The replacement command is not a linear extrapolation. Its target is the centroid reached by the directly solved +3000 A free response; the +4000 A response is measured beside it as an instrument and nonlinearity check.

## Direct free responses

| current change [A] | converged | trips | centroid Z [m] | displacement [m] |
|---:|---|---:|---:|---:|
| 3000 | yes | 3 | 0.00531690879 | 0.0356919134 |
| 4000 | yes | 4 | 0.0142370688 | 0.0446120734 |

## Constrained trip receipt

| trip | applied current [A] | cumulative current [A] | merit | centroid Z [m] | error [mm] | recorded reason |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1000 | 1000 | 0.317274525 | 0.00453255024 | -0.784358549 | active_set_iteration_budget_exhausted |
| 2 | 1000 | 2000 | 0.314347464 | 0.00493755391 | -0.379354877 | active_set_iteration_budget_exhausted |
| 3 | 1000 | 3000 | 0.311252746 | 0.00531690858 | -2.09738026e-07 | active_set_iteration_budget_exhausted |
| 4 | 0 | 3000 | 0.3112526 | 0.00531690838 | -4.08488846e-07 | active_set_iteration_budget_exhausted |
| 5 | 0 | 3000 | 0.311252748 | 0.0053169082 | -5.92315953e-07 | converged |

## Verdict

Overall: **PASS**. The iteration recorded 5 trips and 3 nonzero capped applications. Its final centroid error was -5.92315953e-07 mm and the recorded termination reason was `converged`.

- Converged to within 1 mm: yes
- Raw merit decreased monotonically over every recorded boundary (retained, non-gating): no
- Merit strictly decreased over every trip applying current: yes
- Raw merit first rose on closure trip 5 by 1.48093881e-07; that trip applied 0 A and took 0 Newton steps.
- Three to four capped applications: yes
- No application exceeded 1000 A: yes
- No accepted current step overshot the target: yes
- Total current within 10 percent of the directly measured free value: yes (3000 A against 3000 A; 0 percent).

The termination reason above is reported exactly as returned. Any termination-order change belongs to the separately owned solver node.

Receipt: `/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-handoff-20260915/pfs-constraint-multi-trip-target/docs/figures/playable-forward-solve/constraint-globalisation/multi-trip-target.json`. Figure: `/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-handoff-20260915/pfs-constraint-multi-trip-target/docs/figures/playable-forward-solve/constraint-globalisation/multi-trip-target.svg`.
