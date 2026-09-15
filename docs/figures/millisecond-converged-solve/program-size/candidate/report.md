# Program scope census

Census of the compiled whole-cell certificate solve (weak-rotation-reactor-static row) at 300 and 1000 requested cells, and of one application of the flux map alone at the analytic flux.

The source premise was tested rather than assumed. The production certificate budgets and the compiled reduced-slice budgets are already `jax.lax.fori_loop` bodies. The Python Newton and trip loops exist only on the inspectable host route and create zero copies in this optimised HLO. Replication is therefore attributed below to separately traced map and topology call paths, not to those host loops.

## Captured literal context

The literal census is taken from the optimised HLO, not from StableHLO or Python object sizes. XLA omits source metadata on some closure literals; those rows explicitly name the closure seam rather than claiming an unavailable innermost frame.

| cells | literals >1 KiB | captured bytes | executable context | generated-code context |
|---:|---:|---:|---:|---:|
| 300 | 602 | 364.59 MiB | 462.00 MiB | 450.56 MiB |
| 1000 | 605 | 2.91 GiB | not serialisable | 3.50 GiB |
| 2500 | not recompiled in this node | - | not serialisable | 19.00 GiB |

![Captured bytes by group](/nova/figures/millisecond-converged-solve/program-census/captured-bytes-by-group.svg)

### Captured bytes by group

| group | 300 literals | 300 bytes | 1000 literals | 1000 bytes |
|---|---:|---:|---:|---:|
| interaction-matrix kernel blocks | 270 | 359.06 MiB | 270 | 2.89 GiB |
| mesh connectivity | 70 | 680.54 KiB | 73 | 2.32 MiB |
| moment geometry | 31 | 3.51 MiB | 31 | 11.22 MiB |
| other captured literals | 135 | 602.53 KiB | 135 | 2.06 MiB |
| wall and sample blocks | 96 | 780.39 KiB | 96 | 2.43 MiB |

### Every optimised-HLO literal above 1 KiB

| cells | id | group | shape | dtype | bytes | innermost Nova frame or closure seam |
|---:|---|---|---|---|---:|---|
| 300 | `constant.17378` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17379` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17380` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51854..sunk.10` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51854..sunk2.11` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51855..sunk.10` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51855..sunk2.11` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51856..sunk.10` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51856..sunk2.11` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53810..sunk.10` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53810..sunk2.11` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53811..sunk.10` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53811..sunk2.11` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53812..sunk.10` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53812..sunk2.11` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61774` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61775` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61776` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61801` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61802` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61803` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62061` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62062` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62063` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64138` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64139` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64140` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64167` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64168` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64169` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71930` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71931` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71932` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72131` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72132` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72133` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72390` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72391` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72392` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72429` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72430` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72431` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72777` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72778` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72779` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73359` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73360` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73361` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.74988` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.74989` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.74990` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.75206` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75207` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75208` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75383` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75384` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75385` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75797` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75798` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75799` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77013` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77014` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77015` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77701` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77702` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77703` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79178` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79179` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79180` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80514` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80515` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80516` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85663` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85664` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85665` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86106` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86107` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86108` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86271` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86272` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86273` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86331` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86332` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86333` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86373` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86374` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86375` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9560` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9561` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9563` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17372` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17373` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17374` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51828..sunk.10` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51828..sunk2.11` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51844..sunk.10` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51844..sunk2.11` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51848..sunk.10` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51848..sunk2.11` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53788..sunk.10` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53788..sunk2.11` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53802..sunk.10` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53802..sunk2.11` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53805..sunk.10` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53805..sunk2.11` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61763` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61767` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61770` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61807` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61808` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61809` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62067` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62068` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62069` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64127` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64131` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64134` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64173` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64174` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64175` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71890` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71920` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71924` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72102` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72125` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72127` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72366` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72380` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72384` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72400` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72423` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72425` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72753` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72767` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72771` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73365` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73366` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73367` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.74959` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.74982` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.74984` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.75177` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75200` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75202` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75354` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75377` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75379` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75803` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75804` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75805` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77019` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77020` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77021` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77707` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77708` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77709` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79154` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79168` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79172` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80508` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80509` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80510` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85639` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85653` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85657` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86074` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86096` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86100` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86239` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86261` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86265` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86307` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86321` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86325` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86349` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86363` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86367` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9569` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9570` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9571` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17375` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17376` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17377` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51851..sunk.10` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51851..sunk2.11` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51852..sunk.10` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51852..sunk2.11` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51853..sunk.10` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51853..sunk2.11` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53807..sunk.10` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53807..sunk2.11` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53808..sunk.10` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53808..sunk2.11` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53809..sunk.10` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53809..sunk2.11` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61771` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61772` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61773` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61804` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61805` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61806` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62064` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62065` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62066` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64135` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64136` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64137` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64170` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64171` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64172` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71927` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71928` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71929` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72128` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72129` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72130` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72387` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72388` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72389` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72426` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72427` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72428` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72774` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72775` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72776` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73362` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73363` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73364` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.74985` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.74986` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.74987` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.75203` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75204` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75205` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75380` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75381` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75382` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75800` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75801` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75802` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77016` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77017` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77018` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77704` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77705` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77706` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79175` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79176` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79177` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80511` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80512` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80513` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85660` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85661` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85662` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86103` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86104` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86105` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86268` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86269` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86270` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86328` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86329` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86330` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86370` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86371` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86372` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9565` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9566` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9567` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.69079` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80499` | moment geometry | `f64[201,7,3,7]{3,2,1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.57680` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.80498` | mesh connectivity | `s32[201,7,3,7]{3,2,1,0}` | `s32` | 118,188 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17367` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51833..sunk.10` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51833..sunk2.11` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53793..sunk.10` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53793..sunk2.11` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61750` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61861` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62121` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64114` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64219` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71900` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72110` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72371` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72408` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72758` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73406` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.74967` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.75185` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75362` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.75845` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77037` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77686` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79157` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80502` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85642` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86079` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86244` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86310` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86354` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9630` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.58125` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.58127` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.72892` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.80497` | wall and sample blocks | `f64[201,7,2,2]{3,2,1,0}` | `f64` | 45,024 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17365` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80526` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17350` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.57780` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.80492` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80500` | mesh connectivity | `s64[342,7]{1,0}` | `s64` | 19,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17364` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80527` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.57803` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.80491` | mesh connectivity | `s64[201,7]{1,0}` | `s64` | 11,256 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.58862` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.60381` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61785` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61988` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62180` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64149` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64302` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71963` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72147` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72397` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72445` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72784` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72975` | mesh connectivity | `s32[342,7]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.73004` | mesh connectivity | `s32[342,7]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.73431` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75614` | mesh connectivity | `s32[342,7]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.75653` | mesh connectivity | `s32[342,7]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.75883` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75981` | mesh connectivity | `s32[342,7]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.77058` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79158` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85205` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85643` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86030` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86080` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.86195` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86245` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.86311` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86355` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86652` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86703` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.80507` | other captured literals | `f64[342,3]{1,0}` | `f64` | 8,208 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17354` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 300 | `constant.57449` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:774 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.80496` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17348` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17366` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:3559 `_active_set_newton_krylov.<locals>.reconcile.<locals>.<lambda>` |
| 300 | `constant.58144` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.58857..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.58860..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.58902..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.58905..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.58924..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.58927..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.58947..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.58950..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.58999..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.59002..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.59030..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.59033..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.60376..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.60379..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.61826` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61984..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.61986..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.62008..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.62010..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.62086` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62176..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.62178..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.62205..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.62207..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.64083..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.64085..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.64187` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64298..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.64300..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.69115` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.69116` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.69117` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.69233` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.69234` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.69235` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.69255` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.69256` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.69257` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.69307` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.69308` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.69309` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.69329` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.69330` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.70565` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.70816` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72903..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.72906..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.72933..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.72936..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.72968` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.72969` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.72974` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.72976` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.72997` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.72998` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.73003` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.73005` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.73379` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73449` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.73450` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.73451` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.75607` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.75608` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.75630` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.75631` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.75646` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.75647` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.75817` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75918` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.75919` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.75920` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.75974` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.75975` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.80490` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80504` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80528` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.87036` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.9585` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.57816` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.74792` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.17351` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17352` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80493` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80494` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17228` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17229` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17230` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.42701` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.43756` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51845..sunk.13` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51845..sunk2.16` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51846..sunk.10` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51846..sunk2.11` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51849..sunk.13` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51849..sunk2.16` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53803..sunk.13` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53803..sunk2.16` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53804..sunk.10` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53804..sunk2.11` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53806..sunk.13` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.53806..sunk2.16` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.58115` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 300 | `constant.61764` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61765` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61766` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61768` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61813` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61814` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61815` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61907` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61997` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61998` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62073` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62074` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62075` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62156` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62191` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62192` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64128` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64129` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64130` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64132` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64179` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64180` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64181` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64230` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64316` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64317` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.69245..sunk2.2..sunk.4` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.69246..sunk2.2..sunk.4` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.69247..sunk2.6` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.69248..sunk2.6` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.69258..sunk2.2..sunk.4` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.69259..sunk2.2..sunk.4` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.69260..sunk2.6` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.69261..sunk2.6` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71921` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71922` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71923` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71925` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72155` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72156` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72157` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72158` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72381` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72382` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72383` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72385` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72453` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72454` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72455` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72456` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72768` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72769` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72770` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72772` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72984` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.73013` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.73371` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73372` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73373` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73419` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75623` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75625` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75627` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75629` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75637` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75639` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75641` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75662` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75664` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75666` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75668` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.75809` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75810` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75811` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75863` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75985` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.77023` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77024` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77025` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77048` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77698` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77733` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77740` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77741` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79169` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79170` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79171` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79173` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80525` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85654` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85655` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85656` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85658` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86033` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86097` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.86098` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.86099` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.86101` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.86198` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86262` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.86263` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.86264` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.86266` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.86322` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86323` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86324` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86326` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86364` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86365` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86366` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86368` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86655` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86706` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9576` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9578` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9579` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17353` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.80495` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.17231` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.51832..sunk.10` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.51832..sunk2.11` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.53792..sunk.10` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.53792..sunk2.11` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.61752` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.61858` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.62118` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64116` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.64217` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.71893` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72146` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72370` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72444` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.72757` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.73404` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.75843` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77035` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.77687` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.79156` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.85641` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86116` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86281` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.86309` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.86353` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.9628` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17802` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17803` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17804` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52093..sunk.10` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52093..sunk2.11` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52094..sunk.10` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52094..sunk2.11` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52095..sunk.10` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52095..sunk2.11` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54098..sunk.10` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54098..sunk2.11` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54099..sunk.10` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54099..sunk2.11` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54100..sunk.10` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54100..sunk2.11` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62074` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62075` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62076` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62101` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62102` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62103` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62361` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62362` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62363` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64474` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64475` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64476` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64504` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64505` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64506` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72330` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72331` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72332` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72531` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72532` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72533` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72790` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72791` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72792` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72829` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72830` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72831` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.73177` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73178` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73179` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73764` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73765` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73766` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.75400` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75401` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75402` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75618` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75619` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75620` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75795` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75796` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75797` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.76210` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76211` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76212` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77426` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77427` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77428` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78127` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78128` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78129` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79613` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79614` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79615` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80953` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80954` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80955` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86102` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86103` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86104` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86545` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86546` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86547` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86710` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86711` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86712` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86770` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86771` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86772` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86812` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86813` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86814` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9737` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9738` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9740` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17796` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17797` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17798` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52067..sunk.10` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52067..sunk2.11` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52083..sunk.10` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52083..sunk2.11` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52087..sunk.10` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52087..sunk2.11` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54076..sunk.10` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54076..sunk2.11` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54090..sunk.10` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54090..sunk2.11` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54093..sunk.10` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54093..sunk2.11` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62063` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62067` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62070` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62107` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62108` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62109` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62367` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62368` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62369` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64463` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64467` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64470` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64510` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64511` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64512` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72290` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72320` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72324` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72502` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72525` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72527` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72766` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72780` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72784` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72800` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72823` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72825` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.73153` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73167` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73171` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73770` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73771` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73772` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.75371` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75394` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75396` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75589` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75612` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75614` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75766` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75789` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75791` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.76216` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76217` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76218` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77432` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77433` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77434` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78133` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78134` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78135` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79589` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79603` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79607` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80947` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80948` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80949` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86078` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86092` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86096` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86513` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86535` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86539` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86678` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86700` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86704` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86746` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86760` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86764` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86788` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86802` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86806` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9746` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9747` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9748` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17799` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17800` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17801` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52090..sunk.10` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52090..sunk2.11` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52091..sunk.10` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52091..sunk2.11` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52092..sunk.10` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52092..sunk2.11` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54095..sunk.10` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54095..sunk2.11` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54096..sunk.10` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54096..sunk2.11` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54097..sunk.10` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54097..sunk2.11` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62071` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62072` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62073` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62104` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62105` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62106` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62364` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62365` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62366` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64471` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64472` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64473` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64507` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64508` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64509` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72327` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72328` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72329` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72528` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72529` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72530` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72787` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72788` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72789` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72826` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72827` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72828` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.73174` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73175` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73176` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73767` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73768` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73769` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.75397` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75398` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75399` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75615` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75616` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75617` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75792` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75793` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75794` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.76213` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76214` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76215` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77429` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77430` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77431` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78130` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78131` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78132` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79610` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79611` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79612` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80950` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80951` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80952` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86099` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86100` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86101` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86542` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86543` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86544` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86707` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86708` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86709` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86767` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86768` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86769` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86809` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86810` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86811` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9742` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9743` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9744` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.69464` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80938` | moment geometry | `f64[815,7,3,7]{3,2,1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.57980` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.80937` | mesh connectivity | `s32[815,7,3,7]{3,2,1,0}` | `s32` | 479,220 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17791` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52072..sunk.10` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52072..sunk2.11` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54081..sunk.10` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54081..sunk2.11` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62050` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62161` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62421` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64450` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64555` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72299` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72510` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.72771` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72808` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.73158` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73811` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.75379` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.75597` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.75774` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.76258` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77450` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78112` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79592` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80941` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86081` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86518` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86683` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86749` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86793` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9809` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.58425` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.58427` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.73292` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.80936` | wall and sample blocks | `f64[815,7,2,2]{3,2,1,0}` | `f64` | 182,560 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17789` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80965` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17774` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.58080` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.80931` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80939` | mesh connectivity | `s64[1072,7]{1,0}` | `s64` | 60,032 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.58103` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.80930` | mesh connectivity | `s64[815,7]{1,0}` | `s64` | 45,640 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17788` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80966` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.59162` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.60681` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62085` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62288` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62480` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64485` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64638` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72363` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72547` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72797` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72845` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73184` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73375` | mesh connectivity | `s32[1072,7]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.73404` | mesh connectivity | `s32[1072,7]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.73836` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76026` | mesh connectivity | `s32[1072,7]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.76065` | mesh connectivity | `s32[1072,7]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.76296` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76394` | mesh connectivity | `s32[1072,7]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.77471` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79593` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.85644` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86082` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86469` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86519` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.86634` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86684` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.86750` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86794` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.87091` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.87142` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.80946` | other captured literals | `f64[1072,3]{1,0}` | `f64` | 25,728 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17778` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 1000 | `constant.57749` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:774 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.80935` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.58116` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.75204` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.17772` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17790` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:3559 `_active_set_newton_krylov.<locals>.reconcile.<locals>.<lambda>` |
| 1000 | `constant.58444` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.59157..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.59160..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.59202..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.59205..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.59224..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.59227..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.59247..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.59250..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.59299..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.59302..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.59330..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.59333..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.60676..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.60679..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.62128` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62284..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.62286..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.62308..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.62310..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.62388` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62476..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.62478..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.62505..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.62507..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.64419..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.64421..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.64524` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64634..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.64636..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.69500` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.69501` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.69502` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.69618` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.69619` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.69620` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.69640` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.69641` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.69642` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.69692` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.69693` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.69694` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.69714` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.69715` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.70957` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.71212` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.73303..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.73306..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.73333..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.73336..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.73368` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.73369` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.73374` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.73376` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.73397` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.73398` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.73403` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.73405` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.73784` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73854` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.73855` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.73856` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.76019` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.76020` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.76042` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.76043` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.76058` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.76059` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.76230` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76331` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.76332` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.76333` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.76387` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.76388` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.80929` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80943` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80967` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.87475` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.9762` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17775` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17776` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80932` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80933` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17649` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17650` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17651` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.42912` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.43967` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52084..sunk.13` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52084..sunk2.16` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52085..sunk.10` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52085..sunk2.11` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52088..sunk.13` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52088..sunk2.16` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54091..sunk.13` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54091..sunk2.16` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54092..sunk.10` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54092..sunk2.11` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54094..sunk.13` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.54094..sunk2.16` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.58415` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 1000 | `constant.62064` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62065` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62066` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62068` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62113` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62114` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62115` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62207` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62297` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62298` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62373` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62374` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62375` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62456` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62491` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62492` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64464` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64465` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64466` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64468` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64516` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64517` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64518` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64566` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64652` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64653` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.69630..sunk2.2..sunk.4` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.69631..sunk2.2..sunk.4` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.69632..sunk2.6` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.69633..sunk2.6` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.69643..sunk2.2..sunk.4` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.69644..sunk2.2..sunk.4` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.69645..sunk2.6` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.69646..sunk2.6` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72321` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72322` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72323` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72325` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72555` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72556` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72557` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72558` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72781` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72782` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72783` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72785` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72853` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72854` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72855` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72856` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73168` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73169` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73170` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73172` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73384` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.73413` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.73776` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73777` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73778` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73824` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76035` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76037` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76039` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76041` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76049` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76051` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76053` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76074` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76076` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76078` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76080` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.76222` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76223` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76224` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76276` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76398` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.77436` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77437` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77438` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77461` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78124` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78159` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78166` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78167` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79604` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79605` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79606` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79608` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80964` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86093` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86094` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86095` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86097` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86472` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86536` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.86537` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.86538` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.86540` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.86637` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86701` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2144 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.86702` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2145 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.86703` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2146 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.86705` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2143 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.86761` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86762` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86763` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86765` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86803` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86804` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86805` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86807` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.87094` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.87145` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9753` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9755` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9756` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.17652` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.52071..sunk.10` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.52071..sunk2.11` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.54080..sunk.10` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.54080..sunk2.11` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.62052` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62158` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.62418` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64452` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.64553` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72293` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72546` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72770` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.72844` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73157` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.73809` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.76256` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.77448` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.78113` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.79591` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86080` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86555` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86720` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.86748` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.86792` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.9807` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.57974` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.58105` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.58105..sunk.2` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.17777` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.80934` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |

## Replicated compiled paths

| cells | program | path | traced copies | instructions in path | source |
|---:|---|---|---:|---:|---|
| 300 | map | current-moment path | 1 | 4 | nova/equilibrium/forward_operator.py:2462 `ForwardFluxOperator.normalised_current_moments`; nova/equilibrium/forward_operator.py:2301 `ForwardFluxOperator._internal_on_partition` |
| 300 | map | topology read | 1 | 14 | nova/equilibrium/forward_operator.py:1885 `ForwardFluxOperator._fixed_design_read` |
| 300 | solve | current-moment path | 120 | 480 | nova/equilibrium/forward_operator.py:2462 `ForwardFluxOperator.normalised_current_moments`; nova/equilibrium/forward_operator.py:2301 `ForwardFluxOperator._internal_on_partition` |
| 300 | solve | topology read | 2 | 98 | nova/equilibrium/forward_operator.py:1885 `ForwardFluxOperator._fixed_design_read` |
| 1000 | map | current-moment path | 1 | 5 | nova/equilibrium/forward_operator.py:2462 `ForwardFluxOperator.normalised_current_moments`; nova/equilibrium/forward_operator.py:2301 `ForwardFluxOperator._internal_on_partition` |
| 1000 | map | topology read | 1 | 14 | nova/equilibrium/forward_operator.py:1885 `ForwardFluxOperator._fixed_design_read` |
| 1000 | solve | current-moment path | 120 | 636 | nova/equilibrium/forward_operator.py:2462 `ForwardFluxOperator.normalised_current_moments`; nova/equilibrium/forward_operator.py:2301 `ForwardFluxOperator._internal_on_partition` |
| 1000 | solve | topology read | 2 | 332 | nova/equilibrium/forward_operator.py:1885 `ForwardFluxOperator._fixed_design_read` |

## Loop inventory

| loop | form | source | effect on optimised HLO |
|---|---|---|---|
| Newton steps | `Python for` | nova/equilibrium/reduced_newton.py:1127 `_plain_newton_trip` | host route only; zero optimised-HLO copies |
| active-set trips | `Python for` | nova/equilibrium/reduced_newton.py:1301 `_drive_trips` | host route only; zero optimised-HLO copies |
| compiled Newton steps | `jax.lax.fori_loop` | nova/equilibrium/reduced_newton.py:1512 `_compiled_slice_solver` | one while body in optimised HLO |
| compiled active-set trips | `jax.lax.fori_loop` | nova/equilibrium/reduced_newton.py:1626 `_compiled_slice_solver` | one while body in optimised HLO |
| certificate active-set budget | `jax.lax.fori_loop` | nova/equilibrium/fixed_point.py:3837 `_active_set_newton_krylov` | one while body in optimised HLO |

## Solve over one map application

| cells | solve instr | map instr | instr ratio | solve bytes | map bytes | byte ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 203,078 | 9,930 | 20.5x | 89.80 GiB | 787.26 MiB | 116.8x |
| 1000 | 204,531 | 10,046 | 20.4x | 306.36 GiB | 2.29 GiB | 134.0x |

## Structural summary (solve program)

### 300 cells

- instructions: 203,078  (unparsed-shape: 0)
- while ops: 192  |  conditional ops: 161
- scanned bodies: 171,068 instructions (84.2%)  |  straight-line: 32,010 (15.8%)
- instructions with no nova scope: 53,851
  (no metadata 52,740; metadata but not nova 1,111)
- scan bytes: 86.69 GiB  |  straight-line bytes: 3.12 GiB

while body/condition computation sizes:

| body computation | instructions |
|---:|---:|
| region_537.1032.sunk.clone.clone | 14 |
| region_542.1033.clone.clone | 3 |
| region_554.1046 | 11 |
| region_555.1047 | 3 |
| region_565.1057 | 11 |
| region_566.1058 | 3 |
| wide.wide.region_506.1077.clone.sunk.clone.clone | 137 |
| wide.wide.region_584.1078.clone.clone.clone | 3 |
| wide.region_735.1264.clone | 5 |
| wide.region_736.1265.clone | 2 |
| region_713.1243 | 11 |
| region_714.1244 | 3 |
| wide.wide.region_725.1255.clone.clone.clone | 9 |
| wide.wide.region_726.1256.clone.clone.clone | 3 |
| region_732.1262 | 7 |
| region_733.1263 | 1 |
| region_734.1266 | 18 |
| region_737.1267 | 2 |
| wide.wide.wide.wide.region_709.1271.sunk.clone.sunk.clone.clone.sunk.clone.clone.clone | 133 |
| wide.wide.wide.wide.region_741.1272.clone.clone.clone.clone.clone.clone | 2 |
| region_746.1277 | 11 |
| region_747.1278 | 3 |
| wide.wide.region_758.1289.clone.clone.clone | 9 |
| wide.wide.region_759.1290.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_708.1296.sunk.sunk.sunk.sunk.clone.clone.clone | 137 |
| wide.wide.wide.wide.wide.region_765.1297.clone.clone.clone | 2 |
| region_797.1334.sunk.sunk.clone.clone | 16 |
| region_802.1335.clone.clone | 3 |
| region_814.1348 | 11 |
| region_815.1349 | 3 |
| region_598.1097.sunk.clone | 29 |
| region_603.1098.clone | 3 |
| region_617.1117.sunk.clone | 16 |
| region_622.1118.clone | 3 |
| region_652.1152.clone.clone | 20 |
| region_655.1153.clone.clone | 3 |
| region_685.1221 | 11 |
| region_690.1222 | 3 |
| wide.wide.region_662.1181.clone.clone.clone | 9 |
| wide.wide.region_668.1182.clone.clone.clone | 3 |
| wide.wide.region_701.1233.clone.clone.sunk.clone | 9 |
| wide.wide.region_702.1234.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_673.1360.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone | 156 |
| wide.wide.wide.wide.wide.region_825.1361.clone.clone.clone.clone.clone.clone.clone | 3 |
| region_900.1474.sunk.clone.clone | 14 |
| region_905.1475.clone.clone | 3 |
| region_838.1383.sunk.clone | 28 |
| region_843.1384.clone | 3 |
| region_924.1503.sunk.clone | 16 |
| region_929.1504.clone | 3 |
| wide.wide.region_856.1426.clone.clone.clone | 9 |
| wide.wide.region_862.1427.clone.clone.clone | 3 |
| wide.wide.region_869.1487.clone.clone | 75 |
| wide.wide.region_914.1488.clone.clone | 3 |
| region_1142.1728 | 11 |
| region_1143.1729 | 3 |
| region_1079.1665.sunk.clone.clone | 14 |
| region_1084.1666.clone.clone | 3 |
| wide.region_1138.1737.clone | 68 |
| wide.region_1150.1738.clone | 3 |
| region_1154.1742 | 11 |
| region_1155.1743 | 3 |
| region_1103.1693.sunk.clone | 16 |
| region_1108.1694.clone | 3 |
| wide.wide.region_1048.1677.clone.clone | 75 |
| wide.wide.region_1093.1678.clone.clone | 3 |
| region_1197.1789.sunk.clone.clone | 14 |
| region_1202.1790.clone.clone | 3 |
| region_1214.1803 | 11 |
| region_1215.1804 | 3 |
| region_1225.1814 | 11 |
| region_1226.1815 | 3 |
| wide.wide.region_1166.1834.clone.sunk.clone.clone | 137 |
| wide.wide.region_1244.1835.clone.clone.clone | 3 |
| wide.region_1354.1946.clone | 5 |
| wide.region_1355.1947.clone | 2 |
| region_1332.1925 | 11 |
| region_1333.1926 | 3 |
| wide.wide.region_1344.1937.clone.clone.clone | 9 |
| wide.wide.region_1345.1938.clone.clone.clone | 3 |
| region_1351.1944 | 7 |
| region_1352.1945 | 1 |
| region_1353.1948 | 18 |
| region_1356.1949 | 2 |
| wide.wide.wide.wide.region_1328.1953.sunk.clone.sunk.clone.clone.sunk.clone.clone.clone | 133 |
| wide.wide.wide.wide.region_1360.1954.clone.clone.clone.clone.clone.clone | 2 |
| region_1365.1959 | 11 |
| region_1366.1960 | 3 |
| wide.wide.region_1377.1971.clone.clone.clone | 9 |
| wide.wide.region_1378.1972.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_1327.1978.sunk.sunk.sunk.sunk.clone.clone.clone | 137 |
| wide.wide.wide.wide.wide.region_1384.1979.clone.clone.clone | 2 |
| region_1416.2016.sunk.sunk.clone.clone | 16 |
| region_1421.2017.clone.clone | 3 |
| region_1433.2030 | 11 |
| region_1434.2031 | 3 |
| region_1292.1888.clone.clone | 20 |
| region_1295.1889.clone.clone | 3 |
| region_1308.1903 | 11 |
| region_1309.1904 | 3 |
| region_1257.1853.sunk.clone | 16 |
| region_1262.1854.clone | 3 |
| wide.wide.region_1320.1915.clone.clone.sunk.clone | 9 |
| wide.wide.region_1321.1916.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_1296.2042.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone | 156 |
| wide.wide.wide.wide.wide.region_1444.2043.clone.clone.clone.clone.clone.clone.clone | 3 |
| region_1471.2075.sunk.clone | 16 |
| region_1476.2076.clone | 3 |
| region_1486.2087 | 11 |
| region_1487.2088 | 3 |
| region_1023.1608.clone.clone | 20 |
| region_1026.1609.clone.clone | 3 |
| wide.wide.region_1033.1622.clone.clone.clone | 9 |
| wide.wide.region_1039.1623.clone.clone.clone | 3 |
| wide.wide.wide.region_1044.2109.clone.clone.clone.clone.clone.clone.clone.clone | 154 |
| wide.wide.wide.region_1508.2110.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_482.970 | 11 |
| region_483.971 | 3 |
| region_419.906.sunk.clone.clone | 14 |
| region_424.907.clone.clone | 3 |
| wide.region_478.979.clone | 68 |
| wide.region_490.980.clone | 3 |
| region_495.986 | 11 |
| region_496.987 | 3 |
| region_443.935.sunk.clone | 16 |
| region_448.936.clone | 3 |
| wide.wide.region_388.919.clone.clone | 75 |
| wide.wide.region_433.920.clone.clone | 3 |
| region_984.1571.sunk.clone | 16 |
| region_989.1572.clone | 3 |
| region_999.1582 | 11 |
| region_1000.1583 | 3 |
| wide.region_336.796.clone | 5 |
| wide.region_337.797.clone | 2 |
| region_327.788 | 11 |
| region_328.789 | 3 |
| region_333.794 | 7 |
| region_334.795 | 1 |
| region_335.798 | 18 |
| region_338.799 | 2 |
| wide.wide.region_323.803.clone | 90 |
| wide.wide.region_342.804.clone | 2 |
| region_347.812 | 11 |
| region_348.813 | 3 |
| region_297.738 | 11 |
| region_298.739 | 3 |
| region_316.777 | 11 |
| region_317.778 | 3 |
| wide.wide.region_293.752.clone.clone.sunk.clone.clone | 90 |
| wide.wide.region_307.753.clone.clone.clone | 3 |
| region_279.716 | 11 |
| region_284.717 | 3 |
| wide.wide.region_322.818.clone.clone | 92 |
| wide.wide.region_353.819.clone.clone | 2 |
| region_357.823 | 11 |
| region_358.824 | 3 |
| region_256.664 | 27 |
| region_261.665 | 3 |
| region_1907.2553 | 12 |
| region_1912.2554 | 3 |
| region_1927.2574 | 12 |
| region_1932.2575 | 3 |
| region_1944.2588 | 11 |
| region_1945.2589 | 3 |
| region_1955.2599 | 11 |
| region_1956.2600 | 3 |
| wide.region_1896.2619.clone.sunk.clone.clone | 72 |
| wide.region_1974.2620.clone.clone.clone | 3 |
| region_2008.2659 | 12 |
| region_2013.2660 | 3 |
| region_2028.2680 | 12 |
| region_2033.2681 | 3 |
| wide.region_2126.2776.clone | 5 |
| wide.region_2127.2777.clone | 2 |
| region_2104.2755 | 11 |
| region_2105.2756 | 3 |
| wide.wide.region_2116.2767.clone.clone.clone | 9 |
| wide.wide.region_2117.2768.clone.clone.clone | 3 |
| region_2123.2774 | 7 |
| region_2124.2775 | 1 |
| region_2125.2778 | 18 |
| region_2128.2779 | 2 |
| wide.wide.wide.region_2100.2783.sunk.clone.clone.clone.clone | 118 |
| wide.wide.wide.region_2132.2784.clone.clone.clone.clone | 2 |
| region_2137.2789 | 11 |
| region_2138.2790 | 3 |
| wide.wide.region_2149.2801.clone.clone.clone | 9 |
| wide.wide.region_2150.2802.clone.clone.clone | 3 |
| region_2168.2825 | 12 |
| region_2173.2826 | 3 |
| region_2188.2846 | 12 |
| region_2193.2847 | 3 |
| wide.wide.wide.wide.wide.region_2099.2808.sunk.sunk.sunk.clone.clone.clone | 120 |
| wide.wide.wide.wide.wide.region_2156.2809.clone.clone.clone | 2 |
| region_2205.2860 | 11 |
| region_2206.2861 | 3 |
| region_1988.2639 | 25 |
| region_1993.2640 | 3 |
| region_2042.2694.clone.clone | 20 |
| region_2045.2695.clone.clone | 3 |
| region_2076.2733 | 11 |
| region_2081.2734 | 3 |
| wide.wide.region_2052.2708.clone.clone.clone | 9 |
| wide.wide.region_2058.2709.clone.clone.clone | 3 |
| wide.wide.region_2092.2745.clone.clone.sunk.clone | 9 |
| wide.wide.region_2093.2746.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_2064.2872.sunk.clone.sunk.clone.clone.sunk.clone.clone.clone.clone | 90 |
| wide.wide.wide.wide.wide.region_2216.2873.clone.clone.clone.clone.clone.clone.clone | 3 |
| region_2317.2983 | 12 |
| region_2322.2984 | 3 |
| region_2337.3004 | 12 |
| region_2342.3005 | 3 |
| region_2272.2934 | 12 |
| region_2277.2935 | 3 |
| region_2292.2955 | 12 |
| region_2297.2956 | 3 |
| region_2229.2891.clone | 24 |
| region_2234.2892.clone | 3 |
| wide.wide.region_2247.2911.clone.clone.clone | 9 |
| wide.wide.region_2253.2912.clone.clone.clone | 3 |
| wide.region_2261.2967.clone.clone.clone | 14 |
| wide.region_2306.2968.clone.clone.clone | 3 |
| region_2496.3168 | 12 |
| region_2501.3169 | 3 |
| region_2516.3189 | 12 |
| region_2521.3190 | 3 |
| region_2534.3203 | 11 |
| region_2535.3204 | 3 |
| region_2451.3119 | 12 |
| region_2456.3120 | 3 |
| region_2471.3140 | 12 |
| region_2476.3141 | 3 |
| wide.region_2530.3212.clone | 52 |
| wide.region_2542.3213.clone | 3 |
| wide.region_2440.3152.clone.clone.clone | 14 |
| wide.region_2485.3153.clone.clone.clone | 3 |
| region_2546.3217 | 11 |
| region_2547.3218 | 3 |
| region_2569.3243 | 12 |
| region_2574.3244 | 3 |
| region_2589.3264 | 12 |
| region_2594.3265 | 3 |
| region_2606.3278 | 11 |
| region_2607.3279 | 3 |
| region_2617.3289 | 11 |
| region_2618.3290 | 3 |
| wide.region_2558.3309.clone.sunk.clone.clone | 72 |
| wide.region_2636.3310.clone.clone.clone | 3 |
| region_2650.3328 | 12 |
| region_2655.3329 | 3 |
| region_2670.3349 | 12 |
| region_2675.3350 | 3 |
| wide.region_2746.3421.clone | 5 |
| wide.region_2747.3422.clone | 2 |
| region_2724.3400 | 11 |
| region_2725.3401 | 3 |
| wide.wide.region_2736.3412.clone.clone.clone | 9 |
| wide.wide.region_2737.3413.clone.clone.clone | 3 |
| region_2743.3419 | 7 |
| region_2744.3420 | 1 |
| region_2745.3423 | 18 |
| region_2748.3424 | 2 |
| wide.wide.wide.region_2720.3428.sunk.clone.clone.clone.clone | 118 |
| wide.wide.wide.region_2752.3429.clone.clone.clone.clone | 2 |
| region_2757.3434 | 11 |
| region_2758.3435 | 3 |
| wide.wide.region_2769.3446.clone.clone.clone | 9 |
| wide.wide.region_2770.3447.clone.clone.clone | 3 |
| region_2788.3470 | 12 |
| region_2793.3471 | 3 |
| region_2808.3491 | 12 |
| region_2813.3492 | 3 |
| wide.wide.wide.wide.wide.region_2719.3453.sunk.sunk.sunk.clone.clone.clone | 120 |
| wide.wide.wide.wide.wide.region_2776.3454.clone.clone.clone | 2 |
| region_2825.3505 | 11 |
| region_2826.3506 | 3 |
| region_2684.3363.clone.clone | 20 |
| region_2687.3364.clone.clone | 3 |
| region_2700.3378 | 11 |
| region_2701.3379 | 3 |
| wide.wide.region_2712.3390.clone.clone.sunk.clone | 9 |
| wide.wide.region_2713.3391.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_2688.3517.clone.sunk.clone.clone.sunk.clone.clone.clone.clone | 90 |
| wide.wide.wide.wide.wide.region_2836.3518.clone.clone.clone.clone.clone.clone.clone | 3 |
| region_2863.3550 | 12 |
| region_2868.3551 | 3 |
| region_2878.3562 | 11 |
| region_2879.3563 | 3 |
| region_2414.3082.clone.clone | 20 |
| region_2417.3083.clone.clone | 3 |
| wide.wide.region_2424.3096.clone.clone.clone | 9 |
| wide.wide.region_2430.3097.clone.clone.clone | 3 |
| wide.wide.wide.region_2436.3584.clone.clone.clone.clone.clone.clone | 138 |
| wide.wide.wide.region_2900.3585.clone.clone.clone.clone.clone.clone | 2 |
| region_1835.2480 | 12 |
| region_1840.2481 | 3 |
| region_1855.2501 | 12 |
| region_1860.2502 | 3 |
| region_1873.2515 | 11 |
| region_1874.2516 | 3 |
| region_1790.2431 | 12 |
| region_1795.2432 | 3 |
| region_1810.2452 | 12 |
| region_1815.2453 | 3 |
| wide.region_1869.2524.clone | 52 |
| wide.region_1881.2525.clone | 3 |
| wide.region_1779.2464.clone.clone.clone | 14 |
| wide.region_1824.2465.clone.clone.clone | 3 |
| region_1885.2529 | 11 |
| region_1886.2530 | 3 |
| region_2375.3045 | 12 |
| region_2380.3046 | 3 |
| region_2390.3056 | 11 |
| region_2391.3057 | 3 |
| wide.region_1727.2363.clone | 5 |
| wide.region_1728.2364.clone | 2 |
| region_1718.2355 | 11 |
| region_1719.2356 | 3 |
| region_1724.2361 | 7 |
| region_1725.2362 | 1 |
| region_1726.2365 | 18 |
| region_1729.2366 | 2 |
| wide.wide.region_1714.2370.clone | 74 |
| wide.wide.region_1733.2371.clone | 2 |
| region_1738.2376 | 11 |
| region_1739.2377 | 3 |
| region_1688.2324 | 11 |
| region_1689.2325 | 3 |
| region_1707.2346 | 11 |
| region_1708.2347 | 3 |
| wide.wide.region_1684.2335.clone.clone.sunk.clone.clone | 74 |
| wide.wide.region_1698.2336.clone.clone.clone | 3 |
| region_1671.2312 | 11 |
| region_1676.2313 | 3 |
| wide.wide.region_1713.2382.clone.clone | 76 |
| wide.wide.region_1744.2383.clone.clone | 2 |
| region_1748.2387 | 11 |
| region_1749.2388 | 3 |
| region_1648.2289 | 25 |
| region_1653.2290 | 3 |
| wide.region_17.65.clone.2.clone.1.clone | 8 |
| wide.region_18.66.clone.2.clone.1.clone | 3 |
| wide.region_56.157.clone.clone.1 | 9 |
| wide.region_57.158.clone.clone.1 | 3 |
| wide.region_1637.3589.clone.clone.clone | 130 |
| wide.region_2901.3590.clone.clone.clone | 2 |
| region_2981.3677 | 12 |
| region_2986.3678 | 3 |
| wide.region_65.183.clone.clone.1 | 15 |
| wide.region_69.184.clone.clone.1 | 3 |
| wide.region_198.515.clone.clone.1 | 15 |
| wide.region_202.516.clone.clone.1 | 3 |
| region_2957.3652 | 12 |
| region_2962.3653 | 3 |
| region_2934.3628 | 12 |
| region_2939.3629 | 3 |
| region_3004.3701 | 12 |
| region_3009.3702 | 3 |
| wide.region_17.65.clone.clone.clone | 8 |
| wide.region_18.66.clone.clone.clone | 3 |
| wide.region_17.65.clone.2.clone.clone.clone | 8 |
| wide.region_18.66.clone.2.clone.clone.clone | 3 |
| wide.region_56.157.sunk.clone.clone | 11 |
| wide.region_57.158.clone.1.clone | 3 |
| wide.region_65.183.clone.1 | 15 |
| wide.region_69.184.clone.1 | 3 |
| wide.region_198.515.clone.1 | 15 |
| wide.region_202.516.clone.1 | 3 |
| wide.region_244.2114.clone.clone | 128 |
| wide.region_1509.2115.clone.clone | 2 |
| region_1592.2229.sunk.clone | 16 |
| region_1597.2230.clone | 3 |
| wide.region_65.183.clone.clone.clone | 15 |
| wide.region_69.184.clone.clone.clone | 3 |
| wide.region_198.515.clone.clone.clone | 15 |
| wide.region_202.516.clone.clone.clone | 3 |
| region_1545.2174.sunk.clone | 16 |
| region_1550.2175.clone | 3 |
| region_1568.2203.sunk.clone | 16 |
| region_1573.2204.clone | 3 |
| region_1615.2253.sunk.clone | 16 |
| region_1620.2254.clone | 3 |
| wide.region_1634.3718.clone.clone.clone.clone | 55 |
| wide.region_3023.3719.clone.clone.clone | 3 |

### 1000 cells

- instructions: 204,531  (unparsed-shape: 0)
- while ops: 192  |  conditional ops: 161
- scanned bodies: 170,979 instructions (83.6%)  |  straight-line: 33,552 (16.4%)
- instructions with no nova scope: 54,479
  (no metadata 53,260; metadata but not nova 1,219)
- scan bytes: 297.36 GiB  |  straight-line bytes: 9.00 GiB

while body/condition computation sizes:

| body computation | instructions |
|---:|---:|
| region_538.1038.sunk.clone.clone | 14 |
| region_543.1039.clone.clone | 3 |
| region_555.1052 | 11 |
| region_556.1053 | 3 |
| region_566.1063 | 11 |
| region_567.1064 | 3 |
| wide.wide.region_507.1083.clone.sunk.clone.clone | 137 |
| wide.wide.region_585.1084.clone.clone.clone | 3 |
| wide.region_736.1272.clone | 5 |
| wide.region_737.1273.clone | 2 |
| region_714.1251 | 11 |
| region_715.1252 | 3 |
| wide.wide.region_726.1263.clone.clone.clone | 9 |
| wide.wide.region_727.1264.clone.clone.clone | 3 |
| region_733.1270 | 7 |
| region_734.1271 | 1 |
| region_735.1274 | 18 |
| region_738.1275 | 2 |
| wide.wide.wide.wide.region_710.1279.sunk.clone.sunk.clone.clone.sunk.clone.clone.clone | 133 |
| wide.wide.wide.wide.region_742.1280.clone.clone.clone.clone.clone.clone | 2 |
| region_747.1285 | 11 |
| region_748.1286 | 3 |
| wide.wide.region_759.1297.clone.clone.clone | 9 |
| wide.wide.region_760.1298.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_709.1304.sunk.sunk.sunk.sunk.clone.clone.clone | 137 |
| wide.wide.wide.wide.wide.region_766.1305.clone.clone.clone | 2 |
| region_798.1342.sunk.sunk.clone.clone | 16 |
| region_803.1343.clone.clone | 3 |
| region_815.1356 | 11 |
| region_816.1357 | 3 |
| region_599.1103.sunk.clone | 29 |
| region_604.1104.clone | 3 |
| region_618.1123.sunk.clone | 16 |
| region_623.1124.clone | 3 |
| region_653.1158.clone.clone | 20 |
| region_656.1159.clone.clone | 3 |
| region_686.1229 | 11 |
| region_691.1230 | 3 |
| wide.wide.region_663.1188.clone.clone.clone | 9 |
| wide.wide.region_669.1189.clone.clone.clone | 3 |
| wide.wide.region_702.1241.clone.clone.sunk.clone | 9 |
| wide.wide.region_703.1242.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_674.1368.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone | 156 |
| wide.wide.wide.wide.wide.region_826.1369.clone.clone.clone.clone.clone.clone.clone | 3 |
| region_901.1484.sunk.clone.clone | 14 |
| region_906.1485.clone.clone | 3 |
| region_839.1391.sunk.clone | 28 |
| region_844.1392.clone | 3 |
| region_925.1513.sunk.clone | 16 |
| region_930.1514.clone | 3 |
| wide.wide.region_857.1436.clone.clone.clone | 9 |
| wide.wide.region_863.1437.clone.clone.clone | 3 |
| wide.wide.region_870.1497.clone.clone | 75 |
| wide.wide.region_915.1498.clone.clone | 3 |
| region_1143.1738 | 11 |
| region_1144.1739 | 3 |
| region_1080.1675.sunk.clone.clone | 14 |
| region_1085.1676.clone.clone | 3 |
| wide.region_1139.1747.clone | 68 |
| wide.region_1151.1748.clone | 3 |
| region_1155.1752 | 11 |
| region_1156.1753 | 3 |
| region_1104.1703.sunk.clone | 16 |
| region_1109.1704.clone | 3 |
| wide.wide.region_1049.1687.clone.clone | 75 |
| wide.wide.region_1094.1688.clone.clone | 3 |
| region_1198.1799.sunk.clone.clone | 14 |
| region_1203.1800.clone.clone | 3 |
| region_1215.1813 | 11 |
| region_1216.1814 | 3 |
| region_1226.1824 | 11 |
| region_1227.1825 | 3 |
| wide.wide.region_1167.1844.clone.sunk.clone.clone | 137 |
| wide.wide.region_1245.1845.clone.clone.clone | 3 |
| wide.region_1355.1956.clone | 5 |
| wide.region_1356.1957.clone | 2 |
| region_1333.1935 | 11 |
| region_1334.1936 | 3 |
| wide.wide.region_1345.1947.clone.clone.clone | 9 |
| wide.wide.region_1346.1948.clone.clone.clone | 3 |
| region_1352.1954 | 7 |
| region_1353.1955 | 1 |
| region_1354.1958 | 18 |
| region_1357.1959 | 2 |
| wide.wide.wide.wide.region_1329.1963.sunk.clone.sunk.clone.clone.sunk.clone.clone.clone | 133 |
| wide.wide.wide.wide.region_1361.1964.clone.clone.clone.clone.clone.clone | 2 |
| region_1366.1969 | 11 |
| region_1367.1970 | 3 |
| wide.wide.region_1378.1981.clone.clone.clone | 9 |
| wide.wide.region_1379.1982.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_1328.1988.sunk.sunk.sunk.sunk.clone.clone.clone | 137 |
| wide.wide.wide.wide.wide.region_1385.1989.clone.clone.clone | 2 |
| region_1417.2026.sunk.sunk.clone.clone | 16 |
| region_1422.2027.clone.clone | 3 |
| region_1434.2040 | 11 |
| region_1435.2041 | 3 |
| region_1293.1898.clone.clone | 20 |
| region_1296.1899.clone.clone | 3 |
| region_1309.1913 | 11 |
| region_1310.1914 | 3 |
| region_1258.1863.sunk.clone | 16 |
| region_1263.1864.clone | 3 |
| wide.wide.region_1321.1925.clone.clone.sunk.clone | 9 |
| wide.wide.region_1322.1926.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_1297.2052.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone | 156 |
| wide.wide.wide.wide.wide.region_1445.2053.clone.clone.clone.clone.clone.clone.clone | 3 |
| region_1472.2085.sunk.clone | 16 |
| region_1477.2086.clone | 3 |
| region_1487.2097 | 11 |
| region_1488.2098 | 3 |
| region_1024.1618.clone.clone | 20 |
| region_1027.1619.clone.clone | 3 |
| wide.wide.region_1034.1632.clone.clone.clone | 9 |
| wide.wide.region_1040.1633.clone.clone.clone | 3 |
| wide.wide.wide.region_1045.2119.clone.clone.clone.clone.clone.clone.clone.clone | 154 |
| wide.wide.wide.region_1509.2120.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_483.976 | 11 |
| region_484.977 | 3 |
| region_420.912.sunk.clone.clone | 14 |
| region_425.913.clone.clone | 3 |
| wide.region_479.985.clone | 68 |
| wide.region_491.986.clone | 3 |
| region_496.992 | 11 |
| region_497.993 | 3 |
| region_444.941.sunk.clone | 16 |
| region_449.942.clone | 3 |
| wide.wide.region_389.925.clone.clone | 75 |
| wide.wide.region_434.926.clone.clone | 3 |
| region_985.1581.sunk.clone | 16 |
| region_990.1582.clone | 3 |
| region_1000.1592 | 11 |
| region_1001.1593 | 3 |
| wide.region_337.801.clone | 5 |
| wide.region_338.802.clone | 2 |
| region_328.793 | 11 |
| region_329.794 | 3 |
| region_334.799 | 7 |
| region_335.800 | 1 |
| region_336.803 | 18 |
| region_339.804 | 2 |
| wide.wide.region_324.808.clone | 90 |
| wide.wide.region_343.809.clone | 2 |
| region_348.817 | 11 |
| region_349.818 | 3 |
| region_298.742 | 11 |
| region_299.743 | 3 |
| region_317.782 | 11 |
| region_318.783 | 3 |
| wide.wide.region_294.756.clone.clone.sunk.clone.clone | 90 |
| wide.wide.region_308.757.clone.clone.clone | 3 |
| region_280.719 | 11 |
| region_285.720 | 3 |
| wide.wide.region_323.823.clone.clone | 92 |
| wide.wide.region_354.824.clone.clone | 2 |
| region_358.828 | 11 |
| region_359.829 | 3 |
| region_257.666 | 27 |
| region_262.667 | 3 |
| region_1908.2566 | 12 |
| region_1913.2567 | 3 |
| region_1928.2587 | 12 |
| region_1933.2588 | 3 |
| region_1945.2601 | 11 |
| region_1946.2602 | 3 |
| region_1956.2612 | 11 |
| region_1957.2613 | 3 |
| wide.region_1897.2632.clone.sunk.clone.clone | 72 |
| wide.region_1975.2633.clone.clone.clone | 3 |
| region_2009.2672 | 12 |
| region_2014.2673 | 3 |
| region_2029.2693 | 12 |
| region_2034.2694 | 3 |
| wide.region_2127.2789.clone | 5 |
| wide.region_2128.2790.clone | 2 |
| region_2105.2768 | 11 |
| region_2106.2769 | 3 |
| wide.wide.region_2117.2780.clone.clone.clone | 9 |
| wide.wide.region_2118.2781.clone.clone.clone | 3 |
| region_2124.2787 | 7 |
| region_2125.2788 | 1 |
| region_2126.2791 | 18 |
| region_2129.2792 | 2 |
| wide.wide.wide.region_2101.2796.sunk.clone.clone.clone.clone | 118 |
| wide.wide.wide.region_2133.2797.clone.clone.clone.clone | 2 |
| region_2138.2802 | 11 |
| region_2139.2803 | 3 |
| wide.wide.region_2150.2814.clone.clone.clone | 9 |
| wide.wide.region_2151.2815.clone.clone.clone | 3 |
| region_2169.2838 | 12 |
| region_2174.2839 | 3 |
| region_2189.2859 | 12 |
| region_2194.2860 | 3 |
| wide.wide.wide.wide.wide.region_2100.2821.sunk.sunk.sunk.clone.clone.clone | 120 |
| wide.wide.wide.wide.wide.region_2157.2822.clone.clone.clone | 2 |
| region_2206.2873 | 11 |
| region_2207.2874 | 3 |
| region_1989.2652 | 25 |
| region_1994.2653 | 3 |
| region_2043.2707.clone.clone | 20 |
| region_2046.2708.clone.clone | 3 |
| region_2077.2746 | 11 |
| region_2082.2747 | 3 |
| wide.wide.region_2053.2721.clone.clone.clone | 9 |
| wide.wide.region_2059.2722.clone.clone.clone | 3 |
| wide.wide.region_2093.2758.clone.clone.sunk.clone | 9 |
| wide.wide.region_2094.2759.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_2065.2885.sunk.clone.sunk.clone.clone.sunk.clone.clone.clone.clone | 90 |
| wide.wide.wide.wide.wide.region_2217.2886.clone.clone.clone.clone.clone.clone.clone | 3 |
| region_2318.2996 | 12 |
| region_2323.2997 | 3 |
| region_2338.3017 | 12 |
| region_2343.3018 | 3 |
| region_2273.2947 | 12 |
| region_2278.2948 | 3 |
| region_2293.2968 | 12 |
| region_2298.2969 | 3 |
| region_2230.2904.clone | 24 |
| region_2235.2905.clone | 3 |
| wide.wide.region_2248.2924.clone.clone.clone | 9 |
| wide.wide.region_2254.2925.clone.clone.clone | 3 |
| wide.region_2262.2980.clone.clone.clone | 14 |
| wide.region_2307.2981.clone.clone.clone | 3 |
| region_2497.3181 | 12 |
| region_2502.3182 | 3 |
| region_2517.3202 | 12 |
| region_2522.3203 | 3 |
| region_2535.3216 | 11 |
| region_2536.3217 | 3 |
| region_2452.3132 | 12 |
| region_2457.3133 | 3 |
| region_2472.3153 | 12 |
| region_2477.3154 | 3 |
| wide.region_2531.3225.clone | 52 |
| wide.region_2543.3226.clone | 3 |
| wide.region_2441.3165.clone.clone.clone | 14 |
| wide.region_2486.3166.clone.clone.clone | 3 |
| region_2547.3230 | 11 |
| region_2548.3231 | 3 |
| region_2570.3256 | 12 |
| region_2575.3257 | 3 |
| region_2590.3277 | 12 |
| region_2595.3278 | 3 |
| region_2607.3291 | 11 |
| region_2608.3292 | 3 |
| region_2618.3302 | 11 |
| region_2619.3303 | 3 |
| wide.region_2559.3322.clone.sunk.clone.clone | 72 |
| wide.region_2637.3323.clone.clone.clone | 3 |
| region_2651.3341 | 12 |
| region_2656.3342 | 3 |
| region_2671.3362 | 12 |
| region_2676.3363 | 3 |
| wide.region_2747.3434.clone | 5 |
| wide.region_2748.3435.clone | 2 |
| region_2725.3413 | 11 |
| region_2726.3414 | 3 |
| wide.wide.region_2737.3425.clone.clone.clone | 9 |
| wide.wide.region_2738.3426.clone.clone.clone | 3 |
| region_2744.3432 | 7 |
| region_2745.3433 | 1 |
| region_2746.3436 | 18 |
| region_2749.3437 | 2 |
| wide.wide.wide.region_2721.3441.sunk.clone.clone.clone.clone | 118 |
| wide.wide.wide.region_2753.3442.clone.clone.clone.clone | 2 |
| region_2758.3447 | 11 |
| region_2759.3448 | 3 |
| wide.wide.region_2770.3459.clone.clone.clone | 9 |
| wide.wide.region_2771.3460.clone.clone.clone | 3 |
| region_2789.3483 | 12 |
| region_2794.3484 | 3 |
| region_2809.3504 | 12 |
| region_2814.3505 | 3 |
| wide.wide.wide.wide.wide.region_2720.3466.sunk.sunk.sunk.clone.clone.clone | 120 |
| wide.wide.wide.wide.wide.region_2777.3467.clone.clone.clone | 2 |
| region_2826.3518 | 11 |
| region_2827.3519 | 3 |
| region_2685.3376.clone.clone | 20 |
| region_2688.3377.clone.clone | 3 |
| region_2701.3391 | 11 |
| region_2702.3392 | 3 |
| wide.wide.region_2713.3403.clone.clone.sunk.clone | 9 |
| wide.wide.region_2714.3404.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_2689.3530.clone.sunk.clone.clone.sunk.clone.clone.clone.clone | 90 |
| wide.wide.wide.wide.wide.region_2837.3531.clone.clone.clone.clone.clone.clone.clone | 3 |
| region_2864.3563 | 12 |
| region_2869.3564 | 3 |
| region_2879.3575 | 11 |
| region_2880.3576 | 3 |
| region_2415.3095.clone.clone | 20 |
| region_2418.3096.clone.clone | 3 |
| wide.wide.region_2425.3109.clone.clone.clone | 9 |
| wide.wide.region_2431.3110.clone.clone.clone | 3 |
| wide.wide.wide.region_2437.3597.clone.clone.clone.clone.clone.clone | 138 |
| wide.wide.wide.region_2901.3598.clone.clone.clone.clone.clone.clone | 2 |
| region_1836.2493 | 12 |
| region_1841.2494 | 3 |
| region_1856.2514 | 12 |
| region_1861.2515 | 3 |
| region_1874.2528 | 11 |
| region_1875.2529 | 3 |
| region_1791.2444 | 12 |
| region_1796.2445 | 3 |
| region_1811.2465 | 12 |
| region_1816.2466 | 3 |
| wide.region_1870.2537.clone | 52 |
| wide.region_1882.2538.clone | 3 |
| wide.region_1780.2477.clone.clone.clone | 14 |
| wide.region_1825.2478.clone.clone.clone | 3 |
| region_1886.2542 | 11 |
| region_1887.2543 | 3 |
| region_2376.3058 | 12 |
| region_2381.3059 | 3 |
| region_2391.3069 | 11 |
| region_2392.3070 | 3 |
| wide.region_1728.2376.clone | 5 |
| wide.region_1729.2377.clone | 2 |
| region_1719.2368 | 11 |
| region_1720.2369 | 3 |
| region_1725.2374 | 7 |
| region_1726.2375 | 1 |
| region_1727.2378 | 18 |
| region_1730.2379 | 2 |
| wide.wide.region_1715.2383.clone | 74 |
| wide.wide.region_1734.2384.clone | 2 |
| region_1739.2389 | 11 |
| region_1740.2390 | 3 |
| region_1689.2337 | 11 |
| region_1690.2338 | 3 |
| region_1708.2359 | 11 |
| region_1709.2360 | 3 |
| wide.wide.region_1685.2348.clone.clone.sunk.clone.clone | 74 |
| wide.wide.region_1699.2349.clone.clone.clone | 3 |
| region_1672.2325 | 11 |
| region_1677.2326 | 3 |
| wide.wide.region_1714.2395.clone.clone | 76 |
| wide.wide.region_1745.2396.clone.clone | 2 |
| region_1749.2400 | 11 |
| region_1750.2401 | 3 |
| region_1649.2302 | 25 |
| region_1654.2303 | 3 |
| wide.region_17.65.clone.2.clone.1.clone | 8 |
| wide.region_18.66.clone.2.clone.1.clone | 3 |
| wide.region_56.157.clone.clone.1 | 9 |
| wide.region_57.158.clone.clone.1 | 3 |
| wide.region_1638.3602.clone.clone.clone | 131 |
| wide.region_2902.3603.clone.clone.clone | 2 |
| region_2982.3690 | 12 |
| region_2987.3691 | 3 |
| wide.region_65.183.clone.clone.1 | 15 |
| wide.region_69.184.clone.clone.1 | 3 |
| wide.region_198.515.clone.clone.1 | 15 |
| wide.region_202.516.clone.clone.1 | 3 |
| region_2958.3665 | 12 |
| region_2963.3666 | 3 |
| region_2935.3641 | 12 |
| region_2940.3642 | 3 |
| region_3005.3714 | 12 |
| region_3010.3715 | 3 |
| wide.region_17.65.clone.clone.clone | 8 |
| wide.region_18.66.clone.clone.clone | 3 |
| wide.region_17.65.clone.2.clone.clone.clone | 8 |
| wide.region_18.66.clone.2.clone.clone.clone | 3 |
| wide.region_56.157.sunk.clone.clone | 11 |
| wide.region_57.158.clone.1.clone | 3 |
| wide.region_65.183.clone.1 | 15 |
| wide.region_69.184.clone.1 | 3 |
| wide.region_198.515.clone.1 | 15 |
| wide.region_202.516.clone.1 | 3 |
| wide.region_244.2124.clone.clone | 129 |
| wide.region_1510.2125.clone.clone | 2 |
| region_1593.2242.sunk.clone | 16 |
| region_1598.2243.clone | 3 |
| wide.region_65.183.clone.clone.clone | 15 |
| wide.region_69.184.clone.clone.clone | 3 |
| wide.region_198.515.clone.clone.clone | 15 |
| wide.region_202.516.clone.clone.clone | 3 |
| region_1546.2186.sunk.clone | 16 |
| region_1551.2187.clone | 3 |
| region_1569.2216.sunk.clone | 16 |
| region_1574.2217.clone | 3 |
| region_1616.2266.sunk.clone | 16 |
| region_1621.2267.clone | 3 |
| wide.region_1635.3731.clone.clone.clone.clone | 55 |
| wide.region_3024.3732.clone.clone.clone | 3 |

## Top 30 functions by instructions

| cells | # | function | count | share | file:line |
|---:|---:|---|---:|---:|---|
| 300 | 1 | _quadrature_from_arrays | 37,276 | 18.4% | clip_quadrature.py:91 |
| 300 | 2 | clipped_support_current_moments | 19,065 | 9.4% | clip_quadrature.py:414 |
| 300 | 3 | FluxFieldPolynomial.sample | 7,000 | 3.4% | stencil_mesh.py:145 |
| 300 | 4 | _active_set_newton_krylov | 5,622 | 2.8% | fixed_point.py:3837 |
| 300 | 5 | _integrate_current_points | 5,019 | 2.5% | clip_quadrature.py:279 |
| 300 | 6 | reduce_sum | 3,753 | 1.8% | - |
| 300 | 7 | ForwardFluxOperator.current_normalisation_amplitude | 3,415 | 1.7% | forward_operator.py:2444 |
| 300 | 8 | clipped_support_current_moments.<locals>.scatter | 3,000 | 1.5% | clip_quadrature.py:422 |
| 300 | 9 | IsothermalRotation.centrifugal_exponent_gradient | 2,804 | 1.4% | rotation.py:171 |
| 300 | 10 | clipped_support_current_moments.<locals>.scan_cut | 2,404 | 1.2% | clip_quadrature.py:406 |
| 300 | 11 | _FixedDesignNull2D._compatibility_census | 2,367 | 1.2% | forward_operator.py:787 |
| 300 | 12 | ForwardFluxOperator.scaled_current_moments.<locals>.<genexpr> | 2,241 | 1.1% | forward_operator.py:2422 |
| 300 | 13 | _traced_clip | 2,107 | 1.0% | separatrix_clip.py:1216 |
| 300 | 14 | FluxTarget.internal | 2,016 | 1.0% | target.py:582 |
| 300 | 15 | RotatingDomainProfile.pressure_gradient | 1,928 | 0.9% | rotation.py:269 |
| 300 | 16 | InteriorCurrentMomentStencil.flux_coefficients | 1,892 | 0.9% | stencil_mesh.py:233 |
| 300 | 17 | _smooth_relative_sup_merit | 1,822 | 0.9% | fixed_point.py:875 |
| 300 | 18 | ForwardFluxOperator.coupling_current_moments | 1,754 | 0.9% | forward_operator.py:2152 |
| 300 | 19 | _rebuilt_model_promotion.<locals>.damping_trial | 1,666 | 0.8% | fixed_point.py:1284 |
| 300 | 20 | Topology.boundary | 1,659 | 0.8% | topology.py:635 |
| 300 | 21 | _rebuilt_model_promotion | 1,608 | 0.8% | fixed_point.py:1337 |
| 300 | 22 | reduce_window_sum | 1,387 | 0.7% | - |
| 300 | 23 | clipped_support_current_moments.<locals>.scan_cut.<locals>.integrate | 1,332 | 0.7% | clip_quadrature.py:395 |
| 300 | 24 | Null2D.interpolate | 1,322 | 0.7% | null.py:212 |
| 300 | 25 | _quadratic_flux_design | 1,303 | 0.6% | stencil_mesh.py:435 |
| 300 | 26 | _backtracking_scores | 1,302 | 0.6% | fixed_point.py:996 |
| 300 | 27 | clipped_support_current_moments.<locals>.<genexpr> | 1,216 | 0.6% | clip_quadrature.py:427 |
| 300 | 28 | _backtracked_promotion.<locals>.recover_with_continuation | 1,208 | 0.6% | fixed_point.py:1188 |
| 300 | 29 | ForwardSource.current_moments.<locals>.<genexpr> | 1,163 | 0.6% | source.py:653 |
| 300 | 30 | mul | 1,162 | 0.6% | - |
| 1000 | 1 | _quadrature_from_arrays | 37,276 | 18.2% | clip_quadrature.py:91 |
| 1000 | 2 | clipped_support_current_moments | 19,435 | 9.5% | clip_quadrature.py:414 |
| 1000 | 3 | FluxFieldPolynomial.sample | 7,000 | 3.4% | stencil_mesh.py:145 |
| 1000 | 4 | _active_set_newton_krylov | 5,625 | 2.8% | fixed_point.py:3837 |
| 1000 | 5 | _integrate_current_points | 5,019 | 2.5% | clip_quadrature.py:279 |
| 1000 | 6 | reduce_sum | 4,265 | 2.1% | - |
| 1000 | 7 | ForwardFluxOperator.current_normalisation_amplitude | 3,415 | 1.7% | forward_operator.py:2444 |
| 1000 | 8 | clipped_support_current_moments.<locals>.scatter | 3,006 | 1.5% | clip_quadrature.py:422 |
| 1000 | 9 | IsothermalRotation.centrifugal_exponent_gradient | 2,799 | 1.4% | rotation.py:171 |
| 1000 | 10 | clipped_support_current_moments.<locals>.scan_cut | 2,404 | 1.2% | clip_quadrature.py:406 |
| 1000 | 11 | _FixedDesignNull2D._compatibility_census | 2,367 | 1.2% | forward_operator.py:787 |
| 1000 | 12 | ForwardFluxOperator.scaled_current_moments.<locals>.<genexpr> | 2,295 | 1.1% | forward_operator.py:2422 |
| 1000 | 13 | _traced_clip | 2,107 | 1.0% | separatrix_clip.py:1216 |
| 1000 | 14 | FluxTarget.internal | 2,016 | 1.0% | target.py:582 |
| 1000 | 15 | InteriorCurrentMomentStencil.flux_coefficients | 1,892 | 0.9% | stencil_mesh.py:233 |
| 1000 | 16 | _smooth_relative_sup_merit | 1,822 | 0.9% | fixed_point.py:875 |
| 1000 | 17 | ForwardFluxOperator.coupling_current_moments | 1,754 | 0.9% | forward_operator.py:2152 |
| 1000 | 18 | _rebuilt_model_promotion.<locals>.damping_trial | 1,666 | 0.8% | fixed_point.py:1284 |
| 1000 | 19 | Topology.boundary | 1,659 | 0.8% | topology.py:635 |
| 1000 | 20 | _rebuilt_model_promotion | 1,600 | 0.8% | fixed_point.py:1337 |
| 1000 | 21 | mul | 1,468 | 0.7% | - |
| 1000 | 22 | RotatingDomainProfile.pressure_gradient | 1,389 | 0.7% | rotation.py:272 |
| 1000 | 23 | reduce_window_sum | 1,376 | 0.7% | - |
| 1000 | 24 | clipped_support_current_moments.<locals>.scan_cut.<locals>.integrate | 1,332 | 0.7% | clip_quadrature.py:395 |
| 1000 | 25 | Null2D.interpolate | 1,322 | 0.6% | null.py:212 |
| 1000 | 26 | _backtracking_scores | 1,312 | 0.6% | fixed_point.py:996 |
| 1000 | 27 | _quadratic_flux_design | 1,303 | 0.6% | stencil_mesh.py:435 |
| 1000 | 28 | clipped_support_current_moments.<locals>.<genexpr> | 1,261 | 0.6% | clip_quadrature.py:427 |
| 1000 | 29 | _backtracked_promotion.<locals>.recover_with_continuation | 1,220 | 0.6% | fixed_point.py:1188 |
| 1000 | 30 | wall_height_shadow_mask | 1,141 | 0.6% | connectivity_boundary.py:451 |

## Top 30 functions by bytes

| cells | # | function | count | share | file:line |
|---:|---:|---|---:|---:|---|
| 300 | 1 | _quadrature_from_arrays | 22.71 GiB | 25.3% | clip_quadrature.py:91 |
| 300 | 2 | _integrate_current_points | 4.00 GiB | 4.5% | clip_quadrature.py:279 |
| 300 | 3 | IsothermalRotation.centrifugal_exponent_gradient | 3.73 GiB | 4.2% | rotation.py:171 |
| 300 | 4 | _active_set_newton_krylov | 3.71 GiB | 4.1% | fixed_point.py:3837 |
| 300 | 5 | RotatingDomainProfile.pressure_gradient | 3.02 GiB | 3.4% | rotation.py:269 |
| 300 | 6 | _quadratic_flux_design | 3.02 GiB | 3.4% | stencil_mesh.py:435 |
| 300 | 7 | FluxFieldPolynomial.sample | 2.83 GiB | 3.2% | stencil_mesh.py:145 |
| 300 | 8 | mul | 2.09 GiB | 2.3% | - |
| 300 | 9 | toroidal_current_density | 1.64 GiB | 1.8% | convention.py:70 |
| 300 | 10 | IsothermalRotation.centrifugal_exponent | 1.46 GiB | 1.6% | rotation.py:154 |
| 300 | 11 | _rebuilt_model_promotion | 1.42 GiB | 1.6% | fixed_point.py:1337 |
| 300 | 12 | _backtracked_promotion.<locals>.recover_with_continuation | 1.31 GiB | 1.5% | fixed_point.py:1188 |
| 300 | 13 | _backtracking_scores | 1.31 GiB | 1.5% | fixed_point.py:996 |
| 300 | 14 | _rebuilt_model_promotion.<locals>.damping_trial | 1.02 GiB | 1.1% | fixed_point.py:1284 |
| 300 | 15 | clipped_support_current_moments | 1008.99 MiB | 1.1% | clip_quadrature.py:414 |
| 300 | 16 | IsothermalRotation.centrifugal_factor | 907.76 MiB | 1.0% | rotation.py:182 |
| 300 | 17 | IsothermalRotation.squared_radius_offset | 590.39 MiB | 0.6% | rotation.py:177 |
| 300 | 18 | _qualified_krylov_step | 470.67 MiB | 0.5% | fixed_point.py:1792 |
| 300 | 19 | min | 349.34 MiB | 0.4% | - |
| 300 | 20 | _steepest_descent_promotion | 309.45 MiB | 0.3% | fixed_point.py:1399 |
| 300 | 21 | _projected_krylov_condition | 272.69 MiB | 0.3% | fixed_point.py:822 |
| 300 | 22 | _newton_krylov_inner.<locals>.newton_body.<locals>.attempt_step.<locals>.promoted_state.<locals>.carry_fallback_sequence | 235.41 MiB | 0.3% | fixed_point.py:3151 |
| 300 | 23 | add_any | 139.99 MiB | 0.2% | - |
| 300 | 24 | sub | 135.91 MiB | 0.1% | - |
| 300 | 25 | _newton_krylov_inner | 119.57 MiB | 0.1% | fixed_point.py:2755 |
| 300 | 26 | select_n | 91.01 MiB | 0.1% | - |
| 300 | 27 | div | 89.98 MiB | 0.1% | - |
| 300 | 28 | _traced_clip | 59.12 MiB | 0.1% | separatrix_clip.py:1216 |
| 300 | 29 | hex_edge_admissibility | 49.56 MiB | 0.1% | flux_surface_connectivity.py:341 |
| 300 | 30 | max | 44.99 MiB | 0.0% | - |
| 1000 | 1 | _quadrature_from_arrays | 64.48 GiB | 21.0% | clip_quadrature.py:91 |
| 1000 | 2 | _active_set_newton_krylov | 13.82 GiB | 4.5% | fixed_point.py:3837 |
| 1000 | 3 | _integrate_current_points | 11.37 GiB | 3.7% | clip_quadrature.py:279 |
| 1000 | 4 | IsothermalRotation.centrifugal_exponent_gradient | 10.54 GiB | 3.4% | rotation.py:171 |
| 1000 | 5 | _quadratic_flux_design | 8.54 GiB | 2.8% | stencil_mesh.py:435 |
| 1000 | 6 | FluxFieldPolynomial.sample | 8.04 GiB | 2.6% | stencil_mesh.py:145 |
| 1000 | 7 | clipped_support_current_moments | 7.69 GiB | 2.5% | clip_quadrature.py:414 |
| 1000 | 8 | mul | 7.09 GiB | 2.3% | - |
| 1000 | 9 | _rebuilt_model_promotion | 6.68 GiB | 2.2% | fixed_point.py:1337 |
| 1000 | 10 | RotatingDomainProfile.pressure_gradient | 6.29 GiB | 2.1% | rotation.py:272 |
| 1000 | 11 | _backtracked_promotion.<locals>.recover_with_continuation | 5.31 GiB | 1.7% | fixed_point.py:1188 |
| 1000 | 12 | _backtracking_scores | 4.99 GiB | 1.6% | fixed_point.py:996 |
| 1000 | 13 | toroidal_current_density | 4.66 GiB | 1.5% | convention.py:70 |
| 1000 | 14 | IsothermalRotation.centrifugal_exponent | 4.14 GiB | 1.3% | rotation.py:154 |
| 1000 | 15 | _rebuilt_model_promotion.<locals>.damping_trial | 3.95 GiB | 1.3% | fixed_point.py:1284 |
| 1000 | 16 | IsothermalRotation.centrifugal_factor | 2.52 GiB | 0.8% | rotation.py:182 |
| 1000 | 17 | IsothermalRotation.squared_radius_offset | 1.64 GiB | 0.5% | rotation.py:177 |
| 1000 | 18 | _qualified_krylov_step | 1.63 GiB | 0.5% | fixed_point.py:1792 |
| 1000 | 19 | sub | 1.17 GiB | 0.4% | - |
| 1000 | 20 | _projected_krylov_condition | 1.13 GiB | 0.4% | fixed_point.py:822 |
| 1000 | 21 | _steepest_descent_promotion | 1.11 GiB | 0.4% | fixed_point.py:1399 |
| 1000 | 22 | min | 987.70 MiB | 0.3% | - |
| 1000 | 23 | _newton_krylov_inner.<locals>.newton_body.<locals>.attempt_step.<locals>.promoted_state.<locals>.carry_fallback_sequence | 836.49 MiB | 0.3% | fixed_point.py:3151 |
| 1000 | 24 | add_any | 419.10 MiB | 0.1% | - |
| 1000 | 25 | _newton_krylov_inner | 406.25 MiB | 0.1% | fixed_point.py:2755 |
| 1000 | 26 | select_n | 253.83 MiB | 0.1% | - |
| 1000 | 27 | div | 252.89 MiB | 0.1% | - |
| 1000 | 28 | hex_edge_admissibility | 199.76 MiB | 0.1% | flux_surface_connectivity.py:341 |
| 1000 | 29 | _traced_clip | 170.10 MiB | 0.1% | separatrix_clip.py:1216 |
| 1000 | 30 | max | 126.45 MiB | 0.0% | - |

## Flux-map top 10 by instructions

| cells | # | function | count | share | file:line |
|---:|---:|---|---:|---:|---|
| 300 | 1 | _FixedDesignNull2D._compatibility_census | 802 | 8.1% | forward_operator.py:787 |
| 300 | 2 | _traced_clip | 758 | 7.6% | separatrix_clip.py:1216 |
| 300 | 3 | Topology.boundary | 623 | 6.3% | topology.py:635 |
| 300 | 4 | Null2D.interpolate | 450 | 4.5% | null.py:212 |
| 300 | 5 | _quadrature_from_arrays | 415 | 4.2% | clip_quadrature.py:91 |
| 300 | 6 | Topology.read_qualification | 332 | 3.3% | topology.py:1009 |
| 300 | 7 | wall_height_shadow_mask | 311 | 3.1% | connectivity_boundary.py:451 |
| 300 | 8 | _FixedDesignNull2D.read_census | 224 | 2.3% | forward_operator.py:875 |
| 300 | 9 | Topology.x_point_data | 201 | 2.0% | topology.py:421 |
| 300 | 10 | _canonicalize_reciprocal_hex_edges | 188 | 1.9% | connectivity_boundary.py:693 |
| 1000 | 1 | _FixedDesignNull2D._compatibility_census | 802 | 8.0% | forward_operator.py:787 |
| 1000 | 2 | _traced_clip | 758 | 7.5% | separatrix_clip.py:1216 |
| 1000 | 3 | Topology.boundary | 623 | 6.2% | topology.py:635 |
| 1000 | 4 | Null2D.interpolate | 450 | 4.5% | null.py:212 |
| 1000 | 5 | _quadrature_from_arrays | 415 | 4.1% | clip_quadrature.py:91 |
| 1000 | 6 | Topology.read_qualification | 333 | 3.3% | topology.py:1009 |
| 1000 | 7 | wall_height_shadow_mask | 311 | 3.1% | connectivity_boundary.py:451 |
| 1000 | 8 | _FixedDesignNull2D.read_census | 224 | 2.2% | forward_operator.py:875 |
| 1000 | 9 | Topology.x_point_data | 201 | 2.0% | topology.py:421 |
| 1000 | 10 | clipped_support_current_moments | 191 | 1.9% | clip_quadrature.py:414 |

## Flux-map top 10 by bytes

| cells | # | function | count | share | file:line |
|---:|---:|---|---:|---:|---|
| 300 | 1 | _quadrature_from_arrays | 350.10 MiB | 44.5% | clip_quadrature.py:91 |
| 300 | 2 | _quadratic_flux_design | 62.70 MiB | 8.0% | stencil_mesh.py:435 |
| 300 | 3 | _integrate_current_points | 44.26 MiB | 5.6% | clip_quadrature.py:279 |
| 300 | 4 | FluxFieldPolynomial.sample | 33.25 MiB | 4.2% | stencil_mesh.py:145 |
| 300 | 5 | toroidal_current_density | 25.77 MiB | 3.3% | convention.py:70 |
| 300 | 6 | RotatingDomainProfile.pressure_gradient | 22.11 MiB | 2.8% | rotation.py:269 |
| 300 | 7 | IsothermalRotation.centrifugal_exponent_gradient | 18.42 MiB | 2.3% | rotation.py:171 |
| 300 | 8 | _traced_clip | 17.87 MiB | 2.3% | separatrix_clip.py:1216 |
| 300 | 9 | hex_edge_admissibility | 16.52 MiB | 2.1% | flux_surface_connectivity.py:341 |
| 300 | 10 | _pack_traced_vertices | 8.85 MiB | 1.1% | separatrix_clip.py:536 |
| 1000 | 1 | _quadrature_from_arrays | 995.62 MiB | 42.5% | clip_quadrature.py:91 |
| 1000 | 2 | _quadratic_flux_design | 178.19 MiB | 7.6% | stencil_mesh.py:435 |
| 1000 | 3 | _integrate_current_points | 125.87 MiB | 5.4% | clip_quadrature.py:279 |
| 1000 | 4 | FluxFieldPolynomial.sample | 94.55 MiB | 4.0% | stencil_mesh.py:145 |
| 1000 | 5 | toroidal_current_density | 73.33 MiB | 3.1% | convention.py:70 |
| 1000 | 6 | hex_edge_admissibility | 66.59 MiB | 2.8% | flux_surface_connectivity.py:341 |
| 1000 | 7 | IsothermalRotation.centrifugal_exponent_gradient | 52.39 MiB | 2.2% | rotation.py:171 |
| 1000 | 8 | _traced_clip | 51.47 MiB | 2.2% | separatrix_clip.py:1216 |
| 1000 | 9 | mul | 41.89 MiB | 1.8% | - |
| 1000 | 10 | RotatingDomainProfile.pressure_gradient | 31.45 MiB | 1.3% | rotation.py:272 |

## Cached public-entry host work

The cached-entry profile was not requested in this invocation.

## Exact implementation attack list

| implement node | exact source seams | measured removal target |
|---|---|---|
| Mesh arrays as program arguments | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` closes the operator into the solve; nova/equilibrium/forward_operator.py:2714 `ForwardFluxOperator.traced_flux_map` closes the map. | Move at least 2.91 GiB of 1000-cell interaction, wall/sample, moment-geometry and connectivity literals from constants to arguments; compare against the recorded 462 MiB / 3.50 GiB / 19.00 GiB executable and generated-code ladder. |
| Flux functions and target current as traced arguments | nova/equilibrium/source.py:297 `DomainProfile.pressure_gradient` and nova/equilibrium/source.py:621 `ForwardSource.current_moments` feed the static profile closure; nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` includes target current in the program key. | Remove 0 B of directly classified profile/current literals at 1000 cells plus their folded descendants; the separate coefficient audit identifies two f64 amplitudes (16 B) and the target current as the root traced arguments. |
| Program-size budget with scans | nova/equilibrium/reduced_newton.py:1626 `_compiled_slice_solver` is already the compiled trip `fori_loop`; nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` is the straight-line closure seam. | Do not rewrite host loops as a size fix: they contribute zero HLO copies. Gate the 204,531-instruction solve against the 10,046-instruction map and remove 120 current-moment plus 2 topology-read traced copies by hoisting/reusing those bodies. |
| Cache-entry overhead | nova/equilibrium/reduced_newton.py:3084 `solve_reduced_newton_compiled` derives per-call inputs; nova/equilibrium/reduced_newton.py:356 `reduced_coordinates` builds reduced coordinates before the lookup. | Hoist the measured coordinate and exterior derivations from profile not run; preserve the reusable executable key and terminal identity. |