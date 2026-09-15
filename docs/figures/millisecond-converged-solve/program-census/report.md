# Program scope census

Census of the compiled whole-cell certificate solve (weak-rotation-reactor-static row) at 300 and 1000 requested cells, and of one application of the flux map alone at the analytic flux.

The source premise was tested rather than assumed. The production certificate budgets and the compiled reduced-slice budgets are already `jax.lax.fori_loop` bodies. The Python Newton and trip loops exist only on the inspectable host route and create zero copies in this optimised HLO. Replication is therefore attributed below to separately traced map and topology call paths, not to those host loops.

## Captured literal context

The literal census is taken from the optimised HLO, not from StableHLO or Python object sizes. XLA omits source metadata on some closure literals; those rows explicitly name the closure seam rather than claiming an unavailable innermost frame.

| cells | literals >1 KiB | captured bytes | executable context | generated-code context |
|---:|---:|---:|---:|---:|
| 300 | 856 | 372.55 MiB | 462.00 MiB | 450.56 MiB |
| 1000 | 889 | 2.94 GiB | not serialisable | 3.50 GiB |
| 2500 | not recompiled in this node | - | not serialisable | 19.00 GiB |

![Captured bytes by group](/nova/figures/millisecond-converged-solve/program-census/captured-bytes-by-group.svg)

### Captured bytes by group

| group | 300 literals | 300 bytes | 1000 literals | 1000 bytes |
|---|---:|---:|---:|---:|
| interaction-matrix kernel blocks | 270 | 359.06 MiB | 270 | 2.89 GiB |
| mesh connectivity | 155 | 2.78 MiB | 188 | 10.70 MiB |
| moment geometry | 31 | 3.51 MiB | 31 | 11.22 MiB |
| other captured literals | 161 | 3.89 MiB | 161 | 15.45 MiB |
| wall and sample blocks | 239 | 3.30 MiB | 239 | 10.52 MiB |

### Every optimised-HLO literal above 1 KiB

| cells | id | group | shape | dtype | bytes | innermost Nova frame or closure seam |
|---:|---|---|---|---|---:|---|
| 300 | `constant.21660` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21661` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21663` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788759..sunk.14` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788759..sunk2.14` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788760..sunk.14` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788760..sunk2.14` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788761..sunk.14` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788761..sunk2.14` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796780..sunk.14` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796780..sunk2.16` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796781..sunk.14` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796781..sunk2.16` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796782..sunk.14` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796782..sunk2.16` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988485` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988486` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988487` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989487` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989488` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989489` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989705` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989706` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989707` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989883` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989884` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989885` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991849` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991850` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991851` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997701` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997702` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997703` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005730` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005731` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005732` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006706` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006707` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006708` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006880` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006881` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006882` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008700` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008701` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008702` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062070` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062071` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062072` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064872` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064873` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064874` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065152` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065153` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065154` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065418` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065419` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065420` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072036` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072037` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072038` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074882` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074883` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074884` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3075598` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3075599` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3075600` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3076781` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3076782` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3076783` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3077953` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3077954` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3077955` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3078922` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078923` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078924` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3081540` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081541` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081542` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082150` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082151` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082152` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3084365` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084366` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084367` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085032` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085033` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085034` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.32936` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32937` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32938` | interaction-matrix kernel blocks | `f64[1066,342]{1,0}` | `f64` | 2,916,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21669` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21670` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21671` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788719..sunk.14` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788719..sunk2.14` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788749..sunk.14` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788749..sunk2.14` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788753..sunk.14` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788753..sunk2.14` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796760..sunk.14` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796760..sunk2.16` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796772..sunk.14` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796772..sunk2.16` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796775..sunk.14` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796775..sunk2.16` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988491` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988492` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988493` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989476` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989480` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989483` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989694` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989698` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989701` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989889` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989890` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989891` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991855` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991856` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991857` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997707` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997708` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997709` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005736` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005737` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005738` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006700` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006701` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006702` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006886` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006887` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006888` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008706` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008707` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008708` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062022` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062060` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062064` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064825` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064862` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064866` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065104` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065142` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065146` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065371` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065408` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065412` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3071989` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072026` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072030` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074377` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074872` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074876` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3075045` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3075588` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3075592` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3076228` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3076771` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3076775` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3077400` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3077943` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3077947` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3078875` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078912` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078916` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3080987` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081530` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081534` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081597` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082140` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082144` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3083812` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084355` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084359` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084479` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085022` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085026` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.32930` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32931` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32932` | interaction-matrix kernel blocks | `f64[342,342]{1,0}` | `f64` | 935,712 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21665` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21666` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21667` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788756..sunk.14` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788756..sunk2.14` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788757..sunk.14` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788757..sunk2.14` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788758..sunk.14` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788758..sunk2.14` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796777..sunk.14` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796777..sunk2.16` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796778..sunk.14` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796778..sunk2.16` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796779..sunk.14` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796779..sunk2.16` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988488` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988489` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988490` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989484` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989485` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989486` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989702` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989703` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989704` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989886` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989887` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989888` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991852` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991853` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991854` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997704` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997705` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997706` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005733` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005734` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005735` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006703` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006704` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006705` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006883` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006884` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006885` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008703` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008704` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008705` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062067` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062068` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062069` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064869` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064870` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064871` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065149` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065150` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065151` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065415` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065416` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065417` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072033` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072034` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072035` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074879` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074880` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074881` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3075595` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3075596` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3075597` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3076778` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3076779` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3076780` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3077950` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3077951` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3077952` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3078919` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078920` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078921` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3081537` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081538` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081539` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082147` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082148` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082149` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3084362` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084363` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084364` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085029` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085030` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085031` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.32933` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32934` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32935` | interaction-matrix kernel blocks | `f64[121,342]{1,0}` | `f64` | 331,056 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2895231` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006685` | moment geometry | `f64[201,7,3,7]{3,2,1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008339` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3085080` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3085081` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3085083` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3085084` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3085085` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3085086` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3085087` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3085088` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3085089` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085090` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3085092` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085094` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3085096` | other captured literals | `f64[29547,1]{1,0}` | `f64` | 236,376 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2805086` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.2821434` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.2989085` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.2991135` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.2992959` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.2998402` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.3006684` | mesh connectivity | `s32[201,7,3,7]{3,2,1,0}` | `s32` | 118,188 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3007906` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.3074580` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.3075290` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.3076473` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.3077645` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.3081232` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.3081842` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.3084057` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.3084724` | mesh connectivity | `s32[29547,1]{1,0}` | `s32` | 118,188 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 300 | `constant.21716` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788723..sunk.14` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788723..sunk2.14` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796764..sunk.14` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796764..sunk2.16` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988523` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989463` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989681` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989929` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991895` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997740` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005715` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006694` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006918` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008724` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062027` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064828` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065109` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065376` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3071992` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074839` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3075554` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3076737` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3077909` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3078878` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3081496` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082106` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3084321` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084988` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.32925` | moment geometry | `f64[342,6,7]{2,1,0}` | `f64` | 114,912 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2806064` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2806067` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.2821843` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2821845` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.2914882` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2916290` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2989183` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2989185` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.2989298` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2989641` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2989642` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.2989653` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2989859` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2989860` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.2989871` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2991233` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2991235` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.2991801` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2993057` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2993059` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.2993486` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2998500` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.2998502` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.2998756` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.3008004` | wall and sample blocks | `f64[342,12,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.3008006` | wall and sample blocks | `f64[342,12,1,2]{3,2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 300 | `constant.3008423` | wall and sample blocks | `f64[4104,1,2]{2,1,0}` | `f64` | 65,664 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 300 | `constant.3006683` | wall and sample blocks | `f64[201,7,2,2]{3,2,1,0}` | `f64` | 45,024 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21775` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988555` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989552` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989770` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989961` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991927` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997772` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006689` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006951` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32923` | mesh connectivity | `s64[342,12]{1,0}` | `s64` | 32,832 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21800` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2805186` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2805813..sunk.10` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2821534` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2915168..sunk.14` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2915793..sunk.16` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2938810..sunk.8` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2940022..sunk.8` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2941469..sunk.10` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2943661..sunk.8` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2945147..sunk.10` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2988563` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989110` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2989971` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991160` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2991937` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2992984` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.2997781` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2998427` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.3006678` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006960` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3007931` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 300 | `constant.3074438` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3075130` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3076313` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3077485` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3081072` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081682` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3083897` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084564` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.32908` | wall and sample blocks | `f64[201,7,2]{2,1,0}` | `f64` | 22,512 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006692` | mesh connectivity | `s64[342,7]{1,0}` | `s64` | 19,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21774` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988554` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989564` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989782` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989960` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991926` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997771` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006690` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006950` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32922` | wall and sample blocks | `f64[925,2]{1,0}` | `f64` | 14,800 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2805209` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2805836..sunk.10` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2821557` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2915185..sunk.14` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2915810..sunk.16` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2938827..sunk.8` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2940039..sunk.8` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2941482..sunk.10` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2943674..sunk.8` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2945160..sunk.10` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2989111` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2991161` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2992985` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2998428` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.3006677` | mesh connectivity | `s64[201,7]{1,0}` | `s64` | 11,256 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3007932` | other captured literals | `f64[201,7]{1,0}` | `f64` | 11,256 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 300 | `constant.2807563` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2821865` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989198` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989512` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989730` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991248` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2993072` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2998515` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008019` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008793` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3047062` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062051` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064852` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065133` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065400` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072016` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074840` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3075557` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.3076740` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.3077912` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.3078305` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3078440` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3078772` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3078902` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3081499` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.3082109` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.3082967` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084324` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.3084991` | mesh connectivity | `s32[2394,1]{1,0}` | `s32` | 9,576 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 300 | `constant.3006699` | other captured literals | `f64[342,3]{1,0}` | `f64` | 8,208 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21794` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 300 | `constant.2804855` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.2821203` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.2915080..sunk.14` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 300 | `constant.2915705..sunk.16` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 300 | `constant.2938722..sunk.8` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 300 | `constant.2939934..sunk.8` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 300 | `constant.2941405..sunk.10` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 300 | `constant.2943597..sunk.8` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 300 | `constant.2945083..sunk.10` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 300 | `constant.2988559` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 300 | `constant.2989010` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.2989519` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989737` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989967` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 300 | `constant.2991059` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.2991933` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 300 | `constant.2992883` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.2997777` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 300 | `constant.2998327` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.3006682` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006956` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 300 | `constant.3007831` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.3074410` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.3074575` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 300 | `constant.3074901` | mesh connectivity | `s32[201,7,1]{2,1,0}` | `s32` | 5,628 | nova/biot/null.py:139 `Null2D.__call__` |
| 300 | `constant.3075094` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.3076277` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.3077449` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.3078747` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3081036` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.3081646` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.3082942` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3083861` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.3084528` | mesh connectivity | `s32[1407,1]{1,0}` | `s32` | 5,628 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 300 | `constant.32912` | mesh connectivity | `s32[201,7]{1,0}` | `s32` | 5,628 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 300 | `constant.21682` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21802` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2806079..sunk2.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2806081..sunk2.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2807558..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2807561..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2808948..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2808951..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2809647..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2809650..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2810346..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2810349..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2811908..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2811911..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2812616..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2812619..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2821860..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2821863..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2831551..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2831553..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2831896..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2831898..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2833363..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2833365..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2833713..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2833715..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2863581..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2863583..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2864632..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2864634..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2895273..sunk.7` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.2895304..sunk.7` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.2895415` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2895416` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.2895417` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.2896495` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2896496` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.2914893..sunk.5` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2915426..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2915428..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2915432..sunk.15` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.2916051..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2916053..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2916057..sunk.17` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.2939058..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2939060..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2939064..sunk.9` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.2940270..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2940272..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2940276..sunk.9` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.2941659..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2941661..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2941665..sunk.11` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2943851..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2943853..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2943857..sunk.9` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.2945337..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 300 | `constant.2945339..sunk2.2` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 300 | `constant.2945343..sunk.11` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2961435..sunk.8` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2961437..sunk.10` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2988502` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988564` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989294` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2989295` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.2989436..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2989618` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989654..sunk.3` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2989836` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989872..sunk.3` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2989904` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989972` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991625` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2991626` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.2991627` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.2991870` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991938` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2993428` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2993429` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.2993430` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.2993627..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2997718` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997782` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2998628` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.2998629` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.3006676` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006691` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006696` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006897` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006961` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008365` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.3008366` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.3008367` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.3008564..sunk.1` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 300 | `constant.3074657` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074834` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:122 `_quadrature_from_arrays` |
| 300 | `constant.3075368` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3075549` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3075552` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.3076551` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3076732` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3076735` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.3077723` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3077904` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3077907` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.3081310` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081491` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081494` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.3081920` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082101` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3082104` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.3084135` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084316` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084319` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.3084802` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084983` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084986` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 300 | `constant.3085079` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.3085082` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.3085093` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 300 | `constant.32906` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32924` | wall and sample blocks | `f64[342,2]{1,0}` | `f64` | 5,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2805222` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.2821570` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.2931587` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2932816` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2934042` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2934821` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2936245` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2961374` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2961385` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2961433` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2961485` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2989116` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.2989651` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2989869` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 300 | `constant.2991166` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.2992990` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.2998433` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.3007937` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.3074652` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.3075364` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.3076547` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.3077719` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.3081306` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.3081916` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.3084131` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.3084798` | mesh connectivity | `s32[1206,1]{1,0}` | `s32` | 4,824 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 300 | `constant.21798` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21799` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988561` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988562` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989969` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989970` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991935` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991936` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997779` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997780` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006679` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006680` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006958` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006959` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074421` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074453` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3075105` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3075147` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3076288` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3076330` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3077460` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3077502` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3081047` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081089` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081657` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081699` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3083872` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3083914` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084539` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084581` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.32909` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32910` | wall and sample blocks | `f64[201,2]{1,0}` | `f64` | 3,216 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21676` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21678` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21679` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2760486` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2771290` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788750..sunk.17` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788750..sunk2.19` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788751..sunk.14` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788751..sunk2.14` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788754..sunk.17` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788754..sunk2.19` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796773..sunk.17` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796773..sunk2.21` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796774..sunk.14` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796774..sunk2.16` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796776..sunk.17` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2796776..sunk2.21` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2806054` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 300 | `constant.2821833` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 300 | `constant.2896117..sunk2.2..sunk.10` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2896118..sunk2.2..sunk.10` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2896119..sunk2.10` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2896120..sunk2.10` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2896196..sunk2.2..sunk.8` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2896197..sunk2.2..sunk.8` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2896198..sunk2.10` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2896199..sunk2.10` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988497` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988498` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988499` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988971` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989181` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 300 | `constant.2989477` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989478` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989479` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989481` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989551` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989695` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989696` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989697` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989699` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989769` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.2989895` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989896` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989897` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2990893` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991231` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 300 | `constant.2991256` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991257` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991861` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991862` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991863` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2992822` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2993055` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 300 | `constant.2993081` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2993082` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997713` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997714` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997715` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2998180` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2998498` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 300 | `constant.2998528` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2998529` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005727` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005806` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005819` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005820` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006688` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006892` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006893` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006894` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3007751` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008002` | mesh connectivity | `s64[342]{0}` | `s64` | 2,736 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 300 | `constant.3008710` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008711` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008712` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008760` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062061` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062062` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062063` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062065` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064863` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064864` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064865` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064867` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065143` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065144` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065145` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065147` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065409` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065410` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065411` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065413` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072027` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072028` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072029` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3072031` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074873` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074874` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074875` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074877` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3075589` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3075590` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3075591` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3075593` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3076772` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3076773` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3076774` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3076776` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3077944` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3077945` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3077946` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3077948` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3078308` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078443` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078775` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078913` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078914` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078915` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078917` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3081531` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3081532` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3081533` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3081535` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3082141` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3082142` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3082143` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3082145` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3082970` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3084356` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3084357` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3084358` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3084360` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3085023` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3085024` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3085025` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.3085027` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 300 | `constant.32786` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32787` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.32788` | other captured literals | `f64[342]{0}` | `f64` | 2,736 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21795` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2988560` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989968` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991934` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997778` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006681` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006957` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074379` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3075047` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3076230` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 300 | `constant.3077402` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 300 | `constant.3080989` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3081599` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 300 | `constant.3083814` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3084481` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.32911` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.21715` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2788711..sunk.14` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.2788711..sunk2.14` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.2796757..sunk.14` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.2796757..sunk2.16` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.2988522` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989465` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989683` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2989926` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2991892` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.2997739` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3005716` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3006917` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3008722` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3062026` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3064827` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065108` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3065375` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3071991` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3074838` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3078877` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3081555` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3082165` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 300 | `constant.3084375` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.3085042` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 300 | `constant.32789` | mesh connectivity | `s32[342,1]{1,0}` | `s32` | 1,368 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21655` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21656` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21658` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789522..sunk.14` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789522..sunk2.14` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789523..sunk.14` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789523..sunk2.14` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789524..sunk.14` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789524..sunk2.14` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797135..sunk.14` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797135..sunk2.16` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797136..sunk.14` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797136..sunk2.16` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797137..sunk.14` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797137..sunk2.16` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989153` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989154` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989155` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990157` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990158` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990159` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990375` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990376` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990377` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990553` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990554` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990555` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992522` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992523` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992524` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998390` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998391` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998392` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006427` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006428` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006429` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007406` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007407` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007408` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007580` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007581` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007582` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009402` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009403` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009404` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062772` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062773` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062774` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065574` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065575` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065576` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065854` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065855` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065856` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066120` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066121` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066122` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072738` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072739` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072740` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075584` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075585` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075586` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3076300` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076301` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076302` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3077483` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3077484` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3077485` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3078655` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3078656` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3078657` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3079624` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079625` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079626` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3082242` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082243` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082244` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082852` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082853` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082854` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3085067` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085068` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085069` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085734` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085735` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085736` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.32966` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32967` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32968` | interaction-matrix kernel blocks | `f64[2828,1072]{1,0}` | `f64` | 24,252,928 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21664` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21665` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21666` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789482..sunk.14` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789482..sunk2.14` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789512..sunk.14` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789512..sunk2.14` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789516..sunk.14` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789516..sunk2.14` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797115..sunk.14` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797115..sunk2.16` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797127..sunk.14` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797127..sunk2.16` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797130..sunk.14` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797130..sunk2.16` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989159` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989160` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989161` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990146` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990150` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990153` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990364` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990368` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990371` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990559` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990560` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990561` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992528` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992529` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992530` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998396` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998397` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998398` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006433` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006434` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006435` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007400` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007401` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007402` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007586` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007587` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007588` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009408` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009409` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009410` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062724` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062762` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062766` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065527` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065564` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065568` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065806` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065844` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065848` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066073` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066110` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066114` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072691` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072728` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072732` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075079` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075574` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075578` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075747` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076290` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076294` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076930` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3077473` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3077477` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3078102` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3078645` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3078649` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3079577` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079614` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079618` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3081689` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082232` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082236` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082299` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082842` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082846` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3084514` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085057` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085061` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085181` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085724` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085728` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.32960` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32961` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32962` | interaction-matrix kernel blocks | `f64[1072,1072]{1,0}` | `f64` | 9,193,472 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21660` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21661` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21662` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789519..sunk.14` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789519..sunk2.14` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789520..sunk.14` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789520..sunk2.14` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789521..sunk.14` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789521..sunk2.14` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797132..sunk.14` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797132..sunk2.16` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797133..sunk.14` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797133..sunk2.16` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797134..sunk.14` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797134..sunk2.16` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989156` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989157` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989158` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990154` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990155` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990156` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990372` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990373` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990374` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990556` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990557` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990558` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992525` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992526` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992527` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998393` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998394` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998395` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006430` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006431` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006432` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007403` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007404` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007405` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007583` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007584` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007585` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009405` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009406` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009407` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062769` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062770` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062771` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065571` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065572` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065573` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065851` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065852` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065853` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066117` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066118` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066119` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072735` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072736` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072737` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075581` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075582` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075583` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3076297` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076298` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076299` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3077480` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3077481` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3077482` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3078652` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3078653` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3078654` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3079621` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079622` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079623` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3082239` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082240` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082241` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082849` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082850` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082851` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3085064` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085065` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085066` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085731` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085732` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085733` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.32963` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32964` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32965` | interaction-matrix kernel blocks | `f64[121,1072]{1,0}` | `f64` | 1,037,696 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2895776` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007385` | moment geometry | `f64[815,7,3,7]{3,2,1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009041` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3085782` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3085809` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3085811` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3085812` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3085813` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3085814` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3085815` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3085816` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3085818` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085819` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3085823` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085825` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3085827` | other captured literals | `f64[119805,1]{1,0}` | `f64` | 958,440 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2805453` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.2821801` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.2989755` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.2991808` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.2993635` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.2999092` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.3007384` | mesh connectivity | `s32[815,7,3,7]{3,2,1,0}` | `s32` | 479,220 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3008608` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.3075282` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.3075992` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.3077175` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.3078347` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.3081934` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.3082544` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.3084759` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.3085426` | mesh connectivity | `s32[119805,1]{1,0}` | `s32` | 479,220 | nova/equilibrium/topology.py:831 `Topology.axis_component` |
| 1000 | `constant.21714` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789486..sunk.14` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789486..sunk2.14` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797119..sunk.14` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797119..sunk2.16` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989193` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990133` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990351` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990602` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992571` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998430` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006412` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007394` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007620` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009426` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062729` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065530` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065811` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066078` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072694` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075541` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3076256` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3077439` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3078611` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3079580` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3082198` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082808` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3085023` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085690` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.32955` | moment geometry | `f64[1072,6,7]{2,1,0}` | `f64` | 360,192 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2806431` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2806434` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.2822210` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2822212` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.2915453` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2916863` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2989853` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2989855` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.2989968` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2990311` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2990312` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.2990323` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2990529` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2990530` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.2990541` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2991906` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2991908` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.2992474` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2993733` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2993735` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.2994162` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2999190` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.2999192` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.2999446` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.3008706` | wall and sample blocks | `f64[1072,11,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.3008708` | wall and sample blocks | `f64[1072,11,1,2]{3,2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:1001 `_traced_clip` |
| 1000 | `constant.3009125` | wall and sample blocks | `f64[11792,1,2]{2,1,0}` | `f64` | 188,672 | nova/equilibrium/separatrix_clip.py:888 `_traced_clip` |
| 1000 | `constant.3007383` | wall and sample blocks | `f64[815,7,2,2]{3,2,1,0}` | `f64` | 182,560 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21773` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989225` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990222` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990440` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990634` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992603` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998462` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007389` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007653` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32953` | mesh connectivity | `s64[1072,11]{1,0}` | `s64` | 94,336 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21798` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2805553` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2806180..sunk.10` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2821901` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2915739..sunk.14` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2916365..sunk.16` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2939423..sunk.8` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2940637..sunk.8` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2942086..sunk.10` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2944281..sunk.8` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2945771..sunk.10` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2989233` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989780` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2990644` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2991833` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2992613` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2993660` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.2998471` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2999117` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.3007378` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007662` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3008633` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/topology.py:798 `Topology.axis_component` |
| 1000 | `constant.3075140` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075832` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3077015` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3078187` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3081774` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082384` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3084599` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085266` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.32938` | wall and sample blocks | `f64[815,7,2]{2,1,0}` | `f64` | 91,280 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007392` | mesh connectivity | `s64[1072,7]{1,0}` | `s64` | 60,032 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2805576` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2806203..sunk.10` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2821924` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2915756..sunk.14` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2916382..sunk.16` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2939440..sunk.8` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2940654..sunk.8` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2942099..sunk.10` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2944294..sunk.8` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2945784..sunk.10` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2989781` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2991834` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2993661` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.2999118` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.3007377` | mesh connectivity | `s64[815,7]{1,0}` | `s64` | 45,640 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3008634` | other captured literals | `f64[815,7]{1,0}` | `f64` | 45,640 | nova/equilibrium/topology.py:865 `Topology.axis_component` |
| 1000 | `constant.21772` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989224` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990234` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990452` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990633` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992602` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998461` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007390` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007652` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32952` | wall and sample blocks | `f64[2385,2]{1,0}` | `f64` | 38,160 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2807930` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2822232` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989868` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990182` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990400` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2991921` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2993748` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2999205` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3008721` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009495` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3047764` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062753` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065554` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065835` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066102` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072718` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075542` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3076259` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.3077442` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.3078614` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.3079007` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3079142` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3079474` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3079604` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3082201` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.3082811` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.3083669` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085026` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.3085693` | mesh connectivity | `s32[7504,1]{1,0}` | `s32` | 30,016 | nova/equilibrium/stencil_mesh.py:232 `InteriorCurrentMomentStencil.flux_coefficients` |
| 1000 | `constant.3007399` | other captured literals | `f64[1072,3]{1,0}` | `f64` | 25,728 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21792` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 1000 | `constant.2805222` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.2821570` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.2915651..sunk.14` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2916277..sunk.16` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2939335..sunk.8` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2940549..sunk.8` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2942022..sunk.10` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2944217..sunk.8` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2945707..sunk.10` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2989229` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 1000 | `constant.2989680` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.2990189` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990407` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990640` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 1000 | `constant.2991732` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.2992609` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 1000 | `constant.2993559` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.2998467` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 1000 | `constant.2999017` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.3007382` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007658` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 1000 | `constant.3008533` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.3075112` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.3075277` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:290 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3075603` | mesh connectivity | `s32[815,7,1]{2,1,0}` | `s32` | 22,820 | nova/biot/null.py:139 `Null2D.__call__` |
| 1000 | `constant.3075796` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.3076979` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.3078151` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.3079449` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3081738` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.3082348` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.3083644` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3084563` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.3085230` | mesh connectivity | `s32[5705,1]{1,0}` | `s32` | 22,820 | nova/equilibrium/forward_operator.py:770 `_FixedDesignNull2D._compatibility_census` |
| 1000 | `constant.32942` | mesh connectivity | `s32[815,7]{1,0}` | `s32` | 22,820 | nova/equilibrium/flux_surface_connectivity.py:466 `label_saddle_aware_hex_connected_components` |
| 1000 | `constant.2805589` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2821937` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2932199` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2933428` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2934654` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2935433` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2936857` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2962007` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2962018` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2962066` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2962118` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2989786` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2990321` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2990539` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/connectivity_boundary.py:692 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2991839` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2993666` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2999123` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3008639` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3075354` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3076066` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3077249` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3078421` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3082008` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3082618` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3084833` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3085500` | mesh connectivity | `s32[4890,1]{1,0}` | `s32` | 19,560 | nova/equilibrium/flux_surface_connectivity.py:282 `_propagate_admissible_hex_minima` |
| 1000 | `constant.21678` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21800` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2806446..sunk2.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2806448..sunk2.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2807925..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2807928..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2809315..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2809318..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2810014..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2810017..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2810713..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2810716..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2812275..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2812278..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2812983..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2812986..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2822227..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2822230..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2831923..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2831925..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2832268..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2832270..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2833738..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2833740..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2834088..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2834090..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2864048..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2864050..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2865102..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2865104..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2895818..sunk.7` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2895849..sunk.7` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.2895960` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2895961` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2895962` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.2897040` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2897041` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2915464..sunk.5` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2915998..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2916000..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2916004..sunk.15` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.2916624..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2916626..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2916630..sunk.17` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.2939672..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2939674..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2939678..sunk.9` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.2940886..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2940888..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2940892..sunk.9` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.2942277..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2942279..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2942283..sunk.11` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2944472..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2944474..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2944478..sunk.9` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.2945962..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:470 `flux_field_polynomial` |
| 1000 | `constant.2945964..sunk2.2` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:473 `flux_field_polynomial` |
| 1000 | `constant.2945968..sunk.11` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2962068..sunk.8` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2962070..sunk.10` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2989171` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989234` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989964` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2989965` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2990106..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2990288` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990324..sunk.3` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2990506` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990542..sunk.3` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2990578` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990645` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992298` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2992299` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2992300` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.2992547` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992614` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2994104` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2994105` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2994106` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.2994303..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2998408` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998472` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2999318` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.2999319` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3007376` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007391` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007396` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007598` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007663` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009067` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3009068` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3009069` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.3009266..sunk.1` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:129 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3075359` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075536` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:122 `_quadrature_from_arrays` |
| 1000 | `constant.3076070` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076251` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076254` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3077253` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3077434` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3077437` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3078425` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3078606` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3078609` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3082012` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082193` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082196` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3082622` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082803` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082806` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3084837` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085018` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085021` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3085504` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085685` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085688` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/stencil_mesh.py:131 `FluxFieldPolynomial.sample` |
| 1000 | `constant.3085781` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.3085810` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.3085824` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/clip_quadrature.py:273 `_integrate_current_points` |
| 1000 | `constant.32936` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32954` | wall and sample blocks | `f64[1072,2]{1,0}` | `f64` | 17,152 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21796` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21797` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989231` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989232` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990642` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990643` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992611` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992612` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998469` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998470` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007379` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007380` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007660` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007661` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075123` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075155` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075807` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3075849` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076990` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3077032` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3078162` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3078204` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3081749` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3081791` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082359` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082401` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3084574` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3084616` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085241` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085283` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.32939` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32940` | wall and sample blocks | `f64[815,2]{1,0}` | `f64` | 13,040 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21671` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21673` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21674` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2760823` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2771563` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789513..sunk.17` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789513..sunk2.19` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789514..sunk.14` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789514..sunk2.14` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789517..sunk.17` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789517..sunk2.19` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797128..sunk.17` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797128..sunk2.21` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797129..sunk.14` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797129..sunk2.16` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797131..sunk.17` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2797131..sunk2.21` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2806421` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 1000 | `constant.2822200` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 1000 | `constant.2896662..sunk2.2..sunk.10` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2896663..sunk2.2..sunk.10` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2896664..sunk2.10` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2896665..sunk2.10` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2896741..sunk2.2..sunk.8` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2896742..sunk2.2..sunk.8` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2896743..sunk2.10` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2896744..sunk2.10` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989165` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989166` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989167` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989641` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989851` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 1000 | `constant.2990147` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990148` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990149` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990151` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990221` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990365` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990366` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990367` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990369` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990439` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.2990565` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990566` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990567` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2991566` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2991904` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 1000 | `constant.2991929` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2991930` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992534` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992535` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992536` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2993498` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2993731` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 1000 | `constant.2993757` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2993758` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998402` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998403` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998404` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998870` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2999188` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 1000 | `constant.2999218` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2999219` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006424` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006503` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006516` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006517` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007388` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007592` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007593` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007594` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3008453` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3008704` | mesh connectivity | `s64[1072]{0}` | `s64` | 8,576 | nova/equilibrium/separatrix_clip.py:885 `_traced_clip` |
| 1000 | `constant.3009412` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009413` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009414` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009462` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062763` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062764` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062765` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062767` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065565` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065566` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065567` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065569` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065845` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065846` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065847` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065849` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066111` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066112` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066113` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066115` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072729` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072730` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072731` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072733` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075575` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075576` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075577` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075579` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3076291` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3076292` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3076293` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3076295` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3077474` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3077475` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3077476` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3077478` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3078646` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3078647` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3078648` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3078650` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3079010` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079145` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079477` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079615` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079616` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079617` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079619` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3082233` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3082234` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3082235` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3082237` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3082843` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3082844` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3082845` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3082847` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3083672` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3085058` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3085059` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3085060` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3085062` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3085725` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2119 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3085726` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2120 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3085727` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2121 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.3085729` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward_operator.py:2118 `ForwardFluxOperator.coupling_current_moments` |
| 1000 | `constant.32813` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32814` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.32815` | other captured literals | `f64[1072]{0}` | `f64` | 8,576 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.21713` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2789472..sunk.14` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.2789472..sunk2.14` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.2797112..sunk.14` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.2797112..sunk2.16` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.2989192` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990135` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990353` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990599` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992568` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998429` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3006413` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007619` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3009424` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3062728` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065529` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3065810` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3066077` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3072693` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075540` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3079579` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3082257` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3082867` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3085077` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085744` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.32816` | mesh connectivity | `s32[1072,1]{1,0}` | `s32` | 4,288 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2805447` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2805578` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2805578..sunk.2` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2821795` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2821926` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2821926..sunk.2` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2831795..sunk.2` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2833610..sunk.2` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2864974..sunk.2` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2915655..sunk.14` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2916281..sunk.16` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2939339..sunk.8` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2939442..sunk.1` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2940553..sunk.8` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2942025..sunk.10` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2942100..sunk.1` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2944220..sunk.8` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2944295..sunk.1` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2945710..sunk.10` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2945785..sunk.1` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2989754` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2989782` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2990307` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2990525` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2991807` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2991835` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2993634` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2993662` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.2999091` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.2999119` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.3008607` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.3008635` | mesh connectivity | `s32[815]{0}` | `s32` | 3,260 | nova/equilibrium/connectivity_boundary.py:691 `_canonicalize_reciprocal_hex_edges` |
| 1000 | `constant.3075278` | mesh connectivity | `s32[815,1]{1,0}` | `s32` | 3,260 | nova/equilibrium/flux_surface_connectivity.py:278 `_propagate_admissible_hex_minima` |
| 1000 | `constant.21793` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2989230` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2990641` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2992610` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.2998468` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007381` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3007659` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075081` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |
| 1000 | `constant.3075749` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3076932` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1399 `_steepest_descent_promotion` |
| 1000 | `constant.3078104` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:989 `_backtracking_scores` |
| 1000 | `constant.3081691` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3082301` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1188 `_backtracked_promotion.<locals>.recover_with_continuation` |
| 1000 | `constant.3084516` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.3085183` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/fixed_point.py:1337 `_rebuilt_model_promotion` |
| 1000 | `constant.32941` | wall and sample blocks | `f64[121,2]{1,0}` | `f64` | 1,936 | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` |

## Replicated compiled paths

| cells | program | path | traced copies | instructions in path | source |
|---:|---|---|---:|---:|---|
| 300 | map | current-moment path | 1 | 4 | nova/equilibrium/forward_operator.py:2437 `ForwardFluxOperator.normalised_current_moments` |
| 300 | map | topology read | 1 | 14 | nova/equilibrium/forward_operator.py:1879 `ForwardFluxOperator._fixed_design_read` |
| 300 | solve | current-moment path | 120 | 534 | nova/equilibrium/forward_operator.py:2437 `ForwardFluxOperator.normalised_current_moments` |
| 300 | solve | topology read | 8 | 3,917 | nova/equilibrium/forward_operator.py:1879 `ForwardFluxOperator._fixed_design_read` |
| 1000 | map | current-moment path | 1 | 5 | nova/equilibrium/forward_operator.py:2437 `ForwardFluxOperator.normalised_current_moments` |
| 1000 | map | topology read | 1 | 14 | nova/equilibrium/forward_operator.py:1879 `ForwardFluxOperator._fixed_design_read` |
| 1000 | solve | current-moment path | 120 | 636 | nova/equilibrium/forward_operator.py:2437 `ForwardFluxOperator.normalised_current_moments` |
| 1000 | solve | topology read | 8 | 3,911 | nova/equilibrium/forward_operator.py:1879 `ForwardFluxOperator._fixed_design_read` |

## Loop inventory

| loop | form | source | effect on optimised HLO |
|---|---|---|---|
| Newton steps | `Python for` | nova/equilibrium/reduced_newton.py:1127 `_plain_newton_trip` | host route only; zero optimised-HLO copies |
| active-set trips | `Python for` | nova/equilibrium/reduced_newton.py:1301 `_drive_trips` | host route only; zero optimised-HLO copies |
| compiled Newton steps | `jax.lax.fori_loop` | nova/equilibrium/reduced_newton.py:1496 `_compiled_slice_solver` | one while body in optimised HLO |
| compiled active-set trips | `jax.lax.fori_loop` | nova/equilibrium/reduced_newton.py:1610 `_compiled_slice_solver` | one while body in optimised HLO |
| certificate active-set budget | `jax.lax.fori_loop` | nova/equilibrium/fixed_point.py:3837 `_active_set_newton_krylov` | one while body in optimised HLO |

## Solve over one map application

| cells | solve instr | map instr | instr ratio | solve bytes | map bytes | byte ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 654,832 | 9,908 | 66.1x | 87.50 GiB | 787.26 MiB | 113.8x |
| 1000 | 662,331 | 10,024 | 66.1x | 301.65 GiB | 2.29 GiB | 131.9x |

## Structural summary (solve program)

### 300 cells

- instructions: 654,832  (unparsed-shape: 0)
- while ops: 631  |  conditional ops: 213
- scanned bodies: 572,202 instructions (87.4%)  |  straight-line: 82,630 (12.6%)
- instructions with no nova scope: 152,966
  (no metadata 148,557; metadata but not nova 4,409)
- scan bytes: 85.09 GiB  |  straight-line bytes: 2.42 GiB

while body/condition computation sizes:

| body computation | instructions |
|---:|---:|
| wide.region_3702.6229.clone.5 | 5 |
| wide.region_3703.6230.clone.5 | 3 |
| wide.region_3702.6229.clone.clone.4 | 5 |
| wide.region_3703.6230.clone.clone.4 | 3 |
| wide.region_719.1905.clone.clone.28.clone.clone | 8 |
| wide.region_720.1906.clone.clone.28.clone.clone | 3 |
| wide.region_757.1994.clone.29.clone | 9 |
| wide.region_758.1995.clone.29.clone | 3 |
| wide.region_765.2018.clone.29.clone | 15 |
| wide.region_769.2019.clone.29.clone | 3 |
| wide.region_898.2341.clone.29.clone | 15 |
| wide.region_902.2342.clone.29.clone | 3 |
| region_4132.6751 | 12 |
| region_4137.6752 | 3 |
| wide.region_719.1905.clone.2.clone.62.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.62.clone.clone | 3 |
| wide.region_757.1994.clone.clone.62.clone | 9 |
| wide.region_758.1995.clone.clone.62.clone | 3 |
| wide.region_765.2018.clone.clone.62.clone | 15 |
| wide.region_769.2019.clone.clone.62.clone | 3 |
| wide.region_898.2341.clone.clone.62.clone | 15 |
| wide.region_902.2342.clone.clone.62.clone | 3 |
| region_4188.6808 | 12 |
| region_4193.6809 | 3 |
| wide.region_3702.6229.clone.4 | 5 |
| wide.region_3703.6230.clone.4 | 3 |
| wide.region_3702.6229.clone.clone.3 | 5 |
| wide.region_3703.6230.clone.clone.3 | 3 |
| wide.region_3698.6218.clone.4 | 5 |
| wide.region_3699.6219.clone.4 | 3 |
| wide.region_3698.6218.clone.clone.3 | 5 |
| wide.region_3699.6219.clone.clone.3 | 3 |
| region_4205.6822 | 11 |
| region_4206.6823 | 3 |
| wide.region_3698.6218.clone.5 | 5 |
| wide.region_3699.6219.clone.5 | 3 |
| wide.region_3698.6218.clone.clone.4 | 5 |
| wide.region_3699.6219.clone.clone.4 | 3 |
| region_4216.6833 | 11 |
| region_4217.6834 | 3 |
| wide.wide.region_4084.6853.clone.clone.sunk.clone.clone.clone | 111 |
| wide.wide.region_4235.6854.clone.clone.clone.clone.clone | 3 |
| wide.region_719.1905.clone.clone.30.clone.clone | 8 |
| wide.region_720.1906.clone.clone.30.clone.clone | 3 |
| wide.region_757.1994.clone.31.clone | 9 |
| wide.region_758.1995.clone.31.clone | 3 |
| wide.region_765.2018.clone.31.clone | 15 |
| wide.region_769.2019.clone.31.clone | 3 |
| wide.region_898.2341.clone.31.clone | 15 |
| wide.region_902.2342.clone.31.clone | 3 |
| region_4342.6966 | 12 |
| region_4347.6967 | 3 |
| wide.region_719.1905.clone.2.clone.66.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.66.clone.clone | 3 |
| wide.region_757.1994.clone.clone.66.clone | 9 |
| wide.region_758.1995.clone.clone.66.clone | 3 |
| wide.region_765.2018.clone.clone.66.clone | 15 |
| wide.region_769.2019.clone.clone.66.clone | 3 |
| wide.region_898.2341.clone.clone.66.clone | 15 |
| wide.region_902.2342.clone.clone.66.clone | 3 |
| region_4398.7023 | 12 |
| region_4403.7024 | 3 |
| wide.region_719.1905.clone.clone.32.clone.clone | 8 |
| wide.region_720.1906.clone.clone.32.clone.clone | 3 |
| wide.region_757.1994.clone.33.clone | 9 |
| wide.region_758.1995.clone.33.clone | 3 |
| wide.region_765.2018.clone.33.clone | 15 |
| wide.region_769.2019.clone.33.clone | 3 |
| wide.region_898.2341.clone.33.clone | 15 |
| wide.region_902.2342.clone.33.clone | 3 |
| region_4610.7357 | 12 |
| region_4615.7358 | 3 |
| wide.region_719.1905.clone.2.clone.70.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.70.clone.clone | 3 |
| wide.region_757.1994.clone.clone.70.clone | 9 |
| wide.region_758.1995.clone.clone.70.clone | 3 |
| wide.region_765.2018.clone.clone.70.clone | 15 |
| wide.region_769.2019.clone.clone.70.clone | 3 |
| wide.region_898.2341.clone.clone.70.clone | 15 |
| wide.region_902.2342.clone.clone.70.clone | 3 |
| region_4666.7414 | 12 |
| region_4671.7415 | 3 |
| wide.region_4529.7269.clone | 5 |
| wide.region_4530.7270.clone | 2 |
| wide.region_3698.6218.clone.7 | 5 |
| wide.region_3699.6219.clone.7 | 3 |
| wide.region_3698.6218.clone.clone.6 | 5 |
| wide.region_3699.6219.clone.clone.6 | 3 |
| wide.wide.region_4450.7110.clone.8.clone | 6 |
| wide.wide.region_4451.7111.clone.8.clone | 3 |
| wide.wide.region_4450.7110.clone.clone.7.clone | 6 |
| wide.wide.region_4451.7111.clone.clone.7.clone | 3 |
| region_4505.7246 | 11 |
| region_4506.7247 | 3 |
| wide.wide.region_4517.7258.clone.clone.clone | 9 |
| wide.wide.region_4518.7259.clone.clone.clone | 3 |
| region_4526.7267 | 7 |
| region_4527.7268 | 1 |
| region_4528.7271 | 18 |
| region_4531.7272 | 2 |
| wide.region_3698.6218.clone.8 | 5 |
| wide.region_3699.6219.clone.8 | 3 |
| wide.region_3698.6218.clone.clone.7 | 5 |
| wide.region_3699.6219.clone.clone.7 | 3 |
| wide.wide.region_4450.7110.clone.1.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.1.clone.clone | 3 |
| wide.wide.region_4450.7110.clone.clone.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.clone.clone.clone | 3 |
| wide.wide.wide.wide.region_4501.7276.sunk.clone.clone.clone.sunk.clone.clone.clone.clone.clone | 236 |
| wide.wide.wide.wide.region_4535.7277.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_4540.7282 | 11 |
| region_4541.7283 | 3 |
| wide.wide.region_4552.7294.clone.clone.clone | 9 |
| wide.wide.region_4553.7295.clone.clone.clone | 3 |
| wide.region_3698.6218.clone.10 | 5 |
| wide.region_3699.6219.clone.10 | 3 |
| wide.region_3698.6218.clone.clone.9 | 5 |
| wide.region_3699.6219.clone.clone.9 | 3 |
| wide.wide.wide.wide.wide.wide.region_4500.7303.sunk.sunk.sunk.clone.clone.clone.clone | 229 |
| wide.wide.wide.wide.wide.wide.region_4561.7304.clone.clone.clone.clone | 2 |
| region_4683.7428 | 11 |
| region_4684.7429 | 3 |
| wide.region_226.598.clone.clone.4.clone.sunk.clone | 8 |
| wide.region_227.599.clone.clone.4.clone.clone | 3 |
| wide.region_377.932.clone.5.clone.sunk.clone | 9 |
| wide.region_378.933.clone.5.clone.clone | 3 |
| wide.region_377.932.clone.clone.4.clone.sunk.clone | 9 |
| wide.region_378.933.clone.clone.4.clone.clone | 3 |
| wide.region_379.957.clone.11.clone.sunk.clone.clone | 12 |
| wide.region_380.958.clone.11.clone.clone.clone | 3 |
| wide.region_379.957.clone.clone.4.clone.sunk.clone.clone | 12 |
| wide.region_380.958.clone.clone.4.clone.clone.clone | 3 |
| wide.wide.region_4450.7110.clone.3.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.3.clone.clone | 3 |
| wide.wide.region_4450.7110.clone.clone.2.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.clone.2.clone.clone | 3 |
| wide.region_276.720.clone.4.clone | 9 |
| wide.region_277.721.clone.4.clone | 3 |
| wide.region_284.744.clone.4.clone | 15 |
| wide.region_288.745.clone.4.clone | 3 |
| wide.region_479.1355.clone.4.clone | 15 |
| wide.region_483.1356.clone.4.clone | 3 |
| region_4285.6909 | 25 |
| region_4290.6910 | 3 |
| region_4412.7037.clone.clone | 20 |
| region_4415.7038.clone.clone | 3 |
| region_4475.7222 | 11 |
| region_4480.7223 | 3 |
| wide.wide.region_4422.7066.clone.clone.clone | 9 |
| wide.wide.region_4428.7067.clone.clone.clone | 3 |
| wide.wide.region_4491.7234.clone.clone.sunk.clone | 9 |
| wide.wide.region_4492.7235.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.wide.region_4461.7440.sunk.clone.sunk.clone.clone.clone.sunk.clone.clone.clone.clone.clone.clone | 124 |
| wide.wide.wide.wide.wide.wide.region_4694.7441.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_719.1905.clone.clone.36.clone.clone | 8 |
| wide.region_720.1906.clone.clone.36.clone.clone | 3 |
| wide.region_757.1994.clone.37.clone | 9 |
| wide.region_758.1995.clone.37.clone | 3 |
| wide.region_765.2018.clone.37.clone | 15 |
| wide.region_769.2019.clone.37.clone | 3 |
| wide.region_898.2341.clone.37.clone | 15 |
| wide.region_902.2342.clone.37.clone | 3 |
| region_4943.7699 | 12 |
| region_4948.7700 | 3 |
| wide.region_719.1905.clone.2.clone.78.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.78.clone.clone | 3 |
| wide.region_757.1994.clone.clone.78.clone | 9 |
| wide.region_758.1995.clone.clone.78.clone | 3 |
| wide.region_765.2018.clone.clone.78.clone | 15 |
| wide.region_769.2019.clone.clone.78.clone | 3 |
| wide.region_898.2341.clone.clone.78.clone | 15 |
| wide.region_902.2342.clone.clone.78.clone | 3 |
| region_4999.7756 | 12 |
| region_5004.7757 | 3 |
| wide.region_719.1905.clone.clone.34.clone.clone | 8 |
| wide.region_720.1906.clone.clone.34.clone.clone | 3 |
| wide.region_757.1994.clone.35.clone | 9 |
| wide.region_758.1995.clone.35.clone | 3 |
| wide.region_765.2018.clone.35.clone | 15 |
| wide.region_769.2019.clone.35.clone | 3 |
| wide.region_898.2341.clone.35.clone | 15 |
| wide.region_902.2342.clone.35.clone | 3 |
| region_4825.7577 | 12 |
| region_4830.7578 | 3 |
| wide.region_719.1905.clone.2.clone.74.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.74.clone.clone | 3 |
| wide.region_757.1994.clone.clone.74.clone | 9 |
| wide.region_758.1995.clone.clone.74.clone | 3 |
| wide.region_765.2018.clone.clone.74.clone | 15 |
| wide.region_769.2019.clone.clone.74.clone | 3 |
| wide.region_898.2341.clone.clone.74.clone | 15 |
| wide.region_902.2342.clone.clone.74.clone | 3 |
| region_4881.7634 | 12 |
| region_4886.7635 | 3 |
| wide.region_1803.3450.clone.clone.2.clone.sunk.clone | 8 |
| wide.region_1804.3451.clone.clone.2.clone.clone | 3 |
| wide.region_1953.3778.clone.3.clone.sunk.clone | 9 |
| wide.region_1954.3779.clone.3.clone.clone | 3 |
| wide.region_1953.3778.clone.clone.2.clone.sunk.clone | 9 |
| wide.region_1954.3779.clone.clone.2.clone.clone | 3 |
| wide.region_1955.3802.clone.9.clone.sunk.clone.clone | 12 |
| wide.region_1956.3803.clone.9.clone.clone.clone | 3 |
| wide.region_1955.3802.clone.clone.2.clone.sunk.clone.clone | 12 |
| wide.region_1956.3803.clone.clone.2.clone.clone.clone | 3 |
| wide.wide.region_2163.4418.clone.1.clone.clone | 6 |
| wide.wide.region_2164.4419.clone.1.clone.clone | 3 |
| wide.wide.region_2163.4418.clone.clone.clone.clone | 6 |
| wide.wide.region_2164.4419.clone.clone.clone.clone | 3 |
| wide.region_1852.3568.clone.2.clone | 9 |
| wide.region_1853.3569.clone.2.clone | 3 |
| wide.region_1860.3592.clone.2.clone | 15 |
| wide.region_1864.3593.clone.2.clone | 3 |
| wide.region_2055.4223.clone.2.clone | 15 |
| wide.region_2059.4224.clone.2.clone | 3 |
| region_4743.7495.clone | 24 |
| region_4748.7496.clone | 3 |
| wide.wide.region_4761.7515.clone.clone.clone | 9 |
| wide.wide.region_4767.7516.clone.clone.clone | 3 |
| wide.region_4777.7646.clone.clone.clone | 14 |
| wide.region_4895.7647.clone.clone.clone | 3 |
| wide.wide.region_4448.7099.clone.7.clone.clone | 6 |
| wide.wide.region_4449.7100.clone.7.clone.clone | 3 |
| wide.wide.region_4448.7099.clone.clone.6.clone.clone | 6 |
| wide.wide.region_4449.7100.clone.clone.6.clone.clone | 3 |
| wide.region_3702.6229.clone.12 | 5 |
| wide.region_3703.6230.clone.12 | 3 |
| wide.region_3702.6229.clone.clone.11 | 5 |
| wide.region_3703.6230.clone.clone.11 | 3 |
| wide.region_719.1905.clone.clone.40.clone.clone | 8 |
| wide.region_720.1906.clone.clone.40.clone.clone | 3 |
| wide.region_757.1994.clone.41.clone | 9 |
| wide.region_758.1995.clone.41.clone | 3 |
| wide.region_765.2018.clone.41.clone | 15 |
| wide.region_769.2019.clone.41.clone | 3 |
| wide.region_898.2341.clone.41.clone | 15 |
| wide.region_902.2342.clone.41.clone | 3 |
| region_5306.8068 | 12 |
| region_5311.8069 | 3 |
| wide.region_719.1905.clone.2.clone.86.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.86.clone.clone | 3 |
| wide.region_757.1994.clone.clone.86.clone | 9 |
| wide.region_758.1995.clone.clone.86.clone | 3 |
| wide.region_765.2018.clone.clone.86.clone | 15 |
| wide.region_769.2019.clone.clone.86.clone | 3 |
| wide.region_898.2341.clone.clone.86.clone | 15 |
| wide.region_902.2342.clone.clone.86.clone | 3 |
| region_5362.8125 | 12 |
| region_5367.8126 | 3 |
| wide.region_719.1905.clone.clone.38.clone.clone | 8 |
| wide.region_720.1906.clone.clone.38.clone.clone | 3 |
| wide.region_757.1994.clone.39.clone | 9 |
| wide.region_758.1995.clone.39.clone | 3 |
| wide.region_765.2018.clone.39.clone | 15 |
| wide.region_769.2019.clone.39.clone | 3 |
| wide.region_898.2341.clone.39.clone | 15 |
| wide.region_902.2342.clone.39.clone | 3 |
| region_5188.7946 | 12 |
| region_5193.7947 | 3 |
| wide.region_719.1905.clone.2.clone.82.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.82.clone.clone | 3 |
| wide.region_757.1994.clone.clone.82.clone | 9 |
| wide.region_758.1995.clone.clone.82.clone | 3 |
| wide.region_765.2018.clone.clone.82.clone | 15 |
| wide.region_769.2019.clone.clone.82.clone | 3 |
| wide.region_898.2341.clone.clone.82.clone | 15 |
| wide.region_902.2342.clone.clone.82.clone | 3 |
| region_5244.8003 | 12 |
| region_5249.8004 | 3 |
| wide.region_3702.6229.clone.11 | 5 |
| wide.region_3703.6230.clone.11 | 3 |
| wide.region_3702.6229.clone.clone.10 | 5 |
| wide.region_3703.6230.clone.clone.10 | 3 |
| wide.region_3698.6218.clone.11 | 5 |
| wide.region_3699.6219.clone.11 | 3 |
| wide.region_3698.6218.clone.clone.10 | 5 |
| wide.region_3699.6219.clone.clone.10 | 3 |
| region_5380.8139 | 11 |
| region_5381.8140 | 3 |
| wide.region_3698.6218.clone.12 | 5 |
| wide.region_3699.6219.clone.12 | 3 |
| wide.region_3698.6218.clone.clone.11 | 5 |
| wide.region_3699.6219.clone.clone.11 | 3 |
| wide.region_5140.8015.clone.clone.clone | 14 |
| wide.region_5258.8016.clone.clone.clone | 3 |
| wide.wide.region_5376.8148.clone.clone.clone | 91 |
| wide.wide.region_5388.8149.clone.clone.clone | 3 |
| region_5392.8153 | 11 |
| region_5393.8154 | 3 |
| wide.region_3702.6229.clone.14 | 5 |
| wide.region_3703.6230.clone.14 | 3 |
| wide.region_3702.6229.clone.clone.13 | 5 |
| wide.region_3703.6230.clone.clone.13 | 3 |
| wide.region_719.1905.clone.clone.42.clone.clone | 8 |
| wide.region_720.1906.clone.clone.42.clone.clone | 3 |
| wide.region_757.1994.clone.43.clone | 9 |
| wide.region_758.1995.clone.43.clone | 3 |
| wide.region_765.2018.clone.43.clone | 15 |
| wide.region_769.2019.clone.43.clone | 3 |
| wide.region_898.2341.clone.43.clone | 15 |
| wide.region_902.2342.clone.43.clone | 3 |
| region_5452.8216 | 12 |
| region_5457.8217 | 3 |
| wide.region_719.1905.clone.2.clone.90.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.90.clone.clone | 3 |
| wide.region_757.1994.clone.clone.90.clone | 9 |
| wide.region_758.1995.clone.clone.90.clone | 3 |
| wide.region_765.2018.clone.clone.90.clone | 15 |
| wide.region_769.2019.clone.clone.90.clone | 3 |
| wide.region_898.2341.clone.clone.90.clone | 15 |
| wide.region_902.2342.clone.clone.90.clone | 3 |
| region_5508.8273 | 12 |
| region_5513.8274 | 3 |
| wide.region_3702.6229.clone.13 | 5 |
| wide.region_3703.6230.clone.13 | 3 |
| wide.region_3702.6229.clone.clone.12 | 5 |
| wide.region_3703.6230.clone.clone.12 | 3 |
| wide.region_3698.6218.clone.13 | 5 |
| wide.region_3699.6219.clone.13 | 3 |
| wide.region_3698.6218.clone.clone.12 | 5 |
| wide.region_3699.6219.clone.clone.12 | 3 |
| region_5525.8287 | 11 |
| region_5526.8288 | 3 |
| wide.region_3698.6218.clone.14 | 5 |
| wide.region_3699.6219.clone.14 | 3 |
| wide.region_3698.6218.clone.clone.13 | 5 |
| wide.region_3699.6219.clone.clone.13 | 3 |
| region_5536.8298 | 11 |
| region_5537.8299 | 3 |
| wide.wide.region_5404.8318.clone.clone.sunk.clone.clone.clone | 111 |
| wide.wide.region_5555.8319.clone.clone.clone.clone.clone | 3 |
| wide.region_719.1905.clone.clone.44.clone.clone | 8 |
| wide.region_720.1906.clone.clone.44.clone.clone | 3 |
| wide.region_757.1994.clone.45.clone | 9 |
| wide.region_758.1995.clone.45.clone | 3 |
| wide.region_765.2018.clone.45.clone | 15 |
| wide.region_769.2019.clone.45.clone | 3 |
| wide.region_898.2341.clone.45.clone | 15 |
| wide.region_902.2342.clone.45.clone | 3 |
| region_5606.8374 | 12 |
| region_5611.8375 | 3 |
| wide.region_719.1905.clone.2.clone.94.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.94.clone.clone | 3 |
| wide.region_757.1994.clone.clone.94.clone | 9 |
| wide.region_758.1995.clone.clone.94.clone | 3 |
| wide.region_765.2018.clone.clone.94.clone | 15 |
| wide.region_769.2019.clone.clone.94.clone | 3 |
| wide.region_898.2341.clone.clone.94.clone | 15 |
| wide.region_902.2342.clone.clone.94.clone | 3 |
| region_5662.8431 | 12 |
| region_5667.8432 | 3 |
| wide.region_3702.6229.clone.18 | 5 |
| wide.region_3703.6230.clone.18 | 3 |
| wide.region_3702.6229.clone.clone.17 | 5 |
| wide.region_3703.6230.clone.clone.17 | 3 |
| wide.wide.region_4448.7099.clone.6.clone.clone | 6 |
| wide.wide.region_4449.7100.clone.6.clone.clone | 3 |
| wide.wide.region_4448.7099.clone.clone.5.clone.clone | 6 |
| wide.wide.region_4449.7100.clone.clone.5.clone.clone | 3 |
| wide.region_3702.6229.clone.17 | 5 |
| wide.region_3703.6230.clone.17 | 3 |
| wide.region_3702.6229.clone.clone.16 | 5 |
| wide.region_3703.6230.clone.clone.16 | 3 |
| wide.wide.region_4448.7099.clone.5.clone.clone | 6 |
| wide.wide.region_4449.7100.clone.5.clone.clone | 3 |
| wide.wide.region_4448.7099.clone.clone.4.clone.clone | 6 |
| wide.wide.region_4449.7100.clone.clone.4.clone.clone | 3 |
| wide.region_3702.6229.clone.16 | 5 |
| wide.region_3703.6230.clone.16 | 3 |
| wide.region_3702.6229.clone.clone.15 | 5 |
| wide.region_3703.6230.clone.clone.15 | 3 |
| wide.wide.region_4448.7099.clone.4.clone.clone | 6 |
| wide.wide.region_4449.7100.clone.4.clone.clone | 3 |
| wide.wide.region_4448.7099.clone.clone.3.clone.clone | 6 |
| wide.wide.region_4449.7100.clone.clone.3.clone.clone | 3 |
| wide.region_5742.8507.clone | 5 |
| wide.region_5743.8508.clone | 2 |
| wide.region_3698.6218.clone.16 | 5 |
| wide.region_3699.6219.clone.16 | 3 |
| wide.region_3698.6218.clone.clone.15 | 5 |
| wide.region_3699.6219.clone.clone.15 | 3 |
| wide.wide.region_4450.7110.clone.4.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.4.clone.clone | 3 |
| wide.wide.region_4450.7110.clone.clone.3.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.clone.3.clone.clone | 3 |
| region_5718.8484 | 11 |
| region_5719.8485 | 3 |
| wide.wide.region_5730.8496.clone.clone.clone | 9 |
| wide.wide.region_5731.8497.clone.clone.clone | 3 |
| region_5739.8505 | 7 |
| region_5740.8506 | 1 |
| region_5741.8509 | 18 |
| region_5744.8510 | 2 |
| wide.region_3698.6218.clone.17 | 5 |
| wide.region_3699.6219.clone.17 | 3 |
| wide.region_3698.6218.clone.clone.16 | 5 |
| wide.region_3699.6219.clone.clone.16 | 3 |
| wide.wide.region_4450.7110.clone.5.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.5.clone.clone | 3 |
| wide.wide.region_4450.7110.clone.clone.4.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.clone.4.clone.clone | 3 |
| wide.wide.wide.region_5714.8514.sunk.clone.clone.clone.clone.clone.clone.clone | 248 |
| wide.wide.wide.region_5748.8515.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_5753.8520 | 11 |
| region_5754.8521 | 3 |
| wide.wide.region_5765.8532.clone.clone.clone | 9 |
| wide.wide.region_5766.8533.clone.clone.clone | 3 |
| wide.region_719.1905.clone.clone.46.clone.clone | 8 |
| wide.region_720.1906.clone.clone.46.clone.clone | 3 |
| wide.region_757.1994.clone.47.clone | 9 |
| wide.region_758.1995.clone.47.clone | 3 |
| wide.region_765.2018.clone.47.clone | 15 |
| wide.region_769.2019.clone.47.clone | 3 |
| wide.region_898.2341.clone.47.clone | 15 |
| wide.region_902.2342.clone.47.clone | 3 |
| region_5823.8595 | 12 |
| region_5828.8596 | 3 |
| wide.region_719.1905.clone.2.clone.98.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.98.clone.clone | 3 |
| wide.region_757.1994.clone.clone.98.clone | 9 |
| wide.region_758.1995.clone.clone.98.clone | 3 |
| wide.region_765.2018.clone.clone.98.clone | 15 |
| wide.region_769.2019.clone.clone.98.clone | 3 |
| wide.region_898.2341.clone.clone.98.clone | 15 |
| wide.region_902.2342.clone.clone.98.clone | 3 |
| region_5879.8652 | 12 |
| region_5884.8653 | 3 |
| wide.region_3702.6229.clone.19 | 5 |
| wide.region_3703.6230.clone.19 | 3 |
| wide.region_3702.6229.clone.clone.18 | 5 |
| wide.region_3703.6230.clone.clone.18 | 3 |
| wide.region_3698.6218.clone.19 | 5 |
| wide.region_3699.6219.clone.19 | 3 |
| wide.region_3698.6218.clone.clone.18 | 5 |
| wide.region_3699.6219.clone.clone.18 | 3 |
| wide.wide.wide.wide.wide.wide.region_5713.8541.sunk.sunk.sunk.clone.clone.clone.clone | 259 |
| wide.wide.wide.wide.wide.wide.region_5774.8542.clone.clone.clone.clone | 2 |
| region_5896.8666 | 11 |
| region_5897.8667 | 3 |
| wide.region_3698.6218.clone.18 | 5 |
| wide.region_3699.6219.clone.18 | 3 |
| wide.region_3698.6218.clone.clone.17 | 5 |
| wide.region_3699.6219.clone.clone.17 | 3 |
| wide.wide.region_4450.7110.clone.6.clone.sunk.clone | 6 |
| wide.wide.region_4451.7111.clone.6.clone.clone | 3 |
| wide.wide.region_4450.7110.clone.clone.5.clone.sunk.clone | 6 |
| wide.wide.region_4451.7111.clone.clone.5.clone.clone | 3 |
| region_5676.8445.clone.clone | 20 |
| region_5679.8446.clone.clone | 3 |
| region_5692.8460 | 11 |
| region_5693.8461 | 3 |
| wide.wide.region_5704.8472.clone.clone.sunk.clone | 9 |
| wide.wide.region_5705.8473.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.wide.wide.wide.region_5680.8678.clone.sunk.clone.clone.sunk.clone.clone.clone.clone.clone.clone | 130 |
| wide.wide.wide.wide.wide.wide.wide.wide.region_5907.8679.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_719.1905.clone.2.clone.100.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.100.clone.clone | 3 |
| wide.region_757.1994.clone.clone.100.clone | 9 |
| wide.region_758.1995.clone.clone.100.clone | 3 |
| wide.region_765.2018.clone.clone.100.clone | 15 |
| wide.region_769.2019.clone.clone.100.clone | 3 |
| wide.region_898.2341.clone.clone.100.clone | 15 |
| wide.region_902.2342.clone.clone.100.clone | 3 |
| region_5970.8747 | 12 |
| region_5975.8748 | 3 |
| wide.region_3702.6229.clone.20 | 5 |
| wide.region_3703.6230.clone.20 | 3 |
| wide.region_3702.6229.clone.clone.19 | 5 |
| wide.region_3703.6230.clone.clone.19 | 3 |
| wide.region_3698.6218.clone.20 | 5 |
| wide.region_3699.6219.clone.20 | 3 |
| wide.region_3698.6218.clone.clone.19 | 5 |
| wide.region_3699.6219.clone.clone.19 | 3 |
| region_5985.8759 | 11 |
| region_5986.8760 | 3 |
| wide.wide.region_4450.7110.clone.7.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.7.clone.clone | 3 |
| wide.wide.region_4450.7110.clone.clone.6.clone.clone | 6 |
| wide.wide.region_4451.7111.clone.clone.6.clone.clone | 3 |
| region_5112.7870.clone.clone | 20 |
| region_5115.7871.clone.clone | 3 |
| wide.wide.region_5122.7884.clone.clone.clone | 9 |
| wide.wide.region_5128.7885.clone.clone.clone | 3 |
| wide.wide.wide.wide.region_5136.8781.clone.clone.clone.clone.clone.clone.clone.clone | 177 |
| wide.wide.wide.wide.region_6007.8782.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| wide.region_3702.6229.clone.21 | 5 |
| wide.region_3703.6230.clone.21 | 3 |
| wide.region_3702.6229.clone.clone.20 | 5 |
| wide.region_3703.6230.clone.clone.20 | 3 |
| wide.region_3702.6229.clone.22 | 5 |
| wide.region_3703.6230.clone.22 | 3 |
| wide.region_3702.6229.clone.clone.21 | 5 |
| wide.region_3703.6230.clone.clone.21 | 3 |
| wide.region_719.1905.clone.clone.26.clone.clone | 8 |
| wide.region_720.1906.clone.clone.26.clone.clone | 3 |
| wide.region_757.1994.clone.27.clone | 9 |
| wide.region_758.1995.clone.27.clone | 3 |
| wide.region_765.2018.clone.27.clone | 15 |
| wide.region_769.2019.clone.27.clone | 3 |
| wide.region_898.2341.clone.27.clone | 15 |
| wide.region_902.2342.clone.27.clone | 3 |
| region_3987.6605 | 12 |
| region_3992.6606 | 3 |
| wide.region_719.1905.clone.2.clone.58.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.58.clone.clone | 3 |
| wide.region_757.1994.clone.clone.58.clone | 9 |
| wide.region_758.1995.clone.clone.58.clone | 3 |
| wide.region_765.2018.clone.clone.58.clone | 15 |
| wide.region_769.2019.clone.clone.58.clone | 3 |
| wide.region_898.2341.clone.clone.58.clone | 15 |
| wide.region_902.2342.clone.clone.58.clone | 3 |
| region_4043.6662 | 12 |
| region_4048.6663 | 3 |
| wide.region_719.1905.clone.clone.24.clone.clone | 8 |
| wide.region_720.1906.clone.clone.24.clone.clone | 3 |
| wide.region_757.1994.clone.25.clone | 9 |
| wide.region_758.1995.clone.25.clone | 3 |
| wide.region_765.2018.clone.25.clone | 15 |
| wide.region_769.2019.clone.25.clone | 3 |
| wide.region_898.2341.clone.25.clone | 15 |
| wide.region_902.2342.clone.25.clone | 3 |
| region_3869.6483 | 12 |
| region_3874.6484 | 3 |
| wide.region_719.1905.clone.2.clone.54.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.54.clone.clone | 3 |
| wide.region_757.1994.clone.clone.54.clone | 9 |
| wide.region_758.1995.clone.clone.54.clone | 3 |
| wide.region_765.2018.clone.clone.54.clone | 15 |
| wide.region_769.2019.clone.clone.54.clone | 3 |
| wide.region_898.2341.clone.clone.54.clone | 15 |
| wide.region_902.2342.clone.clone.54.clone | 3 |
| region_3925.6540 | 12 |
| region_3930.6541 | 3 |
| wide.region_3702.6229.clone.3 | 5 |
| wide.region_3703.6230.clone.3 | 3 |
| wide.region_3702.6229.clone.clone.2 | 5 |
| wide.region_3703.6230.clone.clone.2 | 3 |
| wide.region_3698.6218.clone.3 | 5 |
| wide.region_3699.6219.clone.3 | 3 |
| wide.region_3698.6218.clone.clone.2 | 5 |
| wide.region_3699.6219.clone.clone.2 | 3 |
| region_4061.6676 | 11 |
| region_4062.6677 | 3 |
| wide.region_3698.6218.clone.21 | 5 |
| wide.region_3699.6219.clone.21 | 3 |
| wide.region_3698.6218.clone.clone.20 | 5 |
| wide.region_3699.6219.clone.clone.20 | 3 |
| wide.region_3698.6218.clone.22 | 5 |
| wide.region_3699.6219.clone.22 | 3 |
| wide.region_3698.6218.clone.clone.21 | 5 |
| wide.region_3699.6219.clone.clone.21 | 3 |
| wide.region_719.1905.clone.2.clone.102.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.102.clone.clone | 3 |
| wide.region_757.1994.clone.clone.102.clone | 9 |
| wide.region_758.1995.clone.clone.102.clone | 3 |
| wide.region_3821.6552.clone.clone.clone | 14 |
| wide.region_3939.6553.clone.clone.clone | 3 |
| wide.wide.region_4057.6685.clone.clone.clone | 91 |
| wide.wide.region_4069.6686.clone.clone.clone | 3 |
| region_4073.6690 | 11 |
| region_4074.6691 | 3 |
| region_5088.7844 | 11 |
| region_5089.7845 | 3 |
| wide.region_765.2018.clone.clone.102.clone | 15 |
| wide.region_769.2019.clone.clone.102.clone | 3 |
| wide.region_898.2341.clone.clone.102.clone | 15 |
| wide.region_902.2342.clone.clone.102.clone | 3 |
| region_5073.7833 | 12 |
| region_5078.7834 | 3 |
| wide.region_3702.6229.clone.23 | 5 |
| wide.region_3703.6230.clone.23 | 3 |
| wide.region_3702.6229.clone.clone.22 | 5 |
| wide.region_3703.6230.clone.clone.22 | 3 |
| wide.region_3702.6229.clone.24 | 5 |
| wide.region_3703.6230.clone.24 | 3 |
| wide.region_3702.6229.clone.clone.23 | 5 |
| wide.region_3703.6230.clone.clone.23 | 3 |
| wide.region_3702.6229.clone.25 | 5 |
| wide.region_3703.6230.clone.25 | 3 |
| wide.region_3702.6229.clone.clone.24 | 5 |
| wide.region_3703.6230.clone.clone.24 | 3 |
| wide.region_3702.6229 | 5 |
| wide.region_3703.6230 | 3 |
| wide.region_3702.6229.clone | 5 |
| wide.region_3703.6230.clone | 3 |
| wide.region_3698.6218 | 5 |
| wide.region_3699.6219 | 3 |
| wide.region_3698.6218.clone | 5 |
| wide.region_3699.6219.clone | 3 |
| region_3728.6283 | 11 |
| region_3729.6284 | 3 |
| wide.region_3702.6229.clone.2 | 5 |
| wide.region_3703.6230.clone.2 | 3 |
| wide.region_3702.6229.clone.clone.1 | 5 |
| wide.region_3703.6230.clone.clone.1 | 3 |
| wide.region_3702.6229.clone.1 | 5 |
| wide.region_3703.6230.clone.1 | 3 |
| wide.region_3702.6229.clone.clone | 5 |
| wide.region_3703.6230.clone.clone | 3 |
| wide.region_3769.6378.clone | 5 |
| wide.region_3770.6379.clone | 2 |
| wide.region_3698.6218.clone.1 | 5 |
| wide.region_3699.6219.clone.1 | 3 |
| wide.region_3698.6218.clone.clone | 5 |
| wide.region_3699.6219.clone.clone | 3 |
| region_3760.6370 | 11 |
| region_3761.6371 | 3 |
| region_3766.6376 | 7 |
| region_3767.6377 | 1 |
| region_3768.6380 | 18 |
| region_3771.6381 | 2 |
| wide.region_3698.6218.clone.2 | 5 |
| wide.region_3699.6219.clone.2 | 3 |
| wide.region_3698.6218.clone.clone.1 | 5 |
| wide.region_3699.6219.clone.clone.1 | 3 |
| wide.wide.region_3756.6385.clone.clone.clone | 113 |
| wide.wide.region_3775.6386.clone.clone.clone | 2 |
| region_3780.6391 | 11 |
| region_3781.6392 | 3 |
| wide.region_3698.6218.clone.23 | 5 |
| wide.region_3699.6219.clone.23 | 3 |
| wide.region_3698.6218.clone.clone.22 | 5 |
| wide.region_3699.6219.clone.clone.22 | 3 |
| wide.region_3698.6218.clone.24 | 5 |
| wide.region_3699.6219.clone.24 | 3 |
| wide.region_3698.6218.clone.clone.23 | 5 |
| wide.region_3699.6219.clone.clone.23 | 3 |
| wide.region_3698.6218.clone.25 | 5 |
| wide.region_3699.6219.clone.25 | 3 |
| wide.region_3698.6218.clone.clone.24 | 5 |
| wide.region_3699.6219.clone.clone.24 | 3 |
| wide.wide.region_3724.6294.clone.clone.clone.clone.clone.clone | 113 |
| wide.wide.region_3738.6295.clone.clone.clone.clone.clone | 3 |
| region_3711.6263 | 11 |
| region_3716.6264 | 3 |
| region_3749.6361 | 11 |
| region_3750.6362 | 3 |
| wide.wide.wide.wide.region_3755.6397.clone.sunk.clone.clone.clone | 119 |
| wide.wide.wide.wide.region_3786.6398.clone.clone.clone.clone | 2 |
| region_3790.6402 | 11 |
| region_3791.6403 | 3 |
| wide.region_226.598.clone.clone.6.sunk.clone | 8 |
| wide.region_227.599.clone.clone.6.clone | 3 |
| wide.region_377.932.clone.7.sunk.clone | 9 |
| wide.region_378.933.clone.7.clone | 3 |
| wide.region_377.932.clone.clone.6.sunk.clone | 9 |
| wide.region_378.933.clone.clone.6.clone | 3 |
| wide.region_379.957.clone.13.sunk.clone.clone.clone | 12 |
| wide.region_380.958.clone.13.clone.clone.clone | 3 |
| wide.region_379.957.clone.clone.6.sunk.clone.clone.clone | 12 |
| wide.region_380.958.clone.clone.6.clone.clone.clone | 3 |
| wide.region_284.744.clone.6.clone | 15 |
| wide.region_288.745.clone.6.clone | 3 |
| wide.region_479.1355.clone.6.clone | 15 |
| wide.region_483.1356.clone.6.clone | 3 |
| region_3678.6177 | 25 |
| region_3683.6178 | 3 |
| wide.region_17.65.clone.2.clone.5.clone | 8 |
| wide.region_18.66.clone.2.clone.5.clone | 3 |
| wide.region_17.65.clone.2.clone.clone.clone | 8 |
| wide.region_18.66.clone.2.clone.clone.clone | 3 |
| wide.region_17.65.clone.4.clone.1.clone.clone | 8 |
| wide.region_18.66.clone.4.clone.1.clone.clone | 3 |
| wide.region_17.65.clone.4.clone.2.clone.clone | 8 |
| wide.region_18.66.clone.4.clone.2.clone.clone | 3 |
| wide.region_17.65.clone.2.clone.1.clone.clone | 8 |
| wide.region_18.66.clone.2.clone.1.clone.clone | 3 |
| wide.region_276.720.clone.6.clone | 9 |
| wide.region_277.721.clone.6.clone | 3 |
| wide.wide.wide.wide.region_3631.8786.clone.sunk.clone.clone.clone.clone.clone.clone.clone | 575 |
| wide.wide.wide.wide.region_6008.8787.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| wide.region_65.183.clone.1.clone.1.clone | 15 |
| wide.region_69.184.clone.1.clone.1.clone | 3 |
| wide.region_198.515.clone.1.clone.1.clone | 15 |
| wide.region_202.516.clone.1.clone.1.clone | 3 |
| wide.region_65.183.clone.clone.5 | 15 |
| wide.region_69.184.clone.clone.5 | 3 |
| wide.region_65.183.clone.clone.1.clone | 15 |
| wide.region_69.184.clone.clone.1.clone | 3 |
| wide.region_65.183.clone.clone.clone | 15 |
| wide.region_69.184.clone.clone.clone | 3 |
| region_6128.8914 | 12 |
| region_6133.8915 | 3 |
| wide.region_198.515.clone.clone.5 | 15 |
| wide.region_202.516.clone.clone.5 | 3 |
| wide.region_198.515.clone.clone.1.clone | 15 |
| wide.region_202.516.clone.clone.1.clone | 3 |
| wide.region_198.515.clone.clone.clone | 15 |
| wide.region_202.516.clone.clone.clone | 3 |
| region_6040.8824 | 12 |
| region_6045.8825 | 3 |
| region_6084.8869 | 12 |
| region_6089.8870 | 3 |
| wide.region_65.183.clone.1.clone.2.clone | 15 |
| wide.region_69.184.clone.1.clone.2.clone | 3 |
| wide.region_198.515.clone.1.clone.2.clone | 15 |
| wide.region_202.516.clone.1.clone.2.clone | 3 |
| region_6171.8958 | 12 |
| region_6176.8959 | 3 |
| wide.region_577.1614.clone.5 | 5 |
| wide.region_578.1615.clone.5 | 3 |
| wide.region_577.1614.clone.clone.4 | 5 |
| wide.region_578.1615.clone.clone.4 | 3 |
| wide.region_577.1614.clone.4 | 5 |
| wide.region_578.1615.clone.4 | 3 |
| wide.region_577.1614.clone.clone.3 | 5 |
| wide.region_578.1615.clone.clone.3 | 3 |
| wide.region_719.1905.clone.2.clone.8.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.8.clone.clone | 3 |
| wide.region_573.1603.clone.4 | 5 |
| wide.region_574.1604.clone.4 | 3 |
| wide.region_573.1603.clone.clone.3 | 5 |
| wide.region_574.1604.clone.clone.3 | 3 |
| region_1301.2779 | 11 |
| region_1302.2780 | 3 |
| wide.region_765.2018.clone.clone.8.clone | 15 |
| wide.region_769.2019.clone.clone.8.clone | 3 |
| wide.region_898.2341.clone.clone.8.clone | 15 |
| wide.region_902.2342.clone.clone.8.clone | 3 |
| region_1284.2765.sunk.sunk.clone.clone | 16 |
| region_1289.2766.clone.clone | 3 |
| wide.region_573.1603.clone.5 | 5 |
| wide.region_574.1604.clone.5 | 3 |
| wide.region_573.1603.clone.clone.4 | 5 |
| wide.region_574.1604.clone.clone.4 | 3 |
| wide.region_757.1994.clone.clone.8.clone.sunk.clone | 11 |
| wide.region_758.1995.clone.clone.8.clone.clone | 3 |
| region_1312.2790 | 11 |
| region_1313.2791 | 3 |
| wide.wide.wide.wide.wide.wide.region_1180.2810.clone.clone.sunk.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone.clone | 402 |
| wide.wide.wide.wide.wide.wide.region_1331.2811.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_1624.3225.clone | 5 |
| wide.region_1625.3226.clone | 2 |
| wide.region_573.1603.clone.7 | 5 |
| wide.region_574.1604.clone.7 | 3 |
| wide.region_573.1603.clone.clone.6 | 5 |
| wide.region_574.1604.clone.clone.6 | 3 |
| wide.wide.region_1545.3066.clone.4.clone | 6 |
| wide.wide.region_1546.3067.clone.4.clone | 3 |
| wide.wide.region_1545.3066.clone.clone.3.clone | 6 |
| wide.wide.region_1546.3067.clone.clone.3.clone | 3 |
| region_1600.3202 | 11 |
| region_1601.3203 | 3 |
| wide.wide.region_1612.3214.clone.clone.clone | 9 |
| wide.wide.region_1613.3215.clone.clone.clone | 3 |
| region_1621.3223 | 7 |
| region_1622.3224 | 1 |
| region_1623.3227 | 18 |
| region_1626.3228 | 2 |
| wide.region_573.1603.clone.8 | 5 |
| wide.region_574.1604.clone.8 | 3 |
| wide.region_573.1603.clone.clone.7 | 5 |
| wide.region_574.1604.clone.clone.7 | 3 |
| wide.wide.region_1545.3066.clone.1.clone.clone | 6 |
| wide.wide.region_1546.3067.clone.1.clone.clone | 3 |
| wide.wide.region_1545.3066.clone.clone.clone.clone | 6 |
| wide.wide.region_1546.3067.clone.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.wide.region_1596.3232.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone.clone | 251 |
| wide.wide.wide.wide.wide.wide.region_1630.3233.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_1635.3238 | 11 |
| region_1636.3239 | 3 |
| wide.wide.region_1647.3250.clone.clone.clone | 9 |
| wide.wide.region_1648.3251.clone.clone.clone | 3 |
| wide.region_719.1905.clone.2.clone.14.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.14.clone.clone | 3 |
| wide.region_573.1603.clone.10 | 5 |
| wide.region_574.1604.clone.10 | 3 |
| wide.region_573.1603.clone.clone.9 | 5 |
| wide.region_574.1604.clone.clone.9 | 3 |
| wide.wide.wide.wide.wide.wide.wide.region_1595.3259.sunk.sunk.sunk.sunk.sunk.clone.clone.clone.clone | 246 |
| wide.wide.wide.wide.wide.wide.wide.region_1656.3260.clone.clone.clone.clone | 2 |
| region_1778.3384 | 11 |
| region_1779.3385 | 3 |
| wide.region_765.2018.clone.clone.14.clone | 15 |
| wide.region_769.2019.clone.clone.14.clone | 3 |
| wide.region_898.2341.clone.clone.14.clone | 15 |
| wide.region_902.2342.clone.clone.14.clone | 3 |
| region_1761.3370.sunk.sunk.clone.clone | 16 |
| region_1766.3371.clone.clone | 3 |
| wide.region_226.598.clone.clone.clone.sunk.clone | 8 |
| wide.region_227.599.clone.clone.clone.clone | 3 |
| wide.region_377.932.clone.1.clone.sunk.clone | 9 |
| wide.region_378.933.clone.1.clone.clone | 3 |
| wide.region_377.932.clone.clone.clone.sunk.clone | 9 |
| wide.region_378.933.clone.clone.clone.clone | 3 |
| wide.region_719.1905.clone.2.clone.16.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.16.clone.clone | 3 |
| wide.region_379.957.clone.7.clone.sunk.clone.clone | 12 |
| wide.region_380.958.clone.7.clone.clone.clone | 3 |
| wide.region_379.957.clone.clone.clone.sunk.clone.clone | 12 |
| wide.region_380.958.clone.clone.clone.clone.clone | 3 |
| wide.wide.region_1545.3066.clone.3.clone.clone | 6 |
| wide.wide.region_1546.3067.clone.3.clone.clone | 3 |
| wide.wide.region_1545.3066.clone.clone.2.clone.clone | 6 |
| wide.wide.region_1546.3067.clone.clone.2.clone.clone | 3 |
| wide.region_276.720.clone.sunk.clone.clone | 11 |
| wide.region_277.721.clone.clone.clone | 3 |
| wide.region_765.2018.clone.clone.16.clone | 15 |
| wide.region_769.2019.clone.clone.16.clone | 3 |
| wide.region_284.744.clone.clone | 15 |
| wide.region_288.745.clone.clone | 3 |
| wide.region_898.2341.clone.clone.16.clone | 15 |
| wide.region_902.2342.clone.clone.16.clone | 3 |
| wide.region_479.1355.clone.clone | 15 |
| wide.region_483.1356.clone.clone | 3 |
| region_1436.2922.sunk.clone | 16 |
| region_1441.2923.clone | 3 |
| region_1381.2866.sunk.clone | 29 |
| region_1386.2867.clone | 3 |
| region_1508.2994.clone.clone | 20 |
| region_1511.2995.clone.clone | 3 |
| region_1570.3178 | 11 |
| region_1575.3179 | 3 |
| wide.wide.region_1518.3023.clone.clone.clone | 9 |
| wide.wide.region_1524.3024.clone.clone.clone | 3 |
| wide.wide.region_1586.3190.clone.clone.sunk.clone | 9 |
| wide.wide.region_1587.3191.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.wide.wide.wide.region_1556.3396.sunk.clone.sunk.clone.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone.clone.clone | 415 |
| wide.wide.wide.wide.wide.wide.wide.wide.region_1789.3397.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_719.1905.clone.2.clone.20.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.20.clone.clone | 3 |
| wide.region_765.2018.clone.clone.20.clone | 15 |
| wide.region_769.2019.clone.clone.20.clone | 3 |
| wide.region_898.2341.clone.clone.20.clone | 15 |
| wide.region_902.2342.clone.clone.20.clone | 3 |
| region_2280.4567.sunk.sunk.clone.clone | 16 |
| region_2285.4568.clone.clone | 3 |
| wide.region_1803.3450.clone.clone.clone.sunk.clone | 8 |
| wide.region_1804.3451.clone.clone.clone.clone | 3 |
| wide.region_1953.3778.clone.1.clone.sunk.clone | 9 |
| wide.region_1954.3779.clone.1.clone.clone | 3 |
| wide.region_1953.3778.clone.clone.clone.sunk.clone | 9 |
| wide.region_1954.3779.clone.clone.clone.clone | 3 |
| wide.region_719.1905.clone.2.clone.24.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.24.clone.clone | 3 |
| wide.region_1955.3802.clone.7.clone.sunk.clone.clone | 12 |
| wide.region_1956.3803.clone.7.clone.clone.clone | 3 |
| wide.region_1955.3802.clone.clone.clone.sunk.clone.clone | 12 |
| wide.region_1956.3803.clone.clone.clone.clone.clone | 3 |
| wide.wide.region_2163.4418.clone.2.clone | 6 |
| wide.wide.region_2164.4419.clone.2.clone | 3 |
| wide.wide.region_2163.4418.clone.clone.1.clone | 6 |
| wide.wide.region_2164.4419.clone.clone.1.clone | 3 |
| wide.region_1852.3568.clone.sunk.clone.clone | 11 |
| wide.region_1853.3569.clone.clone.clone | 3 |
| wide.region_765.2018.clone.clone.24.clone | 15 |
| wide.region_769.2019.clone.clone.24.clone | 3 |
| wide.region_1860.3592.clone.clone | 15 |
| wide.region_1864.3593.clone.clone | 3 |
| wide.region_898.2341.clone.clone.24.clone | 15 |
| wide.region_902.2342.clone.clone.24.clone | 3 |
| wide.region_2055.4223.clone.clone | 15 |
| wide.region_2059.4224.clone.clone | 3 |
| region_2340.4631.sunk.clone | 16 |
| region_2345.4632.clone | 3 |
| region_2118.4332.sunk.clone | 28 |
| region_2123.4333.clone | 3 |
| wide.wide.region_2136.4375.clone.clone.clone | 9 |
| wide.wide.region_2142.4376.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_2176.4579.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone | 303 |
| wide.wide.wide.wide.wide.region_2294.4580.clone.clone.clone.clone.clone.clone | 3 |
| wide.wide.region_2547.4868.clone.3.clone.clone | 6 |
| wide.wide.region_2548.4869.clone.3.clone.clone | 3 |
| wide.wide.region_2547.4868.clone.clone.2.clone.clone | 6 |
| wide.wide.region_2548.4869.clone.clone.2.clone.clone | 3 |
| wide.region_577.1614.clone.12 | 5 |
| wide.region_578.1615.clone.12 | 3 |
| wide.region_577.1614.clone.clone.11 | 5 |
| wide.region_578.1615.clone.clone.11 | 3 |
| wide.region_577.1614.clone.11 | 5 |
| wide.region_578.1615.clone.11 | 3 |
| wide.region_577.1614.clone.clone.10 | 5 |
| wide.region_578.1615.clone.clone.10 | 3 |
| wide.region_573.1603.clone.11 | 5 |
| wide.region_574.1604.clone.11 | 3 |
| wide.region_573.1603.clone.clone.10 | 5 |
| wide.region_574.1604.clone.clone.10 | 3 |
| region_2804.5163 | 11 |
| region_2805.5164 | 3 |
| wide.region_719.1905.clone.2.clone.28.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.28.clone.clone | 3 |
| wide.region_765.2018.clone.clone.28.clone | 15 |
| wide.region_769.2019.clone.clone.28.clone | 3 |
| wide.region_898.2341.clone.clone.28.clone | 15 |
| wide.region_902.2342.clone.clone.28.clone | 3 |
| region_2668.5027.sunk.sunk.clone.clone | 16 |
| region_2673.5028.clone.clone | 3 |
| wide.region_573.1603.clone.12 | 5 |
| wide.region_574.1604.clone.12 | 3 |
| wide.region_573.1603.clone.clone.11 | 5 |
| wide.region_574.1604.clone.clone.11 | 3 |
| wide.region_719.1905.clone.2.clone.32.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.32.clone.clone | 3 |
| wide.region_757.1994.clone.clone.28.clone.sunk.clone | 11 |
| wide.region_758.1995.clone.clone.28.clone.clone | 3 |
| wide.wide.region_2800.5172.clone.clone.clone | 107 |
| wide.wide.region_2812.5173.clone.clone.clone | 3 |
| region_2816.5177 | 11 |
| region_2817.5178 | 3 |
| wide.wide.wide.wide.wide.region_2564.5039.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone | 303 |
| wide.wide.wide.wide.wide.region_2682.5040.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_765.2018.clone.clone.32.clone | 15 |
| wide.region_769.2019.clone.clone.32.clone | 3 |
| wide.region_898.2341.clone.clone.32.clone | 15 |
| wide.region_902.2342.clone.clone.32.clone | 3 |
| region_2728.5091.sunk.clone | 16 |
| region_2733.5092.clone | 3 |
| wide.region_577.1614.clone.14 | 5 |
| wide.region_578.1615.clone.14 | 3 |
| wide.region_577.1614.clone.clone.13 | 5 |
| wide.region_578.1615.clone.clone.13 | 3 |
| wide.region_577.1614.clone.13 | 5 |
| wide.region_578.1615.clone.13 | 3 |
| wide.region_577.1614.clone.clone.12 | 5 |
| wide.region_578.1615.clone.clone.12 | 3 |
| wide.region_719.1905.clone.2.clone.36.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.36.clone.clone | 3 |
| wide.region_573.1603.clone.13 | 5 |
| wide.region_574.1604.clone.13 | 3 |
| wide.region_573.1603.clone.clone.12 | 5 |
| wide.region_574.1604.clone.clone.12 | 3 |
| region_2949.5311 | 11 |
| region_2950.5312 | 3 |
| wide.region_765.2018.clone.clone.36.clone | 15 |
| wide.region_769.2019.clone.clone.36.clone | 3 |
| wide.region_898.2341.clone.clone.36.clone | 15 |
| wide.region_902.2342.clone.clone.36.clone | 3 |
| region_2932.5297.sunk.sunk.clone.clone | 16 |
| region_2937.5298.clone.clone | 3 |
| wide.region_573.1603.clone.14 | 5 |
| wide.region_574.1604.clone.14 | 3 |
| wide.region_573.1603.clone.clone.13 | 5 |
| wide.region_574.1604.clone.clone.13 | 3 |
| wide.region_757.1994.clone.clone.36.clone.sunk.clone | 11 |
| wide.region_758.1995.clone.clone.36.clone.clone | 3 |
| region_2960.5322 | 11 |
| region_2961.5323 | 3 |
| wide.wide.wide.wide.wide.wide.region_2828.5342.clone.clone.sunk.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone.clone | 402 |
| wide.wide.wide.wide.wide.wide.region_2979.5343.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_577.1614.clone.18 | 5 |
| wide.region_578.1615.clone.18 | 3 |
| wide.region_577.1614.clone.clone.17 | 5 |
| wide.region_578.1615.clone.clone.17 | 3 |
| wide.wide.region_2547.4868.clone.2.clone.clone | 6 |
| wide.wide.region_2548.4869.clone.2.clone.clone | 3 |
| wide.wide.region_2547.4868.clone.clone.1.clone.clone | 6 |
| wide.wide.region_2548.4869.clone.clone.1.clone.clone | 3 |
| wide.region_577.1614.clone.17 | 5 |
| wide.region_578.1615.clone.17 | 3 |
| wide.region_577.1614.clone.clone.16 | 5 |
| wide.region_578.1615.clone.clone.16 | 3 |
| wide.wide.region_2547.4868.clone.1.clone.clone | 6 |
| wide.wide.region_2548.4869.clone.1.clone.clone | 3 |
| wide.wide.region_2547.4868.clone.clone.clone.clone | 6 |
| wide.wide.region_2548.4869.clone.clone.clone.clone | 3 |
| wide.region_577.1614.clone.16 | 5 |
| wide.region_578.1615.clone.16 | 3 |
| wide.region_577.1614.clone.clone.15 | 5 |
| wide.region_578.1615.clone.clone.15 | 3 |
| wide.wide.region_2547.4868.clone.4.clone | 6 |
| wide.wide.region_2548.4869.clone.4.clone | 3 |
| wide.wide.region_2547.4868.clone.clone.3.clone | 6 |
| wide.wide.region_2548.4869.clone.clone.3.clone | 3 |
| wide.region_3168.5587.clone | 5 |
| wide.region_3169.5588.clone | 2 |
| wide.region_573.1603.clone.16 | 5 |
| wide.region_574.1604.clone.16 | 3 |
| wide.region_573.1603.clone.clone.15 | 5 |
| wide.region_574.1604.clone.clone.15 | 3 |
| wide.wide.region_2549.4879.clone.4.clone | 6 |
| wide.wide.region_2550.4880.clone.4.clone | 3 |
| wide.wide.region_2549.4879.clone.clone.3.clone | 6 |
| wide.wide.region_2550.4880.clone.clone.3.clone | 3 |
| region_3144.5564 | 11 |
| region_3145.5565 | 3 |
| wide.wide.region_3156.5576.clone.clone.clone | 9 |
| wide.wide.region_3157.5577.clone.clone.clone | 3 |
| region_3165.5585 | 7 |
| region_3166.5586 | 1 |
| region_3167.5589 | 18 |
| region_3170.5590 | 2 |
| wide.region_573.1603.clone.17 | 5 |
| wide.region_574.1604.clone.17 | 3 |
| wide.region_573.1603.clone.clone.16 | 5 |
| wide.region_574.1604.clone.clone.16 | 3 |
| wide.wide.region_2549.4879.clone.1.clone.clone | 6 |
| wide.wide.region_2550.4880.clone.1.clone.clone | 3 |
| wide.wide.region_2549.4879.clone.clone.clone.clone | 6 |
| wide.wide.region_2550.4880.clone.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_3140.5594.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone | 263 |
| wide.wide.wide.wide.wide.region_3174.5595.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_3179.5600 | 11 |
| region_3180.5601 | 3 |
| wide.wide.region_3191.5612.clone.clone.clone | 9 |
| wide.wide.region_3192.5613.clone.clone.clone | 3 |
| wide.region_577.1614.clone.19 | 5 |
| wide.region_578.1615.clone.19 | 3 |
| wide.region_577.1614.clone.clone.18 | 5 |
| wide.region_578.1615.clone.clone.18 | 3 |
| wide.region_719.1905.clone.2.clone.42.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.42.clone.clone | 3 |
| wide.region_573.1603.clone.19 | 5 |
| wide.region_574.1604.clone.19 | 3 |
| wide.region_573.1603.clone.clone.18 | 5 |
| wide.region_574.1604.clone.clone.18 | 3 |
| wide.wide.wide.wide.wide.wide.wide.region_3139.5621.sunk.sunk.sunk.sunk.sunk.clone.clone.clone.clone | 276 |
| wide.wide.wide.wide.wide.wide.wide.region_3200.5622.clone.clone.clone.clone | 2 |
| region_3322.5746 | 11 |
| region_3323.5747 | 3 |
| wide.region_765.2018.clone.clone.42.clone | 15 |
| wide.region_769.2019.clone.clone.42.clone | 3 |
| wide.region_898.2341.clone.clone.42.clone | 15 |
| wide.region_902.2342.clone.clone.42.clone | 3 |
| region_3305.5732.sunk.sunk.clone.clone | 16 |
| region_3310.5733.clone.clone | 3 |
| wide.region_573.1603.clone.18 | 5 |
| wide.region_574.1604.clone.18 | 3 |
| wide.region_573.1603.clone.clone.17 | 5 |
| wide.region_574.1604.clone.clone.17 | 3 |
| wide.region_719.1905.clone.2.clone.44.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.44.clone.clone | 3 |
| wide.wide.region_2549.4879.clone.2.clone.sunk.clone | 6 |
| wide.wide.region_2550.4880.clone.2.clone.clone | 3 |
| wide.wide.region_2549.4879.clone.clone.1.clone.sunk.clone | 6 |
| wide.wide.region_2550.4880.clone.clone.1.clone.clone | 3 |
| wide.region_757.1994.clone.clone.44.sunk.clone.clone | 11 |
| wide.region_758.1995.clone.clone.44.clone.clone | 3 |
| region_3100.5469.clone.clone | 20 |
| region_3103.5470.clone.clone | 3 |
| region_3118.5540 | 11 |
| region_3119.5541 | 3 |
| wide.region_765.2018.clone.clone.44.clone | 15 |
| wide.region_769.2019.clone.clone.44.clone | 3 |
| wide.wide.region_3130.5552.clone.clone.sunk.clone | 9 |
| wide.wide.region_3131.5553.clone.clone.clone | 3 |
| wide.region_898.2341.clone.clone.44.clone | 15 |
| wide.region_902.2342.clone.clone.44.clone | 3 |
| region_3028.5397.sunk.clone | 16 |
| region_3033.5398.clone | 3 |
| wide.wide.wide.wide.wide.wide.wide.wide.wide.wide.region_3104.5758.clone.sunk.clone.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone.clone.clone | 421 |
| wide.wide.wide.wide.wide.wide.wide.wide.wide.wide.region_3333.5759.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_719.1905.clone.2.clone.46.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.46.clone.clone | 3 |
| wide.region_757.1994.clone.clone.46.sunk.clone.clone | 11 |
| wide.region_758.1995.clone.clone.46.clone.clone | 3 |
| wide.region_765.2018.clone.clone.46.clone | 15 |
| wide.region_769.2019.clone.clone.46.clone | 3 |
| wide.region_898.2341.clone.clone.46.clone | 15 |
| wide.region_902.2342.clone.clone.46.clone | 3 |
| region_3396.5827.sunk.clone | 16 |
| region_3401.5828.clone | 3 |
| wide.region_577.1614.clone.20 | 5 |
| wide.region_578.1615.clone.20 | 3 |
| wide.region_577.1614.clone.clone.19 | 5 |
| wide.region_578.1615.clone.clone.19 | 3 |
| wide.region_573.1603.clone.20 | 5 |
| wide.region_574.1604.clone.20 | 3 |
| wide.region_573.1603.clone.clone.19 | 5 |
| wide.region_574.1604.clone.clone.19 | 3 |
| region_3411.5839 | 11 |
| region_3412.5840 | 3 |
| wide.wide.region_2549.4879.clone.3.clone.clone | 6 |
| wide.wide.region_2550.4880.clone.3.clone.clone | 3 |
| wide.wide.region_2549.4879.clone.clone.2.clone.clone | 6 |
| wide.wide.region_2550.4880.clone.clone.2.clone.clone | 3 |
| region_2512.4807.clone.clone | 20 |
| region_2515.4808.clone.clone | 3 |
| wide.wide.region_2522.4836.clone.clone.clone | 9 |
| wide.wide.region_2528.4837.clone.clone.clone | 3 |
| wide.wide.wide.wide.region_2560.5861.clone.clone.clone.clone.clone.clone.clone.clone | 193 |
| wide.wide.wide.wide.region_3433.5862.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| wide.region_577.1614.clone.21 | 5 |
| wide.region_578.1615.clone.21 | 3 |
| wide.region_577.1614.clone.clone.20 | 5 |
| wide.region_578.1615.clone.clone.20 | 3 |
| wide.region_577.1614.clone.22 | 5 |
| wide.region_578.1615.clone.22 | 3 |
| wide.region_577.1614.clone.clone.21 | 5 |
| wide.region_578.1615.clone.clone.21 | 3 |
| wide.region_577.1614.clone.3 | 5 |
| wide.region_578.1615.clone.3 | 3 |
| wide.region_577.1614.clone.clone.2 | 5 |
| wide.region_578.1615.clone.clone.2 | 3 |
| wide.region_573.1603.clone.3 | 5 |
| wide.region_574.1604.clone.3 | 3 |
| wide.region_573.1603.clone.clone.2 | 5 |
| wide.region_574.1604.clone.clone.2 | 3 |
| region_1156.2630 | 11 |
| region_1157.2631 | 3 |
| wide.region_719.1905.clone.2.clone.2.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.2.clone.clone | 3 |
| wide.region_765.2018.clone.clone.2.clone | 15 |
| wide.region_769.2019.clone.clone.2.clone | 3 |
| wide.region_898.2341.clone.clone.2.clone | 15 |
| wide.region_902.2342.clone.clone.2.clone | 3 |
| region_1020.2493.sunk.sunk.clone.clone | 16 |
| region_1025.2494.clone.clone | 3 |
| wide.region_573.1603.clone.21 | 5 |
| wide.region_574.1604.clone.21 | 3 |
| wide.region_573.1603.clone.clone.20 | 5 |
| wide.region_574.1604.clone.clone.20 | 3 |
| wide.region_573.1603.clone.22 | 5 |
| wide.region_574.1604.clone.22 | 3 |
| wide.region_573.1603.clone.clone.21 | 5 |
| wide.region_574.1604.clone.clone.21 | 3 |
| wide.region_719.1905.clone.2.clone.48.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.48.clone.clone | 3 |
| wide.region_719.1905.clone.2.clone.50.clone.clone | 8 |
| wide.region_720.1906.clone.2.clone.50.clone.clone | 3 |
| wide.region_757.1994.clone.clone.2.clone.sunk.clone | 11 |
| wide.region_758.1995.clone.clone.2.clone.clone | 3 |
| wide.wide.region_1152.2639.clone.clone.clone | 107 |
| wide.wide.region_1164.2640.clone.clone.clone | 3 |
| region_1169.2646 | 11 |
| region_1170.2647 | 3 |
| wide.wide.wide.wide.wide.region_697.2506.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone | 303 |
| wide.wide.wide.wide.wide.region_1034.2507.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_765.2018.clone.clone.48.clone | 15 |
| wide.region_769.2019.clone.clone.48.clone | 3 |
| wide.region_898.2341.clone.clone.48.clone | 15 |
| wide.region_902.2342.clone.clone.48.clone | 3 |
| region_1080.2558.sunk.clone | 16 |
| region_1085.2559.clone | 3 |
| region_2488.4781 | 11 |
| region_2489.4782 | 3 |
| wide.region_765.2018.clone.clone.50.clone | 15 |
| wide.region_769.2019.clone.clone.50.clone | 3 |
| wide.region_898.2341.clone.clone.50.clone | 15 |
| wide.region_902.2342.clone.clone.50.clone | 3 |
| region_2473.4770.sunk.clone | 16 |
| region_2478.4771.clone | 3 |
| wide.region_577.1614.clone.23 | 5 |
| wide.region_578.1615.clone.23 | 3 |
| wide.region_577.1614.clone.clone.22 | 5 |
| wide.region_578.1615.clone.clone.22 | 3 |
| wide.region_577.1614.clone.24 | 5 |
| wide.region_578.1615.clone.24 | 3 |
| wide.region_577.1614.clone.clone.23 | 5 |
| wide.region_578.1615.clone.clone.23 | 3 |
| wide.region_577.1614.clone.25 | 5 |
| wide.region_578.1615.clone.25 | 3 |
| wide.region_577.1614.clone.clone.24 | 5 |
| wide.region_578.1615.clone.clone.24 | 3 |
| wide.region_577.1614 | 5 |
| wide.region_578.1615 | 3 |
| wide.region_577.1614.clone | 5 |
| wide.region_578.1615.clone | 3 |
| wide.region_573.1603 | 5 |
| wide.region_574.1604 | 3 |
| wide.region_573.1603.clone | 5 |
| wide.region_574.1604.clone | 3 |
| region_604.1676 | 11 |
| region_605.1677 | 3 |
| wide.region_577.1614.clone.2 | 5 |
| wide.region_578.1615.clone.2 | 3 |
| wide.region_577.1614.clone.clone.1 | 5 |
| wide.region_578.1615.clone.clone.1 | 3 |
| wide.region_577.1614.clone.1 | 5 |
| wide.region_578.1615.clone.1 | 3 |
| wide.region_577.1614.clone.clone | 5 |
| wide.region_578.1615.clone.clone | 3 |
| wide.region_645.1778.clone | 5 |
| wide.region_646.1779.clone | 2 |
| wide.region_573.1603.clone.1 | 5 |
| wide.region_574.1604.clone.1 | 3 |
| wide.region_573.1603.clone.clone | 5 |
| wide.region_574.1604.clone.clone | 3 |
| region_636.1770 | 11 |
| region_637.1771 | 3 |
| region_642.1776 | 7 |
| region_643.1777 | 1 |
| region_644.1780 | 18 |
| region_647.1781 | 2 |
| wide.region_573.1603.clone.2 | 5 |
| wide.region_574.1604.clone.2 | 3 |
| wide.region_573.1603.clone.clone.1 | 5 |
| wide.region_574.1604.clone.clone.1 | 3 |
| wide.wide.region_632.1785.clone.clone.clone | 129 |
| wide.wide.region_651.1786.clone.clone.clone | 2 |
| region_656.1794 | 11 |
| region_657.1795 | 3 |
| wide.region_573.1603.clone.23 | 5 |
| wide.region_574.1604.clone.23 | 3 |
| wide.region_573.1603.clone.clone.22 | 5 |
| wide.region_574.1604.clone.clone.22 | 3 |
| wide.region_573.1603.clone.24 | 5 |
| wide.region_574.1604.clone.24 | 3 |
| wide.region_573.1603.clone.clone.23 | 5 |
| wide.region_574.1604.clone.clone.23 | 3 |
| wide.region_573.1603.clone.25 | 5 |
| wide.region_574.1604.clone.25 | 3 |
| wide.region_573.1603.clone.clone.24 | 5 |
| wide.region_574.1604.clone.clone.24 | 3 |
| wide.wide.region_600.1690.clone.clone.clone.clone.clone.clone | 129 |
| wide.wide.region_614.1691.clone.clone.clone.clone.clone | 3 |
| region_586.1654 | 11 |
| region_591.1655 | 3 |
| region_625.1759 | 11 |
| region_626.1760 | 3 |
| wide.wide.wide.wide.region_631.1800.clone.sunk.clone.clone.clone | 135 |
| wide.wide.wide.wide.region_662.1801.clone.clone.clone.clone | 2 |
| region_666.1805 | 11 |
| region_667.1806 | 3 |
| wide.region_226.598.clone.clone.2.sunk.clone | 8 |
| wide.region_227.599.clone.clone.2.clone | 3 |
| wide.region_377.932.clone.3.sunk.clone | 9 |
| wide.region_378.933.clone.3.clone | 3 |
| wide.region_377.932.clone.clone.2.sunk.clone | 9 |
| wide.region_378.933.clone.clone.2.clone | 3 |
| wide.region_379.957.clone.9.sunk.clone.clone.clone | 12 |
| wide.region_380.958.clone.9.clone.clone.clone | 3 |
| wide.region_379.957.clone.clone.2.sunk.clone.clone.clone | 12 |
| wide.region_380.958.clone.clone.2.clone.clone.clone | 3 |
| wide.region_284.744.clone.2.clone | 15 |
| wide.region_288.745.clone.2.clone | 3 |
| wide.region_479.1355.clone.2.clone | 15 |
| wide.region_483.1356.clone.2.clone | 3 |
| region_553.1548.sunk.clone | 29 |
| region_558.1549.clone | 3 |
| wide.region_17.65.clone.clone.clone | 8 |
| wide.region_18.66.clone.clone.clone | 3 |
| wide.region_17.65.clone.2.clone.2.clone.clone | 8 |
| wide.region_18.66.clone.2.clone.2.clone.clone | 3 |
| wide.region_17.65.clone.2.clone.3.clone.clone | 8 |
| wide.region_18.66.clone.2.clone.3.clone.clone | 3 |
| wide.region_17.65.clone.4.clone.6.clone.clone | 8 |
| wide.region_18.66.clone.4.clone.6.clone.clone | 3 |
| wide.region_17.65.clone.2.clone.4.clone.clone | 8 |
| wide.region_18.66.clone.2.clone.4.clone.clone | 3 |
| wide.region_56.157.sunk.clone.clone | 11 |
| wide.region_57.158.clone.2.clone | 3 |
| wide.region_65.183.clone.2 | 15 |
| wide.region_69.184.clone.2 | 3 |
| wide.region_198.515.clone.2 | 15 |
| wide.region_202.516.clone.2 | 3 |
| wide.wide.wide.wide.region_214.5866.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone | 508 |
| wide.wide.wide.wide.region_3434.5867.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_3565.6064.sunk.clone | 16 |
| region_3570.6065.clone | 3 |
| wide.region_65.183.clone.clone.4.clone | 15 |
| wide.region_69.184.clone.clone.4.clone | 3 |
| wide.region_65.183.clone.clone.2.clone | 15 |
| wide.region_69.184.clone.clone.2.clone | 3 |
| wide.region_65.183.clone.clone.3.clone | 15 |
| wide.region_69.184.clone.clone.3.clone | 3 |
| wide.region_198.515.clone.clone.4.clone | 15 |
| wide.region_202.516.clone.clone.4.clone | 3 |
| wide.region_198.515.clone.clone.2.clone | 15 |
| wide.region_202.516.clone.clone.2.clone | 3 |
| wide.region_198.515.clone.clone.3.clone | 15 |
| wide.region_202.516.clone.clone.3.clone | 3 |
| region_3477.5968.sunk.clone | 16 |
| region_3482.5969.clone | 3 |
| region_3521.6018.sunk.clone | 16 |
| region_3526.6019.clone | 3 |
| wide.region_65.183.clone.1.clone.6.clone | 15 |
| wide.region_69.184.clone.1.clone.6.clone | 3 |
| wide.region_198.515.clone.1.clone.6.clone | 15 |
| wide.region_202.516.clone.1.clone.6.clone | 3 |
| region_3608.6108.sunk.clone | 16 |
| region_3613.6109.clone | 3 |
| wide.region_3628.8976.clone.clone.clone.clone | 55 |
| wide.region_6191.8977.clone.clone.clone | 3 |

### 1000 cells

- instructions: 662,331  (unparsed-shape: 0)
- while ops: 631  |  conditional ops: 213
- scanned bodies: 573,699 instructions (86.6%)  |  straight-line: 88,632 (13.4%)
- instructions with no nova scope: 156,498
  (no metadata 152,089; metadata but not nova 4,409)
- scan bytes: 294.54 GiB  |  straight-line bytes: 7.10 GiB

while body/condition computation sizes:

| body computation | instructions |
|---:|---:|
| wide.region_586.1665.clone.31 | 5 |
| wide.region_587.1666.clone.31 | 3 |
| wide.region_586.1665.clone.clone.30 | 5 |
| wide.region_587.1666.clone.clone.30 | 3 |
| wide.region_728.1958.clone.clone.28.clone.clone | 8 |
| wide.region_729.1959.clone.clone.28.clone.clone | 3 |
| wide.region_766.2047.clone.29.clone | 9 |
| wide.region_767.2048.clone.29.clone | 3 |
| wide.region_774.2071.clone.29.clone | 15 |
| wide.region_778.2072.clone.29.clone | 3 |
| wide.region_907.2394.clone.29.clone | 15 |
| wide.region_911.2395.clone.29.clone | 3 |
| region_4080.6422 | 12 |
| region_4085.6423 | 3 |
| wide.region_728.1958.clone.2.clone.62.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.62.clone.clone | 3 |
| wide.region_766.2047.clone.clone.62.clone | 9 |
| wide.region_767.2048.clone.clone.62.clone | 3 |
| wide.region_774.2071.clone.clone.62.clone | 15 |
| wide.region_778.2072.clone.clone.62.clone | 3 |
| wide.region_907.2394.clone.clone.62.clone | 15 |
| wide.region_911.2395.clone.clone.62.clone | 3 |
| region_4136.6479 | 12 |
| region_4141.6480 | 3 |
| wide.region_586.1665.clone.30 | 5 |
| wide.region_587.1666.clone.30 | 3 |
| wide.region_586.1665.clone.clone.29 | 5 |
| wide.region_587.1666.clone.clone.29 | 3 |
| wide.region_582.1654.clone.30 | 5 |
| wide.region_583.1655.clone.30 | 3 |
| wide.region_582.1654.clone.clone.29 | 5 |
| wide.region_583.1655.clone.clone.29 | 3 |
| region_4153.6493 | 11 |
| region_4154.6494 | 3 |
| wide.region_582.1654.clone.31 | 5 |
| wide.region_583.1655.clone.31 | 3 |
| wide.region_582.1654.clone.clone.30 | 5 |
| wide.region_583.1655.clone.clone.30 | 3 |
| region_4164.6504 | 11 |
| region_4165.6505 | 3 |
| wide.wide.region_4032.6524.clone.clone.sunk.clone.clone.clone | 111 |
| wide.wide.region_4183.6525.clone.clone.clone.clone.clone | 3 |
| wide.region_728.1958.clone.clone.30.clone.clone | 8 |
| wide.region_729.1959.clone.clone.30.clone.clone | 3 |
| wide.region_766.2047.clone.31.clone | 9 |
| wide.region_767.2048.clone.31.clone | 3 |
| wide.region_774.2071.clone.31.clone | 15 |
| wide.region_778.2072.clone.31.clone | 3 |
| wide.region_907.2394.clone.31.clone | 15 |
| wide.region_911.2395.clone.31.clone | 3 |
| region_4290.6637 | 12 |
| region_4295.6638 | 3 |
| wide.region_728.1958.clone.2.clone.66.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.66.clone.clone | 3 |
| wide.region_766.2047.clone.clone.66.clone | 9 |
| wide.region_767.2048.clone.clone.66.clone | 3 |
| wide.region_774.2071.clone.clone.66.clone | 15 |
| wide.region_778.2072.clone.clone.66.clone | 3 |
| wide.region_907.2394.clone.clone.66.clone | 15 |
| wide.region_911.2395.clone.clone.66.clone | 3 |
| region_4346.6694 | 12 |
| region_4351.6695 | 3 |
| wide.region_728.1958.clone.clone.32.clone.clone | 8 |
| wide.region_729.1959.clone.clone.32.clone.clone | 3 |
| wide.region_766.2047.clone.33.clone | 9 |
| wide.region_767.2048.clone.33.clone | 3 |
| wide.region_774.2071.clone.33.clone | 15 |
| wide.region_778.2072.clone.33.clone | 3 |
| wide.region_907.2394.clone.33.clone | 15 |
| wide.region_911.2395.clone.33.clone | 3 |
| region_4531.6884 | 12 |
| region_4536.6885 | 3 |
| wide.region_728.1958.clone.2.clone.70.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.70.clone.clone | 3 |
| wide.region_766.2047.clone.clone.70.clone | 9 |
| wide.region_767.2048.clone.clone.70.clone | 3 |
| wide.region_774.2071.clone.clone.70.clone | 15 |
| wide.region_778.2072.clone.clone.70.clone | 3 |
| wide.region_907.2394.clone.clone.70.clone | 15 |
| wide.region_911.2395.clone.clone.70.clone | 3 |
| region_4587.6941 | 12 |
| region_4592.6942 | 3 |
| wide.region_4450.6796.clone | 5 |
| wide.region_4451.6797.clone | 2 |
| wide.region_582.1654.clone.33 | 5 |
| wide.region_583.1655.clone.33 | 3 |
| wide.region_582.1654.clone.clone.32 | 5 |
| wide.region_583.1655.clone.clone.32 | 3 |
| wide.wide.region_1554.3121.clone.8.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.8.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.7.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.7.clone.clone | 3 |
| region_4426.6773 | 11 |
| region_4427.6774 | 3 |
| wide.wide.region_4438.6785.clone.clone.clone | 9 |
| wide.wide.region_4439.6786.clone.clone.clone | 3 |
| region_4447.6794 | 7 |
| region_4448.6795 | 1 |
| region_4449.6798 | 18 |
| region_4452.6799 | 2 |
| wide.region_582.1654.clone.34 | 5 |
| wide.region_583.1655.clone.34 | 3 |
| wide.region_582.1654.clone.clone.33 | 5 |
| wide.region_583.1655.clone.clone.33 | 3 |
| wide.wide.region_1554.3121.clone.9.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.9.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.8.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.8.clone.clone | 3 |
| wide.wide.wide.wide.region_4422.6803.sunk.clone.clone.clone.sunk.clone.clone.clone.clone.clone | 236 |
| wide.wide.wide.wide.region_4456.6804.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_4461.6809 | 11 |
| region_4462.6810 | 3 |
| wide.wide.region_4473.6821.clone.clone.clone | 9 |
| wide.wide.region_4474.6822.clone.clone.clone | 3 |
| wide.region_582.1654.clone.36 | 5 |
| wide.region_583.1655.clone.36 | 3 |
| wide.region_582.1654.clone.clone.35 | 5 |
| wide.region_583.1655.clone.clone.35 | 3 |
| wide.wide.wide.wide.wide.wide.region_4421.6830.sunk.sunk.sunk.clone.clone.clone.clone | 229 |
| wide.wide.wide.wide.wide.wide.region_4482.6831.clone.clone.clone.clone | 2 |
| region_4604.6955 | 11 |
| region_4605.6956 | 3 |
| wide.region_234.639.clone.clone.4.clone.sunk.clone | 8 |
| wide.region_235.640.clone.clone.4.clone.clone | 3 |
| wide.region_385.974.clone.5.clone.sunk.clone | 9 |
| wide.region_386.975.clone.5.clone.clone | 3 |
| wide.region_385.974.clone.clone.4.clone.sunk.clone | 9 |
| wide.region_386.975.clone.clone.4.clone.clone | 3 |
| wide.region_387.999.clone.11.clone.sunk.clone.clone | 12 |
| wide.region_388.1000.clone.11.clone.clone.clone | 3 |
| wide.region_387.999.clone.clone.4.clone.sunk.clone.clone | 12 |
| wide.region_388.1000.clone.clone.4.clone.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.11.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.11.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.10.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.10.clone.clone | 3 |
| wide.region_284.761.clone.5.clone | 9 |
| wide.region_285.762.clone.5.clone | 3 |
| wide.region_292.785.clone.5.clone | 15 |
| wide.region_296.786.clone.5.clone | 3 |
| wide.region_487.1404.clone.5.clone | 15 |
| wide.region_491.1405.clone.5.clone | 3 |
| region_4233.6580 | 25 |
| region_4238.6581 | 3 |
| region_4360.6708.clone.clone | 20 |
| region_4363.6709.clone.clone | 3 |
| region_4396.6749 | 11 |
| region_4401.6750 | 3 |
| wide.wide.region_4370.6722.clone.clone.clone | 9 |
| wide.wide.region_4376.6723.clone.clone.clone | 3 |
| wide.wide.region_4412.6761.clone.clone.sunk.clone | 9 |
| wide.wide.region_4413.6762.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.wide.region_4384.6967.sunk.clone.sunk.clone.clone.clone.sunk.clone.clone.clone.clone.clone.clone | 124 |
| wide.wide.wide.wide.wide.wide.region_4615.6968.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_728.1958.clone.clone.36.clone.clone | 8 |
| wide.region_729.1959.clone.clone.36.clone.clone | 3 |
| wide.region_766.2047.clone.37.clone | 9 |
| wide.region_767.2048.clone.37.clone | 3 |
| wide.region_774.2071.clone.37.clone | 15 |
| wide.region_778.2072.clone.37.clone | 3 |
| wide.region_907.2394.clone.37.clone | 15 |
| wide.region_911.2395.clone.37.clone | 3 |
| region_4864.7226 | 12 |
| region_4869.7227 | 3 |
| wide.region_728.1958.clone.2.clone.78.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.78.clone.clone | 3 |
| wide.region_766.2047.clone.clone.78.clone | 9 |
| wide.region_767.2048.clone.clone.78.clone | 3 |
| wide.region_774.2071.clone.clone.78.clone | 15 |
| wide.region_778.2072.clone.clone.78.clone | 3 |
| wide.region_907.2394.clone.clone.78.clone | 15 |
| wide.region_911.2395.clone.clone.78.clone | 3 |
| region_4920.7283 | 12 |
| region_4925.7284 | 3 |
| wide.region_728.1958.clone.clone.34.clone.clone | 8 |
| wide.region_729.1959.clone.clone.34.clone.clone | 3 |
| wide.region_766.2047.clone.35.clone | 9 |
| wide.region_767.2048.clone.35.clone | 3 |
| wide.region_774.2071.clone.35.clone | 15 |
| wide.region_778.2072.clone.35.clone | 3 |
| wide.region_907.2394.clone.35.clone | 15 |
| wide.region_911.2395.clone.35.clone | 3 |
| region_4746.7104 | 12 |
| region_4751.7105 | 3 |
| wide.region_728.1958.clone.2.clone.74.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.74.clone.clone | 3 |
| wide.region_766.2047.clone.clone.74.clone | 9 |
| wide.region_767.2048.clone.clone.74.clone | 3 |
| wide.region_774.2071.clone.clone.74.clone | 15 |
| wide.region_778.2072.clone.clone.74.clone | 3 |
| wide.region_907.2394.clone.clone.74.clone | 15 |
| wide.region_911.2395.clone.clone.74.clone | 3 |
| region_4802.7161 | 12 |
| region_4807.7162 | 3 |
| wide.region_1812.3497.clone.clone.2.clone.sunk.clone | 8 |
| wide.region_1813.3498.clone.clone.2.clone.clone | 3 |
| wide.region_1950.3758.clone.3.clone.sunk.clone | 9 |
| wide.region_1951.3759.clone.3.clone.clone | 3 |
| wide.region_1950.3758.clone.clone.2.clone.sunk.clone | 9 |
| wide.region_1951.3759.clone.clone.2.clone.clone | 3 |
| wide.region_1952.3776.clone.9.clone.sunk.clone.clone | 12 |
| wide.region_1953.3777.clone.9.clone.clone.clone | 3 |
| wide.region_1952.3776.clone.clone.2.clone.sunk.clone.clone | 12 |
| wide.region_1953.3777.clone.clone.2.clone.clone.clone | 3 |
| wide.wide.region_2150.4357.clone.1.clone.clone | 6 |
| wide.wide.region_2151.4358.clone.1.clone.clone | 3 |
| wide.wide.region_2150.4357.clone.clone.clone.clone | 6 |
| wide.wide.region_2151.4358.clone.clone.clone.clone | 3 |
| wide.region_284.761.clone.clone.2.clone | 9 |
| wide.region_285.762.clone.clone.2.clone | 3 |
| wide.region_292.785.clone.clone.2.clone | 15 |
| wide.region_296.786.clone.clone.2.clone | 3 |
| wide.region_487.1404.clone.clone.2.clone | 15 |
| wide.region_491.1405.clone.clone.2.clone | 3 |
| region_4664.7022.clone | 24 |
| region_4669.7023.clone | 3 |
| wide.wide.region_4682.7042.clone.clone.clone | 9 |
| wide.wide.region_4688.7043.clone.clone.clone | 3 |
| wide.region_4698.7173.clone.clone.clone | 14 |
| wide.region_4816.7174.clone.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.15.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.15.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.clone.14.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.clone.14.clone.clone | 3 |
| wide.region_586.1665.clone.38 | 5 |
| wide.region_587.1666.clone.38 | 3 |
| wide.region_586.1665.clone.clone.37 | 5 |
| wide.region_587.1666.clone.clone.37 | 3 |
| wide.region_728.1958.clone.clone.40.clone.clone | 8 |
| wide.region_729.1959.clone.clone.40.clone.clone | 3 |
| wide.region_766.2047.clone.41.clone | 9 |
| wide.region_767.2048.clone.41.clone | 3 |
| wide.region_774.2071.clone.41.clone | 15 |
| wide.region_778.2072.clone.41.clone | 3 |
| wide.region_907.2394.clone.41.clone | 15 |
| wide.region_911.2395.clone.41.clone | 3 |
| region_5227.7595 | 12 |
| region_5232.7596 | 3 |
| wide.region_728.1958.clone.2.clone.86.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.86.clone.clone | 3 |
| wide.region_766.2047.clone.clone.86.clone | 9 |
| wide.region_767.2048.clone.clone.86.clone | 3 |
| wide.region_774.2071.clone.clone.86.clone | 15 |
| wide.region_778.2072.clone.clone.86.clone | 3 |
| wide.region_907.2394.clone.clone.86.clone | 15 |
| wide.region_911.2395.clone.clone.86.clone | 3 |
| region_5283.7652 | 12 |
| region_5288.7653 | 3 |
| wide.region_728.1958.clone.clone.38.clone.clone | 8 |
| wide.region_729.1959.clone.clone.38.clone.clone | 3 |
| wide.region_766.2047.clone.39.clone | 9 |
| wide.region_767.2048.clone.39.clone | 3 |
| wide.region_774.2071.clone.39.clone | 15 |
| wide.region_778.2072.clone.39.clone | 3 |
| wide.region_907.2394.clone.39.clone | 15 |
| wide.region_911.2395.clone.39.clone | 3 |
| region_5109.7473 | 12 |
| region_5114.7474 | 3 |
| wide.region_728.1958.clone.2.clone.82.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.82.clone.clone | 3 |
| wide.region_766.2047.clone.clone.82.clone | 9 |
| wide.region_767.2048.clone.clone.82.clone | 3 |
| wide.region_774.2071.clone.clone.82.clone | 15 |
| wide.region_778.2072.clone.clone.82.clone | 3 |
| wide.region_907.2394.clone.clone.82.clone | 15 |
| wide.region_911.2395.clone.clone.82.clone | 3 |
| region_5165.7530 | 12 |
| region_5170.7531 | 3 |
| wide.region_586.1665.clone.37 | 5 |
| wide.region_587.1666.clone.37 | 3 |
| wide.region_586.1665.clone.clone.36 | 5 |
| wide.region_587.1666.clone.clone.36 | 3 |
| wide.region_582.1654.clone.37 | 5 |
| wide.region_583.1655.clone.37 | 3 |
| wide.region_582.1654.clone.clone.36 | 5 |
| wide.region_583.1655.clone.clone.36 | 3 |
| region_5301.7666 | 11 |
| region_5302.7667 | 3 |
| wide.region_582.1654.clone.38 | 5 |
| wide.region_583.1655.clone.38 | 3 |
| wide.region_582.1654.clone.clone.37 | 5 |
| wide.region_583.1655.clone.clone.37 | 3 |
| wide.region_5061.7542.clone.clone.clone | 14 |
| wide.region_5179.7543.clone.clone.clone | 3 |
| wide.wide.region_5297.7675.clone.clone.clone | 91 |
| wide.wide.region_5309.7676.clone.clone.clone | 3 |
| region_5313.7680 | 11 |
| region_5314.7681 | 3 |
| wide.region_586.1665.clone.40 | 5 |
| wide.region_587.1666.clone.40 | 3 |
| wide.region_586.1665.clone.clone.39 | 5 |
| wide.region_587.1666.clone.clone.39 | 3 |
| wide.region_728.1958.clone.clone.42.clone.clone | 8 |
| wide.region_729.1959.clone.clone.42.clone.clone | 3 |
| wide.region_766.2047.clone.43.clone | 9 |
| wide.region_767.2048.clone.43.clone | 3 |
| wide.region_774.2071.clone.43.clone | 15 |
| wide.region_778.2072.clone.43.clone | 3 |
| wide.region_907.2394.clone.43.clone | 15 |
| wide.region_911.2395.clone.43.clone | 3 |
| region_5373.7743 | 12 |
| region_5378.7744 | 3 |
| wide.region_728.1958.clone.2.clone.90.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.90.clone.clone | 3 |
| wide.region_766.2047.clone.clone.90.clone | 9 |
| wide.region_767.2048.clone.clone.90.clone | 3 |
| wide.region_774.2071.clone.clone.90.clone | 15 |
| wide.region_778.2072.clone.clone.90.clone | 3 |
| wide.region_907.2394.clone.clone.90.clone | 15 |
| wide.region_911.2395.clone.clone.90.clone | 3 |
| region_5429.7800 | 12 |
| region_5434.7801 | 3 |
| wide.region_586.1665.clone.39 | 5 |
| wide.region_587.1666.clone.39 | 3 |
| wide.region_586.1665.clone.clone.38 | 5 |
| wide.region_587.1666.clone.clone.38 | 3 |
| wide.region_582.1654.clone.39 | 5 |
| wide.region_583.1655.clone.39 | 3 |
| wide.region_582.1654.clone.clone.38 | 5 |
| wide.region_583.1655.clone.clone.38 | 3 |
| region_5446.7814 | 11 |
| region_5447.7815 | 3 |
| wide.region_582.1654.clone.40 | 5 |
| wide.region_583.1655.clone.40 | 3 |
| wide.region_582.1654.clone.clone.39 | 5 |
| wide.region_583.1655.clone.clone.39 | 3 |
| region_5457.7825 | 11 |
| region_5458.7826 | 3 |
| wide.wide.region_5325.7845.clone.clone.sunk.clone.clone.clone | 111 |
| wide.wide.region_5476.7846.clone.clone.clone.clone.clone | 3 |
| wide.region_728.1958.clone.clone.44.clone.clone | 8 |
| wide.region_729.1959.clone.clone.44.clone.clone | 3 |
| wide.region_766.2047.clone.45.clone | 9 |
| wide.region_767.2048.clone.45.clone | 3 |
| wide.region_774.2071.clone.45.clone | 15 |
| wide.region_778.2072.clone.45.clone | 3 |
| wide.region_907.2394.clone.45.clone | 15 |
| wide.region_911.2395.clone.45.clone | 3 |
| region_5527.7901 | 12 |
| region_5532.7902 | 3 |
| wide.region_728.1958.clone.2.clone.94.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.94.clone.clone | 3 |
| wide.region_766.2047.clone.clone.94.clone | 9 |
| wide.region_767.2048.clone.clone.94.clone | 3 |
| wide.region_774.2071.clone.clone.94.clone | 15 |
| wide.region_778.2072.clone.clone.94.clone | 3 |
| wide.region_907.2394.clone.clone.94.clone | 15 |
| wide.region_911.2395.clone.clone.94.clone | 3 |
| region_5583.7958 | 12 |
| region_5588.7959 | 3 |
| wide.region_586.1665.clone.44 | 5 |
| wide.region_587.1666.clone.44 | 3 |
| wide.region_586.1665.clone.clone.43 | 5 |
| wide.region_587.1666.clone.clone.43 | 3 |
| wide.wide.region_1552.3110.clone.14.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.14.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.clone.13.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.clone.13.clone.clone | 3 |
| wide.region_586.1665.clone.43 | 5 |
| wide.region_587.1666.clone.43 | 3 |
| wide.region_586.1665.clone.clone.42 | 5 |
| wide.region_587.1666.clone.clone.42 | 3 |
| wide.wide.region_1552.3110.clone.13.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.13.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.clone.12.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.clone.12.clone.clone | 3 |
| wide.region_586.1665.clone.42 | 5 |
| wide.region_587.1666.clone.42 | 3 |
| wide.region_586.1665.clone.clone.41 | 5 |
| wide.region_587.1666.clone.clone.41 | 3 |
| wide.wide.region_1552.3110.clone.12.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.12.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.clone.11.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.clone.11.clone.clone | 3 |
| wide.region_5663.8034.clone | 5 |
| wide.region_5664.8035.clone | 2 |
| wide.region_582.1654.clone.42 | 5 |
| wide.region_583.1655.clone.42 | 3 |
| wide.region_582.1654.clone.clone.41 | 5 |
| wide.region_583.1655.clone.clone.41 | 3 |
| wide.wide.region_1554.3121.clone.12.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.12.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.11.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.11.clone.clone | 3 |
| region_5639.8011 | 11 |
| region_5640.8012 | 3 |
| wide.wide.region_5651.8023.clone.clone.clone | 9 |
| wide.wide.region_5652.8024.clone.clone.clone | 3 |
| region_5660.8032 | 7 |
| region_5661.8033 | 1 |
| region_5662.8036 | 18 |
| region_5665.8037 | 2 |
| wide.region_582.1654.clone.43 | 5 |
| wide.region_583.1655.clone.43 | 3 |
| wide.region_582.1654.clone.clone.42 | 5 |
| wide.region_583.1655.clone.clone.42 | 3 |
| wide.wide.region_1554.3121.clone.13.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.13.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.12.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.12.clone.clone | 3 |
| wide.wide.wide.region_5635.8041.sunk.clone.clone.clone.clone.clone.clone.clone | 248 |
| wide.wide.wide.region_5669.8042.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_5674.8047 | 11 |
| region_5675.8048 | 3 |
| wide.wide.region_5686.8059.clone.clone.clone | 9 |
| wide.wide.region_5687.8060.clone.clone.clone | 3 |
| wide.region_728.1958.clone.clone.46.clone.clone | 8 |
| wide.region_729.1959.clone.clone.46.clone.clone | 3 |
| wide.region_766.2047.clone.47.clone | 9 |
| wide.region_767.2048.clone.47.clone | 3 |
| wide.region_774.2071.clone.47.clone | 15 |
| wide.region_778.2072.clone.47.clone | 3 |
| wide.region_907.2394.clone.47.clone | 15 |
| wide.region_911.2395.clone.47.clone | 3 |
| region_5744.8122 | 12 |
| region_5749.8123 | 3 |
| wide.region_728.1958.clone.2.clone.98.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.98.clone.clone | 3 |
| wide.region_766.2047.clone.clone.98.clone | 9 |
| wide.region_767.2048.clone.clone.98.clone | 3 |
| wide.region_774.2071.clone.clone.98.clone | 15 |
| wide.region_778.2072.clone.clone.98.clone | 3 |
| wide.region_907.2394.clone.clone.98.clone | 15 |
| wide.region_911.2395.clone.clone.98.clone | 3 |
| region_5800.8179 | 12 |
| region_5805.8180 | 3 |
| wide.region_586.1665.clone.45 | 5 |
| wide.region_587.1666.clone.45 | 3 |
| wide.region_586.1665.clone.clone.44 | 5 |
| wide.region_587.1666.clone.clone.44 | 3 |
| wide.region_582.1654.clone.45 | 5 |
| wide.region_583.1655.clone.45 | 3 |
| wide.region_582.1654.clone.clone.44 | 5 |
| wide.region_583.1655.clone.clone.44 | 3 |
| wide.wide.wide.wide.wide.wide.region_5634.8068.sunk.sunk.sunk.clone.clone.clone.clone | 259 |
| wide.wide.wide.wide.wide.wide.region_5695.8069.clone.clone.clone.clone | 2 |
| region_5817.8193 | 11 |
| region_5818.8194 | 3 |
| wide.region_582.1654.clone.44 | 5 |
| wide.region_583.1655.clone.44 | 3 |
| wide.region_582.1654.clone.clone.43 | 5 |
| wide.region_583.1655.clone.clone.43 | 3 |
| wide.wide.region_1554.3121.clone.14.clone.sunk.clone | 6 |
| wide.wide.region_1555.3122.clone.14.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.13.clone.sunk.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.13.clone.clone | 3 |
| region_5597.7972.clone.clone | 20 |
| region_5600.7973.clone.clone | 3 |
| region_5613.7987 | 11 |
| region_5614.7988 | 3 |
| wide.wide.region_5625.7999.clone.clone.sunk.clone | 9 |
| wide.wide.region_5626.8000.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.wide.wide.wide.region_5601.8205.clone.sunk.clone.clone.sunk.clone.clone.clone.clone.clone.clone | 130 |
| wide.wide.wide.wide.wide.wide.wide.wide.region_5828.8206.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_728.1958.clone.2.clone.100.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.100.clone.clone | 3 |
| wide.region_766.2047.clone.clone.100.clone | 9 |
| wide.region_767.2048.clone.clone.100.clone | 3 |
| wide.region_774.2071.clone.clone.100.clone | 15 |
| wide.region_778.2072.clone.clone.100.clone | 3 |
| wide.region_907.2394.clone.clone.100.clone | 15 |
| wide.region_911.2395.clone.clone.100.clone | 3 |
| region_5891.8274 | 12 |
| region_5896.8275 | 3 |
| wide.region_586.1665.clone.46 | 5 |
| wide.region_587.1666.clone.46 | 3 |
| wide.region_586.1665.clone.clone.45 | 5 |
| wide.region_587.1666.clone.clone.45 | 3 |
| wide.region_582.1654.clone.46 | 5 |
| wide.region_583.1655.clone.46 | 3 |
| wide.region_582.1654.clone.clone.45 | 5 |
| wide.region_583.1655.clone.clone.45 | 3 |
| region_5906.8286 | 11 |
| region_5907.8287 | 3 |
| wide.wide.region_1554.3121.clone.15.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.15.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.14.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.14.clone.clone | 3 |
| region_5033.7397.clone.clone | 20 |
| region_5036.7398.clone.clone | 3 |
| wide.wide.region_5043.7411.clone.clone.clone | 9 |
| wide.wide.region_5049.7412.clone.clone.clone | 3 |
| wide.wide.wide.wide.region_5057.8308.clone.clone.clone.clone.clone.clone.clone.clone | 177 |
| wide.wide.wide.wide.region_5928.8309.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| wide.region_586.1665.clone.47 | 5 |
| wide.region_587.1666.clone.47 | 3 |
| wide.region_586.1665.clone.clone.46 | 5 |
| wide.region_587.1666.clone.clone.46 | 3 |
| wide.region_586.1665.clone.48 | 5 |
| wide.region_587.1666.clone.48 | 3 |
| wide.region_586.1665.clone.clone.47 | 5 |
| wide.region_587.1666.clone.clone.47 | 3 |
| wide.region_728.1958.clone.clone.26.clone.clone | 8 |
| wide.region_729.1959.clone.clone.26.clone.clone | 3 |
| wide.region_766.2047.clone.27.clone | 9 |
| wide.region_767.2048.clone.27.clone | 3 |
| wide.region_774.2071.clone.27.clone | 15 |
| wide.region_778.2072.clone.27.clone | 3 |
| wide.region_907.2394.clone.27.clone | 15 |
| wide.region_911.2395.clone.27.clone | 3 |
| region_3935.6276 | 12 |
| region_3940.6277 | 3 |
| wide.region_728.1958.clone.2.clone.58.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.58.clone.clone | 3 |
| wide.region_766.2047.clone.clone.58.clone | 9 |
| wide.region_767.2048.clone.clone.58.clone | 3 |
| wide.region_774.2071.clone.clone.58.clone | 15 |
| wide.region_778.2072.clone.clone.58.clone | 3 |
| wide.region_907.2394.clone.clone.58.clone | 15 |
| wide.region_911.2395.clone.clone.58.clone | 3 |
| region_3991.6333 | 12 |
| region_3996.6334 | 3 |
| wide.region_728.1958.clone.clone.24.clone.clone | 8 |
| wide.region_729.1959.clone.clone.24.clone.clone | 3 |
| wide.region_766.2047.clone.25.clone | 9 |
| wide.region_767.2048.clone.25.clone | 3 |
| wide.region_774.2071.clone.25.clone | 15 |
| wide.region_778.2072.clone.25.clone | 3 |
| wide.region_907.2394.clone.25.clone | 15 |
| wide.region_911.2395.clone.25.clone | 3 |
| region_3817.6154 | 12 |
| region_3822.6155 | 3 |
| wide.region_728.1958.clone.2.clone.54.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.54.clone.clone | 3 |
| wide.region_766.2047.clone.clone.54.clone | 9 |
| wide.region_767.2048.clone.clone.54.clone | 3 |
| wide.region_774.2071.clone.clone.54.clone | 15 |
| wide.region_778.2072.clone.clone.54.clone | 3 |
| wide.region_907.2394.clone.clone.54.clone | 15 |
| wide.region_911.2395.clone.clone.54.clone | 3 |
| region_3873.6211 | 12 |
| region_3878.6212 | 3 |
| wide.region_586.1665.clone.29 | 5 |
| wide.region_587.1666.clone.29 | 3 |
| wide.region_586.1665.clone.clone.28 | 5 |
| wide.region_587.1666.clone.clone.28 | 3 |
| wide.region_582.1654.clone.29 | 5 |
| wide.region_583.1655.clone.29 | 3 |
| wide.region_582.1654.clone.clone.28 | 5 |
| wide.region_583.1655.clone.clone.28 | 3 |
| region_4009.6347 | 11 |
| region_4010.6348 | 3 |
| wide.region_582.1654.clone.47 | 5 |
| wide.region_583.1655.clone.47 | 3 |
| wide.region_582.1654.clone.clone.46 | 5 |
| wide.region_583.1655.clone.clone.46 | 3 |
| wide.region_582.1654.clone.48 | 5 |
| wide.region_583.1655.clone.48 | 3 |
| wide.region_582.1654.clone.clone.47 | 5 |
| wide.region_583.1655.clone.clone.47 | 3 |
| wide.region_728.1958.clone.2.clone.102.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.102.clone.clone | 3 |
| wide.region_766.2047.clone.clone.102.clone | 9 |
| wide.region_767.2048.clone.clone.102.clone | 3 |
| wide.region_3769.6223.clone.clone.clone | 14 |
| wide.region_3887.6224.clone.clone.clone | 3 |
| wide.wide.region_4005.6356.clone.clone.clone | 91 |
| wide.wide.region_4017.6357.clone.clone.clone | 3 |
| region_4021.6361 | 11 |
| region_4022.6362 | 3 |
| region_5009.7371 | 11 |
| region_5010.7372 | 3 |
| wide.region_774.2071.clone.clone.102.clone | 15 |
| wide.region_778.2072.clone.clone.102.clone | 3 |
| wide.region_907.2394.clone.clone.102.clone | 15 |
| wide.region_911.2395.clone.clone.102.clone | 3 |
| region_4994.7360 | 12 |
| region_4999.7361 | 3 |
| wide.region_586.1665.clone.49 | 5 |
| wide.region_587.1666.clone.49 | 3 |
| wide.region_586.1665.clone.clone.48 | 5 |
| wide.region_587.1666.clone.clone.48 | 3 |
| wide.region_586.1665.clone.50 | 5 |
| wide.region_587.1666.clone.50 | 3 |
| wide.region_586.1665.clone.clone.49 | 5 |
| wide.region_587.1666.clone.clone.49 | 3 |
| wide.region_586.1665.clone.51 | 5 |
| wide.region_587.1666.clone.51 | 3 |
| wide.region_586.1665.clone.clone.50 | 5 |
| wide.region_587.1666.clone.clone.50 | 3 |
| wide.region_586.1665.clone.26 | 5 |
| wide.region_587.1666.clone.26 | 3 |
| wide.region_586.1665.clone.clone.25 | 5 |
| wide.region_587.1666.clone.clone.25 | 3 |
| wide.region_582.1654.clone.26 | 5 |
| wide.region_583.1655.clone.26 | 3 |
| wide.region_582.1654.clone.clone.25 | 5 |
| wide.region_583.1655.clone.clone.25 | 3 |
| region_3678.6010 | 11 |
| region_3679.6011 | 3 |
| wide.region_586.1665.clone.28 | 5 |
| wide.region_587.1666.clone.28 | 3 |
| wide.region_586.1665.clone.clone.27 | 5 |
| wide.region_587.1666.clone.clone.27 | 3 |
| wide.region_586.1665.clone.27 | 5 |
| wide.region_587.1666.clone.27 | 3 |
| wide.region_586.1665.clone.clone.26 | 5 |
| wide.region_587.1666.clone.clone.26 | 3 |
| wide.region_3717.6049.clone | 5 |
| wide.region_3718.6050.clone | 2 |
| wide.region_582.1654.clone.27 | 5 |
| wide.region_583.1655.clone.27 | 3 |
| wide.region_582.1654.clone.clone.26 | 5 |
| wide.region_583.1655.clone.clone.26 | 3 |
| region_3708.6041 | 11 |
| region_3709.6042 | 3 |
| region_3714.6047 | 7 |
| region_3715.6048 | 1 |
| region_3716.6051 | 18 |
| region_3719.6052 | 2 |
| wide.region_582.1654.clone.28 | 5 |
| wide.region_583.1655.clone.28 | 3 |
| wide.region_582.1654.clone.clone.27 | 5 |
| wide.region_583.1655.clone.clone.27 | 3 |
| wide.wide.region_3704.6056.clone.clone.clone | 113 |
| wide.wide.region_3723.6057.clone.clone.clone | 2 |
| region_3728.6062 | 11 |
| region_3729.6063 | 3 |
| wide.region_582.1654.clone.49 | 5 |
| wide.region_583.1655.clone.49 | 3 |
| wide.region_582.1654.clone.clone.48 | 5 |
| wide.region_583.1655.clone.clone.48 | 3 |
| wide.region_582.1654.clone.50 | 5 |
| wide.region_583.1655.clone.50 | 3 |
| wide.region_582.1654.clone.clone.49 | 5 |
| wide.region_583.1655.clone.clone.49 | 3 |
| wide.region_582.1654.clone.51 | 5 |
| wide.region_583.1655.clone.51 | 3 |
| wide.region_582.1654.clone.clone.50 | 5 |
| wide.region_583.1655.clone.clone.50 | 3 |
| wide.wide.region_3674.6021.clone.clone.clone.clone.clone.clone | 113 |
| wide.wide.region_3688.6022.clone.clone.clone.clone.clone | 3 |
| region_3661.5998 | 11 |
| region_3666.5999 | 3 |
| region_3697.6032 | 11 |
| region_3698.6033 | 3 |
| wide.wide.wide.wide.region_3703.6068.clone.sunk.clone.clone.clone | 119 |
| wide.wide.wide.wide.region_3734.6069.clone.clone.clone.clone | 2 |
| region_3738.6073 | 11 |
| region_3739.6074 | 3 |
| wide.region_234.639.clone.clone.6.sunk.clone | 8 |
| wide.region_235.640.clone.clone.6.clone | 3 |
| wide.region_385.974.clone.7.sunk.clone | 9 |
| wide.region_386.975.clone.7.clone | 3 |
| wide.region_385.974.clone.clone.6.sunk.clone | 9 |
| wide.region_386.975.clone.clone.6.clone | 3 |
| wide.region_387.999.clone.13.sunk.clone.clone.clone | 12 |
| wide.region_388.1000.clone.13.clone.clone.clone | 3 |
| wide.region_387.999.clone.clone.6.sunk.clone.clone.clone | 12 |
| wide.region_388.1000.clone.clone.6.clone.clone.clone | 3 |
| wide.region_292.785.clone.7.clone | 15 |
| wide.region_296.786.clone.7.clone | 3 |
| wide.region_487.1404.clone.7.clone | 15 |
| wide.region_491.1405.clone.7.clone | 3 |
| region_3638.5975 | 25 |
| region_3643.5976 | 3 |
| wide.region_17.66.clone.2.clone.5.clone | 8 |
| wide.region_18.67.clone.2.clone.5.clone | 3 |
| wide.region_17.66.clone.2.clone.clone.clone | 8 |
| wide.region_18.67.clone.2.clone.clone.clone | 3 |
| wide.region_17.66.clone.4.clone.1.clone.clone | 8 |
| wide.region_18.67.clone.4.clone.1.clone.clone | 3 |
| wide.region_17.66.clone.4.clone.2.clone.clone | 8 |
| wide.region_18.67.clone.4.clone.2.clone.clone | 3 |
| wide.region_17.66.clone.2.clone.1.clone.clone | 8 |
| wide.region_18.67.clone.2.clone.1.clone.clone | 3 |
| wide.region_284.761.clone.7.clone | 9 |
| wide.region_285.762.clone.7.clone | 3 |
| wide.wide.wide.wide.region_3591.8313.clone.sunk.clone.clone.clone.clone.clone.clone.clone | 581 |
| wide.wide.wide.wide.region_5929.8314.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| wide.region_65.184.clone.1.clone.1.clone | 15 |
| wide.region_69.185.clone.1.clone.1.clone | 3 |
| wide.region_206.559.clone.1.clone.1.clone | 15 |
| wide.region_210.560.clone.1.clone.1.clone | 3 |
| wide.region_65.184.clone.clone.5 | 15 |
| wide.region_69.185.clone.clone.5 | 3 |
| wide.region_65.184.clone.clone.1.clone | 15 |
| wide.region_69.185.clone.clone.1.clone | 3 |
| wide.region_65.184.clone.clone.clone | 15 |
| wide.region_69.185.clone.clone.clone | 3 |
| region_6049.8441 | 12 |
| region_6054.8442 | 3 |
| wide.region_206.559.clone.clone.5 | 15 |
| wide.region_210.560.clone.clone.5 | 3 |
| wide.region_206.559.clone.clone.1.clone | 15 |
| wide.region_210.560.clone.clone.1.clone | 3 |
| wide.region_206.559.clone.clone.clone | 15 |
| wide.region_210.560.clone.clone.clone | 3 |
| region_5961.8351 | 12 |
| region_5966.8352 | 3 |
| region_6005.8396 | 12 |
| region_6010.8397 | 3 |
| wide.region_65.184.clone.1.clone.2.clone | 15 |
| wide.region_69.185.clone.1.clone.2.clone | 3 |
| wide.region_206.559.clone.1.clone.2.clone | 15 |
| wide.region_210.560.clone.1.clone.2.clone | 3 |
| region_6092.8485 | 12 |
| region_6097.8486 | 3 |
| wide.region_586.1665.clone.5 | 5 |
| wide.region_587.1666.clone.5 | 3 |
| wide.region_586.1665.clone.clone.4 | 5 |
| wide.region_587.1666.clone.clone.4 | 3 |
| wide.region_586.1665.clone.4 | 5 |
| wide.region_587.1666.clone.4 | 3 |
| wide.region_586.1665.clone.clone.3 | 5 |
| wide.region_587.1666.clone.clone.3 | 3 |
| wide.region_728.1958.clone.2.clone.8.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.8.clone.clone | 3 |
| wide.region_582.1654.clone.4 | 5 |
| wide.region_583.1655.clone.4 | 3 |
| wide.region_582.1654.clone.clone.3 | 5 |
| wide.region_583.1655.clone.clone.3 | 3 |
| region_1310.2833 | 11 |
| region_1311.2834 | 3 |
| wide.region_774.2071.clone.clone.8.clone | 15 |
| wide.region_778.2072.clone.clone.8.clone | 3 |
| wide.region_907.2394.clone.clone.8.clone | 15 |
| wide.region_911.2395.clone.clone.8.clone | 3 |
| region_1293.2819.sunk.sunk.clone.clone | 16 |
| region_1298.2820.clone.clone | 3 |
| wide.region_582.1654.clone.5 | 5 |
| wide.region_583.1655.clone.5 | 3 |
| wide.region_582.1654.clone.clone.4 | 5 |
| wide.region_583.1655.clone.clone.4 | 3 |
| wide.region_766.2047.clone.clone.8.clone.sunk.clone | 11 |
| wide.region_767.2048.clone.clone.8.clone.clone | 3 |
| region_1321.2844 | 11 |
| region_1322.2845 | 3 |
| wide.wide.wide.wide.wide.wide.region_1189.2864.clone.clone.sunk.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone.clone | 405 |
| wide.wide.wide.wide.wide.wide.region_1340.2865.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_1633.3281.clone | 5 |
| wide.region_1634.3282.clone | 2 |
| wide.region_582.1654.clone.7 | 5 |
| wide.region_583.1655.clone.7 | 3 |
| wide.region_582.1654.clone.clone.6 | 5 |
| wide.region_583.1655.clone.clone.6 | 3 |
| wide.wide.region_1554.3121.clone.16.clone | 6 |
| wide.wide.region_1555.3122.clone.16.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.15.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.15.clone | 3 |
| region_1609.3258 | 11 |
| region_1610.3259 | 3 |
| wide.wide.region_1621.3270.clone.clone.clone | 9 |
| wide.wide.region_1622.3271.clone.clone.clone | 3 |
| region_1630.3279 | 7 |
| region_1631.3280 | 1 |
| region_1632.3283 | 18 |
| region_1635.3284 | 2 |
| wide.region_582.1654.clone.8 | 5 |
| wide.region_583.1655.clone.8 | 3 |
| wide.region_582.1654.clone.clone.7 | 5 |
| wide.region_583.1655.clone.clone.7 | 3 |
| wide.wide.region_1554.3121.clone.1.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.1.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.wide.region_1605.3288.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone.clone | 251 |
| wide.wide.wide.wide.wide.wide.region_1639.3289.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_1644.3294 | 11 |
| region_1645.3295 | 3 |
| wide.wide.region_1656.3306.clone.clone.clone | 9 |
| wide.wide.region_1657.3307.clone.clone.clone | 3 |
| wide.region_728.1958.clone.2.clone.14.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.14.clone.clone | 3 |
| wide.region_582.1654.clone.10 | 5 |
| wide.region_583.1655.clone.10 | 3 |
| wide.region_582.1654.clone.clone.9 | 5 |
| wide.region_583.1655.clone.clone.9 | 3 |
| wide.wide.wide.wide.wide.wide.wide.region_1604.3315.sunk.sunk.sunk.sunk.sunk.clone.clone.clone.clone | 246 |
| wide.wide.wide.wide.wide.wide.wide.region_1665.3316.clone.clone.clone.clone | 2 |
| region_1787.3440 | 11 |
| region_1788.3441 | 3 |
| wide.region_774.2071.clone.clone.14.clone | 15 |
| wide.region_778.2072.clone.clone.14.clone | 3 |
| wide.region_907.2394.clone.clone.14.clone | 15 |
| wide.region_911.2395.clone.clone.14.clone | 3 |
| region_1770.3426.sunk.sunk.clone.clone | 16 |
| region_1775.3427.clone.clone | 3 |
| wide.region_234.639.clone.clone.clone.sunk.clone | 8 |
| wide.region_235.640.clone.clone.clone.clone | 3 |
| wide.region_385.974.clone.1.clone.sunk.clone | 9 |
| wide.region_386.975.clone.1.clone.clone | 3 |
| wide.region_385.974.clone.clone.clone.sunk.clone | 9 |
| wide.region_386.975.clone.clone.clone.clone | 3 |
| wide.region_728.1958.clone.2.clone.16.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.16.clone.clone | 3 |
| wide.region_387.999.clone.7.clone.sunk.clone.clone | 12 |
| wide.region_388.1000.clone.7.clone.clone.clone | 3 |
| wide.region_387.999.clone.clone.clone.sunk.clone.clone | 12 |
| wide.region_388.1000.clone.clone.clone.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.3.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.3.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.2.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.2.clone.clone | 3 |
| wide.region_284.761.clone.1.sunk.clone.clone | 11 |
| wide.region_285.762.clone.1.clone.clone | 3 |
| wide.region_774.2071.clone.clone.16.clone | 15 |
| wide.region_778.2072.clone.clone.16.clone | 3 |
| wide.region_292.785.clone.1.clone | 15 |
| wide.region_296.786.clone.1.clone | 3 |
| wide.region_907.2394.clone.clone.16.clone | 15 |
| wide.region_911.2395.clone.clone.16.clone | 3 |
| wide.region_487.1404.clone.1.clone | 15 |
| wide.region_491.1405.clone.1.clone | 3 |
| region_1445.2976.sunk.clone | 16 |
| region_1450.2977.clone | 3 |
| region_1390.2920.sunk.clone | 29 |
| region_1395.2921.clone | 3 |
| region_1517.3048.clone.clone | 20 |
| region_1520.3049.clone.clone | 3 |
| region_1579.3234 | 11 |
| region_1584.3235 | 3 |
| wide.wide.region_1527.3078.clone.clone.clone | 9 |
| wide.wide.region_1533.3079.clone.clone.clone | 3 |
| wide.wide.region_1595.3246.clone.clone.sunk.clone | 9 |
| wide.wide.region_1596.3247.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.wide.wide.wide.region_1565.3452.sunk.clone.sunk.clone.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone.clone.clone | 418 |
| wide.wide.wide.wide.wide.wide.wide.wide.region_1798.3453.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_728.1958.clone.2.clone.20.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.20.clone.clone | 3 |
| wide.region_774.2071.clone.clone.20.clone | 15 |
| wide.region_778.2072.clone.clone.20.clone | 3 |
| wide.region_907.2394.clone.clone.20.clone | 15 |
| wide.region_911.2395.clone.clone.20.clone | 3 |
| region_2267.4506.sunk.sunk.clone.clone | 16 |
| region_2272.4507.clone.clone | 3 |
| wide.region_1812.3497.clone.clone.clone.sunk.clone | 8 |
| wide.region_1813.3498.clone.clone.clone.clone | 3 |
| wide.region_1950.3758.clone.1.clone.sunk.clone | 9 |
| wide.region_1951.3759.clone.1.clone.clone | 3 |
| wide.region_1950.3758.clone.clone.clone.sunk.clone | 9 |
| wide.region_1951.3759.clone.clone.clone.clone | 3 |
| wide.region_728.1958.clone.2.clone.24.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.24.clone.clone | 3 |
| wide.region_1952.3776.clone.7.clone.sunk.clone.clone | 12 |
| wide.region_1953.3777.clone.7.clone.clone.clone | 3 |
| wide.region_1952.3776.clone.clone.clone.sunk.clone.clone | 12 |
| wide.region_1953.3777.clone.clone.clone.clone.clone | 3 |
| wide.wide.region_2150.4357.clone.2.clone | 6 |
| wide.wide.region_2151.4358.clone.2.clone | 3 |
| wide.wide.region_2150.4357.clone.clone.1.clone | 6 |
| wide.wide.region_2151.4358.clone.clone.1.clone | 3 |
| wide.region_284.761.clone.clone.sunk.clone.clone | 11 |
| wide.region_285.762.clone.clone.clone.clone | 3 |
| wide.region_774.2071.clone.clone.24.clone | 15 |
| wide.region_778.2072.clone.clone.24.clone | 3 |
| wide.region_292.785.clone.clone.clone | 15 |
| wide.region_296.786.clone.clone.clone | 3 |
| wide.region_907.2394.clone.clone.24.clone | 15 |
| wide.region_911.2395.clone.clone.24.clone | 3 |
| wide.region_487.1404.clone.clone.clone | 15 |
| wide.region_491.1405.clone.clone.clone | 3 |
| region_2327.4570.sunk.clone | 16 |
| region_2332.4571.clone | 3 |
| region_2105.4269.sunk.clone | 28 |
| region_2110.4270.clone | 3 |
| wide.wide.region_2123.4314.clone.clone.clone | 9 |
| wide.wide.region_2129.4315.clone.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_2163.4518.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone | 306 |
| wide.wide.wide.wide.wide.region_2281.4519.clone.clone.clone.clone.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.7.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.7.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.clone.6.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.clone.6.clone.clone | 3 |
| wide.region_586.1665.clone.12 | 5 |
| wide.region_587.1666.clone.12 | 3 |
| wide.region_586.1665.clone.clone.11 | 5 |
| wide.region_587.1666.clone.clone.11 | 3 |
| wide.region_586.1665.clone.11 | 5 |
| wide.region_587.1666.clone.11 | 3 |
| wide.region_586.1665.clone.clone.10 | 5 |
| wide.region_587.1666.clone.clone.10 | 3 |
| wide.region_582.1654.clone.11 | 5 |
| wide.region_583.1655.clone.11 | 3 |
| wide.region_582.1654.clone.clone.10 | 5 |
| wide.region_583.1655.clone.clone.10 | 3 |
| region_2766.5014 | 11 |
| region_2767.5015 | 3 |
| wide.region_728.1958.clone.2.clone.28.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.28.clone.clone | 3 |
| wide.region_774.2071.clone.clone.28.clone | 15 |
| wide.region_778.2072.clone.clone.28.clone | 3 |
| wide.region_907.2394.clone.clone.28.clone | 15 |
| wide.region_911.2395.clone.clone.28.clone | 3 |
| region_2630.4878.sunk.sunk.clone.clone | 16 |
| region_2635.4879.clone.clone | 3 |
| wide.region_582.1654.clone.12 | 5 |
| wide.region_583.1655.clone.12 | 3 |
| wide.region_582.1654.clone.clone.11 | 5 |
| wide.region_583.1655.clone.clone.11 | 3 |
| wide.region_728.1958.clone.2.clone.32.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.32.clone.clone | 3 |
| wide.region_766.2047.clone.clone.28.clone.sunk.clone | 11 |
| wide.region_767.2048.clone.clone.28.clone.clone | 3 |
| wide.wide.region_2762.5023.clone.clone.clone | 107 |
| wide.wide.region_2774.5024.clone.clone.clone | 3 |
| region_2778.5028 | 11 |
| region_2779.5029 | 3 |
| wide.wide.wide.wide.wide.region_2526.4890.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone | 306 |
| wide.wide.wide.wide.wide.region_2644.4891.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_774.2071.clone.clone.32.clone | 15 |
| wide.region_778.2072.clone.clone.32.clone | 3 |
| wide.region_907.2394.clone.clone.32.clone | 15 |
| wide.region_911.2395.clone.clone.32.clone | 3 |
| region_2690.4942.sunk.clone | 16 |
| region_2695.4943.clone | 3 |
| wide.region_586.1665.clone.14 | 5 |
| wide.region_587.1666.clone.14 | 3 |
| wide.region_586.1665.clone.clone.13 | 5 |
| wide.region_587.1666.clone.clone.13 | 3 |
| wide.region_586.1665.clone.13 | 5 |
| wide.region_587.1666.clone.13 | 3 |
| wide.region_586.1665.clone.clone.12 | 5 |
| wide.region_587.1666.clone.clone.12 | 3 |
| wide.region_728.1958.clone.2.clone.36.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.36.clone.clone | 3 |
| wide.region_582.1654.clone.13 | 5 |
| wide.region_583.1655.clone.13 | 3 |
| wide.region_582.1654.clone.clone.12 | 5 |
| wide.region_583.1655.clone.clone.12 | 3 |
| region_2911.5162 | 11 |
| region_2912.5163 | 3 |
| wide.region_774.2071.clone.clone.36.clone | 15 |
| wide.region_778.2072.clone.clone.36.clone | 3 |
| wide.region_907.2394.clone.clone.36.clone | 15 |
| wide.region_911.2395.clone.clone.36.clone | 3 |
| region_2894.5148.sunk.sunk.clone.clone | 16 |
| region_2899.5149.clone.clone | 3 |
| wide.region_582.1654.clone.14 | 5 |
| wide.region_583.1655.clone.14 | 3 |
| wide.region_582.1654.clone.clone.13 | 5 |
| wide.region_583.1655.clone.clone.13 | 3 |
| wide.region_766.2047.clone.clone.36.clone.sunk.clone | 11 |
| wide.region_767.2048.clone.clone.36.clone.clone | 3 |
| region_2922.5173 | 11 |
| region_2923.5174 | 3 |
| wide.wide.wide.wide.wide.wide.region_2790.5193.clone.clone.sunk.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone.clone | 405 |
| wide.wide.wide.wide.wide.wide.region_2941.5194.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_586.1665.clone.18 | 5 |
| wide.region_587.1666.clone.18 | 3 |
| wide.region_586.1665.clone.clone.17 | 5 |
| wide.region_587.1666.clone.clone.17 | 3 |
| wide.wide.region_1552.3110.clone.6.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.6.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.clone.5.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.clone.5.clone.clone | 3 |
| wide.region_586.1665.clone.17 | 5 |
| wide.region_587.1666.clone.17 | 3 |
| wide.region_586.1665.clone.clone.16 | 5 |
| wide.region_587.1666.clone.clone.16 | 3 |
| wide.wide.region_1552.3110.clone.5.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.5.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.clone.4.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.clone.4.clone.clone | 3 |
| wide.region_586.1665.clone.16 | 5 |
| wide.region_587.1666.clone.16 | 3 |
| wide.region_586.1665.clone.clone.15 | 5 |
| wide.region_587.1666.clone.clone.15 | 3 |
| wide.wide.region_1552.3110.clone.4.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.4.clone.clone | 3 |
| wide.wide.region_1552.3110.clone.clone.3.clone.clone | 6 |
| wide.wide.region_1553.3111.clone.clone.3.clone.clone | 3 |
| wide.region_3128.5382.clone | 5 |
| wide.region_3129.5383.clone | 2 |
| wide.region_582.1654.clone.16 | 5 |
| wide.region_583.1655.clone.16 | 3 |
| wide.region_582.1654.clone.clone.15 | 5 |
| wide.region_583.1655.clone.clone.15 | 3 |
| wide.wide.region_1554.3121.clone.4.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.4.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.3.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.3.clone.clone | 3 |
| region_3104.5359 | 11 |
| region_3105.5360 | 3 |
| wide.wide.region_3116.5371.clone.clone.clone | 9 |
| wide.wide.region_3117.5372.clone.clone.clone | 3 |
| region_3125.5380 | 7 |
| region_3126.5381 | 1 |
| region_3127.5384 | 18 |
| region_3130.5385 | 2 |
| wide.region_582.1654.clone.17 | 5 |
| wide.region_583.1655.clone.17 | 3 |
| wide.region_582.1654.clone.clone.16 | 5 |
| wide.region_583.1655.clone.clone.16 | 3 |
| wide.wide.region_1554.3121.clone.5.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.5.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.4.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.4.clone.clone | 3 |
| wide.wide.wide.wide.wide.region_3100.5389.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone | 263 |
| wide.wide.wide.wide.wide.region_3134.5390.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_3139.5395 | 11 |
| region_3140.5396 | 3 |
| wide.wide.region_3151.5407.clone.clone.clone | 9 |
| wide.wide.region_3152.5408.clone.clone.clone | 3 |
| wide.region_586.1665.clone.19 | 5 |
| wide.region_587.1666.clone.19 | 3 |
| wide.region_586.1665.clone.clone.18 | 5 |
| wide.region_587.1666.clone.clone.18 | 3 |
| wide.region_728.1958.clone.2.clone.42.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.42.clone.clone | 3 |
| wide.region_582.1654.clone.19 | 5 |
| wide.region_583.1655.clone.19 | 3 |
| wide.region_582.1654.clone.clone.18 | 5 |
| wide.region_583.1655.clone.clone.18 | 3 |
| wide.wide.wide.wide.wide.wide.wide.region_3099.5416.sunk.sunk.sunk.sunk.sunk.clone.clone.clone.clone | 276 |
| wide.wide.wide.wide.wide.wide.wide.region_3160.5417.clone.clone.clone.clone | 2 |
| region_3282.5541 | 11 |
| region_3283.5542 | 3 |
| wide.region_774.2071.clone.clone.42.clone | 15 |
| wide.region_778.2072.clone.clone.42.clone | 3 |
| wide.region_907.2394.clone.clone.42.clone | 15 |
| wide.region_911.2395.clone.clone.42.clone | 3 |
| region_3265.5527.sunk.sunk.clone.clone | 16 |
| region_3270.5528.clone.clone | 3 |
| wide.region_582.1654.clone.18 | 5 |
| wide.region_583.1655.clone.18 | 3 |
| wide.region_582.1654.clone.clone.17 | 5 |
| wide.region_583.1655.clone.clone.17 | 3 |
| wide.region_728.1958.clone.2.clone.44.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.44.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.6.clone.sunk.clone | 6 |
| wide.wide.region_1555.3122.clone.6.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.5.clone.sunk.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.5.clone.clone | 3 |
| wide.region_766.2047.clone.clone.44.sunk.clone.clone | 11 |
| wide.region_767.2048.clone.clone.44.clone.clone | 3 |
| region_3062.5320.clone.clone | 20 |
| region_3065.5321.clone.clone | 3 |
| region_3078.5335 | 11 |
| region_3079.5336 | 3 |
| wide.region_774.2071.clone.clone.44.clone | 15 |
| wide.region_778.2072.clone.clone.44.clone | 3 |
| wide.wide.region_3090.5347.clone.clone.sunk.clone | 9 |
| wide.wide.region_3091.5348.clone.clone.clone | 3 |
| wide.region_907.2394.clone.clone.44.clone | 15 |
| wide.region_911.2395.clone.clone.44.clone | 3 |
| region_2990.5248.sunk.clone | 16 |
| region_2995.5249.clone | 3 |
| wide.wide.wide.wide.wide.wide.wide.wide.wide.wide.region_3066.5553.clone.sunk.clone.sunk.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone.clone.clone | 424 |
| wide.wide.wide.wide.wide.wide.wide.wide.wide.wide.region_3293.5554.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_728.1958.clone.2.clone.46.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.46.clone.clone | 3 |
| wide.region_766.2047.clone.clone.46.sunk.clone.clone | 11 |
| wide.region_767.2048.clone.clone.46.clone.clone | 3 |
| wide.region_774.2071.clone.clone.46.clone | 15 |
| wide.region_778.2072.clone.clone.46.clone | 3 |
| wide.region_907.2394.clone.clone.46.clone | 15 |
| wide.region_911.2395.clone.clone.46.clone | 3 |
| region_3356.5622.sunk.clone | 16 |
| region_3361.5623.clone | 3 |
| wide.region_586.1665.clone.20 | 5 |
| wide.region_587.1666.clone.20 | 3 |
| wide.region_586.1665.clone.clone.19 | 5 |
| wide.region_587.1666.clone.clone.19 | 3 |
| wide.region_582.1654.clone.20 | 5 |
| wide.region_583.1655.clone.20 | 3 |
| wide.region_582.1654.clone.clone.19 | 5 |
| wide.region_583.1655.clone.clone.19 | 3 |
| region_3371.5634 | 11 |
| region_3372.5635 | 3 |
| wide.wide.region_1554.3121.clone.7.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.7.clone.clone | 3 |
| wide.wide.region_1554.3121.clone.clone.6.clone.clone | 6 |
| wide.wide.region_1555.3122.clone.clone.6.clone.clone | 3 |
| region_2499.4746.clone.clone | 20 |
| region_2502.4747.clone.clone | 3 |
| wide.wide.region_2509.4760.clone.clone.clone | 9 |
| wide.wide.region_2515.4761.clone.clone.clone | 3 |
| wide.wide.wide.wide.region_2522.5656.clone.clone.clone.clone.clone.clone.clone.clone | 193 |
| wide.wide.wide.wide.region_3393.5657.clone.clone.clone.clone.clone.clone.clone.clone | 2 |
| wide.region_586.1665.clone.21 | 5 |
| wide.region_587.1666.clone.21 | 3 |
| wide.region_586.1665.clone.clone.20 | 5 |
| wide.region_587.1666.clone.clone.20 | 3 |
| wide.region_586.1665.clone.22 | 5 |
| wide.region_587.1666.clone.22 | 3 |
| wide.region_586.1665.clone.clone.21 | 5 |
| wide.region_587.1666.clone.clone.21 | 3 |
| wide.region_586.1665.clone.3 | 5 |
| wide.region_587.1666.clone.3 | 3 |
| wide.region_586.1665.clone.clone.2 | 5 |
| wide.region_587.1666.clone.clone.2 | 3 |
| wide.region_582.1654.clone.3 | 5 |
| wide.region_583.1655.clone.3 | 3 |
| wide.region_582.1654.clone.clone.2 | 5 |
| wide.region_583.1655.clone.clone.2 | 3 |
| region_1165.2684 | 11 |
| region_1166.2685 | 3 |
| wide.region_728.1958.clone.2.clone.2.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.2.clone.clone | 3 |
| wide.region_774.2071.clone.clone.2.clone | 15 |
| wide.region_778.2072.clone.clone.2.clone | 3 |
| wide.region_907.2394.clone.clone.2.clone | 15 |
| wide.region_911.2395.clone.clone.2.clone | 3 |
| region_1029.2547.sunk.sunk.clone.clone | 16 |
| region_1034.2548.clone.clone | 3 |
| wide.region_582.1654.clone.21 | 5 |
| wide.region_583.1655.clone.21 | 3 |
| wide.region_582.1654.clone.clone.20 | 5 |
| wide.region_583.1655.clone.clone.20 | 3 |
| wide.region_582.1654.clone.22 | 5 |
| wide.region_583.1655.clone.22 | 3 |
| wide.region_582.1654.clone.clone.21 | 5 |
| wide.region_583.1655.clone.clone.21 | 3 |
| wide.region_728.1958.clone.2.clone.48.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.48.clone.clone | 3 |
| wide.region_728.1958.clone.2.clone.50.clone.clone | 8 |
| wide.region_729.1959.clone.2.clone.50.clone.clone | 3 |
| wide.region_766.2047.clone.clone.2.clone.sunk.clone | 11 |
| wide.region_767.2048.clone.clone.2.clone.clone | 3 |
| wide.wide.region_1161.2693.clone.clone.clone | 107 |
| wide.wide.region_1173.2694.clone.clone.clone | 3 |
| region_1178.2700 | 11 |
| region_1179.2701 | 3 |
| wide.wide.wide.wide.wide.region_706.2560.clone.clone.sunk.clone.sunk.clone.clone.sunk.clone | 306 |
| wide.wide.wide.wide.wide.region_1043.2561.clone.clone.clone.clone.clone.clone | 3 |
| wide.region_774.2071.clone.clone.48.clone | 15 |
| wide.region_778.2072.clone.clone.48.clone | 3 |
| wide.region_907.2394.clone.clone.48.clone | 15 |
| wide.region_911.2395.clone.clone.48.clone | 3 |
| region_1089.2612.sunk.clone | 16 |
| region_1094.2613.clone | 3 |
| region_2475.4720 | 11 |
| region_2476.4721 | 3 |
| wide.region_774.2071.clone.clone.50.clone | 15 |
| wide.region_778.2072.clone.clone.50.clone | 3 |
| wide.region_907.2394.clone.clone.50.clone | 15 |
| wide.region_911.2395.clone.clone.50.clone | 3 |
| region_2460.4709.sunk.clone | 16 |
| region_2465.4710.clone | 3 |
| wide.region_586.1665.clone.23 | 5 |
| wide.region_587.1666.clone.23 | 3 |
| wide.region_586.1665.clone.clone.22 | 5 |
| wide.region_587.1666.clone.clone.22 | 3 |
| wide.region_586.1665.clone.24 | 5 |
| wide.region_587.1666.clone.24 | 3 |
| wide.region_586.1665.clone.clone.23 | 5 |
| wide.region_587.1666.clone.clone.23 | 3 |
| wide.region_586.1665.clone.25 | 5 |
| wide.region_587.1666.clone.25 | 3 |
| wide.region_586.1665.clone.clone.24 | 5 |
| wide.region_587.1666.clone.clone.24 | 3 |
| wide.region_586.1665 | 5 |
| wide.region_587.1666 | 3 |
| wide.region_586.1665.clone | 5 |
| wide.region_587.1666.clone | 3 |
| wide.region_582.1654 | 5 |
| wide.region_583.1655 | 3 |
| wide.region_582.1654.clone | 5 |
| wide.region_583.1655.clone | 3 |
| region_613.1728 | 11 |
| region_614.1729 | 3 |
| wide.region_586.1665.clone.2 | 5 |
| wide.region_587.1666.clone.2 | 3 |
| wide.region_586.1665.clone.clone.1 | 5 |
| wide.region_587.1666.clone.clone.1 | 3 |
| wide.region_586.1665.clone.1 | 5 |
| wide.region_587.1666.clone.1 | 3 |
| wide.region_586.1665.clone.clone | 5 |
| wide.region_587.1666.clone.clone | 3 |
| wide.region_654.1831.clone | 5 |
| wide.region_655.1832.clone | 2 |
| wide.region_582.1654.clone.1 | 5 |
| wide.region_583.1655.clone.1 | 3 |
| wide.region_582.1654.clone.clone | 5 |
| wide.region_583.1655.clone.clone | 3 |
| region_645.1823 | 11 |
| region_646.1824 | 3 |
| region_651.1829 | 7 |
| region_652.1830 | 1 |
| region_653.1833 | 18 |
| region_656.1834 | 2 |
| wide.region_582.1654.clone.2 | 5 |
| wide.region_583.1655.clone.2 | 3 |
| wide.region_582.1654.clone.clone.1 | 5 |
| wide.region_583.1655.clone.clone.1 | 3 |
| wide.wide.region_641.1838.clone.clone.clone | 129 |
| wide.wide.region_660.1839.clone.clone.clone | 2 |
| region_665.1847 | 11 |
| region_666.1848 | 3 |
| wide.region_582.1654.clone.23 | 5 |
| wide.region_583.1655.clone.23 | 3 |
| wide.region_582.1654.clone.clone.22 | 5 |
| wide.region_583.1655.clone.clone.22 | 3 |
| wide.region_582.1654.clone.24 | 5 |
| wide.region_583.1655.clone.24 | 3 |
| wide.region_582.1654.clone.clone.23 | 5 |
| wide.region_583.1655.clone.clone.23 | 3 |
| wide.region_582.1654.clone.25 | 5 |
| wide.region_583.1655.clone.25 | 3 |
| wide.region_582.1654.clone.clone.24 | 5 |
| wide.region_583.1655.clone.clone.24 | 3 |
| wide.wide.region_609.1742.clone.clone.clone.clone.clone.clone | 129 |
| wide.wide.region_623.1743.clone.clone.clone.clone.clone | 3 |
| region_595.1705 | 11 |
| region_600.1706 | 3 |
| region_634.1812 | 11 |
| region_635.1813 | 3 |
| wide.wide.wide.wide.region_640.1853.clone.sunk.clone.clone.clone | 135 |
| wide.wide.wide.wide.region_671.1854.clone.clone.clone.clone | 2 |
| region_675.1858 | 11 |
| region_676.1859 | 3 |
| wide.region_234.639.clone.clone.2.sunk.clone | 8 |
| wide.region_235.640.clone.clone.2.clone | 3 |
| wide.region_385.974.clone.3.sunk.clone | 9 |
| wide.region_386.975.clone.3.clone | 3 |
| wide.region_385.974.clone.clone.2.sunk.clone | 9 |
| wide.region_386.975.clone.clone.2.clone | 3 |
| wide.region_387.999.clone.9.sunk.clone.clone.clone | 12 |
| wide.region_388.1000.clone.9.clone.clone.clone | 3 |
| wide.region_387.999.clone.clone.2.sunk.clone.clone.clone | 12 |
| wide.region_388.1000.clone.clone.2.clone.clone.clone | 3 |
| wide.region_292.785.clone.3.clone | 15 |
| wide.region_296.786.clone.3.clone | 3 |
| wide.region_487.1404.clone.3.clone | 15 |
| wide.region_491.1405.clone.3.clone | 3 |
| region_562.1598.sunk.clone | 29 |
| region_567.1599.clone | 3 |
| wide.region_17.66.clone.clone.clone | 8 |
| wide.region_18.67.clone.clone.clone | 3 |
| wide.region_17.66.clone.2.clone.2.clone.clone | 8 |
| wide.region_18.67.clone.2.clone.2.clone.clone | 3 |
| wide.region_17.66.clone.2.clone.3.clone.clone | 8 |
| wide.region_18.67.clone.2.clone.3.clone.clone | 3 |
| wide.region_17.66.clone.4.clone.6.clone.clone | 8 |
| wide.region_18.67.clone.4.clone.6.clone.clone | 3 |
| wide.region_17.66.clone.2.clone.4.clone.clone | 8 |
| wide.region_18.67.clone.2.clone.4.clone.clone | 3 |
| wide.region_56.158.sunk.clone.clone | 11 |
| wide.region_57.159.clone.2.clone | 3 |
| wide.region_65.184.clone.2 | 15 |
| wide.region_69.185.clone.2 | 3 |
| wide.region_206.559.clone.2 | 15 |
| wide.region_210.560.clone.2 | 3 |
| wide.wide.wide.wide.region_222.5661.clone.sunk.clone.sunk.clone.sunk.clone.clone.clone.clone | 512 |
| wide.wide.wide.wide.region_3394.5662.clone.clone.clone.clone.clone.clone.clone | 2 |
| region_3525.5862.sunk.clone | 16 |
| region_3530.5863.clone | 3 |
| wide.region_65.184.clone.clone.4.clone | 15 |
| wide.region_69.185.clone.clone.4.clone | 3 |
| wide.region_65.184.clone.clone.2.clone | 15 |
| wide.region_69.185.clone.clone.2.clone | 3 |
| wide.region_65.184.clone.clone.3.clone | 15 |
| wide.region_69.185.clone.clone.3.clone | 3 |
| wide.region_206.559.clone.clone.4.clone | 15 |
| wide.region_210.560.clone.clone.4.clone | 3 |
| wide.region_206.559.clone.clone.2.clone | 15 |
| wide.region_210.560.clone.clone.2.clone | 3 |
| wide.region_206.559.clone.clone.3.clone | 15 |
| wide.region_210.560.clone.clone.3.clone | 3 |
| region_3437.5765.sunk.clone | 16 |
| region_3442.5766.clone | 3 |
| region_3481.5816.sunk.clone | 16 |
| region_3486.5817.clone | 3 |
| wide.region_65.184.clone.1.clone.6.clone | 15 |
| wide.region_69.185.clone.1.clone.6.clone | 3 |
| wide.region_206.559.clone.1.clone.6.clone | 15 |
| wide.region_210.560.clone.1.clone.6.clone | 3 |
| region_3568.5906.sunk.clone | 16 |
| region_3573.5907.clone | 3 |
| wide.region_3588.8503.clone.clone.clone.clone | 55 |
| wide.region_6112.8504.clone.clone.clone | 3 |

## Top 30 functions by instructions

| cells | # | function | count | share | file:line |
|---:|---:|---|---:|---:|---|
| 300 | 1 | _FixedDesignNull2D._compatibility_census | 39,586 | 6.0% | forward_operator.py:783 |
| 300 | 2 | _quadrature_from_arrays | 34,971 | 5.3% | clip_quadrature.py:91 |
| 300 | 3 | _traced_clip | 31,585 | 4.8% | separatrix_clip.py:1216 |
| 300 | 4 | Topology.boundary | 29,424 | 4.5% | topology.py:635 |
| 300 | 5 | Null2D.interpolate | 21,325 | 3.3% | null.py:212 |
| 300 | 6 | clipped_support_current_moments | 18,972 | 2.9% | clip_quadrature.py:414 |
| 300 | 7 | _active_set_newton_krylov | 17,287 | 2.6% | fixed_point.py:3837 |
| 300 | 8 | Topology.read_qualification | 15,351 | 2.3% | topology.py:1077 |
| 300 | 9 | Topology.axis_component | 10,650 | 1.6% | topology.py:868 |
| 300 | 10 | _FixedDesignNull2D.read_census | 10,090 | 1.5% | forward_operator.py:871 |
| 300 | 11 | Topology.x_point_data | 9,856 | 1.5% | topology.py:421 |
| 300 | 12 | wall_coordinate | 9,692 | 1.5% | select.py:112 |
| 300 | 13 | hex_edge_admissibility | 9,294 | 1.4% | flux_surface_connectivity.py:341 |
| 300 | 14 | null_coordinate | 7,777 | 1.2% | select.py:277 |
| 300 | 15 | Topology._wall_anchor_selection | 7,573 | 1.2% | topology.py:496 |
| 300 | 16 | _canonicalize_reciprocal_hex_edges | 7,338 | 1.1% | connectivity_boundary.py:693 |
| 300 | 17 | traced_quadratic_surface | 7,313 | 1.1% | select.py:457 |
| 300 | 18 | reduce_sum | 7,243 | 1.1% | - |
| 300 | 19 | FluxFieldPolynomial.sample | 6,961 | 1.1% | stencil_mesh.py:145 |
| 300 | 20 | wall_height_shadow_mask | 6,488 | 1.0% | connectivity_boundary.py:451 |
| 300 | 21 | _points_inside_polygon | 6,147 | 0.9% | connectivity_boundary.py:208 |
| 300 | 22 | Null2D.categorize | 6,127 | 0.9% | null.py:179 |
| 300 | 23 | Bernstein.coefficent_matrix | 5,828 | 0.9% | interpolant.py:131 |
| 300 | 24 | _pack_traced_vertices | 5,791 | 0.9% | separatrix_clip.py:536 |
| 300 | 25 | Topology.o_point_qualification | 5,762 | 0.9% | topology.py:467 |
| 300 | 26 | Topology.qualified_o_candidates.<locals>.qualify | 5,061 | 0.8% | topology.py:917 |
| 300 | 27 | _integrate_current_points | 5,004 | 0.8% | clip_quadrature.py:279 |
| 300 | 28 | _propagate_admissible_hex_minima | 4,984 | 0.8% | flux_surface_connectivity.py:289 |
| 300 | 29 | reduce_window_sum | 4,876 | 0.7% | - |
| 300 | 30 | _rebuilt_model_promotion | 4,454 | 0.7% | fixed_point.py:1337 |
| 1000 | 1 | _FixedDesignNull2D._compatibility_census | 39,589 | 6.0% | forward_operator.py:783 |
| 1000 | 2 | _quadrature_from_arrays | 34,971 | 5.3% | clip_quadrature.py:91 |
| 1000 | 3 | _traced_clip | 31,578 | 4.8% | separatrix_clip.py:1216 |
| 1000 | 4 | Topology.boundary | 29,424 | 4.4% | topology.py:635 |
| 1000 | 5 | Null2D.interpolate | 21,325 | 3.2% | null.py:212 |
| 1000 | 6 | clipped_support_current_moments | 19,875 | 3.0% | clip_quadrature.py:414 |
| 1000 | 7 | _active_set_newton_krylov | 17,201 | 2.6% | fixed_point.py:3837 |
| 1000 | 8 | Topology.read_qualification | 15,481 | 2.3% | topology.py:1077 |
| 1000 | 9 | Topology.axis_component | 10,617 | 1.6% | topology.py:864 |
| 1000 | 10 | _FixedDesignNull2D.read_census | 10,091 | 1.5% | forward_operator.py:871 |
| 1000 | 11 | Topology.x_point_data | 9,856 | 1.5% | topology.py:421 |
| 1000 | 12 | wall_coordinate | 9,701 | 1.5% | select.py:112 |
| 1000 | 13 | hex_edge_admissibility | 9,406 | 1.4% | flux_surface_connectivity.py:341 |
| 1000 | 14 | reduce_sum | 8,023 | 1.2% | - |
| 1000 | 15 | null_coordinate | 7,777 | 1.2% | select.py:277 |
| 1000 | 16 | Topology._wall_anchor_selection | 7,576 | 1.1% | topology.py:496 |
| 1000 | 17 | _canonicalize_reciprocal_hex_edges | 7,338 | 1.1% | connectivity_boundary.py:693 |
| 1000 | 18 | traced_quadratic_surface | 7,312 | 1.1% | select.py:457 |
| 1000 | 19 | FluxFieldPolynomial.sample | 6,961 | 1.1% | stencil_mesh.py:145 |
| 1000 | 20 | wall_height_shadow_mask | 6,488 | 1.0% | connectivity_boundary.py:451 |
| 1000 | 21 | _points_inside_polygon | 6,147 | 0.9% | connectivity_boundary.py:208 |
| 1000 | 22 | Null2D.categorize | 6,127 | 0.9% | null.py:179 |
| 1000 | 23 | Bernstein.coefficent_matrix | 5,828 | 0.9% | interpolant.py:131 |
| 1000 | 24 | _pack_traced_vertices | 5,791 | 0.9% | separatrix_clip.py:536 |
| 1000 | 25 | Topology.o_point_qualification | 5,762 | 0.9% | topology.py:467 |
| 1000 | 26 | reduce_window_sum | 5,351 | 0.8% | - |
| 1000 | 27 | Topology.qualified_o_candidates.<locals>.qualify | 5,104 | 0.8% | topology.py:919 |
| 1000 | 28 | _integrate_current_points | 5,004 | 0.8% | clip_quadrature.py:279 |
| 1000 | 29 | _propagate_admissible_hex_minima | 4,984 | 0.8% | flux_surface_connectivity.py:289 |
| 1000 | 30 | _rebuilt_model_promotion | 4,441 | 0.7% | fixed_point.py:1337 |

## Top 30 functions by bytes

| cells | # | function | count | share | file:line |
|---:|---:|---|---:|---:|---|
| 300 | 1 | _quadrature_from_arrays | 17.60 GiB | 20.1% | clip_quadrature.py:91 |
| 300 | 2 | _integrate_current_points | 3.94 GiB | 4.5% | clip_quadrature.py:279 |
| 300 | 3 | _active_set_newton_krylov | 3.80 GiB | 4.3% | fixed_point.py:3837 |
| 300 | 4 | IsothermalRotation.centrifugal_exponent_gradient | 3.71 GiB | 4.2% | rotation.py:171 |
| 300 | 5 | _quadratic_flux_design | 3.32 GiB | 3.8% | stencil_mesh.py:435 |
| 300 | 6 | FluxFieldPolynomial.sample | 2.85 GiB | 3.3% | stencil_mesh.py:145 |
| 300 | 7 | mul | 2.49 GiB | 2.9% | - |
| 300 | 8 | RotatingDomainProfile.pressure_gradient | 2.21 GiB | 2.5% | rotation.py:272 |
| 300 | 9 | toroidal_current_density | 1.59 GiB | 1.8% | convention.py:70 |
| 300 | 10 | IsothermalRotation.centrifugal_exponent | 1.46 GiB | 1.7% | rotation.py:154 |
| 300 | 11 | _rebuilt_model_promotion | 1.38 GiB | 1.6% | fixed_point.py:1337 |
| 300 | 12 | _backtracked_promotion.<locals>.recover_with_continuation | 1.23 GiB | 1.4% | fixed_point.py:1188 |
| 300 | 13 | _backtracking_scores | 1.13 GiB | 1.3% | fixed_point.py:989 |
| 300 | 14 | _rebuilt_model_promotion.<locals>.damping_trial | 1.02 GiB | 1.2% | fixed_point.py:1284 |
| 300 | 15 | clipped_support_current_moments | 1010.51 MiB | 1.1% | clip_quadrature.py:414 |
| 300 | 16 | hex_edge_admissibility | 918.47 MiB | 1.0% | flux_surface_connectivity.py:341 |
| 300 | 17 | IsothermalRotation.centrifugal_factor | 907.76 MiB | 1.0% | rotation.py:182 |
| 300 | 18 | _traced_clip | 784.18 MiB | 0.9% | separatrix_clip.py:1216 |
| 300 | 19 | IsothermalRotation.squared_radius_offset | 575.70 MiB | 0.6% | rotation.py:177 |
| 300 | 20 | _qualified_krylov_step | 471.25 MiB | 0.5% | fixed_point.py:1792 |
| 300 | 21 | Topology.axis_component | 442.50 MiB | 0.5% | topology.py:868 |
| 300 | 22 | sub | 423.01 MiB | 0.5% | - |
| 300 | 23 | min | 349.34 MiB | 0.4% | - |
| 300 | 24 | _pack_traced_vertices | 331.22 MiB | 0.4% | separatrix_clip.py:536 |
| 300 | 25 | _steepest_descent_promotion | 299.50 MiB | 0.3% | fixed_point.py:1399 |
| 300 | 26 | _projected_krylov_condition | 272.87 MiB | 0.3% | fixed_point.py:822 |
| 300 | 27 | _newton_krylov_inner.<locals>.newton_body.<locals>.attempt_step.<locals>.promoted_state.<locals>.carry_fallback_sequence | 233.65 MiB | 0.3% | fixed_point.py:3151 |
| 300 | 28 | _traced_polygon_moments | 222.09 MiB | 0.2% | separatrix_clip.py:798 |
| 300 | 29 | _propagate_admissible_hex_minima | 162.33 MiB | 0.2% | flux_surface_connectivity.py:289 |
| 300 | 30 | add_any | 147.34 MiB | 0.2% | - |
| 1000 | 1 | _quadrature_from_arrays | 49.94 GiB | 16.6% | clip_quadrature.py:91 |
| 1000 | 2 | _active_set_newton_krylov | 14.10 GiB | 4.7% | fixed_point.py:3837 |
| 1000 | 3 | _integrate_current_points | 11.19 GiB | 3.7% | clip_quadrature.py:279 |
| 1000 | 4 | IsothermalRotation.centrifugal_exponent_gradient | 10.54 GiB | 3.5% | rotation.py:171 |
| 1000 | 5 | _quadratic_flux_design | 9.41 GiB | 3.1% | stencil_mesh.py:435 |
| 1000 | 6 | FluxFieldPolynomial.sample | 8.09 GiB | 2.7% | stencil_mesh.py:145 |
| 1000 | 7 | clipped_support_current_moments | 7.69 GiB | 2.6% | clip_quadrature.py:414 |
| 1000 | 8 | mul | 7.09 GiB | 2.4% | - |
| 1000 | 9 | _rebuilt_model_promotion | 6.55 GiB | 2.2% | fixed_point.py:1337 |
| 1000 | 10 | RotatingDomainProfile.pressure_gradient | 6.29 GiB | 2.1% | rotation.py:272 |
| 1000 | 11 | _backtracked_promotion.<locals>.recover_with_continuation | 5.08 GiB | 1.7% | fixed_point.py:1188 |
| 1000 | 12 | toroidal_current_density | 4.53 GiB | 1.5% | convention.py:70 |
| 1000 | 13 | _backtracking_scores | 4.49 GiB | 1.5% | fixed_point.py:989 |
| 1000 | 14 | IsothermalRotation.centrifugal_exponent | 4.14 GiB | 1.4% | rotation.py:154 |
| 1000 | 15 | _rebuilt_model_promotion.<locals>.damping_trial | 3.96 GiB | 1.3% | fixed_point.py:1284 |
| 1000 | 16 | hex_edge_admissibility | 3.62 GiB | 1.2% | flux_surface_connectivity.py:341 |
| 1000 | 17 | IsothermalRotation.centrifugal_factor | 2.52 GiB | 0.8% | rotation.py:182 |
| 1000 | 18 | _traced_clip | 2.20 GiB | 0.7% | separatrix_clip.py:1216 |
| 1000 | 19 | Topology.axis_component | 1.64 GiB | 0.5% | topology.py:864 |
| 1000 | 20 | _qualified_krylov_step | 1.64 GiB | 0.5% | fixed_point.py:1792 |
| 1000 | 21 | IsothermalRotation.squared_radius_offset | 1.60 GiB | 0.5% | rotation.py:177 |
| 1000 | 22 | sub | 1.17 GiB | 0.4% | - |
| 1000 | 23 | _projected_krylov_condition | 1.13 GiB | 0.4% | fixed_point.py:822 |
| 1000 | 24 | _steepest_descent_promotion | 1.09 GiB | 0.4% | fixed_point.py:1399 |
| 1000 | 25 | min | 987.70 MiB | 0.3% | - |
| 1000 | 26 | _pack_traced_vertices | 951.94 MiB | 0.3% | separatrix_clip.py:536 |
| 1000 | 27 | _newton_krylov_inner.<locals>.newton_body.<locals>.attempt_step.<locals>.promoted_state.<locals>.carry_fallback_sequence | 835.57 MiB | 0.3% | fixed_point.py:3151 |
| 1000 | 28 | _propagate_admissible_hex_minima | 656.11 MiB | 0.2% | flux_surface_connectivity.py:289 |
| 1000 | 29 | _traced_polygon_moments | 638.52 MiB | 0.2% | separatrix_clip.py:798 |
| 1000 | 30 | _canonicalize_reciprocal_hex_edges | 455.34 MiB | 0.1% | connectivity_boundary.py:693 |

## Flux-map top 10 by instructions

| cells | # | function | count | share | file:line |
|---:|---:|---|---:|---:|---|
| 300 | 1 | _FixedDesignNull2D._compatibility_census | 802 | 8.1% | forward_operator.py:783 |
| 300 | 2 | _traced_clip | 758 | 7.7% | separatrix_clip.py:1216 |
| 300 | 3 | Topology.boundary | 623 | 6.3% | topology.py:635 |
| 300 | 4 | Null2D.interpolate | 450 | 4.5% | null.py:212 |
| 300 | 5 | _quadrature_from_arrays | 415 | 4.2% | clip_quadrature.py:91 |
| 300 | 6 | Topology.read_qualification | 330 | 3.3% | topology.py:1009 |
| 300 | 7 | wall_height_shadow_mask | 314 | 3.2% | connectivity_boundary.py:451 |
| 300 | 8 | _FixedDesignNull2D.read_census | 224 | 2.3% | forward_operator.py:871 |
| 300 | 9 | Topology.x_point_data | 201 | 2.0% | topology.py:421 |
| 300 | 10 | _canonicalize_reciprocal_hex_edges | 188 | 1.9% | connectivity_boundary.py:693 |
| 1000 | 1 | _FixedDesignNull2D._compatibility_census | 802 | 8.0% | forward_operator.py:783 |
| 1000 | 2 | _traced_clip | 758 | 7.6% | separatrix_clip.py:1216 |
| 1000 | 3 | Topology.boundary | 623 | 6.2% | topology.py:635 |
| 1000 | 4 | Null2D.interpolate | 450 | 4.5% | null.py:212 |
| 1000 | 5 | _quadrature_from_arrays | 415 | 4.1% | clip_quadrature.py:91 |
| 1000 | 6 | Topology.read_qualification | 331 | 3.3% | topology.py:1009 |
| 1000 | 7 | wall_height_shadow_mask | 314 | 3.1% | connectivity_boundary.py:451 |
| 1000 | 8 | _FixedDesignNull2D.read_census | 224 | 2.2% | forward_operator.py:871 |
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
| 300 | 6 | IsothermalRotation.centrifugal_exponent_gradient | 18.42 MiB | 2.3% | rotation.py:171 |
| 300 | 7 | _traced_clip | 17.87 MiB | 2.3% | separatrix_clip.py:1216 |
| 300 | 8 | hex_edge_admissibility | 16.52 MiB | 2.1% | flux_surface_connectivity.py:341 |
| 300 | 9 | mul | 14.72 MiB | 1.9% | - |
| 300 | 10 | RotatingDomainProfile.pressure_gradient | 11.06 MiB | 1.4% | rotation.py:272 |
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

The warm-up call took 73.170 s; one subsequent public cache-hit call took 36.506 s on CPU and ended at residual 0.00515597 (converged=False). The table is cProfile cumulative host time; device completion is synchronised by the public result conversion.

| function | calls | self ms | cumulative ms | source |
|---|---:|---:|---:|---|
| `solve_reduced_newton_compiled` | 1 | 0.236 | 36505.235 | reduced_newton.py:2896 |
| `_compiled_result` | 1 | 2.076 | 36503.438 | reduced_newton.py:2824 |
| `_compiled_output_fields` | 1 | 0.094 | 36367.701 | reduced_newton.py:2717 |
| `_compiled_program` | 1 | 0.027 | 88.621 | reduced_newton.py:2739 |
| `reduced_coordinates` | 1 | 0.081 | 88.471 | reduced_newton.py:356 |
| `external` | 1 | 0.271 | 1.142 | forward_operator.py:1896 |

The cache-entry repair must move the coordinate construction at nova/equilibrium/reduced_newton.py:356 `reduced_coordinates` and the exterior-flux derivation at nova/equilibrium/forward_operator.py:1896 `ForwardFluxOperator.external` ahead of the reusable program lookup at nova/equilibrium/reduced_newton.py:2784 `_compiled_program`; it must not change the compiled program key or terminal identity.

## Exact implementation attack list

| implement node | exact source seams | measured removal target |
|---|---|---|
| Mesh arrays as program arguments | nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` closes the operator into the solve; nova/equilibrium/forward_operator.py:2679 `ForwardFluxOperator.traced_flux_map` closes the map. | Move at least 2.92 GiB of 1000-cell interaction, wall/sample, moment-geometry and connectivity literals from constants to arguments; compare against the recorded 462 MiB / 3.50 GiB / 19.00 GiB executable and generated-code ladder. |
| Flux functions and target current as traced arguments | nova/equilibrium/source.py:297 `DomainProfile.pressure_gradient` and nova/equilibrium/source.py:621 `ForwardSource.current_moments` feed the static profile closure; nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` includes target current in the program key. | Remove 0 B of directly classified profile/current literals at 1000 cells plus their folded descendants; the separate coefficient audit identifies two f64 amplitudes (16 B) and the target current as the root traced arguments. |
| Program-size budget with scans | nova/equilibrium/reduced_newton.py:1610 `_compiled_slice_solver` is already the compiled trip `fori_loop`; nova/equilibrium/forward.py:1801 `ForwardProfile._accelerated_history_program` is the straight-line closure seam. | Do not rewrite host loops as a size fix: they contribute zero HLO copies. Gate the 662,331-instruction solve against the 10,024-instruction map and remove 120 current-moment plus 8 topology-read traced copies by hoisting/reusing those bodies. |
| Cache-entry overhead | nova/equilibrium/reduced_newton.py:2896 `solve_reduced_newton_compiled` derives per-call inputs; nova/equilibrium/reduced_newton.py:356 `reduced_coordinates` builds reduced coordinates before the lookup. | Hoist the measured coordinate and exterior derivations from 88.471 ms coordinate and 1.142 ms exterior derivations; preserve the reusable executable key and terminal identity. |