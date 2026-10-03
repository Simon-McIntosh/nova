# Recovery-ladder executions per certificate solve

Fence base revision `52bfefc0a0f02412d8910`.

`nova/equilibrium/fixed_point.py` committing revision `f3a1f8e302b00ff0650aae021a1a236165242522`.

Census module committing revision `5721de656b968e95ec150603911c8fe29e6a4ba5`; worktree head `5721de656b968e95ec150603911c8fe29e6a4ba5`.

Complete: `True`; missing rows: `none`.

**Verdict.** no measured row converged, so ladder activity is reported on non-converging rows only

**Family verdict.** no ladder rung was observed in the measured rows, so no row indicates whether the ladders are a fallback or a hot path

| path | instructions | share of 302553 |
| --- | --- | --- |
| `_backtracking_scores` | 2614 | 0.8640% |
| `_backtracked_promotion` | 0 | 0.0000% |
| `_complete_newton_promotion` | 0 | 0.0000% |
| `_rebuilt_model_promotion` | 40237 | 13.2992% |
| `_steepest_descent_promotion` | 17352 | 5.7352% |

## `diverted-single-null` at -300 requested cells

Trips: 2.
Converged `False`; termination `active_set_settled`.

| path | executions | per trip | trips |
| --- | --- | --- | --- |
| `_backtracking_scores` | 0 | 0.000 | none |
| `_backtracked_promotion` | 0 | 0.000 | none |
| `_complete_newton_promotion` | 0 | 0.000 | none |
| `_rebuilt_model_promotion` | 0 | 0.000 | none |
| `_steepest_descent_promotion` | 0 | 0.000 | none |

Terminal fixed-point residual: `2.3408233665296123`.

