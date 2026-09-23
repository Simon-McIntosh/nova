# Recovery-ladder executions per certificate solve

Fence base revision `52bfefc0a0f02412d8910`.

Worktree head `db634db22585d27ff5990f60fd3d6b780cd253bc`.

Complete: `True`; missing rows: `none`.

**Verdict.** no measured row converged, so ladder activity is reported on non-converging rows only

**Family verdict.** every counted ladder rung fires in at least half the trips on the measured rows, none of which converged, so the ladders are a hot path rather than a rare fallback

| path | instructions | share of 302553 |
| --- | --- | --- |
| `_rebuilt_model_promotion` | 40237 | 13.2992% |
| `_steepest_descent_promotion` | 17352 | 5.7352% |

## `diverted-single-null` at -300 requested cells

Trips: 6.
Converged `None`; termination `None`.

| path | executions | per trip | trips |
| --- | --- | --- | --- |
| `_rebuilt_model_promotion` | 22 | 3.667 | 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 |
| `_steepest_descent_promotion` | 12 | 2.000 | 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6 |

Terminal fixed-point residual: `0.27948523220464483`.

## `moderate-rotation-conventional-static` at -300 requested cells

Trips: 11.
Converged `False`; termination `active_set_settled`.

| path | executions | per trip | trips |
| --- | --- | --- | --- |
| `_rebuilt_model_promotion` | 84 | 7.636 | 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 |
| `_steepest_descent_promotion` | 73 | 6.636 | 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11 |

Terminal fixed-point residual: `0.0497672393883413`.

## `strong-rotation-compact-static` at -300 requested cells

Trips: 8.
Converged `False`; termination `active_set_settled`.

| path | executions | per trip | trips |
| --- | --- | --- | --- |
| `_rebuilt_model_promotion` | 48 | 6.000 | 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8 |
| `_steepest_descent_promotion` | 39 | 4.875 | 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8 |

Terminal fixed-point residual: `0.02510889338452445`.

## `weak-rotation-reactor-static` at -300 requested cells

Trips: 16.
Converged `False`; termination `active_set_settled`.

| path | executions | per trip | trips |
| --- | --- | --- | --- |
| `_rebuilt_model_promotion` | 155 | 9.688 | 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 |
| `_steepest_descent_promotion` | 135 | 8.438 | 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16 |

Terminal fixed-point residual: `0.05529921020198606`.

