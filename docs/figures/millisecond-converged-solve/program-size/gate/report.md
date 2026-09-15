# Solve program size gate

| cells | before solve/map instructions | after solve/map instructions | after ratio | executable bytes | moment copies | topology copies |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 654,832/9,908 | 203,078/9,930 | 20.45 | 436,447,170 | 120 | 2 |
| 1000 | 662,331/10,024 | 204,531/10,046 | 20.36 | unavailable | 120 | 2 |

Verdict: **FAIL**.

Refusals:
- 300-cell solve 'current-moment path' has 120 traced copies
- 300-cell solve 'topology read' has 2 traced copies
- 1000-cell solve 'current-moment path' has 120 traced copies
- 1000-cell solve 'topology read' has 2 traced copies
- 1000-cell solve executable size is absent: JaxRuntimeError: INTERNAL: Failed to serialize CpuAotCompilationResult.
