# Solve program size gate

| cells | solve instructions before / after | map floor before / after | compile seconds before / after | executable bytes before / after | solve copies moment / topology | map copies moment / topology |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 654,832 / 203,078 | 9,908 / 9,930 | 370.224 / 20.058 | 461,724,765 / 436,447,170 | 120 / 2 | 1 / 1 |
| 1000 | 662,331 / 204,531 | 10,024 / 10,046 | 386.402 / 69.721 | unavailable / unavailable | 120 / 2 | 1 / 1 |

Verdict: **PASS**.

Measured findings:
- 1000-cell solve executable size is absent: JaxRuntimeError: INTERNAL: Failed to serialize CpuAotCompilationResult.
