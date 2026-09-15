# Solve program semantic and boundary gate

## Certificate terminal-state identity

| case | cells | state values | baseline / candidate seconds | bit identical |
|---|---:|---:|---:|:---:|
| weak-rotation-reactor-static | 300 | 1,529 | 104.176 / 41.160 | no |
| moderate-rotation-conventional-static | 300 | 1,535 | 102.042 / 40.852 | no |
| strong-rotation-compact-static | 300 | 1,525 | 104.710 / 40.575 | no |
| diverted-single-null | 500 | 2,226 | 164.327 / 46.692 | no |

## MAST compiled-slice boundary and terminal identity

| member | trips | banked boundary [ms/trip] | before compiled [ms/trip] | after compiled [ms/trip] | host difference [ULP] |
|---|---:|---:|---:|---:|---:|
| 21978/35 pure | 1 | 46.1 | 166.046 | 45.666 | 0 |
| 21978/35 mixed | 1 | 46.1 | 50.647 | 45.463 | 0 |
| 21983/35 pure | 1 | 46.1 | 54.761 | 51.089 | 0 |
| 21983/35 mixed | 1 | 46.1 | 60.158 | 51.076 | 0 |
| 21985/51 pure | 1 | 46.1 | 152.319 | 53.319 | 0 |
| 21985/51 mixed | 3 | 46.1 | 83.858 | 51.039 | 0 |
| 21986/46 pure | 2 | 46.1 | 76.520 | 50.451 | 0 |
| 21986/46 mixed | 3 | 46.1 | 53.627 | 51.699 | 0 |
| 21989/55 pure | 1 | 46.1 | 57.263 | 53.794 | 0 |
| 21989/55 mixed | 1 | 46.1 | 108.388 | 55.068 | 0 |
| 22086/43 pure | 1 | 46.1 | 63.666 | 46.148 | 0 |
| 22086/43 mixed | 1 | 46.1 | 50.578 | 45.078 | 0 |

Verdict: **FAIL**.
