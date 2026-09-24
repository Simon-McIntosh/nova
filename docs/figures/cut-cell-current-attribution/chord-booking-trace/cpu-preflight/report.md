# Chord booking operation trace

The whole-cell profile carrier is not itself the current-booking error. The first current moment is written when _flux_selected_current_moments clips that carrier with the cell-local quadratic psi_N=1 level, samples the profile density on the same local polynomial, and integrates the fitted density over that support. The production-local level admits analytic-exterior cells and under-covers analytic separatrix-cut cells.

Production source: `nova/equilibrium/clip_quadrature.py:885`.

Despite the route name, `chord` supplies a whole-cell carrier. The traced admission is the subsequent local quadratic level test; no straight chord segment decides these moments.

| Realised cells | False-positive cells | Max booked-current reproduction error | Base map sup | Analytic-condition map sup |
|---:|---|---:|---:|---:|
| 132 | 2 | 4.729e-14 | 0.0950848225 | 0.103972106 |

## Traced moments

| Cells | Cell | Selection | Carrier | Production crossings | Production density points inside | Analytic points inside | Analytic A | Booked A | Conditioned A |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 132 | 2 | booked-current-with-zero-analytic-current | common_sol | 2 | 20530 | 19578 | 0 | 236.205099 | 235.894657 |
| 132 | 97 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32960 | 658.342494 | 655.521254 | 655.171208 |
| 132 | 73 | largest-separatrix-cut-deficit | core | 2 | 32960 | 25747 | 667.207262 | 664.490631 | 664.038947 |
| 132 | 21 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32960 | 545.03171 | 542.414024 | 542.124377 |
| 132 | 109 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32522 | 581.132306 | 578.703667 | 578.37932 |
| 132 | 110 | largest-separatrix-cut-deficit | core | 2 | 33024 | 1462 | 652.141664 | 649.89559 | 649.508181 |
| 132 | 4 | largest-separatrix-cut-deficit | core | 2 | 166 | 4379 | 293.869993 | 291.762578 | 289.319606 |
| 132 | 122 | largest-separatrix-cut-deficit | core | 2 | 32960 | 29022 | 504.620535 | 502.697811 | 502.385647 |
| 132 | 19 | largest-separatrix-cut-deficit | core | 2 | 33024 | 5875 | 586.300012 | 584.460985 | 583.952393 |
| 132 | 69 | largest-separatrix-cut-deficit | core | 2 | 32896 | 28695 | 410.091148 | 408.508359 | 408.210879 |
| 132 | 11 | largest-separatrix-cut-deficit | core | 2 | 32960 | 12141 | 266.09183 | 264.689612 | 263.943024 |

The full 25-point density values, edge-crossing records, support areas, first moments, and every stage current are retained in `report.json` and the row receipts.

## Declared negative control

apply the analytic in-plasma condition at the named stage and show the exterior current removed
