# Chord booking operation trace

The whole-cell profile carrier is not itself the current-booking error. The first current moment is written when _flux_selected_current_moments clips that carrier with the cell-local quadratic psi_N=1 level, samples the profile density on the same local polynomial, and integrates the fitted density over that support. The production-local level admits analytic-exterior cells and under-covers analytic separatrix-cut cells.

Production source: `nova/equilibrium/clip_quadrature.py:885`.

Despite the route name, `chord` supplies a whole-cell carrier. The traced admission is the subsequent local quadratic level test; no straight chord segment decides these moments.

| Realised cells | False-positive cells | Max booked-current reproduction error | Base map sup | Analytic-condition map sup |
|---:|---|---:|---:|---:|
| 132 | 2 | 4.729e-14 | 0.0950848225 | 0.27436323 |

## Traced moments

| Cells | Cell | Selection | Carrier | Production crossings | Production density points inside | Analytic points inside | Analytic A | Booked A | Conditioned A |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 132 | 2 | booked-current-with-zero-analytic-current | common_sol | 2 | 12 | 11 | 0 | 236.205099 | 82.364447 |
| 132 | 97 | largest-separatrix-cut-deficit | core | 2 | 24 | 24 | 658.342494 | 655.521254 | 486.551329 |
| 132 | 73 | largest-separatrix-cut-deficit | core | 2 | 24 | 24 | 667.207262 | 664.490631 | 504.606229 |
| 132 | 21 | largest-separatrix-cut-deficit | core | 2 | 24 | 24 | 545.03171 | 542.414024 | 444.583321 |
| 132 | 109 | largest-separatrix-cut-deficit | core | 2 | 22 | 22 | 581.132306 | 578.703667 | 721.280376 |
| 132 | 110 | largest-separatrix-cut-deficit | core | 2 | 25 | 25 | 652.141664 | 649.89559 | 687.552573 |
| 132 | 4 | largest-separatrix-cut-deficit | core | 2 | 16 | 16 | 293.869993 | 291.762578 | 309.576841 |
| 132 | 122 | largest-separatrix-cut-deficit | core | 2 | 25 | 25 | 504.620535 | 502.697811 | 531.825694 |
| 132 | 19 | largest-separatrix-cut-deficit | core | 2 | 25 | 25 | 586.300012 | 584.460985 | 618.326482 |
| 132 | 69 | largest-separatrix-cut-deficit | core | 2 | 16 | 16 | 410.091148 | 408.508359 | 412.285559 |
| 132 | 11 | largest-separatrix-cut-deficit | core | 2 | 14 | 14 | 266.09183 | 264.689612 | 442.87287 |

The full 25-point density values, edge-crossing records, support areas, first moments, and every stage current are retained in `report.json` and the row receipts.

## Declared negative control

apply the analytic in-plasma condition at the named stage and show the exterior current removed
