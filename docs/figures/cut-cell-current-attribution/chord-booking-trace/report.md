# Chord booking operation trace

The whole-cell profile carrier is not itself the current-booking error. The first current moment is written when _flux_selected_current_moments clips that carrier with the cell-local quadratic psi_N=1 level, samples the profile density on the same local polynomial, and integrates the fitted density over that support. The production-local level admits analytic-exterior cells and under-covers analytic separatrix-cut cells.

Production source: `nova/equilibrium/clip_quadrature.py:885`.

Despite the route name, `chord` supplies a whole-cell carrier. The traced admission is the subsequent local quadratic level test; no straight chord segment decides these moments.

| Realised cells | False-positive cells | Max booked-current reproduction error | Base map sup | Analytic-condition map sup |
|---:|---|---:|---:|---:|
| 132 | 2 | 2.274e-14 | 0.0950848225 | 0.0611907983 |
| 340 | 1, 12 | 4.636e-16 | 0.022909979 | 0.0764944851 |
| 550 | 3, 7, 64 | 2.552e-15 | 0.0338673254 | 0.00704606286 |

## Traced moments

| Cells | Cell | Selection | Carrier | Production crossings | Production density points inside | Analytic points inside | Analytic A | Booked A | Conditioned A |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 132 | 2 | booked-current-with-zero-analytic-current | common_sol | 2 | 24104 | 0 | 0 | 236.205099 | 0 |
| 132 | 97 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32960 | 658.342494 | 655.521254 | 660.12911 |
| 132 | 73 | largest-separatrix-cut-deficit | core | 2 | 32960 | 25741 | 667.207262 | 664.490631 | 669.063806 |
| 132 | 21 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32960 | 545.03171 | 542.414024 | 546.226815 |
| 132 | 109 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32518 | 581.132306 | 578.703667 | 582.755779 |
| 132 | 110 | largest-separatrix-cut-deficit | core | 2 | 33024 | 1445 | 652.141664 | 649.89559 | 654.423138 |
| 132 | 4 | largest-separatrix-cut-deficit | core | 2 | 166 | 4380 | 293.869993 | 291.762578 | 291.510403 |
| 132 | 122 | largest-separatrix-cut-deficit | core | 2 | 32960 | 28983 | 504.620535 | 502.697811 | 506.1863 |
| 132 | 19 | largest-separatrix-cut-deficit | core | 2 | 33024 | 5883 | 586.300012 | 584.460985 | 588.371363 |
| 132 | 69 | largest-separatrix-cut-deficit | core | 2 | 32896 | 28691 | 410.091148 | 408.508359 | 411.299917 |
| 132 | 11 | largest-separatrix-cut-deficit | core | 2 | 32960 | 12141 | 266.09183 | 264.689612 | 265.94037 |
| 340 | 1 | booked-current-with-zero-analytic-current | common_sol | 2 | 33024 | 0 | 0 | 12.6433959 | 0 |
| 340 | 12 | booked-current-with-zero-analytic-current | common_sol | 2 | 32832 | 0 | 0 | 10.5975119 | 0 |
| 340 | 15 | largest-separatrix-cut-deficit | common_sol | 2 | 73 | 16038 | 37.812056 | 3.81424458 | 6.76520037 |
| 340 | 21 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32960 | 136.792752 | 135.051393 | 135.436123 |
| 340 | 22 | largest-separatrix-cut-deficit | common_sol | 2 | 172 | 4747 | 76.5430542 | 75.9370671 | 76.1870639 |
| 340 | 36 | largest-separatrix-cut-deficit | core | 2 | 192 | 31390 | 154.387139 | 154.083633 | 154.744587 |
| 340 | 17 | largest-separatrix-cut-deficit | core | 2 | 32960 | 21532 | 100.196994 | 99.9564255 | 100.23104 |
| 340 | 24 | largest-separatrix-cut-deficit | common_sol | 2 | 122 | 6153 | 21.8627848 | 21.6495356 | 21.5189017 |
| 340 | 18 | largest-separatrix-cut-deficit | common_sol | 2 | 119 | 29296 | 7.6152844 | 7.4153581 | 7.79694632 |
| 340 | 27 | largest-separatrix-cut-deficit | core | 2 | 256 | 32620 | 195.458462 | 195.273876 | 195.911803 |
| 340 | 126 | largest-separatrix-cut-deficit | common_sol | 2 | 32832 | 30316 | 5.0372115 | 4.93148993 | 4.92970515 |
| 340 | 309 | largest-separatrix-cut-deficit | common_sol | 2 | 32768 | 32692 | 12.0282189 | 11.9323062 | 11.9660842 |
| 550 | 3 | booked-current-with-zero-analytic-current | common_sol | 2 | 32832 | 0 | 0 | 8.6485703 | 0 |
| 550 | 7 | booked-current-with-zero-analytic-current | common_sol | 2 | 32768 | 0 | 0 | 0.21866775 | 0 |
| 550 | 64 | booked-current-with-zero-analytic-current | common_sol | 2 | 32832 | 0 | 0 | 43.1533979 | 0 |
| 550 | 80 | largest-separatrix-cut-deficit | common_sol | 2 | 124 | 32832 | 35.0398564 | 32.5979583 | 33.4011382 |
| 550 | 18 | largest-separatrix-cut-deficit | common_sol | 2 | 117 | 32832 | 8.92277762 | 7.14790395 | 7.72432388 |
| 550 | 21 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32960 | 68.4952064 | 67.0148059 | 67.1035459 |
| 550 | 85 | largest-separatrix-cut-deficit | core | 2 | 256 | 32960 | 97.467488 | 96.196835 | 96.5658311 |
| 550 | 86 | largest-separatrix-cut-deficit | common_sol | 2 | 187 | 32896 | 65.4857184 | 64.3414368 | 64.9703805 |
| 550 | 522 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32960 | 81.8886099 | 80.8853867 | 80.9924938 |
| 550 | 529 | largest-separatrix-cut-deficit | common_sol | 2 | 32896 | 32896 | 55.4934077 | 54.5760036 | 54.6482722 |
| 550 | 432 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32960 | 61.9284761 | 61.0348361 | 61.1156575 |
| 550 | 25 | largest-separatrix-cut-deficit | core | 2 | 32960 | 32960 | 65.3074103 | 64.4248397 | 64.5101501 |
| 550 | 528 | largest-separatrix-cut-deficit | core | 2 | 32896 | 32896 | 79.9213181 | 79.0935557 | 79.1982902 |

Compact quadrature-density summaries, edge-crossing records, support areas, first moments, and every stage current are retained in `report.json` and the row receipts.

## Declared negative control

apply the analytic in-plasma condition at the named stage and show the exterior current removed

At 550 cells, cell 64 changes from 43.153397909 A to 0 A, a fraction 0.000e+00. The resulting map mismatch sup is reported above rather than inferred from the removed current.
