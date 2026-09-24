# Chord booking operation trace

The whole-cell profile carrier is not itself the current-booking error. The first current moment is written when _flux_selected_current_moments clips that carrier with the cell-local quadratic psi_N=1 level, samples the profile density on the same local polynomial, and integrates the fitted density over that support. The production-local level admits analytic-exterior cells and under-covers analytic separatrix-cut cells.

Production source: `nova/equilibrium/clip_quadrature.py:885`.

Despite the route name, `chord` supplies a whole-cell carrier. The traced admission is the subsequent local quadratic level test; no straight chord segment decides these moments.

| Realised cells | False-positive cells | Max booked-current reproduction error | Base map sup | Analytic-condition map sup |
|---:|---|---:|---:|---:|
| 550 | 3, 7, 64 | 1.133e-13 | 0.0338673254 | 0.00704606286 |

## Traced moments

| Cells | Cell | Selection | Carrier | Production crossings | Production density points inside | Analytic points inside | Analytic A | Booked A | Conditioned A |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
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
