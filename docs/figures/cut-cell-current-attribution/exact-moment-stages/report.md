# Exact cut-cell moment stage attribution

booked minus incoming-polygon analytic = secondary_geometry + density_evaluation + moment_reduction

The incoming exact support is compared with the effective polygon after the production quadratic normalized-flux reclip. Density evaluation is then separated from the polynomial moment reduction on that same polygon.

| Case | Cells | Simple cuts | Geometry A | Density A | Moment reduction A | Named stage | Share | Closure relative | Corrected exact fraction |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| weak-rotation-reactor-static | 110 | 41 | -1871.55758 | 6.42103259e-10 | -1009496.07 | moment_reduction | 0.998149478 | 2.3e-16 | 0.999864704 |

## Controls

| Case | Cells | Changed simple-cut vertex sets | Max vertex distance m | Order 8 to 16 L1 A | Interior max geometry/current | Interior max density/current | Interior max reduction/current |
|---|---:|---:|---:|---:|---:|---:|---:|
| weak-rotation-reactor-static | 110 | 34 | 0.0610995113 | 7.51242624e-10 | 0 | 3.43e-16 | 4.01e-16 |

The declared negative control replaces the named stage term with its independent reference. Both weak-rotation rows must then reach an exact support fraction of at least 0.999. Every cell's incoming and effective vertices, area, three physical moments, stage terms, and closure are retained in report.json. No solve and no production source change ran.
