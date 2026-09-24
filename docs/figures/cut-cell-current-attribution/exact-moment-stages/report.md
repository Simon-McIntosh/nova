# Exact cut-cell moment stage attribution

booked minus incoming-polygon analytic = secondary_geometry + density_evaluation + reclip_self_intersection_loss + moment_reduction

The incoming exact support is compared with the effective polygon after the production quadratic normalized-flux reclip. Density evaluation is then separated from the polynomial moment reduction on that same polygon.

| Case | Cells | Simple cuts | Geometry A | Density A | Reclip self-intersection A | Moment reduction A | Named stage | Share | Closure relative | Corrected exact fraction |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| weak-rotation-reactor-static | 110 | 41 | -1871.55758 | -1.04591891e-10 | 0 | -1009496.07 | moment_reduction | 0.998149478 | 5.79e-14 | 0.999864704 |
| weak-rotation-reactor-static | 300 | 72 | -7006.59964 | -1.36601175e-10 | 2.18278728e-11 | -296458.034 | moment_reduction | 0.976911314 | 1.15e-15 | 0.999538490 |
| diverted-single-null | 110 | 35 | -23.7449975 | 3.26849658e-13 | 1.13686838e-13 | -1892.93664 | moment_reduction | 0.987611402 | 8.3e-16 | 1.004899883 |

![Signed stage shares](/nova/figures/cut-cell-current-attribution/exact-moment-stages/stage-attribution.svg)

## Controls

| Case | Cells | Self-intersections | Changed vertex sets | Max vertex distance m | Signed order 8 to 16 L1 A | Union order 8 to 16 L1 A | Interior max geometry/current | Interior max density/current | Interior max self-intersection/current | Interior max reduction/current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weak-rotation-reactor-static | 110 | 0 | 34 | 0.0610995113 | 5.99357008e-10 | 5.99357008e-10 | 0 | 3.43e-16 | 0 | 4.01e-16 |
| weak-rotation-reactor-static | 300 | 7 | 66 | 0.0770969528 | 4.45092407e-10 | 4.48730386e-10 | 0 | 4.19e-16 | 0 | 4.75e-16 |
| diverted-single-null | 110 | 2 | 28 | 0.0468022404 | 1.85806925e-12 | 2.02859951e-12 | 0 | 2.16e-16 | 0 | 4.32e-16 |

The declared negative control replaces the moment-reduction term with its signed-winding independent reference. Both weak-rotation rows must then reach an exact support fraction of at least 0.999. Every cell's incoming and effective vertices, area, three physical moments, stage terms, and closure are retained in report.json. No solve and no production source change ran.
