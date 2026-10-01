# Diverted chord response attribution

Neither X-point current-centroid placement nor Green evaluation explains the non-monotone sequence: the archived booked-versus-analytic current moments reproduce it, with mesh-dependent signed cancellation between interior, separatrix-cut, X-point, wall-cut and exterior contributions.

1 - sum((error-correction)^2)/sum(error^2), uncentered squared field error; no fitted coefficients

Class projection shares are signed and additive, not independent causal percentages. Quadrature refinement bounds its numerical sensitivity, not a rigorous integration error. The exact upstream stage that changes the current moments is not identified by this fixed-current experiment.

| Case | Cells | Map sup | X centroid reduction | Green reduction | Quadrature delta/error |
|---|---:|---:|---:|---:|---:|
| diverted-single-null | 132 | 0.095084823 | 0.02084425 | -9.6687927e-08 | 1.5471498e-05 |
| weak-rotation-reactor-static | 135 | 9.5402893e-05 | 4.4408921e-16 | -4.8186926e-08 | 2.3565156e-06 |
| diverted-single-null | 340 | 0.022909979 | -0.0037490638 | 2.7158187e-09 | 2.6828293e-07 |
| weak-rotation-reactor-static | 342 | 9.596811e-05 | 1.3322676e-15 | -5.9556919e-09 | 3.4727577e-07 |
| diverted-single-null | 550 | 0.033867325 | -0.0019837381 | 7.7443443e-09 | 2.1892185e-06 |
| weak-rotation-reactor-static | 555 | 4.8458873e-05 | 0 | -4.2832204e-10 | 1.6109232e-07 |
| diverted-single-null | 1074 | 0.012675989 | 0.00033979762 | 3.1776256e-09 | 7.2486903e-07 |
| weak-rotation-reactor-static | 1072 | 4.6793186e-05 | 1.110223e-16 | 5.4055405e-10 | 4.5439081e-08 |
| diverted-single-null | 2616 | 0.0040918446 | -0.0050875713 | 2.0974655e-09 | 1.4474829e-07 |
| weak-rotation-reactor-static | 2608 | 4.8213841e-05 | 1.110223e-16 | 7.5531803e-12 | 2.2412679e-09 |

Signed class projection shares (sum to one):

| Cells | Interior | Separatrix-cut | X-point | Wall-cut | Exterior |
|---:|---:|---:|---:|---:|---:|
| 132 | 0.47598363 | 0.10350655 | -0.19809888 | 0.6186087 | 0 |
| 340 | 0.61892859 | 0.45900575 | 0.1424258 | -0.22036013 | 0 |
| 550 | 0.018444519 | 0.65338473 | -0.01949351 | 0.051459089 | 0.29620518 |
| 1074 | 0.7274022 | 0.23064578 | 0.00060767843 | 0.010524465 | 0.030819872 |
| 2616 | 0.97023821 | 0.33604249 | 0.06070106 | -0.023456776 | -0.34352498 |
