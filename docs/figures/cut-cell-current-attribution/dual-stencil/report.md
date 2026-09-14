# Dual-stencil stationary-point census

Two censuses read the analytic single-null flux on the same cached carriers.  The cell-centred census fits a quadratic on each centroid and its six sampling vertices; the vertex-centred dual fits a quadratic on every mesh vertex, its three surrounding cell centroids and its three adjacent vertices.  Each reports the closed-form stationary point and Hessian class, admitted inside or within 0.25 pitch of the owned region (the cell polygon for the cell census, the triangle of the three surrounding centroids for the vertex census).

## Analytic single-null ladder (X-point)

| requested | realised | edge_dist/pitch | vertex_dist/pitch | cell_admitted | dual_admitted | cell_error_m | cell_error/pitch | dual_error_m | dual_error/pitch | position_difference/pitch |
|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|
| 110 | 132 | 0.03521 | 0.07012 | yes | no | 0.009631294188974304 | 0.06359226084086127 | None | None | — |
| 200 | 233 | 0.04641 | 0.2114 | yes | yes | 0.005519659311168668 | 0.0491417416787003 | 0.0013611760243863592 | 0.012118603123617806 | 0.05692093831045778 |
| 300 | 340 | 0.07088 | 0.1101 | yes | yes | 0.0028766933044118105 | 0.03136732357681878 | 0.000609013468746447 | 0.006640653178951441 | 0.03437780886353059 |
| 342 | 382 | 0.1496 | 0.1848 | yes | yes | 0.00290678153701794 | 0.03384143108583095 | 0.0005631566253689956 | 0.00655640125865997 | 0.0288979117833101 |
| 400 | 449 | 0.07027 | 0.1346 | yes | yes | 0.0024767275056051465 | 0.03118397668656098 | 0.00048800167000959534 | 0.006144330640388243 | 0.03404569552361054 |
| 500 | 550 | 0.3037 | 0.4246 | yes | yes | 0.000725194659683926 | 0.010208522107157827 | 0.0009702860212751962 | 0.013658658632112923 | 0.01875540497117706 |
| 750 | 814 | 0.2199 | 0.3116 | yes | yes | 0.0006898824779947501 | 0.011894029750875239 | 0.0006623190387597753 | 0.01141881784630542 | 0.007077160365647527 |
| 1000 | 1074 | 0.01599 | 0.1046 | yes | yes | 0.0013580924762046876 | 0.027036633715538137 | 0.0001731144080478393 | 0.0034463270530380417 | 0.026288165257011827 |
| 2500 | 2616 | 0.04134 | 0.06828 | yes | yes | 0.0005012584747705756 | 0.01577811099102676 | 6.980524155660869e-05 | 0.0021972593072659974 | 0.01705235845776121 |

## Axis admission and position error

| case | requested | realised | edge_dist/pitch | cell_admitted | dual_admitted | cell_error/pitch | dual_error/pitch | position_difference/pitch |
|:---|---:|---:|---:|:---:|:---:|---:|---:|---:|
| diverted-single-null | 110 | 132 | 0.1905 | yes | yes | 0.00423212214272957 | 0.006758718061577306 | — |
| diverted-single-null | 200 | 233 | 0.2813 | yes | yes | 0.01007534259255847 | 0.010031866959745924 | — |
| diverted-single-null | 300 | 340 | 0.211 | yes | yes | 0.007497549669006078 | 0.008630238885927653 | — |
| diverted-single-null | 342 | 382 | 0.4588 | yes | yes | 0.009500784325075314 | 0.010820442666580107 | — |
| diverted-single-null | 400 | 449 | 0.3429 | yes | yes | 0.007403354782491315 | 0.005813643298437964 | — |
| diverted-single-null | 500 | 550 | 0.1311 | yes | yes | 0.009840240672171926 | 0.006844119867683767 | — |
| diverted-single-null | 750 | 814 | 0.2249 | yes | yes | 0.000988851242752107 | 0.0008726587766337276 | — |
| diverted-single-null | 1000 | 1074 | 0.2924 | yes | yes | 0.004932126153327788 | 0.005331876976747046 | — |
| diverted-single-null | 2500 | 2616 | 0.2493 | yes | yes | 0.0027104059677998706 | 0.0034130283926661837 | — |
| weak-rotation-reactor-static | 300 | 342 | 0.1095 | yes | yes | 0.0012566385851719587 | 0.0012215054508312893 | — |
| weak-rotation-reactor-static | 1000 | 1072 | 0.2686 | yes | yes | 0.003589792874510207 | 0.0002941008558638821 | — |
| moderate-rotation-conventional-static | 300 | 341 | 0.1788 | yes | yes | 0.0019683960913502185 | 0.0024801047602233846 | — |
| moderate-rotation-conventional-static | 1000 | 1070 | 0.2686 | yes | yes | 0.003042767728846065 | 0.0024699759889590662 | — |
| strong-rotation-compact-static | 300 | 341 | 0.0924 | yes | yes | 0.0017495949170170174 | 0.0014242668905007964 | — |
| strong-rotation-compact-static | 1000 | 1069 | 0.02361 | yes | yes | 0.002736445053294464 | 0.0018504608232849983 | — |

## False candidates before and after the filters

| case | requested | census | sign_saddle | hessian_saddle | contained_saddle | sign_extremum | hessian_extremum | contained_extremum |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|
| diverted-single-null | 110 | cell | 0 | 51 | 3 | 0 | 0 | 1 |
| diverted-single-null | 110 | vertex | 83 | 24 | 0 | 0 | 3 | 3 |
| diverted-single-null | 200 | cell | 0 | 86 | 3 | 0 | 0 | 1 |
| diverted-single-null | 200 | vertex | 205 | 63 | 2 | 0 | 3 | 5 |
| diverted-single-null | 300 | cell | 0 | 122 | 3 | 0 | 0 | 1 |
| diverted-single-null | 300 | vertex | 311 | 107 | 2 | 0 | 4 | 3 |
| diverted-single-null | 342 | cell | 0 | 137 | 3 | 0 | 0 | 1 |
| diverted-single-null | 342 | vertex | 354 | 129 | 2 | 0 | 0 | 6 |
| diverted-single-null | 400 | cell | 0 | 164 | 3 | 0 | 0 | 1 |
| diverted-single-null | 400 | vertex | 471 | 173 | 2 | 0 | 0 | 6 |
| diverted-single-null | 500 | cell | 0 | 194 | 2 | 0 | 0 | 1 |
| diverted-single-null | 500 | vertex | 561 | 219 | 4 | 0 | 4 | 2 |
| diverted-single-null | 750 | cell | 0 | 284 | 2 | 0 | 0 | 1 |
| diverted-single-null | 750 | vertex | 886 | 344 | 2 | 0 | 3 | 3 |
| diverted-single-null | 1000 | cell | 0 | 371 | 3 | 0 | 0 | 1 |
| diverted-single-null | 1000 | vertex | 1189 | 483 | 2 | 0 | 3 | 4 |
| diverted-single-null | 2500 | cell | 0 | 844 | 3 | 0 | 0 | 1 |
| diverted-single-null | 2500 | vertex | 3199 | 1317 | 1 | 0 | 3 | 3 |
| weak-rotation-reactor-static | 300 | cell | 0 | 0 | 0 | 0 | 0 | 1 |
| weak-rotation-reactor-static | 300 | vertex | 354 | 0 | 0 | 0 | 4 | 2 |
| weak-rotation-reactor-static | 1000 | cell | 0 | 0 | 0 | 0 | 0 | 1 |
| weak-rotation-reactor-static | 1000 | vertex | 1230 | 0 | 0 | 0 | 2 | 4 |
| moderate-rotation-conventional-static | 300 | cell | 0 | 0 | 0 | 0 | 0 | 1 |
| moderate-rotation-conventional-static | 300 | vertex | 342 | 0 | 0 | 0 | 4 | 3 |
| moderate-rotation-conventional-static | 1000 | cell | 0 | 0 | 0 | 0 | 0 | 1 |
| moderate-rotation-conventional-static | 1000 | vertex | 1239 | 0 | 0 | 0 | 3 | 3 |
| strong-rotation-compact-static | 300 | cell | 0 | 0 | 0 | 0 | 0 | 1 |
| strong-rotation-compact-static | 300 | vertex | 355 | 0 | 0 | 0 | 4 | 2 |
| strong-rotation-compact-static | 1000 | cell | 0 | 0 | 0 | 0 | 0 | 1 |
| strong-rotation-compact-static | 1000 | vertex | 1245 | 0 | 0 | 0 | 4 | 2 |

## Position error binned by null-to-boundary distance

| distance bin (pitch) | nulls | cell mean error/pitch | dual mean error/pitch | fraction dual smaller |
|:---|---:|---:|---:|---:|
| <0.125 | 9 | 0.02487 | inf | 0.89 |
| 0.125–0.25 | 8 | 0.00912 | 0.00587 | 0.50 |
| 0.25–0.5 | 7 | 0.00696 | 0.00692 | 0.57 |
| 0.5–1 | 0 | — | — | — |
| ≥1 | 0 | — | — | — |

Across all 24 null observations the dual errors were the smaller of the two on 16 (67%).

## Controls and figures

- Manufactured quadratic stationary points are polished to 1e-10 m by both censuses: cell-centred True, vertex-centred True.
- Smooth perturbation amplitude is 1e-4 of the analytic axis-to-X span; the cell-centred saddle admission survived on 9 of 9 rungs.
- [dual-stencil-error-by-boundary-distance](/nova/figures/cut-cell-current-attribution/dual-stencil/dual-stencil-error-by-boundary-distance.svg)
