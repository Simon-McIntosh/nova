# Polish-method accuracy ladder

Each styled stationary-point polish is seeded at the analytic null's own cell on every requested rung and reports position error in characteristic pitch (realised cell count differs from the requested nominal).  ``field`` is the carrier as authored; ``shifted`` evaluates the same four methods on the analytic flux with all carriers translated by one third of a pitch in a non-axis-aligned direction.

## Position error in pitch per method, rung and null

| case | requested | realised | method | axis err/pitch | saddle err/pitch | shifted axis err/pitch |
|---:|---:|---:|---|---:|---:|---:|
| diverted-single-null | 500 | 550 | ring_quadratic | 0.0201717 | 0.0168823 | 0.0197725 |
| diverted-single-null | 500 | 550 | own_node_quadratic | 0.00984024 | 0.0102085 | 0.0037 |
| diverted-single-null | 500 | 550 | two_ring_cubic | 0.00034656 | 0.000477756 | 0.00036699 |
| diverted-single-null | 500 | 550 | fixed_point_refinement | 0.00984024 | 0.0102085 | 0.0037 |
| diverted-single-null | 750 | 814 | ring_quadratic | 0.0133692 | 0.011739 | 0.0159643 |
| diverted-single-null | 750 | 814 | own_node_quadratic | 0.000988851 | 0.011894 | 0.0092698 |
| diverted-single-null | 750 | 814 | two_ring_cubic | 0.00027943 | 0.000625643 | 0.000371282 |
| diverted-single-null | 750 | 814 | fixed_point_refinement | 0.000988851 | 0.011894 | 0.0092698 |
| diverted-single-null | 1000 | 1074 | ring_quadratic | 0.0159343 | 0.0298499 | 0.00943489 |
| diverted-single-null | 1000 | 1074 | own_node_quadratic | 0.00493213 | 0.0270366 | 0.00320116 |
| diverted-single-null | 1000 | 1074 | two_ring_cubic | 5.56851e-05 | 0.000910376 | 0.000239385 |
| diverted-single-null | 1000 | 1074 | fixed_point_refinement | 0.00493213 | 0.0301175 | 0.00320116 |
| diverted-single-null | 2500 | 2616 | ring_quadratic | 0.00999173 | 0.0127014 | 0.0089002 |
| diverted-single-null | 2500 | 2616 | own_node_quadratic | 0.00271041 | 0.0157781 | 0.00717368 |
| diverted-single-null | 2500 | 2616 | two_ring_cubic | 5.86036e-05 | 7.78288e-05 | 0.000141527 |
| diverted-single-null | 2500 | 2616 | fixed_point_refinement | 0.00271041 | 0.0157781 | 0.00717368 |
| weak-rotation-reactor-static | 300 | 342 | ring_quadratic | 0.0114071 | — | 0.0145102 |
| weak-rotation-reactor-static | 300 | 342 | own_node_quadratic | 0.00125664 | — | 0.0009497 |
| weak-rotation-reactor-static | 300 | 342 | two_ring_cubic | 0.000346667 | — | 0.000305726 |
| weak-rotation-reactor-static | 300 | 342 | fixed_point_refinement | 0.00125664 | — | 0.0009497 |
| weak-rotation-reactor-static | 1000 | 1072 | ring_quadratic | 0.0107803 | — | 0.00692788 |
| weak-rotation-reactor-static | 1000 | 1072 | own_node_quadratic | 0.00358979 | — | 0.000136141 |
| weak-rotation-reactor-static | 1000 | 1072 | two_ring_cubic | 7.62615e-06 | — | 9.87212e-05 |
| weak-rotation-reactor-static | 1000 | 1072 | fixed_point_refinement | 0.00358979 | — | 0.000136141 |
| moderate-rotation-conventional-static | 300 | 341 | ring_quadratic | 0.0150908 | — | 0.011248 |
| moderate-rotation-conventional-static | 300 | 341 | own_node_quadratic | 0.0019684 | — | 0.00284858 |
| moderate-rotation-conventional-static | 300 | 341 | two_ring_cubic | 0.000293924 | — | 0.000390899 |
| moderate-rotation-conventional-static | 300 | 341 | fixed_point_refinement | 0.0019684 | — | 0.00284858 |
| moderate-rotation-conventional-static | 1000 | 1070 | ring_quadratic | 0.0105053 | — | 0.00988555 |
| moderate-rotation-conventional-static | 1000 | 1070 | own_node_quadratic | 0.00304277 | — | 0.00256171 |
| moderate-rotation-conventional-static | 1000 | 1070 | two_ring_cubic | 4.80068e-05 | — | 5.99025e-05 |
| moderate-rotation-conventional-static | 1000 | 1070 | fixed_point_refinement | 0.00304277 | — | 0.00256171 |
| strong-rotation-compact-static | 300 | 341 | ring_quadratic | 0.0087932 | — | 0.0152409 |
| strong-rotation-compact-static | 300 | 341 | own_node_quadratic | 0.00174959 | — | 0.00497428 |
| strong-rotation-compact-static | 300 | 341 | two_ring_cubic | 0.00021892 | — | 4.1985e-05 |
| strong-rotation-compact-static | 300 | 341 | fixed_point_refinement | 0.00174959 | — | 0.00497428 |
| strong-rotation-compact-static | 1000 | 1069 | ring_quadratic | 0.00274929 | — | 0.00742452 |
| strong-rotation-compact-static | 1000 | 1069 | own_node_quadratic | 0.00273645 | — | 0.00177631 |
| strong-rotation-compact-static | 1000 | 1069 | two_ring_cubic | 7.50057e-05 | — | 3.82896e-05 |
| strong-rotation-compact-static | 1000 | 1069 | fixed_point_refinement | 0.00273645 | — | 0.00177631 |

## Fitted order in pitch between successive single-null rungs

For each method and null, the exponent of position error in pitch against characteristic pitch is reported between each successive pair of single-null rungs, and as the overall log-log slope with its residual.

| method | null | 500→750 | 750→1000 | 1000→2500 | overall slope | log residual |
|---|---|---:|---:|---:|---:|---:|
| ring_quadratic | axis | 2.0289 | -1.2202 | 1.0187 | 0.7807 | 0.1111 |
| ring_quadratic | saddle | 1.7923 | -6.4882 | 1.8651 | 0.2247 | 0.3601 |
| own_node_quadratic | axis | 11.3336 | -11.1719 | 1.3067 | 0.8410 | 0.8056 |
| own_node_quadratic | saddle | -0.7538 | -5.7088 | 1.1755 | -0.5425 | 0.3348 |
| two_ring_cubic | axis | 1.0620 | 11.2140 | -0.1115 | 2.3231 | 0.5017 |
| two_ring_cubic | saddle | -1.3302 | -2.6076 | 5.3680 | 2.5037 | 0.5878 |
| fixed_point_refinement | axis | 11.3336 | -11.1719 | 1.3067 | 0.8410 | 0.8056 |
| fixed_point_refinement | saddle | -0.7538 | -6.4590 | 1.4111 | -0.5449 | 0.3815 |

## Reach of one thousandth of a pitch

A method reaches the 1e-03 threshold on a rung when its position error falls below a thousandth of a pitch on that rung.

| requested | realised | method | null | placement | err/pitch | fits threshold | fit points | fit cells |
|---:|---:|---|---|---:|---:|:---:|---:|---:|
| 1000 | 1074 | ring_quadratic | axis | field | 0.0159343 | no | 7 | 7 |
| 1000 | 1074 | ring_quadratic | axis | shifted | 0.00943489 | no | 7 | 7 |
| 1000 | 1074 | ring_quadratic | saddle | field | 0.0298499 | no | 7 | 7 |
| 1000 | 1074 | ring_quadratic | saddle | shifted | 0.0158619 | no | 7 | 7 |
| 1000 | 1074 | own_node_quadratic | axis | field | 0.00493213 | no | 7 | 1 |
| 1000 | 1074 | own_node_quadratic | axis | shifted | 0.00320116 | no | 7 | 1 |
| 1000 | 1074 | own_node_quadratic | saddle | field | 0.0270366 | no | 7 | 1 |
| 1000 | 1074 | own_node_quadratic | saddle | shifted | 0.0189548 | no | 7 | 1 |
| 1000 | 1074 | two_ring_cubic | axis | field | 5.56851e-05 | yes | 49 | 7 |
| 1000 | 1074 | two_ring_cubic | axis | shifted | 0.000239385 | yes | 49 | 7 |
| 1000 | 1074 | two_ring_cubic | saddle | field | 0.000910376 | yes | 49 | 7 |
| 1000 | 1074 | two_ring_cubic | saddle | shifted | 0.000526587 | yes | 49 | 7 |
| 1000 | 1074 | fixed_point_refinement | axis | field | 0.00493213 | no | 7 | 1 |
| 1000 | 1074 | fixed_point_refinement | axis | shifted | 0.00320116 | no | 7 | 1 |
| 1000 | 1074 | fixed_point_refinement | saddle | field | 0.0301175 | no | 14 | 2 |
| 1000 | 1074 | fixed_point_refinement | saddle | shifted | 0.0189548 | no | 7 | 1 |
| 2500 | 2616 | ring_quadratic | axis | field | 0.00999173 | no | 7 | 7 |
| 2500 | 2616 | ring_quadratic | axis | shifted | 0.0089002 | no | 7 | 7 |
| 2500 | 2616 | ring_quadratic | saddle | field | 0.0127014 | no | 7 | 7 |
| 2500 | 2616 | ring_quadratic | saddle | shifted | 0.0160112 | no | 7 | 7 |
| 2500 | 2616 | own_node_quadratic | axis | field | 0.00271041 | no | 7 | 1 |
| 2500 | 2616 | own_node_quadratic | axis | shifted | 0.00717368 | no | 7 | 1 |
| 2500 | 2616 | own_node_quadratic | saddle | field | 0.0157781 | no | 7 | 1 |
| 2500 | 2616 | own_node_quadratic | saddle | shifted | 0.014115 | no | 7 | 1 |
| 2500 | 2616 | two_ring_cubic | axis | field | 5.86036e-05 | yes | 49 | 7 |
| 2500 | 2616 | two_ring_cubic | axis | shifted | 0.000141527 | yes | 49 | 7 |
| 2500 | 2616 | two_ring_cubic | saddle | field | 7.78288e-05 | yes | 49 | 7 |
| 2500 | 2616 | two_ring_cubic | saddle | shifted | 0.000190651 | yes | 49 | 7 |
| 2500 | 2616 | fixed_point_refinement | axis | field | 0.00271041 | no | 7 | 1 |
| 2500 | 2616 | fixed_point_refinement | axis | shifted | 0.00717368 | no | 7 | 1 |
| 2500 | 2616 | fixed_point_refinement | saddle | field | 0.0157781 | no | 7 | 1 |
| 2500 | 2616 | fixed_point_refinement | saddle | shifted | 0.014115 | no | 7 | 1 |
