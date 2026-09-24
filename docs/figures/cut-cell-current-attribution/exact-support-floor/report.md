# Exact support current attribution

booked minus true = self_intersection_loss + simple_moment_integration_error + support_geometry_error + non_simple_integration_residual

The archived diverted target integrates density on traced fixture polygons. It is not an independent true-plasma-region integral.

| Case | Cells | Exact fraction | Vertical shift mm | Missing A (true region) | Integration deficit A | Geometry error A | Chord fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| diverted-single-null | 110 | 0.954177219 | -35.06213 | 1565.3809 | 2007.5968 | 442.21587 | 1.003530216 |
| diverted-single-null | 300 | 0.975269435 | -21.41219 | 788.8995 | 1233.2978 | 444.39834 | 0.999187766 |
| diverted-single-null | 500 | 0.989234093 | -3.59201 | 404.18415 | 620.78837 | 216.60422 | 1.000028359 |
| diverted-single-null | 1000 | 0.981303049 | -8.70746 | 695.42234 | 860.68546 | 165.26312 | 1.000446007 |
| diverted-single-null | 2500 | 0.987085068 | -4.72008 | 473.75038 | 590.55235 | 116.80198 | 1.000107887 |
| weak-rotation-reactor-static | 110 | 0.937988510 | 0.00055 | 1011497.8 | 1011492.9 | -4.9781313 | 0.999928695 |
| weak-rotation-reactor-static | 300 | 0.981367350 | 2.51814 | 303781.9 | 303780.14 | -1.7650378 | 0.999978555 |
| weak-rotation-reactor-static | 500 | 0.990312712 | 0.00002 | 157840.36 | 157839.44 | -0.91931388 | 0.999980645 |
| weak-rotation-reactor-static | 1000 | 0.985876947 | 10.24422 | 230306.28 | 230208.49 | -97.797746 | 0.999985688 |
| weak-rotation-reactor-static | 2500 | 0.989245644 | 0.00000 | 175249.32 | 175249.3 | -0.017910898 | 0.999987024 |

## diverted-single-null, 110 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 50 | -9.9475983e-12 | 8.64019967e-12 | 1.30739863e-12 |
| separatrix-cut | 35 | 1900.31889 | -1916.68163 | 16.3627415 |
| X-point cell | 1 | -103.420575 | 78.4867122 | 24.9338629 |
| wall-cut | 5 | -231.515404 | -4.96322884 | 236.478633 |
| exterior | 41 | -0.00199079713 | -164.438645 | 164.440636 |

Boundary refinement L1 / true current: 6.11e-07; quadrature order doubling L1 / target: 9.33e-17.

## diverted-single-null, 300 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 146 | -7.95807864e-12 | 8.41282599e-12 | -4.54747351e-13 |
| separatrix-cut | 65 | 1188.96433 | -1220.20816 | 31.2438221 |
| X-point cell | 1 | -85.4430992 | -12.4737732 | 97.9168724 |
| wall-cut | 9 | -182.013293 | -0.442776659 | 182.456069 |
| exterior | 119 | -132.608447 | -0.17313339 | 132.78158 |

Boundary refinement L1 / true current: 6.11e-07; quadrature order doubling L1 / target: 1.07e-16.

## diverted-single-null, 500 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 261 | 0.00197011735 | -0.000995153116 | -0.000974964233 |
| separatrix-cut | 75 | 638.116619 | -593.716323 | -44.4002959 |
| X-point cell | 1 | 22.3206286 | -11.7466522 | -10.5739764 |
| wall-cut | 93 | -65.057535 | -0.0724459527 | 65.129981 |
| exterior | 120 | -191.197533 | -15.2519558 | 206.449489 |

Boundary refinement L1 / true current: 6.11e-07; quadrature order doubling L1 / target: 9.65e-17.

## diverted-single-null, 1000 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 533 | -1.06368248e-11 | 1.03952402e-11 | 2.4158453e-13 |
| separatrix-cut | 118 | 846.968757 | -843.382146 | -3.58661136 |
| X-point cell | 1 | 14.089025 | -12.4268845 | -1.66214052 |
| wall-cut | 148 | -51.8951882 | -0.0330217404 | 51.9282099 |
| exterior | 274 | -113.740254 | -4.84340472 | 118.583659 |

Boundary refinement L1 / true current: 6.11e-07; quadrature order doubling L1 / target: 9.09e-17.

## diverted-single-null, 2500 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 1403 | 15.4384084 | -15.4384074 | -9.21227466e-07 |
| separatrix-cut | 176 | 537.694078 | -542.959064 | 5.26498619 |
| X-point cell | 1 | -0.11886483 | -1.17880233 | 1.29766715 |
| wall-cut | 218 | -0.316115562 | -16.0532019 | 16.3693175 |
| exterior | 818 | -78.9471311 | -14.9228789 | 93.87001 |

Boundary refinement L1 / true current: 6.11e-07; quadrature order doubling L1 / target: 1.09e-16.

## weak-rotation-reactor-static, 110 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 78 | -3.40514816e-09 | 3.20142135e-09 | 2.03726813e-10 |
| separatrix-cut | 41 | 1011371.87 | -1011367.63 | -4.24310147 |
| X-point cell | 0 | 0 | 0 | 0 |
| wall-cut | 4 | 125.970631 | -125.235602 | -0.735029865 |
| exterior | 12 | 0 | 0 | 0 |

Boundary refinement L1 / true current: 3.65e-08; quadrature order doubling L1 / target: 9.45e-17.

## weak-rotation-reactor-static, 300 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 232 | -3.23780114e-09 | 3.21597327e-09 | 2.18278728e-11 |
| separatrix-cut | 72 | 303466.325 | -303464.634 | -1.69111968 |
| X-point cell | 0 | 0 | 0 | 0 |
| wall-cut | 4 | 315.579954 | -315.506036 | -0.0739180894 |
| exterior | 34 | 0 | 0 | 0 |

Boundary refinement L1 / true current: 3.65e-08; quadrature order doubling L1 / target: 1.09e-16.

## weak-rotation-reactor-static, 500 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 402 | 0.0526177627 | -0.0526177953 | 3.26654117e-08 |
| separatrix-cut | 88 | 157834.432 | -157833.581 | -0.851118847 |
| X-point cell | 0 | 0 | 0 | 0 |
| wall-cut | 10 | 5.87337084 | -5.80517577 | -0.0681950687 |
| exterior | 55 | 0 | 0 | 0 |

Boundary refinement L1 / true current: 3.65e-08; quadrature order doubling L1 / target: 9e-17.

## weak-rotation-reactor-static, 1000 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 824 | 0.00769501405 | -0.00769504437 | 3.03280103e-08 |
| separatrix-cut | 127 | 230302.94 | -230205.145 | -97.7946124 |
| X-point cell | 0 | 0 | 0 | 0 |
| wall-cut | 13 | 3.33518699 | -3.33205317 | -0.00313381965 |
| exterior | 108 | 0 | 0 | 0 |

Boundary refinement L1 / true current: 7.73e-07; quadrature order doubling L1 / target: 9.84e-17.

## weak-rotation-reactor-static, 2500 cells

moment integration carries the larger signed deficit, concentrated in separatrix-cut cells.

| Cell class | Count | Missing A | Moment integration error A | Geometry error A |
|---|---:|---:|---:|---:|
| interior | 2114 | 0.00038858951 | -0.000388618119 | 2.86090653e-08 |
| separatrix-cut | 209 | 175179.477 | -175179.46 | -0.0172497843 |
| X-point cell | 0 | 0 | 0 | 0 |
| wall-cut | 18 | 69.8396805 | -69.8390193 | -0.000661142416 |
| exterior | 267 | 0 | 0 | 0 |

Boundary refinement L1 / true current: 3.65e-08; quadrature order doubling L1 / target: 9.7e-17.

## Non-simple census and signed attribution

Errors below use booked minus true. Negate them for missing current.
The non-simple remainder is reported separately, never absorbed into self-intersection or simple-cell integration.

| Case | Rung | Non-simple | Current fraction | Self-intersection A | Simple integration A | Geometry A | Non-simple remainder A | Exact missing fraction | Chord missing fraction | Control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| diverted-single-null | 110 | 1 | 0.000291382387 | -0.110578257 | -2086.08351 | 442.215873 | 78.5972904 | 0.0458227809 | -0.00353021616 | True |
Boundary points: 5761; [non-simple cells and pieces](diverted-single-null-cells-110-exact-non-simple.json).
| diverted-single-null | 300 | 7 | 0.0337575552 | 3.10333235e-08 | -1156.78508 | 444.398344 | -76.5127576 | 0.0247305648 | 0.000812233836 | True |
Boundary points: 5761; [non-simple cells and pieces](diverted-single-null-cells-300-exact-non-simple.json).
| diverted-single-null | 500 | 0 | 0 | 0 | -620.788372 | 216.604222 | 0 | 0.0107659066 | -2.83585794e-05 | True |
Boundary points: 5761; [non-simple cells and pieces](diverted-single-null-cells-500-exact-non-simple.json).
| diverted-single-null | 1000 | 0 | 0 | 0 | -860.685457 | 165.263117 | 0 | 0.0186969505 | -0.000446006667 | True |
Boundary points: 5761; [non-simple cells and pieces](diverted-single-null-cells-1000-exact-non-simple.json).
| diverted-single-null | 2500 | 0 | 0 | 0 | -590.552355 | 116.80198 | 0 | 0.0129149321 | -0.000107886745 | True |
Boundary points: 5761; [non-simple cells and pieces](diverted-single-null-cells-2500-exact-non-simple.json).
| weak-rotation-reactor-static | 110 | 0 | 0 | 0 | -1011492.87 | -4.97813133 | 0 | 0.0620114898 | 7.13053005e-05 | True |
Boundary points: 23040; [non-simple cells and pieces](weak-rotation-reactor-static-cells-110-exact-non-simple.json).
| weak-rotation-reactor-static | 300 | 0 | 0 | 0 | -303780.14 | -1.76503777 | 0 | 0.01863265 | 2.14451341e-05 | True |
Boundary points: 23040; [non-simple cells and pieces](weak-rotation-reactor-static-cells-300-exact-non-simple.json).
| weak-rotation-reactor-static | 500 | 0 | 0 | 0 | -157839.439 | -0.919313883 | 0 | 0.00968728843 | 1.93554027e-05 | True |
Boundary points: 23040; [non-simple cells and pieces](weak-rotation-reactor-static-cells-500-exact-non-simple.json).
| weak-rotation-reactor-static | 1000 | 0 | 0 | 0 | -230208.485 | -97.7977462 | 0 | 0.0141230534 | 1.43123543e-05 | True |
Boundary points: 23040; [non-simple cells and pieces](weak-rotation-reactor-static-cells-1000-exact-non-simple.json).
| weak-rotation-reactor-static | 2500 | 0 | 0 | 0 | -175249.299 | -0.0179108981 | 0 | 0.0107543557 | 1.29758828e-05 | True |
Boundary points: 23040; [non-simple cells and pieces](weak-rotation-reactor-static-cells-2500-exact-non-simple.json).

Chord control passes when its absolute deficit is below one tenth of exact. Both signed missing fractions are retained.

Every cell and all three physical moments are retained in report.json and the individual row JSON files. No solver step or production repair is made.

## Interpretation

Moment evaluation on simple separatrix-cut cells carries the main deficit at every measured rung. Non-simple support is present only in diverted 110 (one cell; 0.0291382% of true analytic current) and diverted 300 (seven cells; 3.37576%). Its production-formula minus nonzero-winding lobe-union current is -0.110578 A and +3.1033e-8 A respectively. No non-simple support appears at the three finer diverted rungs or any weak-rotation rung.

The requested three errors do not exactly close the production deficit on the two rows with non-simple cells. The additional booked-minus-analytic-formula current is +78.597290 A at diverted 110 and -76.512758 A at diverted 300. These are measured remainders, not values assigned to another category. The four-term identity closes on every cell and row. This comparison localizes the deficit downstream of the emitted support geometry; it does not distinguish subsequent confinement/reclipping, density evaluation and numerical integration within that moment path.

For diverted cell 9 at 110 cells: production formula on analytic density 35.655323 A; independent signed integral 35.655324 A; union of nonzero-winding lobes 35.765901 A; true plasma region 10.832038 A; actual booked current 114.252613 A. Its original chain, atomic cell and branch pieces are preserved unchanged in the non-simple census. No geometric repair was fed back into production.

The explicit coarse centroid control reads 35.062133 mm low, 0.337867 mm from the reviewed 35.4 mm value. All ten chord controls pass; the largest chord/exact absolute-deficit ratio is 0.0770406, below 0.1. The guarded receipt audit rejects a missing rung and a false chord-control verdict.

The analytic boundary uses the highest successful member of the tested sampling ladder: diverted 5,761 points (23,041 and 11,521 failed), weak rotation 23,041 requested points yielding 23,040 vertices. The radial scan was not repaired and no normal projection was used. Sampling attempts and comparison against the next lower passing resolution are retained per case and row.

H200 job 1276905 completed the full row loop in 265 seconds at revision 150a1398c478e2b11d7a25de5195e280838dd856. Render-only all_debug job 1276907 consumed the stored rows; numerical evaluations were not repeated. The current-error panels omit the zero contour, which would otherwise draw roundoff patterns, and retain shared nonzero levels with the peak magnitude stated on each panel.
