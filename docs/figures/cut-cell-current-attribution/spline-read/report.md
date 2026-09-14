# Global tensor-spline read on the analytic hex carrier

Measurement revision: `18cf78b7bf5808172d2e49ca9486908b66d71572`; control and fit revision: `fffd486602bf5444189ea630140ab5c0c207c848`; report revision: `ae32451b3b8a9a920613755b1f15e79f788c0bfd`. Full machine-readable receipt: `/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/spline-read/receipt.json`.

The production hex route reproduced its fixed-design behavior on every row: `spline_authored=false` and `spline_shape=[0, 0]`. Its saddle is the six-coefficient least-squares quadratic on a centroid ring. The alternatives below project analytic centroid values, or centroid plus direct vertex-sample values, onto the same global cubic knot-value representation used by the coefficient carrier.

## Saddle and ring movement

| requested cells | wall nodes | ring error [mm] | ring error / pitch | origin cell |
|---:|---:|---:|---:|---:|
| 500 | 121 | 1.1993 | 0.01688 | 8 |
| 500 | 241 | 1.2022 | 0.01692 | 8 |
| 500 | 481 | 1.2022 | 0.01692 | 8 |
| 500 | 961 | 1.2017 | 0.01691 | 8 |
| 1000 | 121 | 1.5162 | 0.03019 | 23 |
| 1000 | 241 | 1.5115 | 0.03009 | 23 |
| 1000 | 481 | 1.5127 | 0.03011 | 23 |
| 1000 | 961 | 1.5144 | 0.03014 | 23 |
| 2500 | 121 | 0.4035 | 0.01270 | 1479 |
| 2500 | 241 | 0.4069 | 0.01281 | 1501 |
| 2500 | 481 | 0.4063 | 0.01279 | 1473 |
| 2500 | 961 | 0.4052 | 0.01275 | 1473 |

Cell-by-cell matching uses nearest centroid against the 121-node wall carrier; identifiers alone are not compared across rebuilt meshes.

| requested cells | wall nodes | replaced ring cells | maximum centroid shift [mm] | published saddle move [mm] |
|---:|---:|---:|---:|---:|
| 500 | 121 | 0 | 0.000000 | 0.000000 |
| 500 | 241 | 0 | 0.076532 | 0.003118 |
| 500 | 481 | 0 | 0.081289 | 0.002940 |
| 500 | 961 | 0 | 0.071026 | 0.002412 |
| 1000 | 121 | 0 | 0.000000 | 0.000000 |
| 1000 | 241 | 0 | 0.069127 | 0.005858 |
| 1000 | 481 | 0 | 0.073825 | 0.005471 |
| 1000 | 961 | 0 | 0.064999 | 0.004216 |
| 2500 | 121 | 0 | 0.000000 | 0.000000 |
| 2500 | 241 | 0 | 0.067330 | 0.004660 |
| 2500 | 481 | 0 | 0.069440 | 0.004485 |
| 2500 | 961 | 0 | 0.058589 | 0.003490 |

### Mechanism question closed: the earlier saddle motion was mislabeled

No selected saddle-ring member was replaced across the four wall samplings: all seven cells match one-to-one at every count, with only sub-0.002-pitch coordinate shifts and no wall-clipped member. The published ring-quadratic saddle spread is only **3.118 µm** at 500 cells, **5.858 µm** at 1000, and **4.660 µm** at 2500. There is therefore no cell-by-cell wall-reclipping mechanism to explain.

The limiter audit's reported `8.0, 1.3, 1.9, 0.5 mm` sequence was the wall-contact position error carried under the boundary label, not saddle motion. This receipt reproduces those contact errors as `8.0490, 1.2823, 1.9081, 0.4627 mm`.

Positive control: perturbing ring cell 91 by `1e-4` of the analytic span moved the published saddle by **0.045002 mm** (0.000895893 pitch). The instrument therefore observes a known ring-value change.

## Global spline accuracy and wall-count invariance

| requested cells | method | successful wall counts | maximum saddle spread [mm] |
|---:|---|---:|---:|
| 500 | `ring_quadratic` | 4 | 0.003118 |
| 500 | `centroids__pitch_1` | 4 | 0.370374 |
| 500 | `centroids_and_vertices__pitch_0.5` | 4 | 0.208791 |
| 500 | `centroids_and_vertices__pitch_1` | 4 | 0.042828 |
| 500 | `centroids_vertices_wall__pitch_0.5` | 2 | 6.570767 |
| 500 | `centroids_vertices_wall__pitch_1` | 4 | 4.238882 |
| 1000 | `ring_quadratic` | 4 | 0.005858 |
| 1000 | `centroids__pitch_1` | 4 | 0.490694 |
| 1000 | `centroids_and_vertices__pitch_0.5` | 4 | 1.531843 |
| 1000 | `centroids_and_vertices__pitch_1` | 4 | 0.194776 |
| 1000 | `centroids_vertices_wall__pitch_0.5` | 3 | 12.436044 |
| 1000 | `centroids_vertices_wall__pitch_1` | 4 | 0.876283 |
| 2500 | `ring_quadratic` | 4 | 0.004660 |
| 2500 | `centroids__pitch_0.5` | 2 | 0.070716 |
| 2500 | `centroids__pitch_1` | 4 | 5.948027 |
| 2500 | `centroids_and_vertices__pitch_0.5` | 3 | 6.522817 |
| 2500 | `centroids_and_vertices__pitch_1` | 4 | 2.092299 |
| 2500 | `centroids_vertices_wall__pitch_0.5` | 4 | 6.283476 |
| 2500 | `centroids_vertices_wall__pitch_1` | 4 | 2.255827 |

### Null errors, fit residuals, wall errors, and fit time by cell count

Values are medians over the four wall samplings. A refused saddle means the Newton result did not converge with the required Hessian type.

| case | cells | method | axis error [mm] | saddle error [mm] | fit rms / span | wall rms / span | fit [s] |
|---|---:|---|---:|---:|---:|---:|---:|
| `diverted-single-null` | 500 | `ring_quadratic` | 1.436223 | 1.201936 | not applicable | not applicable | not applicable |
| `diverted-single-null` | 500 | `centroids__pitch_0.5` | refused | refused | 3.104e-11 | 7.122e-01 | 0.025 |
| `diverted-single-null` | 500 | `centroids__pitch_1` | 4.805444 | 11.646299 | 1.087e-03 | 1.096e-01 | 0.070 |
| `diverted-single-null` | 500 | `centroids_and_vertices__pitch_0.5` | refused | 11.389530 | 1.341e-03 | 4.606e-02 | 0.198 |
| `diverted-single-null` | 500 | `centroids_and_vertices__pitch_1` | 1.032167 | 2.169685 | 2.491e-03 | 4.047e-03 | 0.138 |
| `diverted-single-null` | 500 | `centroids_vertices_wall__pitch_0.5` | refused | 17.839598 | 1.819e-03 | 2.670e-03 | 0.223 |
| `diverted-single-null` | 500 | `centroids_vertices_wall__pitch_1` | 3.917853 | 3.870309 | 3.036e-03 | 2.928e-03 | 0.190 |
| `diverted-single-null` | 1000 | `ring_quadratic` | 0.801200 | 1.513569 | not applicable | not applicable | not applicable |
| `diverted-single-null` | 1000 | `centroids__pitch_0.5` | 40.383788 | refused | 2.242e-11 | 6.984e-01 | 0.036 |
| `diverted-single-null` | 1000 | `centroids__pitch_1` | 0.629615 | 5.498059 | 4.403e-04 | 1.317e-01 | 0.109 |
| `diverted-single-null` | 1000 | `centroids_and_vertices__pitch_0.5` | refused | 2.668558 | 1.166e-03 | 5.027e-02 | 0.288 |
| `diverted-single-null` | 1000 | `centroids_and_vertices__pitch_1` | 2.158558 | 1.249895 | 2.137e-03 | 5.317e-03 | 0.217 |
| `diverted-single-null` | 1000 | `centroids_vertices_wall__pitch_0.5` | 10.518440 | 8.055367 | 1.490e-03 | 2.517e-03 | 0.326 |
| `diverted-single-null` | 1000 | `centroids_vertices_wall__pitch_1` | 2.403921 | 0.656556 | 2.492e-03 | 3.843e-03 | 0.281 |
| `diverted-single-null` | 2500 | `ring_quadratic` | 0.315603 | 0.405729 | not applicable | not applicable | not applicable |
| `diverted-single-null` | 2500 | `centroids__pitch_0.5` | 10.049348 | 5.207680 | 4.125e-11 | 7.407e-01 | 0.351 |
| `diverted-single-null` | 2500 | `centroids__pitch_1` | 0.979294 | 6.149148 | 3.459e-04 | 1.377e-01 | 0.285 |
| `diverted-single-null` | 2500 | `centroids_and_vertices__pitch_0.5` | refused | 2.417878 | 9.058e-04 | 4.089e-02 | 1.541 |
| `diverted-single-null` | 2500 | `centroids_and_vertices__pitch_1` | 1.898915 | 1.287089 | 1.472e-03 | 4.250e-03 | 0.631 |
| `diverted-single-null` | 2500 | `centroids_vertices_wall__pitch_0.5` | refused | 4.187036 | 1.119e-03 | 2.808e-03 | 0.614 |
| `diverted-single-null` | 2500 | `centroids_vertices_wall__pitch_1` | 3.090546 | 1.190119 | 1.675e-03 | 3.241e-03 | 0.483 |
| `weak-rotation-reactor-static` | 1000 | `ring_quadratic` | 1.667369 | not applicable | not applicable | not applicable | not applicable |
| `weak-rotation-reactor-static` | 1000 | `centroids__pitch_0.5` | 231.908923 | not applicable | 4.402e-11 | 1.086e-01 | 0.035 |
| `weak-rotation-reactor-static` | 1000 | `centroids__pitch_1` | 1.407715 | not applicable | 1.740e-04 | 1.846e-02 | 0.104 |
| `weak-rotation-reactor-static` | 1000 | `centroids_and_vertices__pitch_0.5` | refused | not applicable | 2.080e-04 | 7.022e-03 | 0.282 |
| `weak-rotation-reactor-static` | 1000 | `centroids_and_vertices__pitch_1` | 3.001108 | not applicable | 4.119e-04 | 8.164e-04 | 0.213 |
| `weak-rotation-reactor-static` | 1000 | `centroids_vertices_wall__pitch_0.5` | refused | not applicable | 2.672e-04 | 4.695e-04 | 0.326 |
| `weak-rotation-reactor-static` | 1000 | `centroids_vertices_wall__pitch_1` | 3.285860 | not applicable | 5.012e-04 | 6.108e-04 | 0.295 |

The ring saddle error is not monotone on the first refinement: **1.20 mm at 550 realised cells** becomes **1.52 mm at 1074**; the 2500 request reaches about **0.405 mm**.

| cells | wall | spline route | knot factor | fit rms / span | condition | saddle error / pitch | wall rms / span | wall max / span |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 500 | 121 | `centroids` | 0.5 | 2.830e-11 | infinite_from_structural_nullspace | refused | 7.127e-01 | 2.204e+00 |
| 500 | 121 | `centroids` | 1.0 | 1.075e-03 | infinite_from_structural_nullspace | 1.599e-01 | 1.096e-01 | 5.191e-01 |
| 500 | 121 | `centroids_and_vertices` | 0.5 | 1.336e-03 | infinite_from_structural_nullspace | 1.629e-01 | 4.256e-02 | 2.310e-01 |
| 500 | 121 | `centroids_and_vertices` | 1.0 | 2.485e-03 | 5.238e+02 | 3.077e-02 | 3.907e-03 | 1.370e-02 |
| 500 | 121 | `centroids_vertices_wall` | 0.5 | 1.666e-03 | infinite_from_structural_nullspace | 2.490e-01 | 3.167e-03 | 1.627e-02 |
| 500 | 121 | `centroids_vertices_wall` | 1.0 | 2.664e-03 | 5.257e+02 | 2.251e-02 | 3.018e-03 | 1.095e-02 |
| 500 | 241 | `centroids` | 0.5 | 3.076e-11 | infinite_from_structural_nullspace | refused | 7.123e-01 | 2.298e+00 |
| 500 | 241 | `centroids` | 1.0 | 1.086e-03 | infinite_from_structural_nullspace | 1.635e-01 | 1.095e-01 | 5.328e-01 |
| 500 | 241 | `centroids_and_vertices` | 0.5 | 1.340e-03 | infinite_from_structural_nullspace | 1.606e-01 | 4.629e-02 | 2.638e-01 |
| 500 | 241 | `centroids_and_vertices` | 1.0 | 2.490e-03 | 5.235e+02 | 3.058e-02 | 4.053e-03 | 1.462e-02 |
| 500 | 241 | `centroids_vertices_wall` | 0.5 | 1.709e-03 | infinite_from_structural_nullspace | 2.532e-01 | 3.000e-03 | 1.569e-02 |
| 500 | 241 | `centroids_vertices_wall` | 1.0 | 2.821e-03 | 5.342e+02 | 3.108e-02 | 2.931e-03 | 1.062e-02 |
| 500 | 481 | `centroids` | 0.5 | 3.132e-11 | infinite_from_structural_nullspace | refused | 7.120e-01 | 2.271e+00 |
| 500 | 481 | `centroids` | 1.0 | 1.088e-03 | infinite_from_structural_nullspace | 1.649e-01 | 1.096e-01 | 5.376e-01 |
| 500 | 481 | `centroids_and_vertices` | 0.5 | 1.342e-03 | infinite_from_structural_nullspace | 1.599e-01 | 4.607e-02 | 2.806e-01 |
| 500 | 481 | `centroids_and_vertices` | 1.0 | 2.493e-03 | 5.235e+02 | 3.050e-02 | 4.048e-03 | 1.552e-02 |
| 500 | 481 | `centroids_vertices_wall` | 0.5 | 1.929e-03 | infinite_from_structural_nullspace | refused | 2.340e-03 | 1.081e-02 |
| 500 | 481 | `centroids_vertices_wall` | 1.0 | 3.252e-03 | 5.306e+02 | 7.939e-02 | 2.925e-03 | 1.124e-02 |
| 500 | 961 | `centroids` | 0.5 | 3.139e-11 | infinite_from_structural_nullspace | refused | 7.119e-01 | 2.357e+00 |
| 500 | 961 | `centroids` | 1.0 | 1.088e-03 | infinite_from_structural_nullspace | 1.643e-01 | 1.096e-01 | 5.409e-01 |
| 500 | 961 | `centroids_and_vertices` | 0.5 | 1.342e-03 | infinite_from_structural_nullspace | 1.599e-01 | 4.605e-02 | 2.866e-01 |
| 500 | 961 | `centroids_and_vertices` | 1.0 | 2.493e-03 | 5.235e+02 | 3.048e-02 | 4.046e-03 | 1.592e-02 |
| 500 | 961 | `centroids_vertices_wall` | 0.5 | 2.299e-03 | 4.617e+02 | refused | 1.776e-03 | 8.019e-03 |
| 500 | 961 | `centroids_vertices_wall` | 1.0 | 3.985e-03 | 4.816e+02 | 7.786e-02 | 2.769e-03 | 1.214e-02 |
| 1000 | 121 | `centroids` | 0.5 | 2.543e-11 | infinite_from_structural_nullspace | refused | 6.386e-01 | 1.673e+00 |
| 1000 | 121 | `centroids` | 1.0 | 4.294e-04 | infinite_from_structural_nullspace | 1.029e-01 | 1.317e-01 | 6.511e-01 |
| 1000 | 121 | `centroids_and_vertices` | 0.5 | 1.165e-03 | infinite_from_structural_nullspace | 3.503e-02 | 5.077e-02 | 2.281e-01 |
| 1000 | 121 | `centroids_and_vertices` | 1.0 | 2.137e-03 | 4.880e+02 | 2.708e-02 | 5.288e-03 | 2.067e-02 |
| 1000 | 121 | `centroids_vertices_wall` | 0.5 | 1.225e-03 | infinite_from_structural_nullspace | 7.113e-02 | 2.760e-03 | 1.095e-02 |
| 1000 | 121 | `centroids_vertices_wall` | 1.0 | 2.267e-03 | 4.896e+02 | 1.938e-02 | 4.206e-03 | 1.693e-02 |
| 1000 | 241 | `centroids` | 0.5 | 2.276e-11 | infinite_from_structural_nullspace | refused | 7.285e-01 | 3.216e+00 |
| 1000 | 241 | `centroids` | 1.0 | 4.408e-04 | infinite_from_structural_nullspace | 1.097e-01 | 1.318e-01 | 6.630e-01 |
| 1000 | 241 | `centroids_and_vertices` | 0.5 | 1.163e-03 | infinite_from_structural_nullspace | 5.186e-02 | 5.212e-02 | 2.269e-01 |
| 1000 | 241 | `centroids_and_vertices` | 1.0 | 2.137e-03 | 4.880e+02 | 2.325e-02 | 5.328e-03 | 2.015e-02 |
| 1000 | 241 | `centroids_vertices_wall` | 0.5 | 1.435e-03 | infinite_from_structural_nullspace | 1.603e-01 | 2.943e-03 | 1.266e-02 |
| 1000 | 241 | `centroids_vertices_wall` | 1.0 | 2.394e-03 | 4.892e+02 | 1.420e-02 | 4.116e-03 | 1.710e-02 |
| 1000 | 481 | `centroids` | 0.5 | 2.208e-11 | infinite_from_structural_nullspace | refused | 6.983e-01 | 2.974e+00 |
| 1000 | 481 | `centroids` | 1.0 | 4.398e-04 | infinite_from_structural_nullspace | 1.092e-01 | 1.317e-01 | 6.619e-01 |
| 1000 | 481 | `centroids_and_vertices` | 0.5 | 1.167e-03 | infinite_from_structural_nullspace | 5.437e-02 | 4.977e-02 | 2.713e-01 |
| 1000 | 481 | `centroids_and_vertices` | 1.0 | 2.137e-03 | 4.883e+02 | 2.363e-02 | 5.326e-03 | 2.379e-02 |
| 1000 | 481 | `centroids_vertices_wall` | 0.5 | 1.544e-03 | infinite_from_structural_nullspace | refused | 2.275e-03 | 1.149e-02 |
| 1000 | 481 | `centroids_vertices_wall` | 1.0 | 2.591e-03 | 4.874e+02 | 3.073e-03 | 3.570e-03 | 1.774e-02 |
| 1000 | 961 | `centroids` | 0.5 | 2.190e-11 | infinite_from_structural_nullspace | refused | 6.985e-01 | 3.202e+00 |
| 1000 | 961 | `centroids` | 1.0 | 4.408e-04 | infinite_from_structural_nullspace | 1.096e-01 | 1.317e-01 | 6.652e-01 |
| 1000 | 961 | `centroids_and_vertices` | 0.5 | 1.167e-03 | infinite_from_structural_nullspace | 5.488e-02 | 4.976e-02 | 2.720e-01 |
| 1000 | 961 | `centroids_and_vertices` | 1.0 | 2.135e-03 | 4.885e+02 | 2.613e-02 | 5.309e-03 | 2.355e-02 |
| 1000 | 961 | `centroids_vertices_wall` | 0.5 | 1.724e-03 | infinite_from_structural_nullspace | 2.481e-01 | 1.637e-03 | 7.051e-03 |
| 1000 | 961 | `centroids_vertices_wall` | 1.0 | 2.966e-03 | 4.572e+02 | 1.194e-02 | 3.114e-03 | 1.563e-02 |
| 1000 | 121 | `centroids` | 0.5 | 4.488e-11 | infinite_from_structural_nullspace | refused | 1.092e-01 | 4.360e-01 |
| 1000 | 121 | `centroids` | 1.0 | 1.744e-04 | infinite_from_structural_nullspace | refused | 2.009e-02 | 8.148e-02 |
| 1000 | 121 | `centroids_and_vertices` | 0.5 | 2.073e-04 | infinite_from_structural_nullspace | refused | 6.453e-03 | 3.012e-02 |
| 1000 | 121 | `centroids_and_vertices` | 1.0 | 4.120e-04 | 4.971e+02 | refused | 7.351e-04 | 2.876e-03 |
| 1000 | 121 | `centroids_vertices_wall` | 0.5 | 2.423e-04 | infinite_from_structural_nullspace | refused | 5.006e-04 | 2.205e-03 |
| 1000 | 121 | `centroids_vertices_wall` | 1.0 | 4.412e-04 | 4.952e+02 | refused | 6.220e-04 | 2.639e-03 |
| 1000 | 241 | `centroids` | 0.5 | 4.414e-11 | infinite_from_structural_nullspace | refused | 1.095e-01 | 4.347e-01 |
| 1000 | 241 | `centroids` | 1.0 | 1.740e-04 | infinite_from_structural_nullspace | refused | 1.849e-02 | 8.845e-02 |
| 1000 | 241 | `centroids_and_vertices` | 0.5 | 2.076e-04 | infinite_from_structural_nullspace | refused | 6.791e-03 | 3.304e-02 |
| 1000 | 241 | `centroids_and_vertices` | 1.0 | 4.125e-04 | 4.965e+02 | refused | 8.166e-04 | 3.128e-03 |
| 1000 | 241 | `centroids_vertices_wall` | 0.5 | 2.517e-04 | infinite_from_structural_nullspace | refused | 4.868e-04 | 1.786e-03 |
| 1000 | 241 | `centroids_vertices_wall` | 1.0 | 4.744e-04 | 4.923e+02 | refused | 6.405e-04 | 2.464e-03 |
| 1000 | 481 | `centroids` | 0.5 | 4.390e-11 | infinite_from_structural_nullspace | refused | 1.080e-01 | 4.610e-01 |
| 1000 | 481 | `centroids` | 1.0 | 1.740e-04 | infinite_from_structural_nullspace | refused | 1.844e-02 | 8.908e-02 |
| 1000 | 481 | `centroids_and_vertices` | 0.5 | 2.086e-04 | infinite_from_structural_nullspace | refused | 7.253e-03 | 3.673e-02 |
| 1000 | 481 | `centroids_and_vertices` | 1.0 | 4.118e-04 | 4.967e+02 | refused | 8.163e-04 | 3.424e-03 |
| 1000 | 481 | `centroids_vertices_wall` | 0.5 | 2.827e-04 | infinite_from_structural_nullspace | refused | 4.522e-04 | 1.904e-03 |
| 1000 | 481 | `centroids_vertices_wall` | 1.0 | 5.281e-04 | 4.933e+02 | refused | 5.997e-04 | 2.379e-03 |
| 1000 | 961 | `centroids` | 0.5 | 4.384e-11 | infinite_from_structural_nullspace | refused | 1.079e-01 | 4.607e-01 |
| 1000 | 961 | `centroids` | 1.0 | 1.740e-04 | infinite_from_structural_nullspace | refused | 1.842e-02 | 8.915e-02 |
| 1000 | 961 | `centroids_and_vertices` | 0.5 | 2.084e-04 | infinite_from_structural_nullspace | refused | 7.254e-03 | 3.704e-02 |
| 1000 | 961 | `centroids_and_vertices` | 1.0 | 4.119e-04 | 4.956e+02 | refused | 8.179e-04 | 3.405e-03 |
| 1000 | 961 | `centroids_vertices_wall` | 0.5 | 3.349e-04 | infinite_from_structural_nullspace | refused | 3.568e-04 | 1.510e-03 |
| 1000 | 961 | `centroids_vertices_wall` | 1.0 | 5.991e-04 | 5.041e+02 | refused | 5.543e-04 | 2.415e-03 |
| 2500 | 121 | `centroids` | 0.5 | 5.951e-11 | infinite_from_structural_nullspace | 1.632e-01 | 7.396e-01 | 2.684e+00 |
| 2500 | 121 | `centroids` | 1.0 | 3.303e-04 | infinite_from_structural_nullspace | 2.300e-01 | 1.548e-01 | 7.802e-01 |
| 2500 | 121 | `centroids_and_vertices` | 0.5 | 9.153e-04 | infinite_from_structural_nullspace | 7.611e-02 | 4.652e-02 | 2.266e-01 |
| 2500 | 121 | `centroids_and_vertices` | 1.0 | 1.457e-03 | 4.647e+02 | 1.968e-02 | 4.220e-03 | 1.459e-02 |
| 2500 | 121 | `centroids_vertices_wall` | 0.5 | 1.005e-03 | infinite_from_structural_nullspace | 8.071e-02 | 2.912e-03 | 1.301e-02 |
| 2500 | 121 | `centroids_vertices_wall` | 1.0 | 1.539e-03 | 4.644e+02 | 4.342e-02 | 3.356e-03 | 1.097e-02 |
| 2500 | 241 | `centroids` | 0.5 | 5.704e-11 | infinite_from_structural_nullspace | 1.646e-01 | 7.397e-01 | 3.260e+00 |
| 2500 | 241 | `centroids` | 1.0 | 3.308e-04 | infinite_from_structural_nullspace | 2.295e-01 | 1.395e-01 | 6.384e-01 |
| 2500 | 241 | `centroids_and_vertices` | 0.5 | 9.250e-04 | infinite_from_structural_nullspace | 7.360e-02 | 4.115e-02 | 1.996e-01 |
| 2500 | 241 | `centroids_and_vertices` | 1.0 | 1.456e-03 | 4.652e+02 | 2.142e-02 | 3.986e-03 | 1.816e-02 |
| 2500 | 241 | `centroids_vertices_wall` | 0.5 | 1.095e-03 | infinite_from_structural_nullspace | 8.125e-02 | 2.945e-03 | 1.630e-02 |
| 2500 | 241 | `centroids_vertices_wall` | 1.0 | 1.604e-03 | 4.624e+02 | 2.670e-02 | 3.230e-03 | 1.451e-02 |
| 2500 | 481 | `centroids` | 0.5 | 2.502e-11 | infinite_from_structural_nullspace | refused | 7.421e-01 | 3.775e+00 |
| 2500 | 481 | `centroids` | 1.0 | 3.610e-04 | infinite_from_structural_nullspace | 1.576e-01 | 1.359e-01 | 7.380e-01 |
| 2500 | 481 | `centroids_and_vertices` | 0.5 | 8.960e-04 | infinite_from_structural_nullspace | refused | 4.063e-02 | 2.125e-01 |
| 2500 | 481 | `centroids_and_vertices` | 1.0 | 1.488e-03 | 4.617e+02 | 6.052e-02 | 4.285e-03 | 2.321e-02 |
| 2500 | 481 | `centroids_vertices_wall` | 0.5 | 1.143e-03 | infinite_from_structural_nullspace | 1.930e-01 | 2.703e-03 | 1.197e-02 |
| 2500 | 481 | `centroids_vertices_wall` | 1.0 | 1.747e-03 | 4.630e+02 | 3.281e-02 | 3.251e-03 | 1.675e-02 |
| 2500 | 961 | `centroids` | 0.5 | 2.546e-11 | infinite_from_structural_nullspace | refused | 7.416e-01 | 3.683e+00 |
| 2500 | 961 | `centroids` | 1.0 | 3.611e-04 | infinite_from_structural_nullspace | 1.573e-01 | 1.357e-01 | 7.633e-01 |
| 2500 | 961 | `centroids_and_vertices` | 0.5 | 8.962e-04 | infinite_from_structural_nullspace | 2.022e-01 | 4.057e-02 | 2.207e-01 |
| 2500 | 961 | `centroids_and_vertices` | 1.0 | 1.487e-03 | 4.615e+02 | 5.959e-02 | 4.281e-03 | 2.409e-02 |
| 2500 | 961 | `centroids_vertices_wall` | 0.5 | 1.299e-03 | infinite_from_structural_nullspace | 1.823e-01 | 2.220e-03 | 1.149e-02 |
| 2500 | 961 | `centroids_vertices_wall` | 1.0 | 1.958e-03 | 4.573e+02 | 4.210e-02 | 2.792e-03 | 1.466e-02 |

## Fit credibility controls

The regular-grid control fits the analytic field on a rectangular grid over the same bounding box at each requested knot pitch. Its worst data-point residual is **7.105e-15** of span, and its worst error when evaluated back on the original scattered coordinates is **8.490e-06 rms**, **8.044e-05 maximum**.

The operator sample tail agrees with direct analytic evaluation to **1.500e-14** of span; gathering `sample_coordinates` through `cell_sample_nodes` agrees with `sampling_vertices` to **1.747e-14 pitch**. The worst explicit per-vertex spline error is **2.791e-02** of span; every point's coordinate, analytic value, spline value, and signed error is retained in the receipt.

The sample ordering is exact and the clean regular-grid spline reaches the expected interpolation floor. Neither control explains the `1e-3` residual: it is specific to the ill-conditioned, iteration-limited scattered LSQR projection, not to a vertex-value misalignment or to the cubic interpolant itself.

## Wall-supported spline and row weighting

Adding the exact Biot wall rows expands the knot rectangle to contain every wall node. The table reports whether an unweighted least-squares fit honors those rows or averages them against the plasma samples, plus the smallest tested wall-equation multiplier that reaches a maximum wall error of `1e-6` of span. Weighting is measured on the conservative 121-node wall for each case and cell count.

| case | cells | knot factor | wall inside lattice | unweighted wall rms / span | unweighted wall max / span | smallest passing wall multiplier |
|---|---:|---:|---|---:|---:|---:|
| `diverted-single-null` | 500 | 1.0 | True | 3.018e-03 | 1.095e-02 | no pass through 1000000 |
| `diverted-single-null` | 500 | 0.5 | True | 3.167e-03 | 1.627e-02 | 10000 |
| `diverted-single-null` | 1000 | 1.0 | True | 4.206e-03 | 1.693e-02 | 1000000 |
| `diverted-single-null` | 1000 | 0.5 | True | 2.760e-03 | 1.095e-02 | 10000 |
| `weak-rotation-reactor-static` | 1000 | 1.0 | True | 6.220e-04 | 2.639e-03 | 10000 |
| `weak-rotation-reactor-static` | 1000 | 0.5 | True | 5.006e-04 | 2.205e-03 | 1000 |
| `diverted-single-null` | 2500 | 1.0 | True | 3.356e-03 | 1.097e-02 | 10000 |
| `diverted-single-null` | 2500 | 0.5 | True | 2.912e-03 | 1.301e-02 | 10000 |

Unweighted wall-supported fits **average the wall rows against the plasma data rather than honoring them**: maximum wall error ranges from **1.510e-03** to **1.774e-02** of span despite every wall node being inside the knot rectangle. Successful tested equation-row multipliers range from `1e3` to `1e6`; the diverted 500-cell one-pitch fit still misses the target at `1e6`.

## Verdict

Across the one-pitch fits, adding vertex values changed the median data-point rms residual from **3.953e-04** to **1.812e-03** of span (ratio 4.583). Fine half-pitch lattices that carry more coefficients than observations are explicitly reported as structurally underdetermined; their projection condition is infinite even when LSQR returns a small data residual.

The largest spline wall-node error is **3.775e+00 of span**. The receipt carries every node's signed error and distance beyond the final knot; the second figure shows the error growth with extrapolation distance. This decides wall usability from the measured errors rather than from successful evaluation outside the lattice.

Shape timing on the single-null 1000-cell, 121-wall-node centroid-plus-vertex one-pitch fit: one matrix-free fit took **0.218 s**; one batched read of 16 spline states took median **0.000440 s** (0.000028 s/state) on the CPU allocation.

## Figures

- `saddle-position-error-vs-cell-pitch.svg` — saddle error in pitch units; bands span all four wall samplings.
- `wall-error-vs-knot-extrapolation.svg` — binned rms spline-minus-exact wall flux against distance beyond the knot rectangle.
