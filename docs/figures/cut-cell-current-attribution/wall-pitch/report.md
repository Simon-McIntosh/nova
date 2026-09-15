# Analytic wall contact at one panel per cell pitch

The production read takes the fixed-design hex-carrier branch and resolves the contact inside one wall panel through the three-node quadratic wall fit, landing on the authored smooth-limiter outboard tangency to sub-micron accuracy (`read_distance_to_true_tangency_m`, positive control).  The remaining position error is therefore the sagitta of the wall polyline itself: the gap between the true tangency `[outboard, 0]` and the closest point the piecewise wall represents.  Each row samples the wall at one, half and a quarter outboard panel per realised cell pitch beside the fixed 121-node baseline.

| Case | Cells | Wall nodes | Pitch [m] | Panel / pitch | Contact error [m] | Error / pitch | Level / span | Build [s] | Wall family [s] | Matrix [MB] | Warm read [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| moderate-rotation-conventional-static | 2500 | 121 | 0.027569 | 2.1777 | 2.84940793e-02 | 1.03354282e+00 | 8.14458022e-04 | 0.0 | 0.0 | 573.9 | 7.8790 |
| moderate-rotation-conventional-static | 1000 | 121 | 0.043591 | 1.3773 | 2.84940793e-02 | 6.53669873e-01 | 8.14475270e-04 | 0.0 | 0.0 | 103.0 | 2.0257 |
| moderate-rotation-conventional-static | 1000 | 167 | 0.043596 | 0.9981 | 2.06736957e-02 | 4.74215843e-01 | 4.27435982e-04 | 169.5 | 15.1 | 104.2 | 2.0094 |
| moderate-rotation-conventional-static | 1000 | 333 | 0.043599 | 0.5006 | 1.03645283e-02 | 2.37722075e-01 | 1.07472315e-04 | 184.0 | 19.4 | 108.7 | 2.1781 |
| moderate-rotation-conventional-static | 1000 | 667 | 0.043600 | 0.2499 | 5.17513301e-03 | 1.18694860e-01 | 2.67858207e-05 | 215.1 | 32.9 | 117.3 | 2.0697 |
| weak-rotation-reactor-static | 2500 | 121 | 0.097762 | 2.1974 | 1.01900674e-01 | 1.04233113e+00 | 8.03757907e-04 | 0.0 | 0.0 | 563.9 | 8.3397 |
| weak-rotation-reactor-static | 2500 | 267 | 0.097780 | 0.9960 | 4.62248468e-02 | 4.72745380e-01 | 1.64981561e-04 | 799.9 | 42.2 | 574.3 | 8.2684 |
| weak-rotation-reactor-static | 2500 | 533 | 0.097783 | 0.4990 | 2.31603884e-02 | 2.36855119e-01 | 4.13959726e-05 | 837.0 | 62.1 | 591.0 | 7.9832 |
| weak-rotation-reactor-static | 2500 | 1065 | 0.097784 | 0.2497 | 1.15916183e-02 | 1.18543371e-01 | 1.03681959e-05 | 945.5 | 116.6 | 624.3 | 7.6883 |
| weak-rotation-reactor-static | 1000 | 121 | 0.154576 | 1.3897 | 1.01900674e-01 | 6.59228088e-01 | 8.03763403e-04 | 0.0 | 0.0 | 103.5 | 1.9093 |
| weak-rotation-reactor-static | 1000 | 169 | 0.154593 | 0.9952 | 7.30015676e-02 | 4.72219268e-01 | 4.11888314e-04 | 178.0 | 15.5 | 104.7 | 1.8838 |
| weak-rotation-reactor-static | 1000 | 337 | 0.154606 | 0.4991 | 3.66268102e-02 | 2.36904610e-01 | 1.03556753e-04 | 179.2 | 18.5 | 109.0 | 2.0214 |
| weak-rotation-reactor-static | 1000 | 673 | 0.154609 | 0.2499 | 1.83428907e-02 | 1.18640468e-01 | 2.59645059e-05 | 214.8 | 32.5 | 118.0 | 2.0165 |

## Per-pitch wall counts

Counts are odd, derived from the realised pitch and the limiter perimeter through the outboard panel factor (outboard panel over average panel at 121 nodes): `N(f) = odd(perimeter x factor / (f x pitch))` for `f in {1, 1/2, 1/4}`.
| Case | Cells | 1 | 1/2 | 1/4 |
|---|---|---:|---:|---:|
| weak-rotation-reactor-static | 1000 | 169 | 337 | 673 |
| weak-rotation-reactor-static | 2500 | 267 | 533 | 1065 |
| moderate-rotation-conventional-static | 1000 | 167 | 333 | 667 |
| moderate-rotation-conventional-static | 2500 | n/a | n/a | n/a |

## One thousandth of pitch

- weak-rotation-reactor-static 1000: measured slope 0.474694 error/pitch per panel/pitch; one thousandth of pitch needs 57275 wall nodes (~2947 s build, 1473.6 MB wall family), so the polyline sagitta does not reach the target at any practical count and requires a curved wall representation.
- weak-rotation-reactor-static 2500: measured slope 0.474701 error/pitch per panel/pitch; one thousandth of pitch needs 90561 wall nodes (~10746 s build, 5672.7 MB wall family), so the polyline sagitta does not reach the target at any practical count and requires a curved wall representation.
- moderate-rotation-conventional-static 1000: measured slope 0.474919 error/pitch per panel/pitch; one thousandth of pitch needs 56927 wall nodes (~2991 s build, 1463.3 MB wall family), so the polyline sagitta does not reach the target at any practical count and requires a curved wall representation.
- moderate-rotation-conventional-static 2500: measured slope 0.474596 error/pitch per panel/pitch; one thousandth of pitch needs 89957 wall nodes (n/a (cached), 5658.7 MB wall family), so the polyline sagitta does not reach the target at any practical count and requires a curved wall representation. (one-point slope from the 121-node baseline; the finer samplings did not land in this run)
