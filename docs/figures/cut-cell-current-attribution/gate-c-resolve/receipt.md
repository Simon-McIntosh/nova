# Solov'ev certificate re-solve with the signed-flux profile-support clip

Sixteen certificate rows re-solved on this tree (revision `b9ac9166ba7757fef7846c54c76f5d750b080375`) and tabled beside the committed rows from `docs/figures/gs-absolute-accuracy/solovev-certificate-production-route.json`.

**Amplitude rule:** the plasma current amplitude at the seed and terminal states is within one percent of unity for the clip in production.

## Per-case verdict

| case | rows landed | residual smaller | axis error smaller | both (toward analytic) | amp rule (landed) | amp rule (committed) |
|---|---|---|---|---|---|---|
| weak | 4/4 | 0 | 0 | 0 | 0 | 0 |
| moderate | 4/4 | 0 | 0 | 0 | 0 | 0 |
| strong | 4/4 | 0 | 0 | 0 | 0 | 0 |
| diverted | 2/4 | 0 | 0 | 0 | 0 | 3 |

The fixed point moved toward the analytic equilibrium on a row when the landed terminal residual *and* the landed axis position error are both smaller than the committed row's. The amplitude rule is met when the plasma current amplitude at the seed and terminal states both lie within one percent of unity.

## Per-row table

| row | cells | res (comm) | res (landed) | conv (comm) | conv (landed) | axis err (comm) | axis err (landed) | x err (comm) | x err (landed) | qual (comm) | qual (landed) | seed amp (comm) | seed amp (landed) | term amp (comm) | term amp (landed) | trips (comm) | trips (landed) | toward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weak | 110 |      0.007 |      0.075 | False      | False      |      0.002 |      0.173 |          — |          — | unqualified | unqualified |      0.930 |      2.054 |      1.050 |      2.034 |          4 |          2 | no |
| weak | 300 |      0.006 |      0.098 | False      | False      |      0.033 |      0.366 |          — |          — | unqualified | unqualified |      0.906 |      2.025 |      0.961 |      2.634 |          4 |          6 | no |
| weak | 500 |      0.001 |      0.090 | False      | False      |      0.056 |      0.236 |          — |          — | unqualified | unqualified |      0.902 |      1.951 |      0.946 |      2.060 |          3 |          3 | no |
| weak | 1000 |      0.005 |      0.091 | False      | False      |      0.064 |      0.228 |          — |          — | unqualified | unqualified |      0.901 |      1.892 |      0.933 |      2.015 |          4 |          3 | no |
| moderate | 110 |      0.012 |      0.068 | False      | False      |      0.003 |      0.073 |          — |          — | unqualified | unqualified |      0.913 |      2.082 |      1.020 |      2.217 |          3 |          3 | no |
| moderate | 300 |      0.008 |      0.097 | False      | False      |      0.011 |      0.081 |          — |          — | unqualified | unqualified |      0.900 |      1.971 |      0.960 |      2.193 |          3 |          4 | no |
| moderate | 500 |      0.007 |      0.080 | False      | False      |      0.015 |      0.153 |          — |          — | unqualified | unqualified |      0.903 |      1.907 |      0.949 |      3.759 |          4 |         12 | no |
| moderate | 1000 |      0.006 |      0.097 | False      | False      |      0.018 |      0.122 |          — |          — | unqualified | unqualified |      0.902 |      1.866 |      0.934 |      2.876 |          4 |          9 | no |
| strong | 110 |      0.014 |      0.038 | False      | False      |      0.003 |      0.027 |          — |          — | unqualified | unqualified |      0.919 |      1.905 |      0.996 |      1.945 |          3 |          6 | no |
| strong | 300 |      0.017 |      0.056 | False      | False      |      0.004 |      0.029 |          — |          — | unqualified | unqualified |      0.901 |      1.804 |      0.960 |      1.923 |          5 |          7 | no |
| strong | 500 |      0.006 |      0.049 | False      | False      |      0.006 |      0.013 |          — |          — | unqualified | unqualified |      0.899 |      1.793 |      0.937 |      1.725 |          3 |          3 | no |
| strong | 1000 |      0.004 |      0.055 | False      | False      |      0.007 |      0.014 |          — |          — | unqualified | unqualified |      0.899 |      1.729 |      0.927 |      1.690 |          3 |          2 | no |
| diverted | 110 |  5.291e-15 |          — | True       |          — |      0.005 |          — |          — |          — | unqualified | None |      1.010 |          — |      1.000 |          — |          2 |          — | — |
| diverted | 300 |  5.627e-15 |          — | True       |          — |      0.003 |          — |          — |          — | unqualified | None |      1.005 |          — |      1.000 |          — |          2 |          — | — |
| diverted | 500 |  2.423e-17 |      0.194 | True       | False      |      0.001 |      0.558 |      0.001 |          — | qualified | unqualified |      1.005 |      0.957 |      1.000 |     21.065 |          3 |          4 | no |
| diverted | 1000 |      0.040 |      0.112 | False      | False      |      0.010 |      0.581 |      0.026 |          — | unqualified | unqualified |      1.006 |      0.957 |      1.002 |     31.038 |          6 |          5 | no |

## Amplitude rule detail (landed rows)

| row | cells | seed amplitude | terminal amplitude | within one percent |
|---|---|---|---|---|
| weak | 110 |      2.054 |      2.034 | no |
| weak | 300 |      2.025 |      2.634 | no |
| weak | 500 |      1.951 |      2.060 | no |
| weak | 1000 |      1.892 |      2.015 | no |
| moderate | 110 |      2.082 |      2.217 | no |
| moderate | 300 |      1.971 |      2.193 | no |
| moderate | 500 |      1.907 |      3.759 | no |
| moderate | 1000 |      1.866 |      2.876 | no |
| strong | 110 |      1.905 |      1.945 | no |
| strong | 300 |      1.804 |      1.923 | no |
| strong | 500 |      1.793 |      1.725 | no |
| strong | 1000 |      1.729 |      1.690 | no |
| diverted | 500 |      0.957 |     21.065 | no |
| diverted | 1000 |      0.957 |     31.038 | no |
