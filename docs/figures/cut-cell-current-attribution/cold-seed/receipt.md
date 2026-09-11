# Solov'ev certificate re-solve with the signed-flux profile-support clip

Sixteen certificate rows re-solved on this tree (revision `1e79e056536f8801178aabc1aab4996e47a7f782`) and tabled beside the committed rows from `docs/figures/gs-absolute-accuracy/solovev-certificate-production-route.json`.

**Amplitude rule:** the plasma current amplitude at the seed and terminal states is within one percent of unity for the clip in production.

## Per-case verdict

| case | rows landed | residual smaller | axis error smaller | both (toward analytic) | amp rule (landed) | amp rule (committed) |
|---|---|---|---|---|---|---|
| weak | 1/4 | 0 | 0 | 0 | 0 | 0 |
| moderate | 1/4 | 1 | 0 | 0 | 0 | 0 |
| strong | 1/4 | 1 | 0 | 0 | 0 | 0 |
| diverted | 0/4 | 0 | 0 | 0 | 0 | 3 |

The fixed point moved toward the analytic equilibrium on a row when the landed terminal residual *and* the landed axis position error are both smaller than the committed row's. The amplitude rule is met when the plasma current amplitude at the seed and terminal states both lie within one percent of unity.

## Per-row table

| row | cells | res (comm) | res (landed) | conv (comm) | conv (landed) | axis err (comm) | axis err (landed) | x err (comm) | x err (landed) | qual (comm) | qual (landed) | seed amp (comm) | seed amp (landed) | term amp (comm) | term amp (landed) | trips (comm) | trips (landed) | toward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weak | 110 |      0.007 |      0.015 | False      | False      |      0.002 |      0.069 |          — |          — | unqualified | unqualified |      0.930 |      1.012 |      1.050 |      0.953 |          4 |          6 | no |
| weak | 300 |      0.006 |          — | False      |          — |      0.033 |          — |          — |          — | unqualified | None |      0.906 |          — |      0.961 |          — |          4 |          — | — |
| weak | 500 |      0.001 |          — | False      |          — |      0.056 |          — |          — |          — | unqualified | None |      0.902 |          — |      0.946 |          — |          3 |          — | — |
| weak | 1000 |      0.005 |          — | False      |          — |      0.064 |          — |          — |          — | unqualified | None |      0.901 |          — |      0.933 |          — |          4 |          — | — |
| moderate | 110 |      0.012 |      0.008 | False      | False      |      0.003 |      0.021 |          — |          — | unqualified | unqualified |      0.913 |      1.024 |      1.020 |      0.962 |          3 |          7 | no |
| moderate | 300 |      0.008 |          — | False      |          — |      0.011 |          — |          — |          — | unqualified | None |      0.900 |          — |      0.960 |          — |          3 |          — | — |
| moderate | 500 |      0.007 |          — | False      |          — |      0.015 |          — |          — |          — | unqualified | None |      0.903 |          — |      0.949 |          — |          4 |          — | — |
| moderate | 1000 |      0.006 |          — | False      |          — |      0.018 |          — |          — |          — | unqualified | None |      0.902 |          — |      0.934 |          — |          4 |          — | — |
| strong | 110 |      0.014 |      0.010 | False      | False      |      0.003 |      0.006 |          — |          — | unqualified | unqualified |      0.919 |      1.031 |      0.996 |      0.955 |          3 |          7 | no |
| strong | 300 |      0.017 |          — | False      |          — |      0.004 |          — |          — |          — | unqualified | None |      0.901 |          — |      0.960 |          — |          5 |          — | — |
| strong | 500 |      0.006 |          — | False      |          — |      0.006 |          — |          — |          — | unqualified | None |      0.899 |          — |      0.937 |          — |          3 |          — | — |
| strong | 1000 |      0.004 |          — | False      |          — |      0.007 |          — |          — |          — | unqualified | None |      0.899 |          — |      0.927 |          — |          3 |          — | — |
| diverted | 110 |  5.291e-15 |          — | True       |          — |      0.005 |          — |          — |          — | unqualified | None |      1.010 |          — |      1.000 |          — |          2 |          — | — |
| diverted | 300 |  5.627e-15 |          — | True       |          — |      0.003 |          — |          — |          — | unqualified | None |      1.005 |          — |      1.000 |          — |          2 |          — | — |
| diverted | 500 |  2.423e-17 |          — | True       |          — |      0.001 |          — |      0.001 |          — | qualified | None |      1.005 |          — |      1.000 |          — |          3 |          — | — |
| diverted | 1000 |      0.040 |          — | False      |          — |      0.010 |          — |      0.026 |          — | unqualified | None |      1.006 |          — |      1.002 |          — |          6 |          — | — |

## Amplitude rule detail (landed rows)

| row | cells | seed amplitude | terminal amplitude | within one percent |
|---|---|---|---|---|
| weak | 110 |      1.012 |      0.953 | no |
| moderate | 110 |      1.024 |      0.962 | no |
| strong | 110 |      1.031 |      0.955 | no |
