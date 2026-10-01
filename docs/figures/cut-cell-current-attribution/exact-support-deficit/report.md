# Exact-support deficit per cut cell, weak 110 (135 realised) row

Revision `1f85669c01eb82f3e6580b78718ce31d0919a9c0`.
Analytic oracle total over the cut cells is 16.314569 MA; the target current is 16.314773 MA.

## State `seed`

Boundary level 35.8528226 Wb, polarity 1, cut cells 43.

Totals over the cut cells: analytic 3.3763 MA, chord 0.6193 MA, exact 0.1892 MA (defined: True).

Cut-cell moment bank: 8 call signature(s), 0 overflowing.

| cells | capacity | boundary reduction | cut count | overflow |
|---|---|---|---|---|
| 135 | 135 | False | 0 | False |
| 135 | 135 | True | 30 | False |
| 135 | 1 | False | 0 | False |
| 135 | 1 | True | 0 | False |
| 1 | 1 | False | 1 | False |
| 1 | 1 | True | 1 | False |
| 1 | 1 | True | 0 | False |
| 1 | 1 | False | 0 | False |

| cause | cells |
|---|---|
| `empty-no-vertex-inside-level` | 35 |
| `shrunk-partial-polygon` | 8 |

### Named cells

| cell | analytic A | chord A | exact A | exact area m2 | chord area m2 | vertices inside | cause |
|---|---|---|---|---|---|---|---|
| 2 | 23242.042 | 0.000 | 0.000 | 0.000000e+00 | 1.148917e-01 | 0/15 | `empty-no-vertex-inside-level` |
| 35 | 8850.331 | 0.000 | 0.000 | 0.000000e+00 | 1.378313e-01 | 0/11 | `empty-no-vertex-inside-level` |
| 75 | 76394.816 | 0.000 | 0.000 | 0.000000e+00 | 1.528228e-01 | 0/13 | `empty-no-vertex-inside-level` |
| 89 | 6228.288 | 0.000 | 0.000 | 0.000000e+00 | 7.867772e-03 | 0/5 | `empty-no-vertex-inside-level` |
| 114 | 6228.288 | 0.000 | 0.000 | 0.000000e+00 | 7.867772e-03 | 0/5 | `empty-no-vertex-inside-level` |
| 126 | 23242.042 | 0.000 | 0.000 | 0.000000e+00 | 1.148917e-01 | 0/15 | `empty-no-vertex-inside-level` |
| 128 | 8850.331 | 0.000 | 0.000 | 0.000000e+00 | 1.378313e-01 | 0/11 | `empty-no-vertex-inside-level` |
| 134 | 76394.816 | 0.000 | 0.000 | 0.000000e+00 | 1.528228e-01 | 0/13 | `empty-no-vertex-inside-level` |

## State `terminal`

Boundary level 10.3951318 Wb, polarity 1, cut cells 43.

Totals over the cut cells: analytic 3.3763 MA, chord 2.9514 MA, exact nan MA (defined: False).

Cut-cell moment bank: 8 call signature(s), 1 overflowing.

| cells | capacity | boundary reduction | cut count | overflow |
|---|---|---|---|---|
| 135 | 135 | False | 0 | False |
| 135 | 135 | True | 41 | False |
| 135 | 1 | False | 0 | False |
| 135 | 1 | True | 2 | True |
| 1 | 1 | False | 1 | False |
| 1 | 1 | True | 1 | False |
| 1 | 1 | True | 0 | False |
| 1 | 1 | False | 0 | False |

| cause | cells |
|---|---|
| `exact-current-undefined-bank-overflow` | 43 |

### Named cells

| cell | analytic A | chord A | exact A | exact area m2 | chord area m2 | vertices inside | cause |
|---|---|---|---|---|---|---|---|
| 2 | 23242.042 | 12590.489 | nan | 0.000000e+00 | 1.148917e-01 | 2/15 | `exact-current-undefined-bank-overflow` |
| 35 | 8850.331 | 0.000 | nan | 0.000000e+00 | 1.378313e-01 | 0/11 | `exact-current-undefined-bank-overflow` |
| 75 | 76394.816 | 60425.556 | nan | 0.000000e+00 | 1.528228e-01 | 3/13 | `exact-current-undefined-bank-overflow` |
| 89 | 6228.288 | 5700.541 | nan | 0.000000e+00 | 7.867772e-03 | 2/5 | `exact-current-undefined-bank-overflow` |
| 114 | 6228.288 | 5700.541 | nan | 0.000000e+00 | 7.867772e-03 | 2/5 | `exact-current-undefined-bank-overflow` |
| 126 | 23242.042 | 12590.489 | nan | 0.000000e+00 | 1.148917e-01 | 2/15 | `exact-current-undefined-bank-overflow` |
| 128 | 8850.331 | 0.000 | nan | 0.000000e+00 | 1.378313e-01 | 0/11 | `exact-current-undefined-bank-overflow` |
| 134 | 76394.816 | 60425.556 | nan | 0.000000e+00 | 1.528228e-01 | 3/13 | `exact-current-undefined-bank-overflow` |

