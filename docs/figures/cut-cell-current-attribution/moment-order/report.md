# Frozen atomic-cell moment-order evaluation

Only order two reaches 0.000266 of span with frozen atomic blocks: its worst measured-reference upper envelope is 0.000133172, while order one's 0.000254939 point maximum widens to 0.000275674; the order-two cut-source near-field floor is 0.000128718.

Quadratic blocks use the same total-flux filament kernel and atomic centroid as the committed linear blocks, integrated over each full atomic polygon by a fixed tensor-Duffy rule. The clipped polygon never changes a block; it enters only through the six projected current moments.

Figures: [weak 110](/nova/figures/cut-cell-current-attribution/moment-order/weak-rotation-reactor-static-110.svg), [weak 300](/nova/figures/cut-cell-current-attribution/moment-order/weak-rotation-reactor-static-300.svg), [moderate 110](/nova/figures/cut-cell-current-attribution/moment-order/moderate-rotation-conventional-static-110.svg), and [moderate 300](/nova/figures/cut-cell-current-attribution/moment-order/moderate-rotation-conventional-static-300.svg).

| case | cells | reference check | order 0 | order 1 | order 2 |
|---|---:|---:|---:|---:|---:|
| weak-rotation-reactor-static | 110 | 1.84923e-05 | 0.00550727 | 0.000240038 | 0.000114679 |
| weak-rotation-reactor-static | 300 | 6.42706e-06 | 0.00202383 | 0.000110122 | 3.24354e-05 |
| moderate-rotation-conventional-static | 110 | 2.0734e-05 | 0.00508666 | 0.000254939 | 0.000103971 |
| moderate-rotation-conventional-static | 300 | 5.91683e-06 | 0.00204198 | 0.000106528 | 3.34821e-05 |

The measured reference-check residual is retained as an additive, fail-closed uncertainty envelope. Worst upper RMS/span values are 0.00552576 for order zero, 0.000275674 for order one and 0.000133172 for order two.

## Four-way source and target split

Each value is `RMS / sup`, normalised by the row's grid flux span. Near means self plus first-ring distance (at most 1.1 carrier pitches).

### weak-rotation-reactor-static / 110 cells

| route | cut near | cut far | whole near | whole far |
|---|---:|---:|---:|---:|
| order zero | 0.00446891 / 0.00898891 | 0.00648728 / 0.007888 | 0.000822115 / 0.00166022 | 0.000812227 / 0.00157058 |
| order one | 0.0002695 / 0.000994 | 2.99747e-05 / 6.55277e-05 | 1.11487e-05 / 5.21487e-05 | 2.07941e-07 / 3.34892e-07 |
| order two | 0.000128718 / 0.000544542 | 5.17999e-06 / 2.35717e-05 | 1.09139e-05 / 5.17163e-05 | 3.0551e-10 / 1.04506e-09 |

### weak-rotation-reactor-static / 300 cells

| route | cut near | cut far | whole near | whole far |
|---|---:|---:|---:|---:|
| order zero | 0.00148191 / 0.00335477 | 0.00228073 / 0.00306429 | 0.000333072 / 0.000693114 | 0.000248577 / 0.000654523 |
| order one | 0.000139827 / 0.000708364 | 1.32478e-05 / 7.00683e-05 | 4.57434e-06 / 2.91413e-05 | 2.45858e-08 / 5.09558e-08 |
| order two | 4.10176e-05 / 0.000182901 | 1.43856e-06 / 1.33717e-05 | 4.53922e-06 / 2.91332e-05 | 2.70082e-11 / 1.26543e-10 |

### moderate-rotation-conventional-static / 110 cells

| route | cut near | cut far | whole near | whole far |
|---|---:|---:|---:|---:|
| order zero | 0.00411269 / 0.00923715 | 0.006046 / 0.00822421 | 0.000864059 / 0.0017517 | 0.000625833 / 0.00158299 |
| order one | 0.000285223 / 0.00148836 | 4.30902e-05 / 0.000102644 | 1.09733e-05 / 7.07719e-05 | 1.90748e-07 / 3.97427e-07 |
| order two | 0.000116353 / 0.000529235 | 4.43729e-06 / 2.1649e-05 | 1.07307e-05 / 7.06417e-05 | 3.73278e-10 / 1.4455e-09 |

### moderate-rotation-conventional-static / 300 cells

| route | cut near | cut far | whole near | whole far |
|---|---:|---:|---:|---:|
| order zero | 0.00152014 / 0.00360292 | 0.0022976 / 0.00323809 | 0.000325265 / 0.00068141 | 0.000232452 / 0.000647956 |
| order one | 0.000135754 / 0.000696742 | 1.02324e-05 / 4.9987e-05 | 4.5211e-06 / 1.30291e-05 | 2.80561e-08 / 5.65406e-08 |
| order two | 4.249e-05 / 0.000210417 | 1.26089e-06 / 9.01282e-06 | 4.48321e-06 / 1.30074e-05 | 2.94086e-11 / 1.52955e-10 |

| case | cells | order 1 / order 0 | cut projection L2 | whole projection L2 | cut near order 1 | untranslated order 1 | sign flipped | axes swapped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| weak-rotation-reactor-static | 110 | 0.04359 | 0.1834 | 0.0001344 | 0.0002695 | 0.004924 | 0.01096 | 0.006236 |
| weak-rotation-reactor-static | 300 | 0.05441 | 0.1972 | 4.842e-05 | 0.0001398 | 0.001818 | 0.004032 | 0.002191 |
| moderate-rotation-conventional-static | 110 | 0.05012 | 0.1729 | 0.000148 | 0.0002852 | 0.004514 | 0.01012 | 0.005307 |
| moderate-rotation-conventional-static | 300 | 0.05217 | 0.1904 | 5.579e-05 | 0.0001358 | 0.001847 | 0.004068 | 0.002089 |

## Diagnosis

The frozen-current measurement does not reproduce a first-order regression: first order is 0.04359 to 0.05441 times the zeroth-order RMS error. The earlier discriminator worsening therefore does not come from the frozen first-order matmul itself; it enters through the coupled state/support path or that earlier instrument. The residual left here is nevertheless localised: cut-cell linear-density projection residuals are 0.173 to 0.197 in relative L2, while whole cells are 4.84e-5 to 1.48e-4, and cut-source near-target errors exceed their far-target errors on every row.

The clipped-to-atomic first-moment translation closes to 2.662e-13 relative and the committed linear conversion matches an independent full-cell Gram inversion to 5.040e-14 relative. Feeding clipped-centred moments without the translation is shown in the table, so the reference-point candidate is tested rather than assumed.

On far targets the numerical total-flux kernel columns agree with the committed uniform, radial and vertical blocks to at worst 2.579e-08 relative RMS. The sign-flipped and axis-swapped results in the table test the remaining convention candidates. Full cut/whole and self-plus-first-ring/far RMS and sup metrics are retained in receipt.json.
