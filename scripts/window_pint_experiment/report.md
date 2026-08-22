# Eight-window Parareal experiment

Tree: `e4582716ae1d2c5bd0761bdbb155aabd516acf73`. Backend: `cpu`. Verdict: **QUALIFIED NEGATIVE**.

## Outcome

The serial fine baseline stopped at zero-based window `4` (the fifth contiguous window), exchange `1`, sample `1` with the public typed `ConvergedNonConfinedError`. The limited/diverted core counts were `0` / `157`; the selector reason was `no_admissible_alternative` and availability was `True` / `False`.
Mechanism: accumulated transport evolution removed the selectable confined limited branch. The limited portfolio converged to a zero-core state while the 157-cell diverted candidate was not a topology-consistent available alternative, so causal branch selection refused the horizon exactly as designed.

| measure | result | gate |
|---|---:|---|
| converged serial windows | `4 / 8` | 8 / 8 |
| serial wall through refusal | `296.57999578327872` s | baseline |
| Parareal corrections used | `0` | <= 2 |
| outer residual | `not available` | <= 0.005 |
| speedup | `not measurable` | >= 2x |
| branch equivalence | `not evaluable beyond refusal` | identical |

The pre-registered experiment therefore ends before the coarse prediction: an eight-window serial physical baseline does not exist on this tree, so neither an end-to-end speedup denominator nor a valid Parareal correction trajectory can be formed. No tolerance, physics, branch policy, source multiplier, device count or surrogate was changed to manufacture a pass.

## Completed per-window receipts

| window | wall (s) | iterations | exit gate | flux closure | current closure | branch |
|---:|---:|---:|---:|---:|---:|---|
| 0 | `78.647500595077872` | 4 | `0.0012913749878219105` | `0` | `2.0883354249038394e-16` | `limited` |
| 1 | `68.831688209204003` | 4 | `0.00193242437491533` | `0` | `2.0531854294182247e-16` | `limited` |
| 2 | `81.613263159990311` | 5 | `0.0013572787835761781` | `0` | `2.0094890469354284e-16` | `limited` |
| 3 | `65.462497838074341` | 4 | `0.0025453110735676131` | `0` | `1.9662100454112779e-16` | `limited` |

One-time fixture preparation was `46.057385600870475` s. Every completed window retained its full residual trace, exit fields, valid extraction count, branch receipt, boundary hashes and conservation closures in `results.tsv`; the refusal row carries the exact exchange, sample, core counts, selector reason, availability and converged residual pair.
