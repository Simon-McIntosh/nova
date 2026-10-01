# Full-state unit-amplitude census after the bank-capacity repair

The final gate ran in one `all_debug` allocation, job `1276584`, at revision
`4b7b59ad6357d7621460f12e10620dab07495545`. The payload used `TMPDIR=/tmp`,
`JAX_PLATFORMS=cpu`, the shared Nova interpreter directly, and no `uv` on the
compute node. The fixed-state census passed the complete 772-row state to
`operator._fixed_design_read`: 135 grid rows, 121 wall rows, and the authored
direct-sampling rows. The gate exited 0.

## Terminal result

The bank-overflow count is zero in every state/mode pair, and the exact total
is finite at the terminal state. The unit-amplitude totals are in amperes.

| state | analytic cell total | chord total | exact total |
| --- | ---: | ---: | ---: |
| production seed | 16,314,568.877 | 8,010,834.264 | 7,575,142.892 |
| committed chord terminal | 16,314,568.877 | 15,875,439.765 | 14,013,835.437 |

The analytic target current is 16,314,773.312 A. The final terminal exact total
is therefore finite but remains 2,300,937.875 A below the target; the terminal
chord total remains 439,333.547 A below it. The amplitude gate is false at both
states (`seed`: chord 2.03658855, exact 2.15372483; `terminal`: chord
1.02767379, exact 1.16419044). These are measured deficits, not normalisation
or gate-censoring artefacts. The seed digest does not match the committed bank
digest, so the banked-amplitude identity is retained as provenance and is not
used as an acceptance claim.

The per-cell receipt records 135 realised cells, 43 analytic cut cells, 78
interior cells, and 14 exterior cells. The two ranked and contour figures show
the per-cell difference and its poloidal distribution:

- `difference-ranked.svg` / `difference-ranked.png`
- `difference-contours.svg` / `difference-contours.png`

## Focused gates

Fresh processes passed all named focused modules:

- `tests/test_moment_path_separatrix_test.py`: 1 passed in 5.81 s
- `tests/test_continuous_confined_moments.py`: 3 passed in 25.75 s
- `tests/test_exact_bank_callback_capacity.py`: 3 passed in 19.09 s

The declared negative control passed the sliced state to
`_fixed_design_read` again and refused as required:

```text
ValueError: own-node null census requires the direct sampling flux values
```

The negative-control log also confirms that the full-state read succeeds before
the intentionally sliced call is diverted. Its exit status is 1 by design.

The source probe change is in commit `87d11db6c`; the per-cell receipt/report
driver correction is in `4b7b59ad6`. The output receipt, logs, exit markers,
and figures are committed alongside this report. Nothing under `nova/` changed.
