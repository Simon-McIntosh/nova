# Net-current scaling census per forward route

Revision: `26fc98cff318fd9c0cab291fb9869fa7130a9e9c`

lambda is `declared-current amplitude target_current / sum(unscaled cell current); the amplitude admitted within SCALAR_CURRENT_AMPLITUDE_BAND`

Amplitude band: `[1e-06, 1000000.0]`; neighbourhood of unity: `[0.9, 1.1]`; certificate resolution: `-110` cells.

## Per-row lambda at seed and terminal

| route | clip | case | seed lambda | terminal lambda | converged | termination | core cells | support current (A) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newton_krylov | chord | weak-rotation-reactor-static | 2.0366 | 1.8112 | False | active_set_settled | 54 | 9007823.304 |
| newton_krylov | chord | moderate-rotation-conventional-static | 2.0619 | 1.7776 | False | active_set_settled | 56 | 914126.030 |
| newton_krylov | chord | strong-rotation-compact-static | 1.8908 | 3.0344 | False | active_set_settled | 32 | 82800.386 |
| newton_krylov | chord | diverted-single-null | 1.0004 | 11.9871 | False | active_set_settled | 4 | 3113.282 |
| reduced_newton | chord | weak-rotation-reactor-static | 2.0366 | 1.7156 | False | active_set_settled | 62 | 9509911.739 |
| reduced_newton | chord | moderate-rotation-conventional-static | 2.0619 | 1.6008 | False | active_set_settled | 63 | 1015090.424 |
| reduced_newton | chord | strong-rotation-compact-static | 1.8908 | 1.8194 | False | sufficient_decrease_refused | 55 | 138096.275 |
| reduced_newton | chord | diverted-single-null | 1.0004 | 53.8026 | False | active_set_settled | 1 | 693.635 |

## How each route was read

- **assembly** (assembled): per-row files merged over the rows already measured in report.json
- **cold-seed** (derived-from-row): read off the certificate row above, which records the seed state and the terminal state of the same forward solve
- **cold-seed** (derived-from-row): read off the certificate row above, which records the seed state and the terminal state of the same forward solve
- **cold-seed** (derived-from-row): read off the certificate row above, which records the seed state and the terminal state of the same forward solve
- **cold-seed** (derived-from-row): read off the certificate row above, which records the seed state and the terminal state of the same forward solve
- **cold-seed** (derived-from-row): read off the certificate row above, which records the seed state and the terminal state of the same forward solve
- **cold-seed** (derived-from-row): read off the certificate row above, which records the seed state and the terminal state of the same forward solve
- **cold-seed** (derived-from-row): read off the certificate row above, which records the seed state and the terminal state of the same forward solve
- **cold-seed** (derived-from-row): read off the certificate row above, which records the seed state and the terminal state of the same forward solve
- **sol-ledger-census** (read-from-committed-receipt): the SOL ledger census books a terminal plasma current and its common-SOL split; it records no seed state and no declared-current amplitude, so a seed-to-terminal lambda pair cannot be read from it without re-evaluating the operator on its fixture
- **mast-bank-twelve-rows** (read-from-committed-receipt): each bank row carries a terminal residual, termination reason and class verdict, not a declared-current amplitude; deriving seed and terminal lambda needs the MAST machine and corpus, which this node is not scoped or equipped to reach on the CPU lane

## Planned rows that did not run

- **newton_krylov/exact/weak-rotation-reactor-static**: the exact clip mode did not finish inside a one-hour all_debug allocation; its job log under logs/ records 'CANCELLED ... DUE TO TIME LIMIT' and no row file was written, so its seed and terminal lambda are absent from this census
- **newton_krylov/exact/moderate-rotation-conventional-static**: the exact clip mode did not finish inside a one-hour all_debug allocation; its job log under logs/ records 'CANCELLED ... DUE TO TIME LIMIT' and no row file was written, so its seed and terminal lambda are absent from this census
- **newton_krylov/exact/strong-rotation-compact-static**: the exact clip mode did not finish inside a one-hour all_debug allocation; its job log under logs/ records 'CANCELLED ... DUE TO TIME LIMIT' and no row file was written, so its seed and terminal lambda are absent from this census
- **newton_krylov/exact/diverted-single-null**: the exact clip mode did not finish inside a one-hour all_debug allocation; its job log under logs/ records 'CANCELLED ... DUE TO TIME LIMIT' and no row file was written, so its seed and terminal lambda are absent from this census

## Reconciliation against the banked history

- **chord weak-rotation-reactor-static -110 [newton_krylov]**: banked seed 0.93 / terminal 1.05 (measured seed 2.0366 / terminal 1.8112, converged False); delta seed 1.1066, delta terminal 0.7612.
- **chord weak-rotation-reactor-static -110 [reduced_newton]**: banked seed 0.93 / terminal 1.05 (measured seed 2.0366 / terminal 1.7156, converged False); delta seed 1.1066, delta terminal 0.6656.
- **exact support weak-rotation-reactor-static -110**: banked seed 2.201 / terminal 5.561; no row in this run matches case=weak-rotation-reactor-static requested_cells=-110 clip_mode=exact (banked note: exact-support deficit, a symptom of the dropped clip).
- **diverted production seed**: banked seed 0.98 / terminal None; no row in this run matches case=diverted-single-null requested_cells=None clip_mode=chord (banked note: eighteen S18 trip panels, eighty-three-cell diverted production seed).

## Verdict per route on converged rows

- **newton_krylov/chord**: 0/4 converged; terminal lambda range None to None; all within neighbourhood: False.
- **reduced_newton/chord**: 0/4 converged; terminal lambda range None to None; all within neighbourhood: False.
