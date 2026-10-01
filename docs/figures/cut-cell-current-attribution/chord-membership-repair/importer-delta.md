# Importer delta — every `tests/` file importing `clip_quadrature`

One fresh pytest process per file per arm, run in the foreground on the fleet
allocation (`TMPDIR=/tmp`, `JAX_PLATFORMS=cpu`), against the repo's default fast
lane (`-m 'not slow'`; an explicit test path otherwise clears the inherited
filter and collects the heavy exact-mode solves). Base is a `git archive` of
`b0a0676df` under `$TMPDIR`; head is the worktree at `9acda6b6d`. Each log header
names the revision, the tree, the command, and the imported
`nova.equilibrium.clip_quadrature.__file__`, which confirms the base arm resolved
the base module (the predecessor's "base" logs resolved the worktree).

| tests/ file | base `b0a0676df` | head `9acda6b6d` | added |
|---|---|---|---|
| test_xpoint_cell_wedge_clip.py | 13 passed | 14 passed | — |
| test_clipped_support_quadrature.py | 3 passed | 3 passed | — |
| test_exact_clip_memory.py | 13 passed, 2 deselected | 13 passed, 2 deselected | — |
| test_exact_clip_moments.py | 13 passed | 13 passed | — |
| test_exact_clip_closed_form_budget.py | 5 passed | 5 passed | — |
| test_exact_bank_callback_capacity.py | 3 passed | 3 passed | — |
| test_solovev_certificate_builder.py | 4 passed | 4 passed | — |
| test_continuous_confined_moments.py | 2 failed, 1 passed | 3 passed | — |

**Added failures at head: 0.** Two base failures are fixed by the head change
(both synthetic identities in `test_continuous_confined_moments.py`):
`test_confined_moments_carry_continuous_boundary_motion` and
`test_open_closure_uses_the_complement_of_moving_support`.

## Caveats

- `test_xpoint_cell_wedge_clip.py` (0 slow marks) and
  `test_clipped_support_quadrature.py` (1 slow mark, passed either way) were run
  without an explicit `-m`; the file sets collected are unchanged.
- The base extraction is a `git archive` with no `.git`. The first base arm of
  `test_exact_clip_moments.py` failed on `test_order_study_receipt_carries_the_
  map_the_record_is_transcribed_from` only because the test shells
  `git rev-parse HEAD` in a tree with no repository. The extraction was made a
  git repository (`git init` + commit, HEAD `09af948429e5ba8d759bbebb415fdfe
  022dc28c9`) and the arm re-run: 13 passed. No other file shells git.

Logs: `base-arm-*.log` and `head-arm-*.log` in this directory.