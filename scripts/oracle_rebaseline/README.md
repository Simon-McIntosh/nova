# Closed-form recovery rebaseline

This lane solves each exact-exterior analytic fixture from Nova's production
current-centroid uniform-disc seed. The plasma part of the seed carries only
the aggregate zeroth current moment and current centroid; it never evaluates
the closed-form flux field. The exact exterior supply remains the fixture's
declared free-boundary condition.

Run the carriers in dependency order, then merge their receipts:

```bash
scripts/oracle_rebaseline/run-measure.sh --fixture coarse
scripts/oracle_rebaseline/run-measure.sh --fixture fine
scripts/oracle_rebaseline/run-measure.sh --merge
```

The fixed solver budget is ten undamped Newton steps with thirty fixed GMRES
iterations per step and the unchanged relative residual criterion is
`1e-10`. A direct moment-seed root outside the analytic basin triggers the
recorded source-strength continuation. Every trial and accepted continuation
state remains in the fixture receipt.

`results.json` carries the measured floors and owner-reviewable proposed
bounds. `root-coarse.npz` and `root-fine.npz` contain the terminal, oracle and
seed states, residual traces, locally anchored normalized flux arrays, axes
and cell currents. `recovery-floors.png` compares recovery values to proposed
bounds and separates absolute deviations from the representation/reference
floor.

The measured outcome is an alternate-root hold. Coarse and fine both converge
below `2.5e-15` relative residual, but their flux sup deviations are `0.53331`
and `0.53384` of analytic span. The low-strength continuation states collapse
to the zero-core vacuum branch, while full strength returns to the same
limited alternate root. Six recovery gates reject the result, and their
coarse-to-fine excess is h-independent; no proposed bound was widened to admit
it.
