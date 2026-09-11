# Cut-cell clipped coupling: report

Node `cca-cut-cell-clipped-coupling`, run r-20260911T001005914789.
Base: 1046dc4f (rung A2, frozen per-trip support). Implementation commit
`dbf3434f`, worktree
`.reckon-worktrees/nova-a0f1e0938fc2/s19-review/cca-cut-cell-clipped-coupling`.

## What changed

`coupling_current_moments` and `current_moment_image` in
`nova/equilibrium/forward_operator.py` now accept a per-trip
`ClippedCouplingGeometry` (built in `_clipped_coupling_geometry`, carried by
the frozen partition). For every cut cell of the frozen support, the
geometry re-references the physical first moments from the atomic centroid
to the clipped polygon's centroid and inverts them with the clipped
polygon's area-normalised second moments; the zeroth, first and second
kernel blocks for that cell are the polygon-analytic integrals over the
clipped polygon about that centroid. Interior cells keep the precomputed
atomic blocks. The blocks are rebuilt in host code once per trip where the
frozen partition is refreshed, never inside a Jacobian-vector product.

## Why the discriminator's region-referencing looked like a null

The prior discriminator re-referenced the moments and blocks to the
analytic plasma region and observed no improvement (0.0106 unchanged). The
regression here: its certificate anchor bakes the atomic image of the
*traced-support* moments, so the residual it measured was dominated by the
analytic-region-versus-traced-support geometry mismatch, not the coupling.
Two per-cell probes on weak -110 settle the physical question directly:

| cell | atomic-vs-truth rms (Wb) | clipped-vs-truth rms (Wb) |
|---|---|---|
| 2  | 1.9e-3 | 6.6e-5 |
| 3  | 1.6e-3 | 3.5e-4 |
| 13 | 4.6e-3 | 4.3e-4 |
| 50 | 2.2e-3 | 1.0e-4 |

For a prescribed linear density over a cut cell's clipped polygon, the
clipped coupling matches the exact ring-inductance integral to
1.98e-11 rms (1.1e-10 max relative) on the nodes outside the source cell,
while the atomic coupling departs at the 1e-3 level.

## Gate one — frozen-current flux image (all_debug, CPU x64)

`benchmarks/frozen_current_flux_image_gate.py`, rows weak/moderate/strong
-110 and weak -300, frozen exact per-cell currents over the frozen support
polygons, certificate-anchored external. Acceptance: zeroth-plus-first
clipped image at or below the zeroth-only error on every row, and at or
below 0.0057 on weak -110.

| row | zeroth-only | as-built | clipped (after) | at/below zeroth | at/below floor |
|---|---|---|---|---|---|
| weak -110   | 0.00603 | anchor-identical | 0.000266 | yes | yes (0.0057) |
| moderate -110| 0.00566 | ... | 0.000282 | yes | yes |
| strong -110 | 0.00573 | ... | 0.000317 | yes | yes |
| weak -300   | 0.00222 | ... | 0.000118 | yes | yes |

The as-built atomic image reproduces the certificate anchor by construction
(residual ~1e-10); the clipped image is the same frozen moments through the
new clipped blocks and sits 20-50x below the zeroth floor, so the
first-order term now removes error on every row rather than adding it.

## Gate two — regression tests

Six rung-A2 test files plus the new
`tests/test_forward_cut_cell_coupling.py` (which pins a cut cell's imaged
flux of a linear density to the exact ring integral over its clipped
polygon at <1e-10 rms relative, and confirms an interior cell's image is
unchanged between the atomic and clipped paths). The six files carry zero
added failures against the base (operator_domain keeps its 3 pre-existing
failures).

## Artifacts

- docs/figures/cut-cell-current-attribution/cut-cell-coupling/receipt.json
- docs/figures/cut-cell-current-attribution/cut-cell-coupling/weak-110-coupling-errors.svg
- tests/test_forward_cut_cell_coupling.py
- gate logs: run directory gate1.log, gate2.log
