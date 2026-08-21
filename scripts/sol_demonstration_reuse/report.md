# SOL current demonstration reuse map

Repository snapshot: `b15455305c1b` (2026-08-21).

## Executive verdict

Seven required seams were audited. Two are directly reusable (domain masks and
matched-level contour plotting), two are reusable after a local contract repair
(continuation and exterior-moment routing), and three are missing or blocked
(Eich width law, X-point saddle partition, and equal-total-current plus SOL
flux-surface-average current comparison). The existing continued-equilibrium
test proves that a common-SOL source can move a converged solution, but it does
not yet prove the commissioned demonstration: its continuation is finitely
truncated, its private-flux fixture is empty, and its confined current changes by
up to 5% (`tests/test_equilibrium_sol.py:797-869`).

## Reuse inventory

| Required seam | Existing entry points and evidence | Fitness verdict |
|---|---|---|
| Eich `lambda_q` / heat-flux width | **No implementation exists.** An exhaustive case-insensitive search of source, tests, benchmarks, and scripts for `Eich`, `lambda_q`, `lambda q`, heat-flux-width spellings, and `q_width` returned 0 physics-expression hits; the only fall-off hits were unrelated numerical prose. | **HOLD — 0 reusable expressions.** Supply the Eich scaling, its required equilibrium inputs and units, and the outboard-midplane physical-width-to-`psi_N` mapping before constructing either one- or two-length SOL closures. |
| Common-SOL and private-flux masks | `nova/equilibrium/domain.py:55-124` defines the four total labels and combines closed, axis-connected, and inside-material tests. `nova/equilibrium/topology.py:310-354` publishes core, private-flux, and common-SOL labels from the live axis/X-point/wall read. `nova/equilibrium/flux_surface_connectivity.py:12-19,97-141` supplies the accelerator-native axis-connected flood fill used by the smooth topology/FSA route. `nova/equilibrium/wall_mask.py:242-323` builds `inside_limiter = inside_vessel & ~material`; `nova/equilibrium/labels.py:1-41` pins the fixed X-point slots and LCFS angular labels. | **FIT — 5 named suppliers are reusable.** Select current by `DomainMasks.common_sol` and assert exact zero on `DomainMasks.private_flux`; retain the topology label as authority rather than reconstructing either domain from `psi_N`. |
| SOL continuation contract | `nova/equilibrium/continuation.py:57-105` requires value and first-derivative continuity and describes common-SOL/private-flux policies as independent. The constructor enforces `VALUE_AND_GRADIENT` or better (`nova/equilibrium/continuation.py:344-423`), builds each branch from the core independently (`nova/equilibrium/continuation.py:425-469`), and revalidates the anchor (`nova/equilibrium/continuation.py:566-610`). `ForwardSource` refuses undeclared or cross-wired branch closures (`nova/equilibrium/source.py:430-484`). However, the exponential form requires positive finite `support` and is explicitly truncated there (`nova/equilibrium/continuation.py:305-341,344-401`). | **QUALIFIED — continuity and branch independence fit; support semantics do not.** Reuse the anchoring and independent policy, but replace or extend finite truncation so the exponential is evaluated on the entire material-bounded common-SOL mask and support extent is measured at an insignificance floor rather than imposed. |
| Exterior moments for `psi_N > 1` | The forward operator traces complementary core and exterior clips once (`nova/equilibrium/forward_operator.py:262-292`), asks the source for current plus radial and vertical first moments, and converts them into the three fixed coupling vectors (`nova/equilibrium/forward_operator.py:181-237`). `ForwardSource.current_moments` integrates core and common-SOL profiles through those supports and keeps private flux separately selected (`nova/equilibrium/source.py:568-639`). The resulting three vectors feed every plasma coupling evaluation (`nova/equilibrium/forward_operator.py:354-403`). | **QUALIFIED — the 3-vector coupling route is reusable without a new block.** A `psi_N > 1` source can ride the per-iteration vectors today, but the route is demonstration-safe only after the saddle cell is partitioned by branch rather than represented by one complement polygon. |
| X-point saddle-cell clip | The traced clip identifies a boundary cell only when it has exactly two unique crossings (`nova/equilibrium/separatrix_clip.py:598-608`); the eager clip raises on more than two and requests mesh refinement (`nova/equilibrium/separatrix_clip.py:872-880`). The banked saddle audit found 4 analytic crossings in support cell 19, while production uses 1 chord with endpoints displaced by up to 14.95 mm and a currently small approximately 0.2 A error because confined current is weak there (`docs/evidence/archive/boundary-ring-source-completion-landed.html:482-488`). It names the required explicit X-point clip vertex, branch-paired chords, and per-branch profile integration (`docs/evidence/archive/boundary-ring-source-completion-landed.html:489-494`). | **HOLD — current clip is 2-crossing/single-chord, required clip is 4-crossing/branch-paired.** Implement and test the named saddle partition before treating common-SOL current adjacent to divertor legs as physically attributed. |
| Equal net current and flux-surface-average `j_phi` | A hard measured-current equality exists in the reconstruction KKT path (`nova/equilibrium/profile.py:455-521`), but `ForwardSource` is absolute-only and explicitly says target-current normalisation is a separate policy (`nova/equilibrium/source.py:453-463`). `ForwardProfile` exposes current observations and their Jacobian (`nova/equilibrium/forward.py:316-324,573-585`), while its `plasma_current` observation is confined-core current and open current is separate in the total ledger (`nova/equilibrium/observation.py:33-41,335-375`). The transport path computes confined flux-surface-average toroidal current density as `dI/dS` (`nova/transport/current_diffusion.py:1629-1655`), and the accelerator FSA kernel currently returns a fixed set of geometric averages rather than an arbitrary-field average (`nova/equilibrium/flux_surface_connectivity.py:220-249`). | **HOLD — useful pieces exist, but no production forward seam enforces equal core+SOL total current or emits SOL-leg `j_phi` averages.** Condition the two source states against the same `CurrentLedger.total` (without silent rescaling inside the GS solve), then add a topology-qualified average of the solved current density over common-SOL levels; do not relabel the confined transport `j_tor` as the requested SOL profile. |
| Matched-level contour comparison | `benchmarks/solovev_contour_overlay.py:108-184` samples piecewise-linear contour radii and scores both solutions on common rays. It verifies both checkpoints share the exact 21-level `0:0.05:1` grid (`benchmarks/solovev_contour_overlay.py:340-347`), overlays analytic and solved paths with level-matched colours (`benchmarks/solovev_contour_overlay.py:233-270`), and writes both SVG and machine-readable findings (`benchmarks/solovev_contour_overlay.py:443-455`). The banked precedent is `docs/figures/plasma-edge-current-representation/solovev_contour_overlay.svg`. | **FIT — reuse the 21 matched levels, common-ray extraction, per-level radial distances, and SVG/findings pairing.** Substitute confined-control and SOL-current checkpoints and extend levels above 1 far enough to show the measured decay/support extent. |

## Composition boundary

The shortest valid implementation is composition plus three focused additions:

1. Add an Eich-width evaluator and map its outboard-midplane physical width to
   separatrix distance; compare the single width with a declared broader
   spreading length.
2. Generalise the exponential continuation from a mandatory truncation bound to
   material-mask-bounded evaluation, while retaining the existing value-and-first-
   derivative anchor and independent private-flux declaration.
3. Land the four-crossing saddle partition, condition the control and SOL source
   states to identical ledger total, and publish topology-qualified common-SOL
   `j_phi` averages plus exact private-flux-zero and strike-adjacent-current
   assertions.

Everything downstream of those additions can reuse the existing
`ForwardProfile` fixed-point solve, three-vector plasma coupling, domain ledger,
and matched-level contour benchmark. Until all three land, a converged
continued solve is evidence for route viability, not the SOL current
demonstration itself.
