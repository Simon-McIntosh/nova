# Why the local model refuses the analytic direction

The historical refusal reproduces, and the same model-trust veto rejects the full analytic direction on both authoritative terminal rows. The large discrepancy is a current-moment prediction error imaged by the fixed coupling. The eighth-norm numerator exposes that error; its denominator moderates it. The coarse local tangent is correct by a small-step finite difference, but a full analytic move crosses many support classifications and has substantial curvature. The fine terminal has an additional unresolved local JVP/finite-difference discrepancy, so tangent correctness cannot be generalized from the coarse witness.

The supported repair for the measured refusal is actual-candidate acceptance when the unchanged sufficient-merit and strict residual-decrease conditions pass, even if the model predicts an increase; record the model as unreliable and rebuild it for subsequent direction selection. The actual merit is already evaluated before the veto. This proposes correcting its use, not adding an evaluation production lacks. A smaller trust radius alone does not address this witness: every sampled smaller analytic fraction increases actual merit. No production code changes are delivered, and no cold-seed convergence is claimed.


## Reproduction and state identities

Reproduction at revision a2c8836289ab399e3ed836f94c904ce111852ea4: the historical 135-cell stalled state has incumbent merit 0.0626216340089. Its half-defect candidate is accepted at actual 0.0621299454943, predicted 0.0624428163974. Its full analytic candidate is refused at actual 9.62674026395e-5, predicted 0.348071530520, a factor 3615.67385. The unmodified production _backtracking_scores supplies both verdicts; assertions compare both recorded merit arrays with the archived witness, and assert the two opposite decisions.

That witness uses the earlier stalled state through the repaired map. It is not the later 135-cell continuous-support terminal. The two primary rows here are the hash-verified rows-titan-adjoint receipts: incumbent merits 0.0399490213834 and 0.0546254885059, sup residuals 0.0451931554866 and 0.0518511705161. The 135-cell row exhausted sixteen outer trips; the 342-cell row settled at six. CPU re-evaluation of their stored states reproduces those residuals.


## Measurement

The instrument runs the unchanged production pinned map and merit/trust functions.
It first replays the historical 135-cell witness from the earlier discontinuous
map's terminal state, then diagnoses the authoritative terminal states of the
repaired continuous-support map separately. No cold solve is rerun or repaired.

Let M(x) be the three unnormalised current vectors returned by
cell_current_moments: cell current, radial coupling coefficient and vertical
coupling coefficient. The physical first moments have already passed through
the fixed atomic second-moment conversion at this seam. Let N(x) be those vectors
after the unchanged declared-current pin, C the fixed linear coupling and e the
fixture exterior. On states with no residual shadows, g(x) = e + C N(x).

The local model is g(x) + t Dg(x)[d]. The instrument compares M and N with their
own JVP predictions cell by cell, images each of the three N prediction-error
vectors separately, and checks that their sum reproduces the entire map-model
error. This makes a coupling remainder an observed quantity. It also separates
curvature in M from the nonlinear scalar current pin by normalising M's linear
prediction before coupling it.

Merit is ||g(y)-y||_8 / ||[g(y), 1e-30]||_8. These are numerator and denominator,
not independent additive penalties. Both factors and their eighth powers are
retained, plus the two counterfactual ratios obtained by replacing one factor at
a time. The actual/predicted decrease ratio and predicted/actual merit ratio are
recorded separately.

For every requested fraction, the unchanged sufficient-decrease slope is 1e-4,
the own-mask sup residual must strictly decrease, and trust requires positive
predicted decrease plus actual decrease at least one tenth of it. Refusals are
reported per rule, even when several fail together. The continuation fallback
uses the same inequalities; its map-defect direction is a separate measured
ladder. Fractions 0.01 and 0.001 are diagnostic probes outside the six-factor
production ladder; no budget or factor list is changed in production.

Newton means the production qualified GMRES(30) direction computed at the named
terminal state, with no preceding condition baseline and no history-dependent
cap. This is a state-local measurement, not a replay of the solver's hidden
globalization carry. The analytic direction is a diagnostic oracle and is not
asserted to be a candidate the production solver can generate.

Curvature is measured by central finite differences of the JVP at two fraction
increments, 1e-4 and 1e-5. Confined/open centroid classifications, nonzero current
supports and residual shadows are compared against the incumbent. Local endpoint
classification flips accompany the curvature measurements. A seeded single-bit
flip and known nonzero supported current verify that absence instruments see
something present before zero counts are interpreted.

The proposed repair must retain the locked cold-seed acceptance on both rows:
relative sup residual at most 1e-12, converged true, analytic axis and every
admitted X-point within one characteristic pitch, and joint qualification.
The pitches are 0.4660634103582032 m (135 cells) and 0.28221541 m (342 cells).
Per-trip poloidal panels must retain the wall and both null sets. The sixteen
outer trips, ten Newton steps and thirty Krylov iterations remain unchanged;
neither sufficient decrease nor the current pin may be relaxed. Reaching the
analytic field's approximately 9.5e-5 one-map residual is not this acceptance.


## Complete fraction ladders

135 cells

| Direction | Fraction | Actual merit | Predicted merit | Pred/actual | Actual/predicted decrease | Confined flips | Refusing rules |
|---|---:|---:|---:|---:|---:|---:|---|
| analytic | 1 | 9.626740264e-05 | 0.2710495257 | 2815.59 | -0.172448 | 49 | model trust |
| analytic | 0.5 | 0.0829338135 | 0.1277443218 | 1.54032 | 0.489602 | 16 | sufficient decrease; model trust |
| analytic | 0.25 | 0.06754149107 | 0.07447165908 | 1.10261 | 0.799257 | 2 | sufficient decrease; model trust |
| analytic | 0.125 | 0.05297400198 | 0.05395133326 | 1.01845 | 0.930202 | 2 | sufficient decrease; model trust |
| analytic | 0.0625 | 0.04555146625 | 0.04569736875 | 1.0032 | 0.974618 | 0 | sufficient decrease; model trust |
| analytic | 0.01 | 0.0405348888 | 0.04053697557 | 1.00005 | 0.996451 | 0 | sufficient decrease; model trust |
| analytic | 0.001 | 0.03999932943 | 0.03999934492 | 1 | 0.999692 | 0 | sufficient decrease; model trust |
| newton | 1 | 0.6017436265 | 1.330302507e-14 | 2.21075e-14 | -14.0628 | 82 | sufficient decrease; model trust |
| newton | 0.5 | 0.7118339591 | 0.04213978304 | 0.0591989 | 306.69 | 49 | sufficient decrease; model trust |
| newton | 0.25 | 0.1199558643 | 0.04440994433 | 0.370219 | 17.935 | 31 | sufficient decrease; model trust |
| newton | 0.125 | 0.05172827562 | 0.04202524915 | 0.812423 | 5.67339 | 14 | sufficient decrease; model trust |
| newton | 0.0625 | 0.04349600509 | 0.04093634931 | 0.941152 | 3.59251 | 2 | sufficient decrease; model trust |
| newton | 0.01 | 0.04052192328 | 0.04010018331 | 0.989592 | 3.78999 | 0 | sufficient decrease; model trust |
| newton | 0.001 | 0.03996453542 | 0.03996402287 | 0.999987 | 1.03417 | 0 | sufficient decrease; model trust |
| map_defect | 1 | 0.06856679745 | 0.06632794396 | 0.967348 | 1.08487 | 6 | sufficient decrease; model trust |
| map_defect | 0.5 | 0.04962369208 | 0.04842253436 | 0.975795 | 1.14175 | 4 | sufficient decrease; model trust |
| map_defect | 0.25 | 0.04255693777 | 0.04225033926 | 0.992796 | 1.13323 | 0 | sufficient decrease; model trust |
| map_defect | 0.125 | 0.04060385745 | 0.04053531188 | 0.998312 | 1.11691 | 0 | sufficient decrease; model trust |
| map_defect | 0.0625 | 0.04011272272 | 0.04009559396 | 0.999573 | 1.11686 | 0 | sufficient decrease; model trust |
| map_defect | 0.01 | 0.03995312137 | 0.03995269193 | 0.999989 | 1.117 | 0 | sufficient decrease; model trust |
| map_defect | 0.001 | 0.03994905516 | 0.03994905088 | 1 | 1.14503 | 0 | sufficient decrease; model trust |

342 cells

| Direction | Fraction | Actual merit | Predicted merit | Pred/actual | Actual/predicted decrease | Confined flips | Refusing rules |
|---|---:|---:|---:|---:|---:|---:|---|
| analytic | 1 | 7.580669184e-05 | 0.291954563 | 3851.3 | -0.229848 | 129 | model trust |
| analytic | 0.5 | 0.09050093564 | 0.1461894099 | 1.61534 | 0.391808 | 44 | sufficient decrease; model trust |
| analytic | 0.25 | 0.08280937585 | 0.09377604063 | 1.13243 | 0.719885 | 17 | sufficient decrease; model trust |
| analytic | 0.125 | 0.07016454707 | 0.07252786392 | 1.03368 | 0.867989 | 10 | sufficient decrease; model trust |
| analytic | 0.0625 | 0.06254629152 | 0.06308318803 | 1.00858 | 0.93652 | 4 | sufficient decrease; model trust |
| analytic | 0.01 | 0.05586407431 | 0.05587710106 | 1.00023 | 0.989592 | 0 | sufficient decrease; model trust |
| analytic | 0.001 | 0.0547479077 | 0.05474803401 | 1 | 0.998969 | 0 | sufficient decrease; model trust |
| newton | 1 | 1.3018572 | 1.627217811e-14 | 1.24992e-14 | -22.8324 | 169 | sufficient decrease; model trust |
| newton | 0.5 | 0.6033506679 | 0.03729121694 | 0.0618069 | -31.6555 | 185 | sufficient decrease; model trust |
| newton | 0.25 | 0.6588080778 | 0.08123486859 | 0.123306 | 22.7056 | 112 | sufficient decrease; model trust |
| newton | 0.125 | 0.1208574071 | 0.06779272058 | 0.560931 | 5.03006 | 81 | sufficient decrease; model trust |
| newton | 0.0625 | 0.05754857353 | 0.06039303021 | 1.04943 | 0.506816 | 35 | sufficient decrease; model trust |
| newton | 0.01 | 0.05503997129 | 0.05545099476 | 1.00747 | 0.502095 | 8 | sufficient decrease; model trust |
| newton | 0.001 | 0.05470098925 | 0.05470652292 | 1.0001 | 0.931712 | 0 | sufficient decrease; model trust |
| map_defect | 1 | 0.06586860587 | 0.06217962617 | 0.943995 | 1.48834 | 9 | sufficient decrease; model trust |
| map_defect | 0.5 | 0.05735221935 | 0.05652491597 | 0.985575 | 1.43555 | 7 | sufficient decrease; model trust |
| map_defect | 0.25 | 0.05536545071 | 0.05517889786 | 0.996631 | 1.3371 | 5 | sufficient decrease; model trust |
| map_defect | 0.125 | 0.05484663737 | 0.05480063334 | 0.999161 | 1.26266 | 2 | sufficient decrease; model trust |
| map_defect | 0.0625 | 0.05469711514 | 0.05468592866 | 0.999795 | 1.18508 | 2 | sufficient decrease; model trust |
| map_defect | 0.01 | 0.05463152209 | 0.0546312796 | 0.999996 | 1.04187 | 1 | sufficient decrease; model trust |
| map_defect | 0.001 | 0.05462600089 | 0.05462599848 | 1 | 1.00473 | 1 | sufficient decrease; model trust |

Fractions 0.01 and 0.001 are diagnostic; production also has 0.03125, which is not in this requested ladder. The map-defect rows test continuation's rule at the stated fractions, not a replay of the hidden recovery radius. Every measured map-defect candidate fails both decrease and trust; no unmeasured fallback exhaustion is asserted.

## Decomposition

For the historical full analytic step, the predicted eighth-norm numerator is 39.9193900 Wb versus actual 0.00740588401 Wb; the denominator is 114.687317 Wb versus 76.9303399 Wb. The map prediction error is 23.8329617 Wb. Its zeroth-current, radial-coefficient and vertical-coefficient contributions have maxima 22.0682811, 1.59475204 and 2.49452272 Wb; their signed sum reconstructs the map error to 9.24e-14 Wb. Per-cell unnormalised and pinned current errors reach 184595.824 A and 395986.101 A. These maxima are not additive: their locations and signs differ.

On the authoritative 135-cell full analytic step, the numerator is predicted 27.9129853 versus actual 0.00740588401 Wb, a factor 3769.03. The denominator is 102.981126 versus 76.9303399 Wb, only 1.33863 times, reducing the merit error to 2815.59 times. On 342 cells those factors are 5326.23 and 1.38297, leaving 3851.30 times. Substituting the actual numerator into the predicted denominator gives 7.19150e-5 and 5.48145e-5; substituting only the actual denominator leaves 0.362835 and 0.403764. The eighth norm is applying the wrong predicted residual, not creating a three-order discrepancy from an accurate flux prediction.

The authoritative current error is dominated by the zeroth-current image: maxima 17.3127758 / 18.8602119 Wb (135 / 342). Radial coefficients contribute 0.752220 / 0.916077 Wb and vertical coefficients 1.528222 / 1.202314 Wb. Their signed sums reproduce the full analytic map error, 16.5949356 / 19.0191689 Wb, to 8.88e-14 / 1.12e-13 Wb. Thus neither a nonlinear coupling nor a changed residual shadow is needed to explain either full analytic refusal.

Normalising the linear raw-current prediction separates two signed contributions: raw-current curvature gives 22.9427581 / 25.4497374 Wb, and scalar-current-pin curvature gives 6.39020497 / 6.46293651 Wb, with partial cancellation. The predicted raw totals are 12.586408 / 12.215064 MA against actual 16.313610 / 16.314423 MA. Both pinned actual and predicted totals remain 16.3147733118 MA: the failure is the distribution, despite the correct total.

The full-step raw-current error maxima are 202707.613 / 122247.309 A; pinned maxima are 417127.383 / 246693.907 A. Coarse cell 15 is actually 169840.381 A but predicted 586967.764 A, and fine cell 255 is actually 61252.1084 A but predicted 307946.015 A. Neither changes its centroid classification: continuous support motion and the common pin affect even cells whose labels do not flip. On coarse cells, 5.06052 MA of the raw L1 error is in cells whose confined label changes, 2.08293 MA in cells whose label does not; on fine cells, 6.27503 and 1.31350 MA respectively. The cellwise vectors and ranked examples are retained in the receipts.

Large Newton moves can additionally change residual shadows. Coarse half-Newton flips eleven shadow entries and leaves a 62.1862 Wb remainder in the unshadowed coupling identity; fine full-Newton and quarter-Newton flip thirteen and five entries, with remainders 88.3279 and 50.7089 Wb. These are not evidence that the fixed coupling became nonlinear: the instrument's plain e + C N identity omits the candidate's copy-through shadow term. They are explicitly branch-changing candidates and already fail actual sufficient decrease. The full analytic candidates have zero shadow flips and close that identity at rounding precision.


## Curvature and branch mechanisms

The historical analytic direction flips 63 confined/open centroid classifications. The authoritative full analytic directions flip 49 and 129; their nonzero-current supports change in 47 and 130 cells. The branch at the candidate therefore differs materially from that at the linearization point. This establishes extrapolation across branch changes, not a stale or incorrectly selected base branch.

For the historical witness, the analytic JVP agrees with a central finite difference at fraction increment 1e-6 to relative error 1.6794e-9. Its second directional derivative, measured from JVP differences, is 34.995898 Wb per squared fraction at the origin and 113.494946 at the full analytic endpoint; increments 1e-4 and 1e-5 agree in the shown digits. At the authoritative coarse terminal, the analytic and Newton JVP errors are 1.13064e-9 and 2.11781e-10. Along the coarse analytic ladder, second directional maxima at fractions 0.001, 0.0625, 0.5 and 1 are 24.89098, 31.29904, 59.35442 and 71.91993 Wb per squared fraction. Each local curvature pair has zero confined-label flips, and its two increments agree closely. Thus smooth within-branch curvature is present as well as the many full-path label changes; a label-flip-only explanation is insufficient.

The fine terminal does NOT pass that local tangent check. Central finite-difference discrepancies at increment 1e-6 are 0.999565 (analytic), 0.995716 (Newton) and 1.000092 (map defect). They are measured failures, not rounded passes. At fine analytic fractions 0.001 and 0.5, the JVP-difference curvature is stable at 59.0121 and 91.1154 Wb per squared fraction. At the full analytic endpoint it changes from 331.316 to 290.285 when the increment shrinks from 1e-4 to 1e-5, despite zero local centroid-label flips. Fine Newton at fraction 0.125 changes from 73440.1 to 1885.06. Those estimates are not converged second derivatives. The origin finite-difference endpoint states were not retained, so the receipt cannot attribute its mismatch to a particular root, topology read or subcell quadrature transition. A branch-resolved tangent audit is a required follow-on; a corrected-linearization repair on this row is not ruled out.

The classification detector sees a deliberately flipped bit (count one) and each current detector sees nonzero support. The same detector records one local confined flip in each fine Newton curvature pair at fraction 0.0625. This is a physical positive control for the zero local counts elsewhere. The moment-image JVP identities agree to 3.55e-15 / 2.84e-14 Wb or better on both rows; that verifies the decomposition's arithmetic, not the differentiability of the nonlinear map.


## Supported repair and locked acceptance

Implement actual-candidate acceptance for a pessimistic model: keep finite checks, merit <= incumbent_merit * (1 - 1e-4 * fraction), and strict own-mask sup-residual decrease. If those actual tests pass and predicted decrease is nonpositive, promote the measured candidate while marking the model unreliable and rebuilding it for future proposals. When predicted decrease is positive, the existing realized-versus-predicted decrease check can remain. Do not replace the actual merit with the predicted merit, relax the sufficient-decrease slope, alter the current pin or increase the budgets.

The direct falsifiable acceptance for that refusal repair is that the historical full analytic witness and both authoritative full analytic candidates pass the candidate selector, while every actually worsening Newton candidate in this measured ladder remains refused. The historical half-defect acceptance must remain. Curvature-aware trust radii may prevent a large bad Newton proposal, but shrinking the sampled analytic direction rejects every tested fraction below one and cannot alone recover this witness.

This is not proof of a complete convergence repair. All seven sampled Newton fractions increase actual merit on both terminals. The production qualified Newton solves are well resolved (achieved linear-residual ratios 2.30e-13 and 1.93e-13) but their full nonlinear merits are 0.601744 and 1.301857, against model predictions 1.33e-14 and 1.63e-14. The analytic direction is an oracle diagnostic, not a production proposal. Direction generation and the fine-terminal derivative discrepancy remain separate work. The analytic candidate itself has sup residual 9.53950e-5 / 9.59627e-5, still above the locked 1e-12 bound, and is labelled converged=false in the panel.

Cold-seed acceptance remains locked on BOTH rows: converged=true, sup residual <= 1e-12, analytic axis and every admitted X-point inside one characteristic pitch (0.4660634103582032 / 0.28221540988057975 m), and the joint null qualification. Retain per-trip shared-level contours, both null sets and the wall; neither row may be dropped. Keep sixteen outer trips, ten Newton steps, thirty Krylov iterations, sufficient decrease and the current pin unchanged. The separately dispatched merged-head verification owns that acceptance. This investigation leaves the driving followup and all plan status fields unchanged.


## Execution and artifacts

One all_debug allocation, job 1275504: COMPLETED, scheduler exit 0:0, elapsed 00:03:46, eight CPUs, 96 GiB requested, batch peak RSS 2133872K. Both row processes return zero and the allocation prints GATE_EXIT=0. Their measured process walls including figures are 95.75 and 76.85 seconds. TMPDIR=/tmp is set in submit and payload; JAX_PLATFORMS=cpu and x64 are asserted; the root interpreter is used directly with no uv on the node. Both logs begin with the exact revision, worktree and command. No additional allocation, cold solve or suite run was performed.

The numerical gate completing is not a claim that the fine JVP check passes: its failed values are part of the result. The complete raw receipts remain in the assigned reports directory; published compact receipts point to those paths and carry SHA-256 digests. The PNG and SVG files exist both in that report directory and under the project figures path. All three PNGs were visually inspected. The poloidal panel uses nova.media painters, a single declared contour-level array, the actual fixture wall, both axis-marker sets, and no axes; these limited reference states have no admitted saddle.

Commits: a2c8836289ab399e3ed836f94c904ce111852ea4 (measurement); 492b895770bd10b7e14ec379b4f9d89fc428977f (coarse evidence); 4303ed756cf6cd7beab7729b27d1cf5cdcbfdb98 (fine evidence). The final landing commit is recorded in the manifest after it exists. Nothing under nova/ was changed.

