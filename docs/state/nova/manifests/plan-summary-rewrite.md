node: plan-summary-rewrite
status: complete
commits: 31194e40beab12b218e2af245935776894deb285
changed_paths:
  - docs/mast-catalog-gpu-solve.html
  - docs/plans/biot-element-conditioning.html
  - docs/plans/boundary-ring-source-completion.html
  - docs/plans/coefficient-space-newton.html
  - docs/plans/coil-circuit-discovery.html
  - docs/plans/coil-geometry-inversion.html
  - docs/plans/coupling-surface-contract.html
  - docs/plans/current-constrained-forward-solve.html
  - docs/plans/curved-clip-global-surface.html
  - docs/plans/derived-observable-parity.html
  - docs/plans/device-kernel-build.html
  - docs/plans/diagnostic-correction-schema.html
  - docs/plans/diiid-forward-onboarding.html
  - docs/plans/diiid-pulse-replay.html
  - docs/plans/dual-basin-solve.html
  - docs/plans/efit-flux-decomposition.html
  - docs/plans/efit-forward-parity.html
  - docs/plans/efit-map-inversion.html
  - docs/plans/ensemble-forward-seam.html
  - docs/plans/equilibrium-metric-parity.html
  - docs/plans/flux-function-forward-equilibrium.html
  - docs/plans/flux-function-forward-transport.html
  - docs/plans/forward-operator-refinement.html
  - docs/plans/forward-profile-validator-set.html
  - docs/plans/machine-description-retirement.html
  - docs/plans/mast-catalog-geometry-recovery.html
  - docs/plans/mast-corrected-read-layer.html
  - docs/plans/mast-flux-loop-calibration.html
  - docs/plans/mast-misfit-toroidal-harmonics.html
  - docs/plans/mast-p4p5-circuit-provenance.html
  - docs/plans/mast-vacuum-floor-recovery.html
  - docs/plans/moment-conditioned-basin-entry.html
  - docs/plans/moment-map-root-finding.html
  - docs/plans/nova-calibrate-package.html
  - docs/plans/nova-jax-dissolution.html
  - docs/plans/nova-jax-hygiene.html
  - docs/plans/nova-runtime-dependencies.html
  - docs/plans/nova-test-lane-health.html
  - docs/plans/plasma-edge-current-representation.html
  - docs/plans/plateau-input-attribution.html
  - docs/plans/roundoff-scale-acceptance-bounds.html
  - docs/plans/same-device-label-determinism.html
  - docs/plans/scoring-criteria-derivation.html
  - docs/plans/sol-current-demonstration.html
  - docs/plans/solovev-exact-recovery.html
  - docs/plans/spine-boundary-accelerator.html
  - docs/plans/spine-efit-parity.html
  - docs/plans/test-lane-marker-policy.html
  - docs/plans/topology-preserving-continuation.html
  - docs/plans/vacuum-signature-extraction.html
tests: reckon audit-doc over all 50 touched documents — PASS; summary-too-long 50 before / 0 after; all touched files exit cleanly
test_logs:
  - /tmp/plan-summary-rewrite-before-audit.log
  - /tmp/plan-summary-rewrite-after-audit.log
  - /tmp/plan-summary-rewrite-naming-check.log
  - /tmp/plan-summary-rewrite-commit.log
artifacts:
  - 50 summaries rewritten; decoded length range 127–159 characters, maximum 159
  - Finding counts on touched documents: before 0 error / 50 warning / 11 info (61 total); after 0 error / 0 warning / 11 info (11 total)
  - Every file changed by exactly one removed and one added line: only the plan-summary meta content
  - Review table (alphabetical by slug):

    | slug | old chars | new chars | new summary |
    |---|---:|---:|---|
    | biot-element-conditioning | 213 | 148 | Audited every Biot element for cancellation and route accuracy; subdivision remains for plasma-force targets, while the fan route is safe elsewhere. |
    | boundary-ring-source-completion | 161 | 149 | Replaced the boundary-ring stencil seam with one clip-independent degree-nine source; local gates pass, while global analytic gates remain qualified. |
    | coefficient-space-newton | 444 | 152 | Measured 16–36 spline coefficients as sufficient on banked maps and roots; the reported order deficit was withdrawn, and no reduced solver is built yet. |
    | coil-circuit-discovery | 359 | 151 | Flux-space tests showed recovered coefficients are not conductor currents; only the independently measured ohmic circuit relation is current authority. |
    | coil-geometry-inversion | 724 | 159 | Use autodiff-ready Green kernels to recover coil positions and shapes from vacuum measurements, after bringing every kernel behind one traced section contract. |
    | coupling-surface-contract | 395 | 145 | Shipped a 60-field typed coupling contract used by window exchange, time-parallel boundaries, ensemble state and surrogate I/O; 15/15 tests pass. |
    | current-constrained-forward-solve | 251 | 156 | Exact current-amplitude elimination removes the vacuum root on 5/5 frames, but cold-start convergence qualifies only 1/5; reachability remains the obstacle. |
    | curved-clip-global-surface | 390 | 150 | Global-surface clipping passes both STEP gates at 2.349% and 3.885% and matches ITER chord accuracy; the earlier local-curve refutation remains valid. |
    | derived-observable-parity | 294 | 146 | Shared-state tests locate three compiled/eager disagreements in observable computations, not terminal state; their presence is platform-dependent. |
    | device-kernel-build | 295 | 150 | Shipped single-source exact Green builds at 26.58 s on H200 and 58.45 s on CPU, with provenance-keyed caches; compiled geometry gradients remain held. |
    | diagnostic-correction-schema | 598 | 145 | Shipped a versioned, pulse-scoped correction schema and MAST instance covering gains, offsets, scale steps, pair states, exclusions and evidence. |
    | diiid-forward-onboarding | 347 | 141 | Onboard DIII-D from its shipped description with topology-aware solves, response kernels and measured H200 throughput up to 470.561 slices/s. |
    | diiid-pulse-replay | 420 | 147 | Replay a real DIII-D discharge through the coupled forward under recorded coil currents, scoring trajectory drift against labeled frames over time. |
    | dual-basin-solve | 307 | 140 | The topology-pinned portfolio recovers banked roots and batches under jit/vmap; scalar homotopies do not solve cold MAST basin reachability. |
    | efit-flux-decomposition | 244 | 145 | Conductor fields cancel 95.14–98.20% of exterior current; residual current localises mainly in the solenoid, and the P4-lower claim is retracted. |
    | efit-forward-parity | 254 | 143 | ForwardProfile reproduces one MAST reference at DINA-class fidelity but converges on only 1/6; earlier vacuum-collapse findings are superseded. |
    | efit-map-inversion | 292 | 143 | Nova reads native EFIT maps with 6/6 current-sign agreement and axis parity; surviving LCFS disagreement is attributed to the stored reference. |
    | ensemble-forward-seam | 499 | 151 | Shipped typed batched coupled forwards and deterministic observation kernels for Ambix; H200 batch-8 is 134.2 member-windows/s, 21% slower than scalar. |
    | equilibrium-metric-parity | 291 | 149 | Compared derived quantities from identical stored maps: matched profiles agree at median 1e-4 to 1e-5, while absent conventions remain explicit gaps. |
    | flux-function-forward-equilibrium | 216 | 148 | ForwardProfile now provides prescribed-source free-boundary GS solves with explicit current closure; four named DINA-deviation causes were excluded. |
    | flux-function-forward-transport | 269 | 127 | Shipped typed native and TORAX transport forwards plus equilibrium coupling; geometry parity tightened from rtol 0.12 to 1e-10. |
    | forward-operator-refinement | 372 | 133 | Mesh refinement lowers the floor 72.9% but costs 13.6× memory; raised order moves it 7.7% at zero rebuild or memory and is preferred. |
    | forward-profile-validator-set | 184 | 152 | Pin a digest-identified cross-section of independent IMAS equilibrium validators and build a banded reference lane for prescribed-source ForwardProfile. |
    | machine-description-retirement | 11 | 147 | Retire Nova registry, cache and machine-specific IDS wrappers once the Ambix map producer proves equivalent; preserve validated geometry unchanged. |
    | mast-catalog-geometry-recovery | 197 | 143 | Shipped one immutable MAST physical configuration with three representation aliases and 17 explicit evidence gaps; 3,307 integrated tests pass. |
    | mast-catalog-gpu-solve | 283 | 130 | Reconstruct all 1,341,435 equilibrium slices from 11,573 MAST L2 shots in about one hour on 8×H200, with a DIII-D capability demo. |
    | mast-corrected-read-layer | 246 | 151 | All MAST sensor reads now use one correction chain: 290/3,395 banked reads shift, nine bypass doors are closed and 102/102 sensors report dispositions. |
    | mast-flux-loop-calibration | 247 | 140 | Calibrate every MAST flux loop against exact vacuum flux and admit only validated loops; the suspected 2π convention mismatch was ruled out. |
    | mast-misfit-toroidal-harmonics | 256 | 155 | Fit source-free toroidal-harmonic maps to probes and calibrated loops to localise missing conductors; inconsistent EFIT maps are disagreement, not support. |
    | mast-p4p5-circuit-provenance | 363 | 142 | Resolve P4/P5 currents and geometry from engineering provenance; the claimed 17.3 mm P4 asymmetry was an arithmetic artifact and is withdrawn. |
    | mast-vacuum-floor-recovery | 353 | 140 | Measured the MAST sensor floor at 387 µT versus 4.26 mT residual and promoted ten published turn counts; the remaining floor is model error. |
    | moment-conditioned-basin-entry | 509 | 136 | Predicted current moments across 55 MAST and DIII-D rows; only centroid is boundary-insensitive enough to seed or constrain basin entry. |
    | moment-map-root-finding | 341 | 152 | Consistent support quadrature removed the two-path kink seam; both treatment routes were retired, and remaining gates moved to boundary-ring completion. |
    | nova-calibrate-package | 477 | 153 | Shipped nine machine-agnostic calibration capabilities behind arrays-only APIs; synthetic and adapter lanes pass, with 1,159 MAST-specific lines removed. |
    | nova-jax-dissolution | 239 | 139 | Dissolved the technology-organised JAX namespace into domain packages, centralised precision and validated retained routes on CPU and H200. |
    | nova-jax-hygiene | 258 | 131 | Removed five dead modules, centralised 19 x64 configuration sites, rehomed two NumPy modules and fixed three silent parity defects. |
    | nova-runtime-dependencies | 268 | 146 | Declare plain JAX as required by Nova physics while keeping CUDA wheels in a GPU-only extra, so imports work without forcing accelerator packages. |
    | nova-test-lane-health | 185 | 156 | The monolithic lane now completes all 5,842 tests in one process: 5,655 pass, 50 fail visibly, 141 skip and 4 xfail; retained JAX compilation owns the cost. |
    | plasma-edge-current-representation | 459 | 142 | Shipped nine exact fixed-centroid current-moment blocks and conservative separatrix clipping; 5,199 tests pass and analytic roots reach 1e-15. |
    | plateau-input-attribution | 420 | 155 | None of four input substitutions distinguishes the converged frame from five plateaus; conductor omission is withdrawn and historical attribution declined. |
    | roundoff-scale-acceptance-bounds | 1209 | 156 | Replaced invalid zero-reference bounds: divergence_b keeps its absolute envelope; divergence_j uses a failable sqrt(binary64 epsilon) floor, not truncation. |
    | same-device-label-determinism | 309 | 155 | Persistent executable reuse made all 12 repeated states and 69 observables bitwise identical; prior verdict drift was a process/executable boundary effect. |
    | scoring-criteria-derivation | 351 | 139 | Derived leave-one-out mesh bounds re-score MAST 4/6; the circular estimator is excluded, and DIII-D remains without a defensible tolerance. |
    | sol-current-demonstration | 402 | 152 | Two equal-current solves demonstrate finite continuous SOL current to both divertor targets, zero private-flux current and 2.7e-5 m axis/X-point motion. |
    | solovev-exact-recovery | 395 | 145 | Made exact section kernels the production default and corrected polygon routing; 2,852 tests pass, while analytic-recovery attribution continues. |
    | spine-boundary-accelerator | 311 | 146 | Shipped the differentiable LCFS boundary push at 5.82× faster per map, exact accelerated reads and a hex-ring null fit; all 20 behavior pins pass. |
    | spine-efit-parity | 330 | 139 | The assembled MAST spine scored 9,573 of 45,000 slices and failed 10/12 metrics with zero converged fraction; the GPU catalog remains held. |
    | test-lane-marker-policy | 275 | 130 | Named test paths now collect slow tests and empty collections fail: the forward-solve file moved from 0 selected to 28/28 passing. |
    | topology-preserving-continuation | 378 | 151 | Predictor-corrector continuation lengthened traversal but regressed residuals on 4/5 H200 frames; topology held, so the result is a qualified negative. |
    | vacuum-signature-extraction | 594 | 144 | Harvest pre-breakdown and post-pulse vacuum windows into offsets, drift, scale-state flags and gain checks for the diagnostic-correction schema. |
evidence_inputs: Audited all 208 HTML files under docs before editing. Found 50 in-scope violators: 49/52 docs/plans files plus docs/mast-catalog-gpu-solve.html. The targeted audit is fully clean for summary length and introduces no finding; 11 pre-existing informational findings are byte-for-byte unchanged.
follow_ons: Ten additional flat-layout plan summaries are over 160 characters but outside this node fence: biot-operator-assembly, coil-conductor-fidelity, mast-machine-description, norma-data-provenance, nova-3d-native, nova-heritage-recovery, nova-spine-refactor, physics-spine-adoption, polybeam-prism-section, polybow-arc-section.
blockers: none

