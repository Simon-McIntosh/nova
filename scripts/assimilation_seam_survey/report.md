# Ambix assimilation-seam survey

Surveyed repository: `/home/ITER/mcintos/Code/imas-ambix` at tracked commit
`2f3dcd0c87be908f3f7912c50168c6fc0dae2b79` on 2026-08-22. The checkout had
many user-owned untracked artifacts but no tracked modifications, so every
source and documentation reference below resolves to that commit. The four
questions come from nova's live `diiid-forward-onboarding` followup
`f-dfo-seam-survey` (`docs/plans/diiid-forward-onboarding.html:506-509`).

Verdicts in this report mean:

- **TESTED** — executable focused tests or a banked end-to-end run exercise the
  claimed capability.
- **UNTESTED** — implementation exists, but the surveyed tree contains no
  direct executable or banked evidence for the claimed composition.
- **ABSENT** — no implementation was found after path inventory and repository
  searches over `imas_ambix`, `tests`, and `docs`; nearby but semantically
  different machinery is identified explicitly.

## Executive boundary

Nova should author the deterministic coupled forward, deterministic diagnostic
response kernels, convergence receipts, and forward-only twin truth generator.
It should not author an assimilation algorithm, posterior state container,
signal uncertainty/validity policy, innovation classifier, or estimator-product
semantics. Ambix already owns and tests the per-member flux-function state,
causal/non-causal handoffs, ensemble identity, conditioning layer, and four
Thomson measurement-side operators. The one important deterministic hole is a
composite Thomson synthesizer: no Ambix function currently maps a solved flux
map plus `Te(psi_N)`/`ne(psi_N)` and chord geometry to predicted Thomson
signals.

The governing split is explicit: “Nova produces every deterministic response
kernel and fits with none of them,” while “everything statistical about a
signal stays in Ambix”; the document's litmus test assigns differentiable
geometry-bearing physics to Nova and anything carrying sigma, a prior, or a
decision to Ambix (`imas-ambix/docs/plans/flux-function-state-interface.html:70-81`).

## 1. `imas_ambix/thomson`: four operators and the missing forward

The package exports exactly four operator-like public classes alongside their
value types: `PedestalFootDetector`, `ChannelValidityPolicy`, `IsofluxPairer`,
and `IsothermAsymmetryOperator`
(`imas-ambix/imas_ambix/thomson/__init__.py:15-43`). Its package contract also
states that chord-level `psi_N` calibration is derivative-level label use and
that the asymmetry operator has no fitted parameters
(`imas-ambix/imas_ambix/thomson/__init__.py:1-8`).

| Capability | Physics and signature | Exact Nova consumption | Verdict |
|---|---|---|---|
| Pedestal-foot separatrix detector | `PedestalFootDetector.locate(coordinate_m, temperature_ev, *, topology, sigma_multiplier=None) -> SeparatrixEstimate`. It converts measured temperature along one ordered chord family to calibrated `psi_N`, then interpolates the `psi_N=1` crossing and propagates calibration sigma to coordinate sigma (`imas-ambix/imas_ambix/thomson/calibration.py:144-204`). Its `PedestalCalibration` is a fitted, topology-specific piecewise-linear map from log temperature ratio to `psi_N` (`imas-ambix/imas_ambix/thomson/calibration.py:21-92`). | **No Nova call or protocol.** Training samples get `psi_N` from the local challenge-bank bilinear sampler, not from a Nova import (`imas-ambix/imas_ambix/thomson/bank.py:28-80`, `:133-182`). A production use may consume a Nova chord-to-`psi_N` result as an input array, but that adapter is not implemented here. | **TESTED.** Synthetic separatrix recovery is explicit (`imas-ambix/tests/thomson/test_models.py:18-43`); the banked calibration used 94,846 train and 30,150 held-out samples with 0.6222 one-sigma coverage inside the registered band (`imas-ambix/docs/plans/magnetics-free-equilibrium-recovery.html:81-96`). |
| Channel validity and uncertainty policy | `ChannelValidityPolicy.assess(temperature_ev, density_m3, *, topology, elm_phase=QUIESCENT, saturation_temperature_ev=5e4) -> ChannelAssessment`. It retains every channel and multiplies sigma by topology, ELM phase, and health factors (`imas-ambix/imas_ambix/thomson/models.py:47-101`). | **None.** It consumes measured scalar `Te`, `ne`, topology and phase; these are statistical/quality inputs on the Ambix side, not Nova interfaces. | **TESTED.** Valid, ELM-affected and non-finite cases retain the channel and yield multipliers 1, 5, and 36 (`imas-ambix/tests/thomson/test_models.py:46-71`). |
| Non-collinear isoflux pairing | `IsofluxPairer.pair(first_temperature_ev, second_temperature_ev, first_psi_n, second_psi_n, *, first_direction_rz, second_direction_rz, first_sigma=None, second_sigma=None) -> tuple[IsofluxPair, ...]`. It greedily matches equal-log-temperature samples on two non-collinear chord families and inverse-variance combines their `psi_N` values (`imas-ambix/imas_ambix/thomson/models.py:123-206`). | **Array-level only.** A caller must already supply both chord families' `psi_N` values and R-Z directions. Those `psi_N` arrays are the natural output of Nova's deterministic chord sampler, but no Nova symbol is imported or called. | **TESTED.** Four known pairs recover `psi_N` to `1e-12`, and collinear geometry fails loudly (`imas-ambix/tests/thomson/test_models.py:74-106`). |
| Isotherm-asymmetry/Shafranov-shift moment | `IsothermAsymmetryOperator.measure(inboard_radius_m, outboard_radius_m, *, reference_major_radius_m, minor_radius_m) -> float` implements `delta_R = a^2/R_ref * (beta_p + li/2)`; `synthesize_radii(...)` is its analytic inverse (`imas-ambix/imas_ambix/thomson/models.py:209-245`). | **No Nova call.** It consumes already identified inboard/outboard isotherm radii and scalar reference geometry. Nova could deterministically supply surface geometry or moment receipts, but no such interface appears in this operator. | **TESTED, scientifically qualified.** The analytic round trip is tested to `1e-12` (`imas-ambix/tests/thomson/test_models.py:109-125`) and a bank-conditioned test checks banked moments (`imas-ambix/tests/thomson/test_banked_evidence.py:106-131`). The later loop ablation found genuine direct `FF'` sensitivity but no update traction under one-sided chord coverage, so it is not evidence of a useful production constraint (`imas-ambix/docs/plans/magnetics-free-equilibrium-recovery.html:316`). |

### Does the requested predicted-Thomson operator already exist?

**ABSENT.** There is no function that accepts `(flux map, Te flux function,
ne flux function, chord geometry)` and returns synthetic temperature/density
signals. The existing package runs the inverse/measurement side: it consumes
measured temperatures, measured densities, pre-sampled `psi_N`, chord
directions, or already identified isotherm radii. The only map sampler in the
package is the private challenge-label `_sample_psi_n`, which bilinearly samples
`shot.labels.psirz` and normalizes with banked axis/boundary flux
(`imas-ambix/imas_ambix/thomson/bank.py:133-182`). It neither consumes a Nova
solve nor samples `Te(psi_N)`/`ne(psi_N)`.

This absence matches the locked seam rather than contradicting it: deterministic
chord-to-flux sampling belongs in Nova, while the isoflux claim and confidence
belong in Ambix (`imas-ambix/docs/plans/flux-function-state-interface.html:70-81`).
The Nova plan may therefore add a pure deterministic synthesis kernel and its
geometry/units receipts. Ambix should own the likelihood, calibration, sigma
policy, innovation, and conditioning wrapper around that kernel.

### Where do `Te` and `ne` flux functions live?

The requested pair is **ABSENT as a typed Ambix state representation**.
`FluxFunctionState` carries pressure, `F`, their physical derivatives, optional
domain profiles, and optional flow, but no electron-density field and no
electron-temperature field (`imas-ambix/imas_ambix/fluxstate/contract.py:658-680`).
The nearest temperature array is
`IsothermalToroidalFlow.temperature_ev`, whose semantics are a temperature
constant on each surface for a toroidal-rotation closure, not an explicitly
typed electron-temperature diagnostic profile; it has no paired density
(`imas-ambix/imas_ambix/fluxstate/contract.py:225-250`). The Nova handoff merely
passes this optional flow temperature through
(`imas-ambix/imas_ambix/fluxstate/consumer_contract.py:56-75`, `:156-174`).

A separate MAST baseline constructs transient `te_t` and `ne_t` dictionaries
directly from measured Thomson profiles for TORAX
(`imas-ambix/imas_ambix/statespace/enkf_baseline.py:353-447`), but those are
baseline-specific drive dictionaries, not reusable `FluxFunctionState` fields
and not a Thomson forward representation. Nova should accept typed array/profile
inputs at its deterministic kernel boundary; deciding the durable inferred
`Te`/`ne` state schema remains Ambix assimilation authority.

## 2. Classical current-recovery ensemble-Kalman baseline

### Premise correction

There is no recorded learned-filter win over this baseline in the MSE-free
current-recovery study. The plan remains `active`
(`imas-ambix/docs/mse-free-current-recovery-v0.html:8-17`), and its final oracle
decision is a **qualified negative / no-go**: the near-axis interior stayed
model-failure-dominated, the bolometer gain was small and off-interior, SXR was
not significant, and the neural training/H200 run was deferred
(`imas-ambix/docs/mse-free-current-recovery-v0.html:394-395`). Consequently,
“baseline that the learned filter beat” is **ABSENT/refuted**, not an achieved
comparison.

### The implemented v0 comparator

- **Module:** `imas_ambix/statespace/enkf_baseline.py`. It identifies itself as
  a parameter-space ensemble smoother over TORAX, not a sequential state EnKF:
  each member runs one full trajectory, then one ensemble-Kalman-inversion
  update changes uncertain parameters and reruns the trajectory
  (`imas-ambix/imas_ambix/statespace/enkf_baseline.py:1-9`, `:38-50`).
- **Assimilated state vector:** four parameters,
  `[log(Zeff), log(resistivity_multiplier), current_peaking, Ip_fraction]`
  (`imas-ambix/imas_ambix/statespace/enkf_baseline.py:633-670`). The forward
  trajectory contains `j_total(rho,t)` and `q(rho,t)`, but those profiles are
  not directly updated (`imas-ambix/imas_ambix/statespace/enkf_baseline.py:513-551`).
- **Transition:** TORAX current diffusion driven by measured `Ip(t)`, Thomson
  `Te(rho,t)` and `ne(rho,t)`, with frozen heat/density evolution and uncertain
  `Zeff`, effective resistivity, initial-current peaking and `Ip` scale
  (`imas-ambix/imas_ambix/statespace/enkf_baseline.py:453-510`).
- **Observation operator:** `MagneticsObs` embeds TORAX `j(rho)` into the GS
  operator's plasma nodes using `c_plasma = j * cell_area`, adds known PF
  currents, calls `ForwardOperator.predict`, and selects trustworthy `amb`
  rows (`imas-ambix/imas_ambix/statespace/enkf_baseline.py:572-627`).
- **Analysis:** one global, perturbed-observation EKI update over up to five
  assimilation slices. It uses sample `C_theta,y`, sample `C_y,y`, whitened
  sensor residuals, and reruns TORAX after updating all members
  (`imas-ambix/imas_ambix/statespace/enkf_baseline.py:697-797`).
- **Measured ensemble/config:** 32 members, five assimilation slices,
  `eki_inflation=1.0`, full step `eki_step=1.0`, on 112 held-out shots
  (`imas-ambix/imas_ambix/statespace/artifacts/enkf_baseline_metrics_v0.json:1-30`).
- **Inflation:** the named v0 knob is observation-error inflation in whitened
  space, fixed to 1.0; it is not multiplicative ensemble-spread inflation
  (`imas-ambix/imas_ambix/statespace/enkf_baseline.py:145-150`, `:782-793`).
- **Localization:** **ABSENT in v0.** `EnKFConfig` has no localization field and
  the EKI update uses the full four-parameter cross covariance
  (`imas-ambix/imas_ambix/statespace/enkf_baseline.py:122-159`, `:762-795`).
- **Evidence verdict:** **TESTED as a banked study, not as a generic unit.** The
  112-shot artifact reports a real innovation drop, analysis pitch close to
  forecast pitch, and a persistence win, while attributing recovery primarily
  to TORAX plus `sigma(Te)` rather than the magnetics update
  (`imas-ambix/imas_ambix/statespace/artifacts/enkf_baseline_metrics_v0.json:1419-1420`).
  There is no dedicated `test_enkf_baseline.py`; the strongest evidence is the
  banked run.

### Reusability

The v0 comparator is **bespoke, not generic**. It hard-codes MAST nominal
geometry, MAST signal families, a TORAX configuration, the Ambix GS operator,
the held-out-MSE time/channel contract, and MSE pitch/q readouts
(`imas-ambix/imas_ambix/statespace/enkf_baseline.py:103-189`, `:205-371`). Its
local covariance algebra is mathematically reusable, but it is not exposed as
a state/observation protocol or general filter class. Reusing it for a new
state/observation pair would require extracting the update kernel and replacing
the loader, transition, observation adapter, and readout.

The successor strong baseline is a distinct implementation,
`imas_ambix/statespace/sequential_da.py`: it filters full `psi(rho)` around a
once-per-shot TORAX prior and localizes correction to leading observable modes
(`imas-ambix/imas_ambix/statespace/sequential_da.py:1-22`). Its defaults are
rank 6, correction inflation 1.02, observation inflation 1.0, and 32 posterior
samples (`imas-ambix/imas_ambix/statespace/sequential_da.py:60-78`). The locked
decisions explicitly retire the v0 parameter smoother, choose `psi(rho)`, and
make 5-6-mode localization mandatory
(`imas-ambix/docs/sequential-current-da-v1.html:218-241`). This successor is
also MAST/current-recovery specific; it is evidence and prior art for Ambix, not
a generic Nova assimilation substrate. Verdict: **TESTED** for its focused
substrate — profile transforms, direct observation linearisation, residual
reduction and conformal scaling are executable tests
(`imas-ambix/tests/statespace/test_sequential_da.py:24-94`).

## 3. Locked assimilation authority and innovation classification

### Structured locked decisions

The following decision IDs and operative choices are present in Ambix plan
state:

1. **`ownership-boundary = nova-owns-physics`** — “Nova owns physics; Ambix
   consumes it.” The rationale assigns physics-spine, equilibrium, transport,
   circuits/passives, IMAS and GPU physics to Nova, with Ambix primarily the
   label/ML consumer (`imas-ambix/docs/nova-physics-consumer-integration.html:153-159`).
2. **`transition-authority = nova-led-ensemble`** — “Nova advances every
   reference ensemble member; learning supplies uncertain closures and
   observations” (`imas-ambix/docs/physics-world-model-strategy.html:274-280`).
3. **`estimator-products = forecast-analysis-smoothing`** — “Causal forecast,
   causal analysis/reconstruction and explicitly non-causal smoothing”
   (`imas-ambix/docs/physics-world-model-strategy.html:282-288`).
4. **`normalisation-owner = absolute-sources`** — Ambix emits authoritative
   absolute sources by default; target-current normalization is only an
   explicit, declared closure, never implicit consumer scaling
   (`imas-ambix/docs/plans/flux-function-state-interface.html:109-115`).
5. **`moment-constraint-authority = ambix-profile-conditioning`** — moment
   matching changes profile degrees of freedom and belongs to Ambix inference;
   Nova supplies the deterministic differentiable moment map
   (`imas-ambix/docs/plans/flux-function-state-interface.html:117-124`).
6. **`uncertainty-handoff = weighted-ensemble`** — weighted members are the
   authoritative payload because member-specific, non-Gaussian topology changes
   must not be collapsed to a mean/covariance pair
   (`imas-ambix/docs/plans/flux-function-state-interface.html:134-141`).
7. **`derivative-level-firewall = derivative-level-consumption`** — label maps
   may provide extracted typed derivatives/flux functions and chord `psi_N` for
   calibration, while map-level contact is quarantined
   (`imas-ambix/docs/plans/magnetics-free-equilibrium-recovery.html:194-205`).

These boundaries are **TESTED where they have executable seam semantics**:
`FluxFunctionState` and its causal handoffs passed 9/9 tests; the producer
adapters passed 14/14; and the pinned cross-repo consumer contract passed six
focused plus 20 package tests
(`imas-ambix/docs/evidence/archive/flux-function-state-interface-landed.html:18-56`).
The consumer payload freezes every source/profile and carries targets only as
requests, never renormalizing the supplied state
(`imas-ambix/imas_ambix/fluxstate/consumer_contract.py:126-174`).

### Innovation classification: rule present, typed classifier absent

No structured decision with an `innovation-*` or `assimilation-*` decision ID
exists, and no `InnovationClass`/typed-error classifier implementation was
found. The operative rule is nevertheless explicit in the informing research:

> “Ambix must classify these alternatives before calling any residual
> ‘discovered physics.’”

The alternatives named are missing closure/transferable coefficient versus
calibration drift, bad channel, geometry error, numerical approximation, or
out-of-support regime
(`imas-ambix/docs/research/hybrid-generative-state-estimation.html:29-32`).
The DIII-D smoother plan similarly requires innovations to be attributed across
coil-current nuisances, position/turns corrections, passive currents and
flux-function deviations, with only converged members admitted
(`imas-ambix/docs/plans/magnetics-free-equilibrium-recovery.html:129-139`).

Verdict: **ABSENT as executable classification machinery; present as binding
prose.** Nova should emit typed numerical/convergence/geometry receipts that
make classification possible, but should not implement the classifier or turn
innovation into profile correction. A future Ambix plan should lock a
structured decision ID and implement the typed classifier if that distinction
must be machine-enforced.

## 4. Ambix overlap with coupled transport-equilibrium windows and waveforms

| Ambix asset | What exists | Overlap with Nova coupled-window work | Verdict |
|---|---|---|---|
| `FluxFunctionState` | Immutable per-member state with radial coordinate/Jacobian, pressure and `F` primitives/derivatives, COCOS/sign provenance, domain profiles, weighted-ensemble identity, temporal handoff and moment ledger (`imas-ambix/imas_ambix/fluxstate/contract.py:658-710`). | This is the state **at** the seam and should be consumed, not recreated, by Nova-side coupled-forward adapters. It is per-slice, not a time-dependent coupled-window state. | **TESTED** — 9/9 contract tests and lossless array-tree round trip (`imas-ambix/docs/evidence/archive/flux-function-state-interface-landed.html:18-33`). |
| `TransportForwardAdapter` | Accepts a time grid, requested-current waveform, source state, causal handoff, optional learned correction, and a Nova-like `CurrentDiffusion.evolve`; it returns the final `FluxFunctionState` plus a transport receipt (`imas-ambix/imas_ambix/fluxstate/adapters.py:78-97`, `:126-175`, `:200-269`). | It already defines Ambix's thin consumer/producer edge for deterministic Nova transport. It does **not** iterate transport with a free-boundary equilibrium or preserve a whole evolving coordinate trajectory. Nova should keep the deterministic coupled solve; Ambix should adapt its output here. | **TESTED** — included in the 14/14 adapter gate (`imas-ambix/docs/evidence/archive/flux-function-state-interface-landed.html:35-51`). |
| Temporal and ensemble handoffs | `ForecastHandoff`, `AnalysisHandoff`, `FixedLagSmoothing`, and `FullSequenceSmoothing` have distinct causal semantics; smoothing is rejected at online handoff (`imas-ambix/imas_ambix/fluxstate/contract.py:371-456`). `SequentialEstimatorBatch` preserves every weighted member and exact temporal identity (`imas-ambix/imas_ambix/fluxstate/adapters.py:330-428`). | Nova receipts should carry timestamps/member IDs through unchanged. Nova must not define forecast/analysis/smoothing policy or collapse the ensemble. | **TESTED** — causal rejection and complete three-member identity are in the adapter evidence (`imas-ambix/docs/evidence/archive/flux-function-state-interface-landed.html:35-51`). |
| `NovaEnsembleEstimator` smoke | Ambix advances each member through Nova current diffusion, conditions observable trajectories, and emits causal forecast/analysis plus fixed/full smoothing (`imas-ambix/imas_ambix/statespace/nova_ensemble_estimator.py:488-669`). | This is already an Ambix-owned assimilation/twin harness. It must not be rebuilt as production assimilation in Nova. It is only a synthetic 1-D radial current-diffusion smoke: even its plan says the “equilibrium” product is a radial flux history, not a 2-D Grad-Shafranov map (`imas-ambix/docs/plans/hybrid-state-estimation-smoke.html:38-40`). | **TESTED, qualified** — 46 tests plus CPU and 1/5/8-H200 receipts, but the analysis lost to the same-cohort conventional EnKF and the camera output is a proxy (`imas-ambix/docs/plans/hybrid-state-estimation-smoke.html:38-44`). |
| Learned patch transport prior | `patch_transport.transport_prior_terms` maps patch-current flux at two adjacent slices into a learned `FluxDiffusionPrior` with dissipation, volt-second and positive-diffusivity terms (`imas-ambix/imas_ambix/latent/patch_transport.py:1-23`, `:58-88`). | It overlaps conceptually with temporal coupling, but it is a learned **soft prior/regularizer**, not a deterministic transport-equilibrium window solver. Under the ownership decisions it stays Ambix-side as closure/conditioning research; it must not become a second production physics forward. | **TESTED** — focused tests exercise finite terms, positive diffusivity and firewall imports (`imas-ambix/tests/latent/test_patch_transport.py:80-123`). |
| Flux-function waveform + coupled-window receipt | The shipped plan prose specifies time-varying coordinate maps, interpolation policy and convergence receipts containing iterations, contraction estimate, exit residual and damping (`imas-ambix/docs/plans/flux-function-state-interface.html:58-66`). | This is precisely the adapter contract Nova's coupled-window machinery should satisfy. | **ABSENT in Ambix code.** Searches of `imas_ambix/fluxstate` find no waveform/window/convergence type beyond current-waveform validation in the transport adapter. The plan's acceptance list asks for a waveform fixture (`imas-ambix/docs/plans/flux-function-state-interface.html:83-92`), but the landed evidence enumerates only static/SOL solves, ledgers, no-renormalization and eager/jitted/batched parity (`imas-ambix/docs/evidence/archive/flux-function-state-interface-landed.html:53-80`). |
| DIII-D windowed iterated ensemble smoother | The plan specifies approximately 12-16 inferred parameters per window, typed error attribution, and converged-member admission (`imas-ambix/docs/plans/magnetics-free-equilibrium-recovery.html:129-139`). | This is the eventual Ambix consumer of Nova coupled-window forwards and receipts. | **ABSENT/UNTESTED as implementation.** No matching production smoother or coupled-window code was found in `imas_ambix`; this remains plan prose. |

### Concrete scope for Nova's next plans

1. **Ensemble coupled forward:** build the deterministic, batched member/window
   evaluator in Nova. Accept Ambix member/source payloads, actuator waveforms,
   and geometry; return member-preserving fields, transport state, topology,
   moment/conservation ledgers, and convergence receipts. Do not implement the
   Kalman/ensemble update, smoothing, weighting, innovation attribution, or
   learned closure correction.
2. **Observation-operator receipts:** add pure geometry-bearing deterministic
   kernels, including the missing Thomson synthesizer, with units, COCOS,
   interpolation/support, and numerical-error receipts. Reuse Ambix's four
   measurement models for calibration, sigma, validity, isoflux confidence and
   asymmetry inference.
3. **Thomson-informed forward constraint:** expose a differentiable Thomson
   observation/moment map and Jacobian from Nova, but apply any likelihood,
   prior, profile correction, nuisance fit, or posterior update in Ambix. The
   missing typed `Te`/`ne` state schema is an Ambix-side contract decision; Nova
   can initially accept explicit typed arrays without claiming state authority.
4. **Twin-experiment feasibility:** Nova may generate deterministic truth and
   counterfactual forward receipts. The filter/twin assimilation experiment
   should call that forward from Ambix or remain explicitly a Nova forward-only
   benchmark; do not create a second production estimator beside
   `NovaEnsembleEstimator` or `sequential_da`.

## Capability verdict count

Across 20 capabilities explicitly surveyed above:

- **TESTED:** 12 — four Thomson operators; v0 TORAX/EKI study; typed state;
  transport adapter; temporal/ensemble handoffs; Nova ensemble smoke; learned
  patch transport prior; seam decision enforcement; successor sequential DA.
- **UNTESTED:** 0 — no surveyed implementation lacked all executable or banked
  evidence; qualified scientific losses remain qualified under **TESTED**.
- **ABSENT:** 8 — composite Thomson signal synthesis; typed electron `Te/ne`
  flux-function pair; v0 localization; a reusable generic v0 filter interface;
  a learned-filter win in the named study; executable innovation classification;
  flux-function waveform/coupled-window receipt types; and the DIII-D windowed
  smoother implementation.

The quantitative done-when is therefore met: all four numbered questions are
answered with file:line evidence; seven structured decision IDs are quoted;
and 20 capability rows/findings carry explicit TESTED, UNTESTED, or ABSENT
verdicts.
