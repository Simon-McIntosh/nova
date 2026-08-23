# Derived-observable parity reuse map

## Outcome

The three failing observables already share one production evaluation spine.
`ForwardProfile.solve` is the per-slice entry and `ForwardProfile.solve_batch`
is the leading-axis entry, but both call `_solve_accelerated`, which always
builds the published labels through `ForwardProfile._receipt`. There is no
second batched definition of volume, major radius, or current divergence to
reconcile.

The shortest discriminator is therefore:

1. obtain one terminal flux state and hold the profile, current inputs,
   target current, dtype, and solver budget fixed;
2. evaluate that state once with `ForwardProfile.observe` and once through a
   jitted `vmap` of `ForwardProfile.observe` over a leading axis;
3. flatten the two `ForwardEquilibrium` trees with the existing parity helpers
   and apply the committed per-observable registrations;
4. if a leaf differs, retain intermediate values from the shared receipt spine
   to name the first differing operation.

This separates label computation from terminal-state drift without creating a
new physical definition. The current six-case receipt cannot answer the shared-
state question retrospectively: it retained differences and pass counts, not
the two raw label values or either terminal flux state.

## Quantitative inventory

- Observables covered: **3 of 3**.
- For every observable below: registration site, bound derivation, scalar
  route, batched route, shared-state entry, and localisation seam are named.
- Existing failing evaluations: **7 of 414**, leaving the banked **407 of 414**
  passes and **66 of 69** observables passing across the cohort.
- Explicit empty-search results: **7 no-candidate findings**.

## Shared route and harness map

| Capability | Existing site | Fitness verdict |
|---|---|---|
| Per-slice production route | `ForwardProfile.solve`, `nova/equilibrium/forward.py:1147-1208` | **REUSE.** The accelerated routes call `_solve_accelerated`; its returned `ForwardEquilibrium` includes all three target leaves. Host routes are a separate solver family and should not be mixed into a batch-layout discriminator. |
| Batched production route | `ForwardProfile.solve_batch`, `nova/equilibrium/forward.py:1418-1449` | **REUSE WITH DOMAIN LIMIT.** It is exactly a `jax.vmap` of `_solve_accelerated`, so it publishes the same full receipt. It batches seeds, conductor currents, and target currents for one static `ForwardProfile`; it is not a ready six-case batch of six independently built profiles. |
| Common receipt | `ForwardProfile._receipt`, `nova/equilibrium/forward.py:970-1010` | **REUSE AS THE AUTHORITY.** It evaluates moments and conservation once and constructs the named `ForwardEquilibrium` tree consumed by both solve routes. |
| Shared-state full-label entry | `ForwardProfile.observe`, `nova/equilibrium/forward.py:700-718` | **BEST FIT.** It accepts an already chosen flux state, performs no iteration, and returns the same `_receipt` contract as a solve. Compose `jax.jit(jax.vmap(...))` around it for the leading-axis arm. |
| Shared-state moment-only entry | `ForwardProfile.integral_observation`, `nova/equilibrium/forward.py:720-730` | **PARTIAL FIT.** It is a compact differentiable entry for `moments.volume` and `moments.major_radius`, but it cannot emit `conservation.divergence_j`; use the full `observe` entry for one three-observable discriminator. |
| Result-tree naming and flattening | `_named_tree` and `_leaves`, `benchmarks/jitted_eager_parity_gate.py:130-157` | **REUSE.** These preserve paths such as `moments.volume` and reject silent structural omissions when paired with the existing key check. They are benchmark-private helpers, so reuse in the discriminator is preferable to creating a second path vocabulary, but promotion can wait until there is a second production consumer. |
| Bound scoring | `_finite_difference`, `_bound_ratio`, and `_score_observable`, `benchmarks/conditioned_convergence_observables.py:237-304` | **REUSE.** This is already the exact-equality/dual-envelope scorer used for the banked 69-observable reapplication. It needs one improvement in a later implementation: exact-equality failures currently report no finite utilisation ratio, so the receipt must retain the raw difference explicitly. |
| Case/profile construction | `_case_rows`, `build_profile`, `_with_moment_geometry`, and `_stored_lcfs` as used in `benchmarks/conditioned_convergence_observables.py:405-413` | **REUSE.** This is the frozen six-case identity and profile reconstruction already accepted by the evidence chain. |
| Existing eager/compiled measurement | `benchmarks/conditioned_convergence_observables.py:415-459` | **ADAPT, DO NOT MISTAKE FOR THE DISCRIMINATOR.** It solves separately in eager and JIT modes, then scores the two terminal result trees. It does not evaluate one shared terminal state and it does not exercise `solve_batch`. |
| Earlier parity bank | `benchmarks/jitted_eager_parity_gate.py:303-410` | **REUSE FOR PROVENANCE AND GENERIC HELPERS ONLY.** It supplies the calibration cohort and quantity paths, but its original common tolerance was replaced by the 69 per-observable registrations. |
| Same-flux derived-quantity precedent | `benchmarks/efit_metric_parity_comparison.py:202-397` | **DESIGN PRECEDENT, NOT AN ENGINE.** It demonstrates the discipline of deriving metrics from one stored flux map, but it compares `FluxSurfaceGeometry` to EFM fields. It does not evaluate `ForwardEquilibrium.conservation`, and its volume is a flux-surface profile rather than the terminal clipped-cell moment used here. |
| Shot-wide heterogeneous `vmap` | `_accelerated_profile_solve`, `nova/imas/mast_parity_chain.py:310-358` | **STATE SUPPLIER ONLY.** It genuinely maps varying per-slice source, plasma, measurement, mask, and seed rows, but returns only flux, residual, and trace (`AcceleratedProfileSolve`, lines 130-143). It uses the magnetics `ProfileSolver`, not `ForwardProfile`, and cannot evaluate any of the three registered leaves. |

## Observable-by-observable map

### `moments.volume`

**Registration and bound.** The committed registration is the
`moments.volume` row under
`docs/figures/forward-operator-refinement/criterion-family.json` →
`terminal_compiled_parity.terminal_observable_registration.bounds`. It is a
float64 scalar with `criterion_kind = exact_equality`; the calibration maxima
are both zero. The generating site is `_terminal_observable_bounds` in
`benchmarks/derived_criterion_family.py:419-451`. That function assigns exact
equality to any integer/boolean leaf **or any leaf whose calibration absolute
difference happened to be zero**. Volume is floating-point, so its exact bound
comes from the latter rule, not from a discrete semantic type. This is an
empirical zero-width calibration envelope and leaves the open exact-equality
decision genuinely load-bearing.

**Per-slice evaluation path.** `ForwardProfile.solve` →
`_solve_accelerated` (`nova/equilibrium/forward.py:1117-1145`) → `_receipt` →
`_integral_state` →
`ForwardFluxOperator.normalised_current_moments_and_observation` when a target
current is supplied (`nova/equilibrium/forward_operator.py:654-669`) →
`_support_partition` → `_clipped_integral_measure` → `observe_moments`.
`_clipped_integral_measure` forms each cell's exact toroidal volume from clipped
area and its radial first moment at
`nova/equilibrium/forward_operator.py:513-570`; `observe_moments` performs the
terminal `jnp.sum(support_integrals.volume)` at
`nova/equilibrium/observation.py:511-540`.

**Batched evaluation path.** `ForwardProfile.solve_batch` vmaps the same
`_solve_accelerated`, hence the same `_receipt`, support partition, per-cell
volume expression, and terminal reduction. The leaf is
`equilibrium.moments.volume` with one leading batch axis; there is no alternate
batched volume kernel.

**Current evidence.** Two cases fail exact equality: 21978/35 and 22086/43.
The largest absolute difference is `4.440892098500626e-16`; four of six cases
pass. The registration bank had exactly zero difference on all six calibration
comparisons.

**Fitness verdict — STRONG TARGET, BOUND QUALIFICATION REQUIRED.** The shared
receipt makes a same-state scalar-versus-vmap test direct. Localise a mismatch
in this order: `core_support` identity and inclusion, `closed_branch`, per-cell
`area`/`first_area_moment`, per-cell `volume`, then the final sum. Do not infer
that a floating scalar is intrinsically discrete merely because its calibration
maximum was zero.

### `moments.major_radius`

**Registration and bound.** The committed row is beside volume in the same
criterion-family registration. It is a float64 scalar with a banked dual
envelope: absolute `1.1102230246251565e-16`, relative
`1.1742075037825516e-16`. `_terminal_observable_bounds` copies those maxima
directly from the six-case `profile_solve.quantities` bank; the registration is
explicitly an empirical future-pair envelope rather than a physics tolerance
(`benchmarks/derived_criterion_family.py:535-551`).

**Per-slice evaluation path.** The path is identical to volume through
`_clipped_integral_measure`. That function also forms `radial_volume` from the
clipped area, radial first moment, and radial second moment at
`nova/equilibrium/forward_operator.py:540-566`. `observe_moments` then evaluates
`jnp.sum(radial_volume) / jnp.sum(volume)` at
`nova/equilibrium/observation.py:516-540`.

**Batched evaluation path.** `ForwardProfile.solve_batch` vmaps the same
`_solve_accelerated` and `_receipt`; the published leaf is
`equilibrium.moments.major_radius` with a leading batch axis. There is no
route-specific centroid or major-radius implementation.

**Current evidence.** Cases 21989/55 and 22086/43 fail. The maximum absolute
difference is `2.220446049250313e-16`, the maximum relative difference is
`3.2132739511434644e-16`, and maximum envelope utilisation is
`2.7365469397805193`; four of six cases pass.

**Fitness verdict — STRONGEST SHARED-SUPPORT COMPANION.** Evaluate volume and
major radius together. If volume agrees but major radius does not, the first
candidate is the per-cell radial-volume numerator or its reduction; if both
differ, start at the shared clipped support and volume denominator. The public
receipt does not retain the radial-volume numerator, so localisation requires a
temporary instrument around `_clipped_integral_measure` or `_integral_state`.

### `conservation.divergence_j`

**Registration and bound.** The committed row is in the same registration and
is a float64 scalar with a dual envelope: absolute
`1.3703738919113413e-10`, relative `0.15286615188530017`. The generator copied
the two calibration maxima from the original eager/JIT `profile_solve` bank,
using the same `_terminal_observable_bounds` rule.

**Per-slice evaluation path.** `ForwardProfile.solve` → `_solve_accelerated` →
`_receipt`, which passes the terminal grid flux, source, topology masks, and
flux span into `conservation_ledger` at
`nova/equilibrium/forward.py:982-1000`. `conservation_ledger` derives declared
support and erodes it by the stencil width, evaluates the field-function
squared, takes the guarded square root, differentiates it into poloidal current,
forms the axisymmetric divergence, and publishes the checked-cell sup norm at
`nova/equilibrium/conservation.py:324-399`.

**Batched evaluation path.** `ForwardProfile.solve_batch` vmaps that same
`_receipt` and `conservation_ledger`; the leaf is
`equilibrium.conservation.divergence_j` with a leading batch axis. There is no
alternate batched finite-difference or support rule.

**Current evidence.** Cases 21985/51, 21989/55, and 22086/43 fail. The largest
absolute difference, `1.1641532182693481e-10`, remains inside the absolute
envelope; the failure is driven by maximum relative difference `0.6` against
`0.15286615188530017`, for maximum utilisation `3.9250023147714024`. Three of
six cases pass.

**Fitness verdict — DIRECTLY REUSABLE, BUT NOT THE MOMENT SUPPORT.** This
observable does **not** reduce the clipped toroidal-volume support used by the
two moments. Its domain is the source's declared support eroded by the
difference stencil, as documented by `ClippedIntegralMeasure` at
`nova/equilibrium/observation.py:306-331` and implemented in
`conservation_ledger`. Localise a same-state mismatch through: masks and
`flux_span`, declared/eroded support, field-function squared, square root,
field-function gradients, poloidal-current components, divergence field, then
the checked-cell sup reduction. The relative bound also requires retaining
`divergence_j_scale`, because the current failure can arise from the numerator,
the scale, or both.

## Registration and provenance chain

The active chain is:

`jitted-eager-parity-gate.json` profile-solve maxima →
`benchmarks/derived_criterion_family.py::_terminal_observable_bounds` →
the 69 rows in `criterion-family.json` →
`benchmarks/conditioned_convergence_observables.py::_measure_terminal_observables`
→ `_score_observable` → the 66-of-69 / 407-of-414 reapplication receipt.

The registration is therefore evidence-backed but artifact-owned. The three
literal observable names do not occur in `nova/**/*.py`; they are paths obtained
by flattening the typed `ForwardEquilibrium` tree. This is useful for the
discriminator because the names already align with the full receipt, but a
consumer must load the committed criterion artifact rather than expect a
package registry constant.

## Explicit no-candidate results

- **NO CANDIDATE — package-level terminal-observable registry.** Searching all
  Nova Python sources for the three literal paths and for
  `terminal_observable_registration` found no package registration; only the
  benchmark generator, tests, and committed JSON artifact own it.
- **NO CANDIDATE — public batched observation method.** There is no
  `observe_batch`, `batch_observe`, or `observation_batch` on `ForwardProfile`.
  The available public batch method performs a solve; a same-state label arm
  must compose `vmap` over `observe`.
- **NO CANDIDATE — separate route-specific derived-label definitions.** Both
  `solve` and `solve_batch` terminate in `_solve_accelerated` and `_receipt`.
  A repair that authors a second batched volume, radius, or divergence function
  would create the duplication the live plan forbids.
- **NO CANDIDATE — committed raw shared-state values.** Neither
  `jitted-eager-parity-gate.json` nor
  `conditioned-convergence-and-observables.json` retains the two raw values for
  a case, and neither retains terminal flux arrays. The discriminator must run;
  it cannot be reconstructed from the aggregate receipts.
- **NO CANDIDATE — first-differing-operation trace.** Existing receipts retain
  terminal leaves, maxima, bounds, and counts, but not clipped-support moments,
  radial-volume numerators, eroded support masks, intermediate current fields,
  divergence fields, or reduction indices. Localisation needs a focused trace
  around the shared operation spine.
- **NO CANDIDATE — heterogeneous six-profile `ForwardProfile.solve_batch`.**
  The method batches dynamic arrays for one profile. The shot-wide MAST
  `vmap` accepts heterogeneous slice inputs but belongs to another solver and
  emits no `ForwardEquilibrium` labels. Neither is alone the requested six-case
  batch-versus-per-slice discriminator.
- **NO CANDIDATE — existing regression for all three labels.** The direct
  `solve_batch` parity test in
  `tests/test_equilibrium_forward_solve.py:746-763` checks flux agreement and
  aggregate finiteness, not volume, major radius, or current divergence. The
  continued-source batch test checks a ledger value, not these three labels.

## Recommended consumption boundary

Keep the discriminator in benchmark/test code and consume existing production
interfaces:

- frozen case construction and criterion loading from the current observable
  measurement benchmark;
- one selected shared terminal flux per case;
- scalar `profile.observe(shared_flux, target_current=...)`;
- leading-axis `jax.jit(jax.vmap(lambda state, target:
  profile.observe(state, target_current=target)))` where one static profile is
  valid;
- `_named_tree`/`_leaves` for stable observable paths;
- `_score_observable` for the committed exact/dual-envelope rules;
- focused intermediate receipts only after a terminal leaf differs.

For the six heterogeneous frozen profiles, run the scalar/shared-state pair per
profile first. A true multi-profile batch arm requires stacking the static
profile/operator data or exposing a functional observation kernel; that
capability does not exist today and should not be invented until the scalar
versus transformed shared-state discriminator shows that layout, rather than
state inheritance, is actually implicated.
