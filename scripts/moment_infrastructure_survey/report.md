# Basin-entry moment infrastructure survey

## Scope and verdict vocabulary

This is a read-only inventory of the current Nova package plus the named DIII-D
evaluation scripts. It distinguishes three evidence states:

- **Tested** — the capability is implemented and has a directly relevant test;
  the focused subset run for this inventory is recorded at the end.
- **Untested** — an implementation exists, but no direct test of the stated
  end-to-end capability was found.
- **Absent** — a repository-wide symbol and call-site scan found no implementation
  of the stated capability. Nearby machinery is identified so that it is not
  mistaken for the missing capability.

The six requested questions are all answered below. The central result is that
Nova already has tested scalar-current normalization, per-cell zeroth/first
current moments, converged-state integral observations, and two-class cold-seed
and hysteresis machinery. It does **not** have a predictor that maps prescribed
flux functions to a basin-discriminating vector of global current centroid,
global second current moments, and internal inductance without first supplying a
full flux state.

## Capability summary

| Question | Capability | Verdict | Short answer |
| --- | --- | --- | --- |
| 1 | Net current | **Tested** | Summed from per-cell current in both reconstruction and forward observation paths. |
| 1 | Global current centroid | **Tested** | Computed by `ReconstructMoment` and the transport cold-start adapter, but not exposed by `IntegralObservation`. |
| 1 | Global second current moments | **Absent** | Degree-two reconstruction coefficients and local geometric second moments exist; no global current-weighted second-moment tensor is computed. |
| 1 | Internal inductance | **Tested**, with one legacy formula **untested** | The forward clipped-support observation and 1-D transport diagnostic are tested; the legacy `Plasma.li_3` property has no direct test. |
| 1 | Biot source-moment operators | **Tested** | Exact uniform/radial/vertical companion blocks consume per-section zeroth and first current moments. |
| 2 | Exact scalar target current | **Tested** | `target_current` is threaded through the public map/solve and uniformly scales all per-cell current-moment components. |
| 2 | Constraints beyond scalar current | **Absent** from the solve | `I_p`, `beta_p`, and `l_i` residual/Jacobian observations exist, but the shipped source declares zero closure degrees and enforcement rejects them. Centroid and second moments are not target types. |
| 3 | Two-class cold seeds and hysteresis | **Tested** | Limited/diverted seeds, topology qualification, policy, history, and persistence are implemented. Selection does not rank branches by predicted moments or residual magnitude. |
| 4 | Source-domain continuation | **Tested** | Continuation extends profile derivatives over named flux domains; it is not a homotopy in the solve state. |
| 5 | Flux-functions-only moment prediction | **Absent** | Trial-flux and labelled-map evaluators exist, but they require a complete flux map. DIII-D gates recover current from a labelled EFIT flux map and project it onto filaments. |
| 6 | Newton--Krylov initial plasma-current guess | **Absent** as an independent state | The initial unknown is caller-supplied flux. Current is recomputed from each trial flux; portfolio cold starts optionally construct a uniform-disc current image. |

## 1. Existing current-moment computations

### Reconstruction moment ladder

`ReconstructMoment` is the only package component that explicitly reconstructs
global current moments from magnetic measurements. Its documented ladder fixes
the total current at the Rogowski value, fits a current centroid, then admits
zero-sum degree-two and degree-three polynomial corrections
(`nova/equilibrium/moment.py:13-38`). `build_moment_basis` constructs monomial
columns and projects every higher-order column to zero sum, so those columns
cannot change total current (`nova/equilibrium/moment.py:127-199`).

The fit computes total current as `sum(cell_current)` and the two global
current-centroid coordinates as current-weighted means
(`nova/equilibrium/moment.py:479-534`). `fit_centroid` is the basin-entry seed:
it fits a single filament with current fixed to the Rogowski measurement and
uses that fitted position to seed the boundary reconstruction
(`nova/equilibrium/moment.py:541-590`,
`nova/equilibrium/moment.py:783-813`). The self-sized uniform disc is then
solved by a boundary fixed point, and the optional quadrupole stage fits three
zero-sum degree-two corrections
(`nova/equilibrium/moment.py:700-773`).

Verdicts:

- Net current and centroid: **Tested** by anchor/centroid recovery and basis
  recovery tests (`tests/test_equilibrium_moment.py:104-146`) and by the
  boundary-centroid and ladder receipts
  (`tests/test_equilibrium_moment_boundary.py:112-155`).
- Degree-two reconstruction coefficients: **Tested** as a zero-sum basis and
  low-order span (`tests/test_equilibrium_moment.py:61-98`,
  `tests/test_equilibrium_moment.py:104-126`). They are coefficients of a
  fitted current image, not an emitted global covariance or second-moment
  tensor.
- Global second current moments: **Absent**. `MomentInversion` exposes cell
  current, coefficients, total current, and centroid only
  (`nova/equilibrium/moment.py:363-389`). The boundary solve applies the
  quadrupole stage but has no calculation of
  `sum(I (R-Rc)^2)`, `sum(I (Z-Zc)^2)`, or
  `sum(I (R-Rc)(Z-Zc))` (`nova/equilibrium/moment.py:783-828`).
- The declared octupole order is usable by the generic basis fit but is not
  applied by the boundary `solve`, whose last conditional stage is the
  quadrupole fit (`nova/equilibrium/moment.py:737-773`,
  `nova/equilibrium/moment.py:783-828`). That boundary-stage capability is
  therefore **Absent**, rather than silently inferred from the enum.

### Production forward per-cell moments

The production forward operator represents each cell by three quantities:
integrated current and radial/vertical first moments about the cell's fixed
geometric centroid (`nova/equilibrium/stencil_mesh.py:103-108`). Fixed Duffy
quadrature and clipped-cell integration compute those three arrays
(`nova/equilibrium/stencil_mesh.py:383-416`,
`nova/equilibrium/separatrix_clip.py:255-370`). `ForwardFluxOperator` converts
the physical first moments into the local linear density basis and contracts
the resulting source representation with the field operator
(`nova/equilibrium/forward_operator.py:241-297`,
`nova/equilibrium/forward_operator.py:344-353`).

The mesh also stores polygon **geometric** second area moments
(`nova/equilibrium/stencil_mesh.py:255-301`). They are used to invert the local
linear-density representation; they are not plasma-current second moments and
are never globally aggregated (`nova/equilibrium/forward_operator.py:241-297`).

Verdicts:

- Per-cell zeroth and first current moments: **Tested** for affine-density
  exactness, topology masking, JIT, and vectorization
  (`tests/test_equilibrium_density_moments.py:50-115`). The production route is
  also compared with direct stencil/clip construction
  (`tests/test_equilibrium_forward_solve.py:300-341`).
- Local polygon second area moments: **Tested** as geometry in the production
  conversion (`tests/test_equilibrium_stencil_mesh.py:193-193`,
  `tests/test_equilibrium_forward_solve.py:320-331`).
- Global current centroid and global second current moments in the forward
  receipt: **Absent**. `ForwardEquilibrium` retains cell moments and integral
  observations, but no aggregation of the first moments or second current
  tensor is performed (`nova/equilibrium/forward.py:670-680`,
  `nova/equilibrium/observation.py:126-146`).

### Net current and integral observations

The current ledger sums already-integrated cell current by physical domain and
as a total (`nova/equilibrium/observation.py:339-351`). The converged-state
observation independently sums clipped-support current and computes poloidal
beta and internal inductance from pressure and poloidal-field volume integrals
(`nova/equilibrium/observation.py:354-379`). Its `major_radius` is a
volume-weighted major radius, not a current centroid
(`nova/equilibrium/observation.py:359-375`). These formulas are **Tested** by
direct observation checks (`tests/test_equilibrium_forward.py:434-463`) and by
moment residual/Jacobian checks (`tests/test_equilibrium_forward_solve.py:818-847`).

There are two additional internal-inductance computations outside that forward
observation:

- `poloidal_field_energy_li` evaluates `li3 = 4 Wpol/(mu0 Ip^2 R0)` from the
  one-dimensional transport geometry (`nova/transport/current_diffusion.py:318-327`).
  It is **Tested** for a positive order-unity result
  (`tests/test_current_diffusion.py:299-302`) and is used by the shot flux
  ledger (`nova/transport/parity_flux_ledger.py:199-210`).
- The legacy mutable `biot.Plasma.li_3` property forms a volume average of
  `Bp^2` and a hard-coded-radius boundary normalization
  (`nova/biot/plasma.py:118-128`). No direct package test of this property was
  found, so it is **Untested** and should not be treated as the forward
  observation's authority.

### Biot operator moment stacks

The exact polygon operator publishes flux blocks `(G0, GR, GZ)` for uniform,
radial-first, and vertical-first source moments
(`nova/biot/polygonanalytic.py:1743-1756`) and matching field blocks
(`nova/biot/polygonanalytic.py:2014-2030`). `PolySection` assembles these
companions and exposes `PsiR`, `PsiZ`, `BrR`, `BrZ`, `BzR`, and `BzZ`
(`nova/biot/polysection.py:223-314`); `FluxTarget.internal` contracts them with
the three per-section current-moment arrays
(`nova/biot/target.py:570-585`). This machinery is **Tested** against numerical
quadrature and translation identities (`tests/test_biottargetmoments.py:27-109`,
`tests/test_biotpolysection.py:54-89`).

The nearby `second_moments`/`third_moments` and `moment_filament` routines are
cross-section geometry corrections for conductor kernels
(`nova/biot/greens.py:1097-1266`), not global plasma-moment diagnostics. They
are **Tested** as geometric/kernel quantities
(`tests/test_biotgreens.py:840-861`, `tests/test_biotgreens.py:916-926`) but do
not fill the missing basin-entry predictor.

### Cold-start adapter centroid

The time-coupled transport adapter supplies the other package-level global
current centroid. At the first sample it observes the caller's seed, sums its
cell current, computes the current-weighted lattice centroid, and passes both
to `cold_seed_portfolio` (`nova/transport/coupled_window.py:1156-1183`). This
calculation is **Tested indirectly** by the two-class recovery and waveform
portfolio tests (`tests/test_two_class_recovery.py:85-175`), but it is an
observation of an already supplied full flux seed, not a prediction from flux
functions.

## 2. Constraint seams and admitted structure

`ForwardProfile.flux_map` forwards the optional `target_current` into the
operator map (`nova/equilibrium/forward.py:428-432`). The operator sums the
unscaled cell current, computes one guarded scalar amplitude, and multiplies
all zeroth and first per-cell moment arrays by that same amplitude
(`nova/equilibrium/forward_operator.py:434-475`). Both the internal-current
image and fixed-point map retain the scalar target
(`nova/equilibrium/forward_operator.py:520-574`). The public solve documents
the same policy: one common source amplitude is eliminated inside every map
evaluation (`nova/equilibrium/forward.py:847-904`).

That scalar seam is **Tested** for exact current under JIT, guard failures, and
consistent scaling of all local moment arrays
(`tests/test_equilibrium_forward_constrained.py:47-77`), as well as for the
public receipt (`tests/test_equilibrium_forward_constrained.py:95-104`). The
DIII-D constrained driver calls this exact public map and solve seam
(`benchmarks/diiid_constrained_cold_start.py:222-256`), with an IMAS integration
test (`tests/imas/test_diiid_current_pinned_forward.py:103-145`).

Beyond scalar current, the structure stops at observation:

- `MomentTargets` and `IntegralObservation` name only plasma current, poloidal
  beta, and internal inductance (`nova/equilibrium/observation.py:102-146`).
  Their scale-normalized residual vector is implemented and **Tested**
  (`nova/equilibrium/observation.py:382-399`,
  `tests/test_equilibrium_forward_solve.py:818-847`).
- The source declares zero scalar closure degrees
  (`nova/equilibrium/source.py:480-521`). Public `enforce` therefore validates
  and rejects every non-empty request before solving; it never changes source
  coefficients (`nova/equilibrium/observation.py:402-427`,
  `nova/equilibrium/forward.py:1138-1150`). These negative boundaries are
  **Tested**, including rejection of unsupported and unknown targets
  (`tests/test_equilibrium_forward.py:511-545`,
  `tests/test_equilibrium_forward_solve.py:903-911`).
- Current centroid and second current moments are not members of either target
  or observation type, and there is no vector constraint callback in the
  fixed-point map. Those constraints are **Absent**. Internal inductance is
  structurally admitted only as a differentiable observed residual/Jacobian,
  not as a production solve constraint.

Consequently `target_current` can change only the common magnitude of a
trial-current shape. It cannot independently move the centroid, alter a
second-moment tensor, or tune `l_i`.

## 3. Cold-seed portfolio and branch selection

`ForwardColdSeedPortfolio` contains fixed-order limited and diverted branches,
with receipts declaring how each seed was constructed
(`nova/equilibrium/forward.py:193-261`). `cold_seed_portfolio` requires a
caller-supplied plasma current and current centroid. The limited branch is the
external field plus a uniform current disc; the diverted branch is either the
same neutral seed or a declared axis/saddle cubic field when geometry is
available (`nova/equilibrium/forward.py:434-510`). Both seeds receive topology
read receipts (`nova/equilibrium/forward.py:512-550`). The portfolio solver
then evaluates limited and diverted branches in a fixed order while pinning the
requested class (`nova/equilibrium/forward.py:1050-1103`).

Branch availability requires convergence and topology consistency. On a cold
start, policy selects a declared preferred class when both are admissible or
the sole available class otherwise. On later samples, history retains the
selected class, changes immediately if it disappears, and otherwise requires
the alternative to remain admissible for the persistence threshold
(`nova/equilibrium/branch_selection.py:104-175`,
`nova/equilibrium/branch_selection.py:223-323`). Residuals are recorded in the
candidate receipt but are not used to rank two admissible branches.

The portfolio and hysteresis machinery is **Tested** by the policy transition
matrix (`tests/test_branch_selection.py:48-218`), seed/solve parity
(`tests/test_equilibrium_forward_solve.py:766-815`), and two-class recovery
(`tests/test_two_class_recovery.py:85-175`).

Exactly what selects a branch today is therefore:

1. caller policy, or on the first coupled-window sample the topology observed
   from the caller-supplied seed (`nova/transport/coupled_window.py:1156-1194`);
2. post-solve convergence and agreement with the requested topology class
   (`nova/equilibrium/branch_selection.py:223-250`);
3. caller-declared admissibility and the previous selected class
   (`nova/equilibrium/branch_selection.py:252-323`); and
4. persistence count before a still-available incumbent is replaced
   (`nova/equilibrium/branch_selection.py:290-323`).

No predicted global moments, no moment-distance metric, and no lowest-residual
competition select between two admissible branches. Those selectors are
**Absent**.

## 4. Source-domain continuation versus solve-state homotopy

`SourceContinuation` extends the profile derivatives `p'` and `F F'` over an
explicit open flux domain. It declares the functional form, continuity class,
support, and optional decay/spreading parameters
(`nova/equilibrium/continuation.py:1-111`,
`nova/equilibrium/continuation.py:386-460`). Calling `extend` returns an
independent `ContinuedDomainProfile`; separatrix distance is a function of
normalized flux (`nova/equilibrium/continuation.py:462-543`,
`nova/equilibrium/continuation.py:558-571`). The source owns domain-labelled
closures and a ledger for their participation
(`nova/equilibrium/source.py:121-218`,
`nova/equilibrium/source.py:480-521`).

This capability is **Tested** for declaration/continuity and for participation
in a solved fixed point (`tests/test_equilibrium_sol.py:412-571`,
`tests/test_equilibrium_sol.py:797-890`).

It is not solve-state homotopy because its input and output are source profiles,
not successive flux states. There is no homotopy parameter joining two roots,
no predictor/corrector state, and no path-following loop in this interface.
Newton--Krylov still receives one map and one initial flux
(`nova/equilibrium/fixed_point.py:202-274`). Solve-state homotopy in this
interface is therefore **Absent**, even though changing the continued source
can of course change the eventual equilibrium.

## 5. Pre-convergence prediction and DIII-D differencing gates

Nova can evaluate current from a supplied full flux map without first proving
that map is a converged Nova solution:

- `extract_flux_map_profiles` applies the centred Grad--Shafranov operator to a
  structured supplied flux map to obtain toroidal current density, then
  projects `p'` and `F F'` over supplied normalized-flux shells
  (`nova/equilibrium/map_extraction.py:1-18`,
  `nova/equilibrium/map_extraction.py:128-170`,
  `nova/equilibrium/map_extraction.py:181-339`). The operator and profile
  round-trip are **Tested** (`tests/test_map_extraction.py:111-136`,
  `tests/test_map_extraction.py:206-240`).
- `ForwardFluxOperator.cell_current_moments` and
  `ForwardProfile.integral_observation` can evaluate a trial flux, but they
  require the complete trial flux and topology read, not flux functions alone
  (`nova/equilibrium/forward_operator.py:415-424`,
  `nova/equilibrium/forward.py:630-680`). Their trial-state formulas are
  **Tested** as part of the forward operator and observation suites
  (`tests/test_equilibrium_forward_solve.py:300-341`,
  `tests/test_equilibrium_forward.py:434-463`).

The DIII-D vacuum/quiescent gate converts the EFIT label from webers per radian
to Nova total flux, applies the Grad--Shafranov operator, integrates density
over grid-node areas, and projects those currents through a filament response
matrix (`benchmarks/diiid_vacuum_quiescent_gate.py:235-266`,
`benchmarks/diiid_vacuum_quiescent_gate.py:293-321`). Its conversion/operator
boundary is **Tested** (`tests/imas/test_diiid_quiescent_gate.py:79-93`). The
plasma-subtraction gate likewise extracts flux functions from the labelled
flux, reconstructs grid current, integrates it, and projects it onto filaments
(`benchmarks/diiid_plasma_subtraction_gate.py:135-213`). No direct end-to-end
test of that latter projection function was found, so that specific script
path is **Untested**.

These gates do not predict moments from prescribed flux functions: their input
is a complete labelled EFIT flux map, effectively an already reconstructed
equilibrium state. They report/integrate net plasma current, not global current
centroid, global second current moments, or internal inductance. A
flux-functions-only predictor for those four quantities is **Absent**.

## 6. Newton--Krylov cold-start current guess

The public forward solve requires `initial_flux`; Newton--Krylov is the default
route (`nova/equilibrium/forward.py:847-904`). The accelerator's state is that
flux array. Each residual evaluation calls `map_fn(state)`, and the exact
tangent is taken with respect to the same state
(`nova/equilibrium/fixed_point.py:202-274`). There is no independently initialized
plasma-current vector in the Newton--Krylov state.

For every trial flux, `ForwardFluxOperator.internal` reads topology and
recomputes per-cell current moments from the source closures; when
`target_current` is present, the single normalization amplitude is applied at
that evaluation (`nova/equilibrium/forward_operator.py:520-538`,
`nova/equilibrium/forward.py:630-668`). This state contract is **Tested** by the
forward solve and constrained-map tests
(`tests/test_equilibrium_forward_constrained.py:47-104`,
`tests/test_equilibrium_forward_solve.py:300-388`).

If the caller explicitly uses `cold_seed_portfolio`, the limited seed's initial
plasma flux is generated from a uniform current disc at the caller-supplied
current and centroid; an axis/saddle field may replace the diverted branch
(`nova/equilibrium/forward.py:434-510`). That seed construction is **Tested**
(`tests/test_two_class_recovery.py:85-175`). Otherwise Nova imposes no default
current shape: the caller owns the initial flux.

The DIII-D constrained driver is named a cold start, but `prepare_frame` obtains
the seed from `build_frame_case`, which downsamples the same-frame labelled EFIT
flux and interpolates it to the wall
(`benchmarks/diiid_constrained_cold_start.py:116-164`,
`benchmarks/diiid_forward_gs_match.py:587-642`). It is therefore a labelled-flux
seed, not a moment-predicted plasma-current guess.

For comparison, the separate reconstruction path `ReconstructProfile` does
construct a Gaussian current image around a static `axis_seed`, normalizes it
to the prescribed plasma current, and adds the known-conductor field
(`nova/equilibrium/profile.py:390-411`). That is **Tested** reconstruction
machinery (`tests/test_profile_accelerated.py:166-185`,
`tests/test_equilibrium_profile.py:220-240`), not the forward Newton--Krylov
initializer and not a source of basin moment predictions.

Final verdict: an automatic initial plasma-current guess inside the forward
Newton--Krylov solver is **Absent**. The actual initial unknown is caller-supplied
flux; current is induced by the map from that flux and the declared source.

## Focused verification for this inventory

The focused fast-lane selection completed with **56 passed, 1 deselected, 0
failed in 32.74 s**. The deselected item is the explicitly selected centroid
boundary test because its module is marked `slow` and the repository default is
`-m 'not slow'`; the directly cited broader test remains present but was not
silently forced into this fast evidence run.

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/test_equilibrium_moment.py tests/test_equilibrium_moment_boundary.py::test_the_centroid_fit_recovers_the_truth_centroid tests/test_equilibrium_density_moments.py tests/test_equilibrium_forward_constrained.py tests/test_branch_selection.py tests/test_map_extraction.py::test_delta_star_receipt_marks_the_centred_stencil_only tests/test_equilibrium_sol.py::test_a_continuation_publishes_the_class_form_and_support_it_declared tests/test_equilibrium_sol.py::test_the_continuation_meets_the_core_at_the_separatrix tests/imas/test_diiid_quiescent_gate.py::test_label_map_current_converts_per_radian_flux
```

Captured log: `scripts/moment_infrastructure_survey/focused-tests.log`.
Existing test references above record the broader repository coverage even when
the focused selection does not execute every cited test.
