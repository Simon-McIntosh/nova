# Continuation parameter adjudication

## Recommendation

Use **state-space arclength** as a numerical gauge over successive
topology-admissible flux states. Do not continue in either source amplitude or
constrained plasma current.

This is a design recommendation, not convergence evidence. It identifies the
only candidate whose predictor direction is formed from states already admitted
on the requested topology branch and whose continuation coordinate does not
change the plasma-source magnitude. The production implementation must still be
measured on the five score-blind frames against the banked residual,
admission, advance-length, topology, finite-X-point, and total-traversal
comparators before the solve blocker can close.

The arclength coordinate is deliberately a *gauge*, not a claim that the
Boolean topology class is itself differentiable. The topology read remains a
fail-closed predicate. Its derivative must not be invented.

## Decisive banked negative

The durable receipt is
`docs/evidence/archive/dual-basin-solve-landed.html#s2-two-class-closed`
(evidence commit `d3093912`, integrated by `0546453d`). It records:

> Two bounded composed-map continuation campaigns (21 rungs; 41 rungs at 5
> Newton / 30 GMRES per rung) both terminated converged on the dominating
> limited root even against the well-separated fixture (parity error 5.06e-2,
> contradictions clustered over fractions 0.475-1.0).

That is a negative about path selection, not budget exhaustion: both campaigns
converged, the finer campaign doubled the rung count, and the terminal parity
error remained about `5.06e-2` rather than the `1e-10` recovery floor. A scalar
source-strength path therefore has direct evidence that it follows the wrong
coexisting basin.

A second durable receipt,
`docs/figures/efit-forward-parity/fixed-boundary-double-seed.json`, states the
same distinction explicitly: source-strength homotopy scales plasma source from
zero and converged to the dominating limited root, whereas the separate
fixed-boundary diagnostic changed boundary conditions at full source strength.
Nothing inspected for this adjudication overturns the source-strength negative.

`nova/equilibrium/continuation.py` is not counter-evidence and is not an
implementation seam for this work. It continues `p'` and `F F'` spatially from
the confined domain onto declared open-flux domains. It has no solve-state
path, no homotopy parameter joining roots, and no predictor-corrector loop.

## Candidate constructions and verdicts

Let `x` be the existing flat flux-state vector, `g(x; q)` the production
forward map at fixed conductor inputs and source declaration `q`, and
`r(x; q) = g(x; q) - x`. Let `A(x)` be the existing requested-topology and
finite-state admissibility predicate. All constructions below can have static
array shapes and fixed iteration counts; that property alone does not make
their path scientifically acceptable.

### State-space arclength — recommend

Carry the last two admitted corrected states, `x_prev` and `x_now`, plus the
previous oriented tangent, all with the same fixed shape as `x`. Form a secant
tangent in flux-state space:

```
delta = x_now - x_prev
t_raw = delta / max(weighted_norm(delta), tiny)
t = where(dot(t_raw, t_prev) >= 0, t_raw, -t_raw)
x_predict = x_now + ds * t
```

The metric must be declared once and used identically for every frame. A
dimensionless RMS norm after dividing flux differences by the current
axis-to-boundary flux span is the natural starting choice; it prevents grid
size and absolute Weber scale from changing what one arclength unit means. The
orientation `where` prevents an arbitrary secant sign flip. A zero secant is a
named refusal, not a fallback to a source direction.

Bootstrap the first secant from the current state and the largest trial on the
existing fixed ladder that passes `A(x)`. The measured `0.03125` rung therefore
supplies an initial local direction once; it does not remain the advance rule.
If no nonzero trial is admitted, refuse the bootstrap.

At every later step, correct normal to the secant rather than backtracking the
predictor to zero. If `d_newton` is the already-qualified Krylov correction,
use

```
d_normal = d_newton - t * dot(t, d_newton)
candidate[j] = x_predict + correction_factor[j] * d_normal
```

for a fixed, compile-time correction ladder. Evaluate every candidate, its map
residual, and `A(candidate)` with `jax.lax.map` or `vmap`; select with masks and
`where`, never a data-dependent loop. Run a fixed number of predictor-corrector
iterations with `jax.lax.fori_loop`. Update the secant only from a promoted candidate.
The corrector may reuse the existing `admissibility_fn` contract in
`nova/equilibrium/fixed_point.py`; it must not copy or weaken the topology read.

This construction advances by `ds` along a direction evidenced by two states
on the requested branch, while the normal correction reduces cross-track
fixed-point residual. Full `r(x) = 0` remains the terminal convergence test; a
projected residual is not sufficient. The receipt must expose both the full
residual and achieved state-space advance so a topology-holding stall remains
an honest negative.

Why it survives adjudication: source declaration and target current remain
fixed, so the continuation coordinate cannot trace the banked
source-strength-from-zero curve under another name. The existing predicate
continues to decide admission. The construction is also compatible with the
production state contract: `fixed_point.py` already carries a flat one-
dimensional state, exact JAX tangents, fixed-shape GMRES, fixed ladders, and
device-side masked selection.

### Source amplitude — reject

The implied tangent augments the flux state with a scalar source amplitude
`a`. For `R(x, a) = x - g(x; a q)`, a source-parameter tangent sets `t_a = 1`
and solves

```
R_x t_x = -R_a
```

with `R_x` and `R_a` obtained from one fixed-shape JAX linearisation. A
pseudo-arclength variant would instead solve the same bordered system with an
orientation row. Either version is `jit`-safe with a fixed Krylov budget.

**Refutation:** this is the mechanism already measured. Its parameter changes
only the scalar source strength, and two bounded campaigns of 21 and 41 rungs
converged to the dominating limited root. A tangent implementation or finer
step schedule changes numerical traversal, not the one-dimensional physical
path or its basin endpoint. No new branch-selecting constraint is present.
Selecting it would directly repeat the banked failure.

### Constrained plasma current — reject

The implied tangent augments the state with target current `I`. For the
production current-constrained map `R(x, I)`, set `t_I = 1` and solve

```
R_x t_x = -R_I
```

or the equivalent oriented bordered system. The target axis can remain scalar
or carry the existing batch axis, so JVP construction, a fixed GMRES budget,
and fixed continuation rungs are all `jit`- and `vmap`-safe.

**Refutation:** constrained current is not an independent continuation degree
in the shipped map. `ForwardFluxOperator.current_normalisation_amplitude`
computes

```
a(x, I) = I / sum(unscaled_cell_current(x))
```

after a guarded sign-and-band check, and `scaled_current_moments` multiplies
every zeroth and first current moment by that same `a`. It cannot change source
shape, current centroid, topology, or any other branch coordinate independently.
Where `a(x, I)` is regular and monotone, current and amplitude are merely two
coordinates on the same one-scalar source-scaling path. Where it is not
regular, the guarded normalisation becomes non-finite and is refused; the
singularity does not create a branch-selection mechanism.

Starting the current ladder away from zero would avoid reproducing the exact
rung sequence, but it would not answer the measured causal failure: the only
physical degree being advanced would still be common source amplitude, and
the dominating limited root would remain admissible. No receipt shows current
parameterisation selecting the requested coexisting basin. It therefore
reintroduces the banked mechanism under an implicit coordinate and is rejected.

## Implementation contract carried by the recommendation

The implementation owner should treat these as non-negotiable properties of
the recommended route:

1. Keep `current`, `target_current`, conductor inputs, source closures, and
   requested topology fixed during the arclength path.
2. Carry `x_prev`, `x_now`, `t_prev`, arclength `s`, step `ds`, refusal state,
   and fixed-length receipts in the loop carry; no Python termination based on
   traced values.
3. Reuse the production requested-class/finite-state predicate for every
   predictor and corrector candidate. Do not differentiate the Boolean class
   and do not add a second topology classifier.
4. Retain the existing Krylov-action qualification. A refused or zero
   material step promotes no candidate.
5. Use fixed predictor and corrector budgets and compile-time candidate axes;
   `vmap` over frames or ensemble members must see identical shapes.
6. Record full residual, admitted-advance count, mean advance divided by the
   corresponding Newton-step norm, total Newton-step-equivalents, terminal
   topology, finite X point, secant degeneracies, and all refusal reasons.

## Decision boundary and quantitative outcome

**Adjudication: 1 of 3 candidates recommended; 2 of 3 explicitly rejected.**

- **Recommend:** state-space arclength, because its tangent is constructed from
  consecutive admitted states while the physical source remains fixed.
- **Reject:** source amplitude, directly refuted by the 21-rung and 41-rung
  campaigns ending at the converged limited root with `5.06e-2` parity error.
- **Reject:** constrained plasma current, because the production implementation
  eliminates it through the same single common source amplitude and supplies no
  independent branch-selecting direction.

This report was adjudicated against worktree `HEAD bacbbd81`. It ran no
candidate production solve and therefore makes no comparative performance or
convergence attribution. The inspected production paths were unchanged between
the three paper constructions; their differences above are algebraic contract
differences, not measured runtime deltas. The five-frame production measurement
remains the next evidence gate.
