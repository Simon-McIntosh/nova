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
