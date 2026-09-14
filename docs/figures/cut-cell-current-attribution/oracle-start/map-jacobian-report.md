# Analytic map floor and booked-moment decomposition

No Newton, linear solve, or Jacobian evaluation was entered. Every row-mode
part records one production-map application, the exterior construction, the
production topology read, and the booked internal image against the fixture's
analytic moments.

Comparison anchors: the operator refinement ladder closes the analytic field to 3e-15 relative sup when analytically integrated moments are imaged with their implied exterior; the committed weak-110 whole-cell certificate row is self-consistent at residual 0.0074.

| Row | Mode | Map rms / sup | Affine rms / remainder (energy) | External mismatch rms | Internal mismatch rms | Booked / analytic | Worst JVP discrepancy at 1e-5 |
|---|---|---:|---:|---:|---:|---:|---:|
| weak-rotation-reactor-static 1000 | exact | 1.361e-01 / 1.901e-01 | 1.334e-01 / 2.673e-02 (96.1%) | 1.361e-01 | 1.361e-01 | 0.99998738 | not_requested_for_one_application_decomposition |
| weak-rotation-reactor-static 1000 | chord | 2.334e-01 / 3.114e-01 | 2.253e-01 / 6.105e-02 (93.2%) | 2.334e-01 | 2.334e-01 | 1.08573216 | not_requested_for_one_application_decomposition |

exact floor 0.136088 rms of span is carried by the booked internal-image difference (0.136088); the external-minus-implied-exterior value is the same incompatibility viewed from the boundary supply (0.136088). The affine fit explains 96.1% of residual rms energy and leaves 0.0267317 rms of span. The certificate exterior closes analytically integrated moments, not the production booking evaluated by this map. chord floor 0.233437 rms of span is carried by the booked internal-image difference (0.233437); the external-minus-implied-exterior value is the same incompatibility viewed from the boundary supply (0.233437). The affine fit explains 93.2% of residual rms energy and leaves 0.0610453 rms of span. The certificate exterior closes analytically integrated moments, not the production booking evaluated by this map.

| moderate-rotation-conventional-static 1000 | exact | 1.275e-01 / 1.795e-01 | 1.245e-01 / 2.731e-02 (95.4%) | 1.275e-01 | 1.275e-01 | 0.99998563 | not_requested_for_one_application_decomposition |
| moderate-rotation-conventional-static 1000 | chord | 2.255e-01 / 3.027e-01 | 2.176e-01 / 5.915e-02 (93.1%) | 2.255e-01 | 2.255e-01 | 1.08590932 | not_requested_for_one_application_decomposition |

exact floor 0.127486 rms of span is carried by the booked internal-image difference (0.127486); the external-minus-implied-exterior value is the same incompatibility viewed from the boundary supply (0.127486). The affine fit explains 95.4% of residual rms energy and leaves 0.0273067 rms of span. The certificate exterior closes analytically integrated moments, not the production booking evaluated by this map. chord floor 0.225459 rms of span is carried by the booked internal-image difference (0.225459); the external-minus-implied-exterior value is the same incompatibility viewed from the boundary supply (0.225459). The affine fit explains 93.1% of residual rms energy and leaves 0.059148 rms of span. The certificate exterior closes analytically integrated moments, not the production booking evaluated by this map.

| diverted-single-null 1000 | exact | 2.077e+00 / 4.312e+00 | 1.145e+00 / 1.733e+00 (30.4%) | 2.077e+00 | 2.077e+00 | 0.60888480 | not_requested_for_one_application_decomposition |
| diverted-single-null 1000 | chord | 4.980e-15 / 1.911e-14 | 3.641e-15 / 3.397e-15 (53.5%) | 4.820e-15 | 4.789e-15 | 1.00000000 | not_requested_for_one_application_decomposition |

exact floor 2.07713 rms of span is carried by the booked internal-image difference (2.07713); the external-minus-implied-exterior value is the same incompatibility viewed from the boundary supply (2.07713). The affine fit explains 30.4% of residual rms energy and leaves 1.7332 rms of span. The certificate exterior closes analytically integrated moments, not the production booking evaluated by this map. chord floor 4.97989e-15 rms of span is carried by the booked internal-image difference (4.78899e-15); the external-minus-implied-exterior value is the same incompatibility viewed from the boundary supply (4.82034e-15). The affine fit explains 53.5% of residual rms energy and leaves 3.39721e-15 rms of span. The certificate exterior closes analytically integrated moments, not the production booking evaluated by this map.

| diverted-single-null 500 | exact | 2.063e+00 / 4.317e+00 | 1.107e+00 / 1.741e+00 (28.8%) | 2.063e+00 | 2.063e+00 | 0.60878945 | not_requested_for_one_application_decomposition |
| diverted-single-null 500 | chord | 2.932e-15 / 1.216e-14 | 5.740e-16 / 2.875e-15 (3.8%) | 2.622e-15 | 2.623e-15 | 1.00000000 | not_requested_for_one_application_decomposition |

exact floor 2.06323 rms of span is carried by the booked internal-image difference (2.06323); the external-minus-implied-exterior value is the same incompatibility viewed from the boundary supply (2.06323). The affine fit explains 28.8% of residual rms energy and leaves 1.74102 rms of span. The certificate exterior closes analytically integrated moments, not the production booking evaluated by this map. chord floor 2.93168e-15 rms of span is carried by the booked internal-image difference (2.62295e-15); the external-minus-implied-exterior value is the same incompatibility viewed from the boundary supply (2.62243e-15). The affine fit explains 3.8% of residual rms energy and leaves 2.87492e-15 rms of span. The certificate exterior closes analytically integrated moments, not the production booking evaluated by this map.

## Resolution verdict

The limited-row floors do not converge away with pitch.  Weak exact moves from
0.0787 RMS of span at 300 cells to 0.1361 at 1000, while whole-cell moves from
0.1267 to 0.2334.  Moderate exact is 0.1275 and whole-cell 0.2255 at 1000,
also above their 300-cell values of 0.0761 and 0.1315.  In contrast, admitted
single-null whole-cell remains at roundoff: 2.93e-15 at 500 and 4.98e-15 at
1000, while exact remains 2.063 and 2.077.  A one-cell-thick discrepancy
between whole-cell and clipped boundary booking would shrink with pitch.  These
limited-row floors do not, so their production support differs from the
fixture's analytic cell partition over a resolution-independent current, not
through an incorrect boundary flux level and not through the density evaluated
in fully participating cells.

Booked current divided by the analytic target at 1000 cells is:

| Row | Exact | Whole-cell |
|---|---:|---:|
| weak | 0.9999873776966958 | 1.0857321614725386 |
| moderate | 0.9999856253304152 | 1.0859093185720008 |
| single-null | 0.6088847979441562 | 1.0000000000000002 |

## Weak-1000 exact cell attribution

The difference is not present on every cell and it is not carried by cut cells
alone.  Of 2.244463 MA total absolute cell-current difference, 118 inactive
cells carry 1.130264 MA (50.36%) and image to 0.124559 of span RMS; 130 cut
cells carry 1.100684 MA (49.04%) and image to 0.032425; 824 whole cells carry
only 13.515 kA (0.602%), imaging to 0.002965.  The category images are signed
and non-orthogonal, so their RMS values do not add; their frozen-block sum is
the measured all-cell image of 0.136088.

The whole-cell ratio directly falsifies a different production density in the
interior.  For all 823 whole cells with nonzero analytic current, spanning
R = 4.1880156 to 7.6403902 m, the unnormalised production cell current divided
by the analytic integral over the identical polygon is 1.0 with a maximum
departure of 6.66e-16.  After the certificate's one scalar current
normalisation, the ratio is the R-independent constant
1.000012622462629 (range only 6.66e-16); there is no radial trend.  Thus
`case.toroidal_current_density` and the production profile agree on whole
cells.  The resolution-independent mismatch is the production participation
and support choice: analytic-current cells made inactive, together with the
different integration over cells classified as cut.

The production path is
`ForwardFluxOperator._partitioned_current_moments` at
`nova/equilibrium/forward_operator.py:2278-2289`, which calls
`ForwardSource.current_moments` at `nova/equilibrium/source.py:621-658`.
That selects the core profile and calls
`ForwardFluxOperator.support_current_moments` at
`nova/equilibrium/forward_operator.py:2043-2065`.  Each carried polygon is
integrated by `fixed_profile_current_moments` and
`_direct_profile_current_moments` at
`nova/equilibrium/stencil_mesh.py:405-495`; the actual evaluation is
`profile.current_density(points[..., 0], flux)` at line 485.  The part stores
every cell's R,Z centre, support fraction, class, booked and analytic current,
coupling coefficients, differences, and category image.

## Single-null topology census and X-point cell

| Cells | Production class at analytic flux | x_candidate_count |
|---:|---|---:|
| 300 | limited | 0 |
| 500 | diverted | 1 |
| 1000 | diverted | 1 |

The 300 count is independently present in the committed production-route
certificate part; the 500 and 1000 counts are reproduced by this CPU receipt.
The 300 failure is therefore absence of any retained X candidate, not rejection
of an otherwise retained candidate.  At 500 the analytic integral in the
X-point cell is 109.884308 A: whole-cell books the same value, while exact
books 36.813781 A before global normalisation and 60.470465 A after it.  At
1000 the analytic value is 55.530027 A: whole-cell again books it exactly,
while exact books 33.452150 A before normalisation and 54.940032 A after it.
The exact clip's large two-span error is therefore not explained by the
zeroth current of the X-point cell alone at 1000—the normalised zeroth moment is
within 1.06% there—and must be read with the exact clip's changes to the other
cell moments and their frozen-block image.
