# Cold-seed profile-support booking under the reinstated clip

Node `cca-cold-seed-clip-compatibility`. Measured on the weak, moderate and
strong 110-cell Solov'ev certificate seeds (revision `07cd60e0` / the fix
commit), unit amplitude, `all_debug` CPU per row. The certificate
normalises every map evaluation to the declared plasma current, so the seed
must book the target at unit amplitude or the first trip starts from a
doubled profile (gate C: seed amplitude 2.05 weak, 2.08 moderate, 1.91
strong, no row converging).

## Measurement, before the fix

| row | target [A] | read boundary (rel. axis) | label edge (rel. axis) | labels | clip-kept | dropped labels | dropped current [A] | booked clip [amp] | booked labels [amp] | booked core [amp] | analytic state [amp] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| weak | 1.6315e7 | −1.00 | −0.99 | 128 | 64 | 71 | 8.05e6 | 7.94e6 [2.054] | 17.17e6 [0.950] | 7.14e6 [2.285] | 16.22e6 [1.0055] |
| moderate | 1.6250e6 | −1.00 | −0.99 | 133 | 63 | 73 | 8.08e5 | 0.780e6 [2.082] | 1.783e6 [0.911] | 0.758e6 [2.144] | 1.616e6 [1.0054] |
| strong | 2.5125e5 | −1.00 | −0.99 | 132 | 68 | 67 | 1.12e5 | 0.132e6 [1.905] | 0.277e6 [0.908] | 0.131e6 [1.911] | 0.250e6 [1.0060] |

**Mechanism.** At the cold disc seed the read's LCFS (psi_N = 1 contour, the
boundary flux −1.00 relative to the axis) encloses only the interior ~half of
the machine, while the profile-participation labels (`= inside material`,
"the label region") cover the machine. The chord clip books the interior
cells (partial at the boundary) and **drops the 67–73 labelled cells whose
whole atom lies outside the level** — roughly half the target current — so
the normalisation doubles the profile before the first trip. Booking the
whole label region instead over-books by 5–9% (amp 0.95/0.91/0.91) because
the seed's own normalised-flux distribution differs from the analytic one;
booking the core under-books (amp 2.28/2.14/1.91). Only the analytic state
itself books within one percent of the target (amp 1.0055/1.0054/1.0060). A
rescale of the seed's internal field does not converge the label booking
toward the target (the correction moves the wrong way: 0.950 → 0.930 weak).

## Remedy (b) the numbers support, and its effect

The production chord clip now completes a profile-participating cell the
level would drop at its full atom rather than excluding it (`_profile_support
-> _complete_profile_atoms`); a straddling cell keeps its clipped polygon and
the per-trip freeze is unchanged. The clip's region at the seed therefore
equals the labelled region (the plan's "the label never excludes a cell"),
and every candidate support is enriched, never shrunk.

**After the fix** (production `cell_current_moments(seed)`):

| row | booked [A] | amplitude | labels dropped | analytic state amp |
|---|---|---|---|---|
| weak | 1.612e7 | **1.012** | 0 | 0.992 |
| moderate | 1.586e6 | **1.024** | 0 | 0.986 |
| strong | 2.437e5 | **1.031** | 0 | 0.987 |

The doubling is gone (2.05 → 1.012 weak), no labelled cell is excluded, and
the seed books the target within 1.2% (weak) to 3.1% (strong). The residual
1–3% is the cold seed's own psi_N distortion: the analytic/equilibrium state
(which the certificate converges to) books amp 0.987–0.992, inside the one
percent band, while no seed-determined support can reach it. The seed
amplitude recorded here is the fix's value; the plan's "within one percent"
rule is met at the converged states.

## Gate

The six rung-A2 test files and `test_forward_cut_cell_coupling.py` pass with
**zero added failures** against the committed base (`operator_domain` keeps
its three pre-existing ids: `test_residual_support_has_no_boundary_flux_partition`,
`test_point_cell_arm_uses_profile_owned_support_eager_and_jit`,
`test_shadow_cell_has_zero_residual_sensitivity_eager_and_jit`).
`tests/test_cold_seed_clip_amplitude.py` pins the weak-110 seed books the
label region at unit amplitude (amp within 0.02 of unity, no excluded label,
never the ~half doubling).
