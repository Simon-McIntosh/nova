# Coupled-window receipts across two regimes

This is one locked strong-window run and a bounded gentle search. Every row is from `solve_window`; typed refusals are reported and were not retried with altered tolerances.

## Experiment contract

The equilibrium is the repository's 25 x 25 free-boundary fixture. Every accepted sample was evaluated exactly on a 49 x 49 lattice and extracted into 8 TORAX radial cells with 14 surface bins. TORAX advances all four transport channels in one fixed step per window.
Fixture-scale execution backend: `cpu`.
The auxiliary source multiplier scales the transport-returned equilibrium-source change about the fixture source: 0 keeps the fixture source and 1 applies the full returned source. Coordinate maps still come from the evolving geometry waveform.
Common knobs: iteration cap `10`, contraction threshold `0.80000000000000004`, hard iteration ceiling `20`, convergence and conservation tolerance `0.0050000000000000001`, initial damping `1`, damping floor `0.125`, equilibrium portfolio tolerance `9.9999999999999995e-07`. One-time fixture preparation: `43.355170930037275` s.

| regime | candidate | window (s) | auxiliary multiplier | outcome type | terminal limited/diverted core |
|---|---:|---:|---:|---|---:|
| strong | - | `0.01` | `1` | `ConvergedNonConfinedError` | `0/157` |
| gentle | 1 | `0.0025000000000000001` | `0.5` | `WindowReceipt` | `126/157` |

## Strong regime

Typed outcome: `ConvergedNonConfinedError` — `equilibrium portfolio converged without a selectable confined branch at exchange 3, sample 1; core cells limited/diverted = (0, 157); selection reason = no_admissible_alternative; availability limited/diverted = (True, False); residuals limited/diverted = (7.607266075789316e-17, 4.440379312410689e-16)`

The selector receipts are reproduced for every completed coarse sample:

| exchange | sample | limited core | diverted core | selected | verdict | limited/diverted available | limited/diverted residual |
|---:|---:|---:|---:|---|---|---|---|
| 1 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 1 | 1 | 133 | 157 | `limited` | `history_continuity` | `true/false` | `7.8711617624257372e-16/2.7524271573400452e-16` |
| 2 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 2 | 1 | 145 | 157 | `limited` | `history_continuity` | `true/false` | `9.8578413385004064e-16/4.3284153332244623e-16` |
| 3 | 1 | 0 | 157 | `limited` | `no_admissible_alternative` | `true/false` | `7.6072660757893158e-17/4.440379312410689e-16` |

Exact dense-lattice core counts for the strong trajectory:

| exchange | sample | core cells | extraction valid | exact evaluation (s) |
|---:|---:|---:|---|---:|
| 0 | 0 | 496 | `true` | `3.9895363501273096` |
| 1 | 0 | 496 | `true` | `4.5533757449593395` |
| 1 | 1 | 543 | `true` | `4.6035709080751985` |
| 2 | 0 | 496 | `true` | `4.7957245698198676` |
| 2 | 1 | 571 | `true` | `4.6915806508623064` |

## Gentle regime

Candidate 1 is the first converging candidate: window `0.0025000000000000001` s, auxiliary multiplier `0.5`.

- Iterations used: `4`
- Iterations past ordinary cap: `0`
- Measured gating-norm contraction estimate: `0.23619506584727215`
- Exit gating norm: `0.0012913749878219105`
- Exit all-field norm: `0.032597119023903123`
- Final damping applied: `1`

| iteration | gating norm | all-field norm |
|---:|---:|---:|
| 1 | `1` | `1.7437508268752859` |
| 2 | `0.024544392673843753` | `0.54435218854303113` |
| 3 | `0.0054674088266388095` | `0.13513523071726641` |
| 4 | `0.0012913749878219105` | `0.032597119023903123` |

Damping backoffs:

None.

Post-cap continuation licenses:

None.

| exchanged field | exit relative residual |
|---|---:|
| `geometry.radial_grid` | `0` |
| `geometry.phi_boundary` | `0.00030535540841767451` |
| `geometry.axis_reference` | `0.00010178972315078167` |
| `geometry.boundary_reference` | `0.00010331971991737405` |
| `geometry.b2_cell` | `6.6671701429416678e-05` |
| `geometry.b2_face` | `7.0723501738044484e-05` |
| `geometry.clipped_vertex_capacity` | `0` |
| `geometry.clipped_vertex_count_max` | `0` |
| `geometry.clipped_vertex_count_required` | `0` |
| `geometry.delta_lower_face` | `0.032597119023903123` |
| `geometry.delta_upper_face` | `0.032597119014098605` |
| `geometry.elongation_face` | `0.0087551597651442017` |
| `geometry.f_cell` | `7.4285631721010037e-06` |
| `geometry.f_face` | `7.7052621444414198e-06` |
| `geometry.flux_sign` | `0` |
| `geometry.g2_face` | `0.0002855851160726054` |
| `geometry.g3_cell` | `6.9856510375633239e-05` |
| `geometry.g3_face` | `7.4706723269202165e-05` |
| `geometry.grad_psi2_face` | `0.0004961433773465996` |
| `geometry.grad_psi2_over_r2_face` | `0.00046680466622597844` |
| `geometry.grad_psi_face` | `0.0002644587389717233` |
| `geometry.gradient_moment_scale` | `0` |
| `geometry.int_dl_over_bp_face` | `0.00031144257171987185` |
| `geometry.inv_b2_face` | `4.2994427616033168e-05` |
| `geometry.inv_r_cell` | `3.2483254852154883e-05` |
| `geometry.inv_r_face` | `3.3851808245314621e-05` |
| `geometry.ip_amperes` | `4.4874954805996797e-05` |
| `geometry.ip_profile_face` | `7.1707940344344109e-05` |
| `geometry.psi_face` | `0.00010202797586481682` |
| `geometry.psi_n_cell` | `9.8362039521557133e-06` |
| `geometry.psi_n_face` | `9.6608795928654345e-06` |
| `geometry.q_face` | `0.00038612602813525225` |
| `geometry.r0` | `1.7071198373957844e-05` |
| `geometry.r_in_face` | `5.3617395997331489e-05` |
| `geometry.r_out_face` | `8.6225012009799986e-06` |
| `geometry.rho_cell` | `0` |
| `geometry.shape_axis_expansion_face` | `0` |
| `geometry.shape_boundary_cell_count_face` | `0` |
| `geometry.surface_arc_first_invalid_cell` | `0` |
| `geometry.surface_arc_first_invalid_level` | `0` |
| `geometry.surface_arc_invalid_count` | `0` |
| `geometry.surface_arc_max_coarea_weight` | `0.00032428786542330179` |
| `geometry.surface_arc_min_ordinate_derivative` | `7.1025233507831145e-05` |
| `geometry.surface_arc_valid` | `0` |
| `geometry.surface_cell_band_capacity` | `0` |
| `geometry.surface_cell_band_max_count` | `0` |
| `geometry.surface_cell_band_overflow` | `0` |
| `geometry.valid` | `0` |
| `geometry.volume` | `0.00025577672549128568` |
| `geometry.volume_face` | `0.00025577672549128568` |
| `geometry.vpr_cell` | `0.00023528879507596305` |
| `geometry.vpr_face` | `0.00023066591744312875` |
| `source.radial_grid` | `0` |
| `source.phi_boundary` | `0.0012913749878219105` |
| `source.axis_reference` | `0.00043226015952088084` |
| `source.boundary_reference` | `0.0004306267833712184` |
| `source.boundary_field_function` | `0` |
| `source.boundary_pressure` | `0` |
| `source.ff_prime` | `0.00054968164897094989` |
| `source.p_prime` | `5.7945631980519032e-05` |

### Conservation ledgers

- Flux consumption boundary/resistive/internal: `0.0031404662217260348` / `0.00043152532506729457` / `0.0027089408966587403` Wb.
- Flux closure absolute/relative: `0` Wb / `0`.
- Plasma current requested initial/final: `557455.09288717515` / `557455.09288717515` A; achieved initial/final: `557455.09288717504` / `557455.09288717504` A.
- Boundary current continuity absolute/relative: `1.1641532182693481e-10` A / `2.0883354249038394e-16`.

## Wall time per exchange sweep

| regime | candidate | exchange | side | wall time (s) |
|---|---:|---:|---|---:|
| strong | - | 1 | transport | `7.0271244239993393` |
| strong | - | 1 | equilibrium_plus_fsa | `22.377076998120174` |
| strong | - | 2 | transport | `0.097786694997921586` |
| strong | - | 2 | equilibrium_plus_fsa | `18.19521766086109` |
| strong | - | 3 | transport | `0.072609805967658758` |
| strong | - | 3 | equilibrium | `2.509356256108731` |
| gentle | 1 | 1 | transport | `0.040485315956175327` |
| gentle | 1 | 1 | equilibrium_plus_fsa | `18.070479260990396` |
| gentle | 1 | 2 | transport | `0.060937131987884641` |
| gentle | 1 | 2 | equilibrium_plus_fsa | `17.488553053932264` |
| gentle | 1 | 3 | transport | `0.050803480902686715` |
| gentle | 1 | 3 | equilibrium_plus_fsa | `16.373083757935092` |
| gentle | 1 | 4 | transport | `0.054468614980578423` |
| gentle | 1 | 4 | equilibrium_plus_fsa | `15.928435167996213` |

The TSV is the machine-readable record of every declared knob, attempt, selector receipt, exit residual, timing, exact extraction diagnostic and available transport ledger.
