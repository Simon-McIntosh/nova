# Coupled-window receipts across two regimes

This is one locked strong-window run and a bounded gentle search. Every row is from `solve_window`; typed refusals are reported and were not retried with altered tolerances.

## Experiment contract

The equilibrium is the repository's 25 x 25 free-boundary fixture. Every accepted sample was evaluated exactly on a 49 x 49 lattice and extracted into 8 TORAX radial cells with 14 surface bins. TORAX advances all four transport channels in one fixed step per window.
Fixture-scale execution backend: `cpu`.
The auxiliary source multiplier scales the transport-returned equilibrium-source change about the fixture source: 0 keeps the fixture source and 1 applies the full returned source. Coordinate maps still come from the evolving geometry waveform.
Common knobs: iteration cap `10`, contraction threshold `0.80000000000000004`, hard iteration ceiling `20`, convergence and conservation tolerance `0.0050000000000000001`, damping `0.5`, equilibrium portfolio tolerance `9.9999999999999995e-07`. One-time fixture preparation: `48.193888757145032` s.

| regime | candidate | window (s) | auxiliary multiplier | outcome type | terminal limited/diverted core |
|---|---:|---:|---:|---|---:|
| strong | - | `0.01` | `1` | `ConvergedNonConfinedError` | `0/157` |
| gentle | 1 | `0.0025000000000000001` | `0.5` | `WindowReceipt` | `126/157` |

## Strong regime: typed boundary outcome

`equilibrium portfolio converged without a selectable confined branch at exchange 5, sample 1; core cells limited/diverted = (0, 157); selection reason = no_admissible_alternative; availability limited/diverted = (True, False); residuals limited/diverted = (7.607266075789316e-17, 2.942137576018049e-16)`

The selector receipts are reproduced for every completed coarse sample:

| exchange | sample | limited core | diverted core | selected | verdict | limited/diverted available | limited/diverted residual |
|---:|---:|---:|---:|---|---|---|---|
| 1 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 1 | 1 | 133 | 157 | `limited` | `history_continuity` | `true/false` | `7.8711617624257372e-16/2.7524271573400452e-16` |
| 2 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 2 | 1 | 135 | 157 | `limited` | `history_continuity` | `true/false` | `7.8886971127455822e-16/4.2285691316753316e-16` |
| 3 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 3 | 1 | 145 | 157 | `limited` | `history_continuity` | `true/false` | `1.1825041348092582e-15/1.435221847925257e-16` |
| 4 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 4 | 1 | 145 | 157 | `limited` | `history_continuity` | `true/false` | `9.8676334731307494e-16/4.3669431375303254e-16` |
| 5 | 1 | 0 | 157 | `limited` | `no_admissible_alternative` | `true/false` | `7.6072660757893158e-17/2.9421375760180488e-16` |

Exact dense-lattice core counts for the strong trajectory:

| exchange | sample | core cells | extraction valid | exact evaluation (s) |
|---:|---:|---:|---|---:|
| 0 | 0 | 496 | `true` | `3.9142980920150876` |
| 1 | 0 | 496 | `true` | `4.9090882882010192` |
| 1 | 1 | 543 | `true` | `4.4507547961547971` |
| 2 | 0 | 496 | `true` | `4.4026689489837736` |
| 2 | 1 | 553 | `true` | `4.9027759011369199` |
| 3 | 0 | 496 | `true` | `5.1078897209372371` |
| 3 | 1 | 569 | `true` | `4.3043134671170264` |
| 4 | 0 | 496 | `true` | `4.2100274572148919` |
| 4 | 1 | 576 | `true` | `4.2802224180195481` |

## Gentle regime

Candidate 1 is the first converging candidate: window `0.0025000000000000001` s, auxiliary multiplier `0.5`.

- Iterations used: `14`
- Iterations past ordinary cap: `4`
- Measured contraction estimate: `0.62828535026498111`
- Maximum exit residual: `0.0044476157928988042`
- Damping applied: `0.5`

| licensed iteration | licensing contraction |
|---:|---:|
| 11 | `0.64062864104278661` |
| 12 | `0.63670887334799353` |
| 13 | `0.6333718270632761` |
| 14 | `0.63058448998687555` |

| exchanged field | exit relative residual |
|---|---:|
| `geometry.radial_grid` | `0` |
| `geometry.phi_boundary` | `4.4415406401616936e-05` |
| `geometry.axis_reference` | `1.3017393491389535e-05` |
| `geometry.boundary_reference` | `1.3120434430809395e-05` |
| `geometry.b2_cell` | `9.403588034699377e-06` |
| `geometry.b2_face` | `9.939480319772365e-06` |
| `geometry.clipped_vertex_capacity` | `0` |
| `geometry.clipped_vertex_count_max` | `0` |
| `geometry.clipped_vertex_count_required` | `0` |
| `geometry.delta_lower_face` | `0.0044476157874473758` |
| `geometry.delta_upper_face` | `0.0044476157928988042` |
| `geometry.elongation_face` | `0.0012451026743311568` |
| `geometry.f_cell` | `7.8876212455853512e-07` |
| `geometry.f_face` | `8.0766739264411392e-07` |
| `geometry.flux_sign` | `0` |
| `geometry.g2_face` | `4.1589395441188818e-05` |
| `geometry.g3_cell` | `9.8302685653433845e-06` |
| `geometry.g3_face` | `1.0473329498453768e-05` |
| `geometry.grad_psi2_face` | `6.9101811135697429e-05` |
| `geometry.grad_psi2_over_r2_face` | `6.436869105343849e-05` |
| `geometry.grad_psi_face` | `3.6741883192629592e-05` |
| `geometry.gradient_moment_scale` | `0` |
| `geometry.int_dl_over_bp_face` | `4.3204460316037642e-05` |
| `geometry.inv_b2_face` | `5.9272676787802358e-06` |
| `geometry.inv_r_cell` | `4.557638426190045e-06` |
| `geometry.inv_r_face` | `4.7282064426450598e-06` |
| `geometry.ip_amperes` | `3.7975151443426934e-06` |
| `geometry.ip_profile_face` | `9.1415302245400597e-06` |
| `geometry.psi_face` | `1.3036548296696459e-05` |
| `geometry.psi_n_cell` | `1.1896610669677813e-06` |
| `geometry.psi_n_face` | `1.2156908001520472e-06` |
| `geometry.q_face` | `5.3663904255854159e-05` |
| `geometry.r0` | `2.71467923714066e-06` |
| `geometry.r_in_face` | `7.7649093536762816e-06` |
| `geometry.r_out_face` | `1.4451808423876361e-06` |
| `geometry.rho_cell` | `0` |
| `geometry.shape_axis_expansion_face` | `0.0001220703125` |
| `geometry.shape_boundary_cell_count_face` | `7.1806066176470587e-06` |
| `geometry.surface_arc_first_invalid_cell` | `1.1839991513094084e-07` |
| `geometry.surface_arc_first_invalid_level` | `0` |
| `geometry.surface_arc_invalid_count` | `0` |
| `geometry.surface_arc_max_coarea_weight` | `3.9816249558440556e-05` |
| `geometry.surface_arc_min_ordinate_derivative` | `8.8757140138995891e-06` |
| `geometry.surface_arc_valid` | `0` |
| `geometry.surface_cell_band_capacity` | `0` |
| `geometry.surface_cell_band_max_count` | `7.1806066176470587e-06` |
| `geometry.surface_cell_band_overflow` | `0` |
| `geometry.valid` | `0` |
| `geometry.volume` | `3.7215671834995143e-05` |
| `geometry.volume_face` | `3.7215671834995143e-05` |
| `geometry.vpr_cell` | `3.4574092388290741e-05` |
| `geometry.vpr_face` | `3.3961718546384382e-05` |
| `source.radial_grid` | `0` |
| `source.phi_boundary` | `0.00017757883562106327` |
| `source.axis_reference` | `3.5292582080487258e-05` |
| `source.boundary_reference` | `3.5538361691327351e-05` |
| `source.boundary_field_function` | `0` |
| `source.boundary_pressure` | `0.00012207031250004681` |
| `source.ff_prime` | `5.7486222829240755e-05` |
| `source.p_prime` | `5.8469225687579077e-05` |

### Conservation ledgers

- Flux consumption boundary/resistive/internal: `0.003139013452177597` / `0.00043145493144480795` / `0.002707558520732789` Wb.
- Flux closure absolute/relative: `0` Wb / `0`.
- Plasma current requested initial/final: `557455.09288717515` / `557455.09288717515` A; achieved initial/final: `557455.09288717515` / `557455.09288717515` A.
- Boundary current continuity absolute/relative: `0` A / `0`.

## Wall time per exchange sweep

| regime | candidate | exchange | side | wall time (s) |
|---|---:|---:|---|---:|
| strong | - | 1 | transport | `5.9345995120238513` |
| strong | - | 1 | equilibrium_plus_fsa | `21.841352099087089` |
| strong | - | 2 | transport | `0.066359722055494785` |
| strong | - | 2 | equilibrium_plus_fsa | `17.715575237059966` |
| strong | - | 3 | transport | `0.066712794126942754` |
| strong | - | 3 | equilibrium_plus_fsa | `17.518458364997059` |
| strong | - | 4 | transport | `0.06476939283311367` |
| strong | - | 4 | equilibrium_plus_fsa | `16.492603438906372` |
| strong | - | 5 | transport | `0.060617461102083325` |
| strong | - | 5 | equilibrium | `1.9969096800778061` |
| gentle | 1 | 1 | transport | `0.043094367953017354` |
| gentle | 1 | 1 | equilibrium_plus_fsa | `16.620705739129335` |
| gentle | 1 | 2 | transport | `0.060765814036130905` |
| gentle | 1 | 2 | equilibrium_plus_fsa | `16.052286321064457` |
| gentle | 1 | 3 | transport | `0.054909802973270416` |
| gentle | 1 | 3 | equilibrium_plus_fsa | `15.84835394192487` |
| gentle | 1 | 4 | transport | `0.049401703057810664` |
| gentle | 1 | 4 | equilibrium_plus_fsa | `15.734567251987755` |
| gentle | 1 | 5 | transport | `0.063791248947381973` |
| gentle | 1 | 5 | equilibrium_plus_fsa | `15.844337963033468` |
| gentle | 1 | 6 | transport | `0.062693137908354402` |
| gentle | 1 | 6 | equilibrium_plus_fsa | `15.958449165103957` |
| gentle | 1 | 7 | transport | `0.066495470935478806` |
| gentle | 1 | 7 | equilibrium_plus_fsa | `15.981956381816417` |
| gentle | 1 | 8 | transport | `0.055696059949696064` |
| gentle | 1 | 8 | equilibrium_plus_fsa | `16.148211194900796` |
| gentle | 1 | 9 | transport | `0.056016575079411268` |
| gentle | 1 | 9 | equilibrium_plus_fsa | `16.030918519943953` |
| gentle | 1 | 10 | transport | `0.053867449052631855` |
| gentle | 1 | 10 | equilibrium_plus_fsa | `20.661359562072903` |
| gentle | 1 | 11 | transport | `0.078426192980259657` |
| gentle | 1 | 11 | equilibrium_plus_fsa | `17.205525039928034` |
| gentle | 1 | 12 | transport | `0.050097409868612885` |
| gentle | 1 | 12 | equilibrium_plus_fsa | `16.783641833811998` |
| gentle | 1 | 13 | transport | `0.052862870041280985` |
| gentle | 1 | 13 | equilibrium_plus_fsa | `16.649420274887234` |
| gentle | 1 | 14 | transport | `0.063108789036050439` |
| gentle | 1 | 14 | equilibrium_plus_fsa | `17.402722535189241` |

The TSV is the machine-readable record of every declared knob, attempt, selector receipt, exit residual, timing, exact extraction diagnostic and available transport ledger.
