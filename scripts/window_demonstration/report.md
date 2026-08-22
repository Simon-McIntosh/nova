# Coupled-window receipts across two regimes

This is one locked strong-window run and a bounded gentle search. Every row is from `solve_window`; typed refusals are reported and were not retried with altered tolerances.

## Experiment contract

The equilibrium is the repository's 25 x 25 free-boundary fixture. Every accepted sample was evaluated exactly on a 49 x 49 lattice and extracted into 8 TORAX radial cells with 14 surface bins. TORAX advances all four transport channels in one fixed step per window.
Fixture-scale execution backend: `cpu`.
The auxiliary source multiplier scales the transport-returned equilibrium-source change about the fixture source: 0 keeps the fixture source and 1 applies the full returned source. Coordinate maps still come from the evolving geometry waveform.
Common knobs: iteration cap `10`, convergence and conservation tolerance `0.0050000000000000001`, damping `0.5`, equilibrium portfolio tolerance `9.9999999999999995e-07`. One-time fixture preparation: `55.001441712956876` s.

| regime | candidate | window (s) | auxiliary multiplier | outcome type | terminal limited/diverted core |
|---|---:|---:|---:|---|---:|
| strong | - | `0.01` | `1` | `ConvergedNonConfinedError` | `0/157` |
| gentle | 1 | `0.0025000000000000001` | `0.5` | `WindowReceipt` | `126/157` |

## Strong regime: typed boundary outcome

`equilibrium portfolio converged without a selectable confined branch at exchange 5, sample 1; core cells limited/diverted = (0, 157); selection reason = no_admissible_alternative; availability limited/diverted = (True, False); residuals limited/diverted = (7.607266075789316e-17, 4.4197234248998465e-16)`

The selector receipts are reproduced for every completed coarse sample:

| exchange | sample | limited core | diverted core | selected | verdict | limited/diverted available | limited/diverted residual |
|---:|---:|---:|---:|---|---|---|---|
| 1 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 1 | 1 | 133 | 157 | `limited` | `history_continuity` | `true/false` | `1.1807326829759295e-15/4.1314148218640533e-16` |
| 2 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 2 | 1 | 139 | 157 | `limited` | `history_continuity` | `true/false` | `1.9700427553286153e-15/8.464117776286397e-16` |
| 3 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 3 | 1 | 145 | 157 | `limited` | `history_continuity` | `true/false` | `7.8841951755872654e-16/1.0063048270005709e-15` |
| 4 | 0 | 126 | 157 | `limited` | `sole_valid` | `true/false` | `1.4566846697679485e-14/4.0756513704185692e-16` |
| 4 | 1 | 145 | 157 | `limited` | `history_continuity` | `true/false` | `8.8820024954242939e-16/5.8316109414077684e-16` |
| 5 | 1 | 0 | 157 | `limited` | `no_admissible_alternative` | `true/false` | `7.6072660757893158e-17/4.4197234248998465e-16` |

Exact dense-lattice core counts for the strong trajectory:

| exchange | sample | core cells | extraction valid | exact evaluation (s) |
|---:|---:|---:|---|---:|
| 0 | 0 | 496 | `true` | `3.8709240639582276` |
| 1 | 0 | 496 | `true` | `4.4076768769882619` |
| 1 | 1 | 543 | `true` | `4.4080723340157419` |
| 2 | 0 | 496 | `true` | `4.3863747150171548` |
| 2 | 1 | 559 | `true` | `4.4531851599458605` |
| 3 | 0 | 496 | `true` | `4.4531181808561087` |
| 3 | 1 | 569 | `true` | `4.4946686390321702` |
| 4 | 0 | 496 | `true` | `4.4403217760846019` |
| 4 | 1 | 576 | `true` | `4.3498209430836141` |

## Gentle regime

Candidate 1 is the first converging candidate: window `0.0025000000000000001` s, auxiliary multiplier `0.5`.

- Iterations used: `10`
- Measured contraction estimate: `0.53710396334179378`
- Maximum exit residual: `0.0049860186161842365`
- Damping applied: `0.5`

| exchanged field | exit relative residual |
|---|---:|
| `geometry.radial_grid` | `0` |
| `geometry.phi_boundary` | `0.00030926807100344961` |
| `geometry.axis_reference` | `7.4297520823139713e-05` |
| `geometry.boundary_reference` | `7.4003504924477503e-05` |
| `geometry.b2_cell` | `6.2746177844996744e-05` |
| `geometry.b2_face` | `6.6064887965290622e-05` |
| `geometry.clipped_vertex_capacity` | `0` |
| `geometry.clipped_vertex_count_max` | `0` |
| `geometry.clipped_vertex_count_required` | `0` |
| `geometry.delta_lower_face` | `0.0049860186156783313` |
| `geometry.delta_upper_face` | `0.0049860186161842365` |
| `geometry.elongation_face` | `4.6548105492172183e-05` |
| `geometry.f_cell` | `2.9596536732913224e-06` |
| `geometry.f_face` | `2.997263250536417e-06` |
| `geometry.flux_sign` | `0` |
| `geometry.g2_face` | `0.00028988151251776941` |
| `geometry.g3_cell` | `6.5404044924835241e-05` |
| `geometry.g3_face` | `6.9362267820176929e-05` |
| `geometry.grad_psi2_face` | `0.00045482379636984554` |
| `geometry.grad_psi2_over_r2_face` | `0.0004175832288633455` |
| `geometry.grad_psi_face` | `0.0002409795382426533` |
| `geometry.gradient_moment_scale` | `0` |
| `geometry.int_dl_over_bp_face` | `0.00028070088351724319` |
| `geometry.inv_b2_face` | `3.8316545743940907e-05` |
| `geometry.inv_r_cell` | `3.0192484533605124e-05` |
| `geometry.inv_r_face` | `3.1148316757702154e-05` |
| `geometry.ip_amperes` | `1.6644983812728374e-06` |
| `geometry.ip_profile_face` | `5.2626104848436899e-05` |
| `geometry.psi_face` | `7.4322481651418264e-05` |
| `geometry.psi_n_cell` | `6.2737209200017782e-06` |
| `geometry.psi_n_face` | `6.8391454284694175e-06` |
| `geometry.q_face` | `0.00034984822950073503` |
| `geometry.r0` | `2.1014131864567882e-05` |
| `geometry.r_in_face` | `5.3780313940384109e-05` |
| `geometry.r_out_face` | `1.1861546284144304e-05` |
| `geometry.rho_cell` | `0` |
| `geometry.surface_arc_first_invalid_cell` | `0` |
| `geometry.surface_arc_first_invalid_level` | `0` |
| `geometry.surface_arc_invalid_count` | `0` |
| `geometry.surface_arc_max_coarea_weight` | `0.00020971265553732091` |
| `geometry.surface_arc_min_ordinate_derivative` | `0.00013101663180541413` |
| `geometry.surface_arc_valid` | `0` |
| `geometry.valid` | `0` |
| `geometry.volume` | `0.00025994534260029593` |
| `geometry.volume_face` | `0.00025994534260029593` |
| `geometry.vpr_cell` | `0.00024399981314617825` |
| `geometry.vpr_face` | `0.00024021759224857056` |
| `source.radial_grid` | `0` |
| `source.phi_boundary` | `0.0011318871778522089` |
| `source.axis_reference` | `0.0001111532673261033` |
| `source.boundary_reference` | `0.00011295514933840653` |
| `source.boundary_field_function` | `0` |
| `source.boundary_pressure` | `0.0019531250000000395` |
| `source.ff_prime` | `0.00037457637327342389` |
| `source.p_prime` | `0.00096568121897497414` |

### Conservation ledgers

- Flux consumption boundary/resistive/internal: `0.0031385773159375852` / `0.00043144854900178942` / `0.0027071287669357957` Wb.
- Flux closure absolute/relative: `0` Wb / `0`.
- Plasma current requested initial/final: `557455.09288717515` / `557455.09288717515` A; achieved initial/final: `557455.09288717515` / `557455.09288717515` A.
- Boundary current continuity absolute/relative: `0` A / `0`.

## Wall time per exchange sweep

| regime | candidate | exchange | side | wall time (s) |
|---|---:|---:|---|---:|
| strong | - | 1 | transport | `5.8675248131621629` |
| strong | - | 1 | equilibrium_plus_fsa | `46.657166521064937` |
| strong | - | 2 | transport | `0.05290511786006391` |
| strong | - | 2 | equilibrium_plus_fsa | `42.651266149943694` |
| strong | - | 3 | transport | `0.050963010871782899` |
| strong | - | 3 | equilibrium_plus_fsa | `43.026880852179602` |
| strong | - | 4 | transport | `0.059855333995074034` |
| strong | - | 4 | equilibrium_plus_fsa | `43.833987721940503` |
| strong | - | 5 | transport | `0.06599286594428122` |
| strong | - | 5 | equilibrium | `1.7779766409657896` |
| gentle | 1 | 1 | transport | `0.047859359998255968` |
| gentle | 1 | 1 | equilibrium_plus_fsa | `42.905863158870488` |
| gentle | 1 | 2 | transport | `0.062670045997947454` |
| gentle | 1 | 2 | equilibrium_plus_fsa | `43.112658370984718` |
| gentle | 1 | 3 | transport | `0.061839377973228693` |
| gentle | 1 | 3 | equilibrium_plus_fsa | `42.818313711090013` |
| gentle | 1 | 4 | transport | `0.060834288131445646` |
| gentle | 1 | 4 | equilibrium_plus_fsa | `42.003369207959622` |
| gentle | 1 | 5 | transport | `0.059510410064831376` |
| gentle | 1 | 5 | equilibrium_plus_fsa | `41.927456183126196` |
| gentle | 1 | 6 | transport | `0.061132893897593021` |
| gentle | 1 | 6 | equilibrium_plus_fsa | `42.253127231029794` |
| gentle | 1 | 7 | transport | `0.064742129994556308` |
| gentle | 1 | 7 | equilibrium_plus_fsa | `42.113762564025819` |
| gentle | 1 | 8 | transport | `0.057275077793747187` |
| gentle | 1 | 8 | equilibrium_plus_fsa | `41.957216871203855` |
| gentle | 1 | 9 | transport | `0.044893278041854501` |
| gentle | 1 | 9 | equilibrium_plus_fsa | `41.84535274701193` |
| gentle | 1 | 10 | transport | `0.057390824891626835` |
| gentle | 1 | 10 | equilibrium_plus_fsa | `41.517448354978114` |

The TSV is the machine-readable record of every declared knob, attempt, selector receipt, exit residual, timing, exact extraction diagnostic and available transport ledger.
