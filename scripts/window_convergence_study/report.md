# Window convergence composition study

## Outcome

The band-only Git tree `c3d3d0b7e7bb7393ec040fe78f392bb6bc580f0d` measured contraction `0.53758678955385919`; the band-plus-near-axis-shape tree `c05a83a6cc1ee0368c7c9e97d7656ecb778ce8bd` measured `0.64062864104278661`. The isolated shape contribution is therefore `+0.10304185148892742`.

The study changes no product file. Every run used the same gentle window (0.0025 s, source multiplier 0.5, tolerance 0.005, ordinary cap 10); current-tree runs additionally used the landed contraction-licensed hard ceiling 20. Runs were fresh processes and strictly serialized on the login node.

## Attribution

| state | Git tree SHA | contraction | interpretation |
|---|---|---:|---|
| band only | `c3d3d0b7e7bb7393ec040fe78f392bb6bc580f0d` | `0.53758678955385919` | sparsified contraction without the near-axis shape route |
| band + near-axis shape | `c05a83a6cc1ee0368c7c9e97d7656ecb778ce8bd` | `0.64062864104278661` | same band plus the shape representation landing |

The band-only value remains on the 0.537 branch while adding the shape route moves it to the 0.641 branch; the shift is attributed to the near-axis shape landing, not band sparsification.

## Gating composition

All variants are post-processing of the same exchanged waveforms. Shape fields remain exchanged; only their contribution to the hypothetical stopping norm changes. No criterion was implemented in the product.

| trace | all fields | shape weight 0.25 | shape excluded | saved |
|---|---:|---:|---:|---:|
| landed converged | `14` | `11` | `9` | `0 / 3 / 5` |
| landed cap-10 | `>10` | `>10` | `9` | `n/a` |
| fresh instrumented | `14` | `11` | `9` | `0 / 3 / 5` |

The TSV names the gating field at every iteration for the landed and fresh traces. The weakly coupled shape set is exactly `delta_lower_face`, `delta_upper_face`, `elongation_face`, `r_in_face`, and `r_out_face`, plus the representation channels `shape_axis_expansion_face` and `shape_boundary_cell_count_face` on the geometry waveform.
Under the all-field norm, `delta_lower_face` gates iterations 1-12 and `delta_upper_face` gates 13-14. At shape weight 0.25, `source.boundary_pressure` gates 1-5 before the triangularity channels take over. With shape excluded, `source.boundary_pressure` gates 1-12 and `source.phi_boundary` gates 13-14.

## Damping and acceleration

| damping | iterations to tolerance | window wall (s) | contraction | outcome |
|---:|---:|---:|---:|---|
| `0.5` | `14` | `233.024273` | `0.62828535026498111` | `WindowReceipt` |
| `0.69999999999999996` | `10` | `172.386360` | `0.4718021749442266` | `WindowReceipt` |
| `1` | `6` | `108.237422` | `0.23723366557491143` | `WindowReceipt` |

Against damping 0.5, damping 0.7 saves four exchanges and is 1.35176x faster by window wall; damping 1.0 saves eight exchanges and is 2.15290x faster. These are one-run measurements, not stability statistics.

Existing Anderson compatibility: `TracerArrayConversionError`. The probe returned `The numpy.ndarray conversion method __array__() was called on traced array with shape float64[2]`. The existing accelerator requires a flat traced JAX state and traces its map through a fixed-shape loop; the window exchange owns immutable NumPy `Waveform` objects and host-bearing equilibrium/TORAX sweeps. It therefore does not accept this exchange without product adaptation, so no Anderson timing is claimed.

## Provenance and decision boundary

The band-only arm is a synthetic Git tree formed from the pre-shape tree `80a4aa2b67a656af5ae25cbfb3dfaa2ff66809f1` plus the exact product patch from band commit `32942ac3861af0b95dd19cf279e4b3ab211fa705`. The combined arm is the tree of that band commit, whose ancestry contains the near-axis shape landing. Both were executed from `git archive` overlays against the repository root environment. The criterion variants are measurements for an owner decision; this study neither selects nor implements one.
