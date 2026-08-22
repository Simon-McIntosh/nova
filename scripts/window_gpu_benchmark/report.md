# CPU--H200 single-sweep sensitivity isolation

SLURM job `1253092` replayed one host-serialized iteration-1 input (`sha256:fee1c18196eec3e018404480d4083cc3bcfad2222629a0d49c3a5f30e90b1d57`) through one equilibrium sweep and exact dense-lattice extraction on CPU and NVIDIA H200 in float64.

## Solved flux maps first

| sample | map | bitwise | max absolute | max relative | max ulp scale | location |
|---:|---|---|---:|---:|---:|---|
| 0 | `equilibrium_flux` | `False` | `9.681144774731365e-14` | `5.7455539967499685e-14` | `436` | `(342,)` |
| 0 | `evaluated_flux` | `False` | `1.0658141036401503e-13` | `6.4400483171195618e-14` | `480` | `(34, 25)` |
| 0 | `fixed_point_residual` | `False` | `3.1142913629523239e-15` | `0.176136363636371` | `986958328883026` | `()` |
| 0 | `axis_flux` | `False` | `1.7763568394002505e-15` | `8.5275178828318296e-16` | `4` | `()` |
| 0 | `boundary_flux` | `False` | `6.6613381477509392e-16` | `4.3400453755346216e-16` | `3` | `()` |
| 0 | `axis_radius` | `False` | `6.6613381477509392e-16` | `6.5826377494067725e-16` | `3` | `()` |
| 0 | `axis_height` | `False` | `2.5270204765531767e-14` | `2.3273902500647198e-10` | `1864612` | `()` |
| 0 | `core_cells` | `True` | `0` | `0` | `0` | `()` |
| 1 | `equilibrium_flux` | `False` | `4.9071857688431919e-14` | `2.879587110614374e-14` | `221` | `(307,)` |
| 1 | `evaluated_flux` | `False` | `4.9737991503207013e-14` | `2.9353647706078582e-14` | `224` | `(33, 25)` |
| 1 | `fixed_point_residual` | `False` | `4.9303806576313238e-30` | `4.9281246421977366e-15` | `25` | `()` |
| 1 | `axis_flux` | `False` | `1.7763568394002505e-15` | `8.4665978709275379e-16` | `4` | `()` |
| 1 | `boundary_flux` | `False` | `2.2204460492503131e-16` | `1.4369534672959723e-16` | `1` | `()` |
| 1 | `axis_radius` | `False` | `2.2204460492503131e-16` | `2.2014069332034902e-16` | `1` | `()` |
| 1 | `axis_height` | `False` | `1.3130216857358545e-14` | `1.584992583243289e-10` | `968839` | `()` |
| 1 | `core_cells` | `True` | `0` | `0` | `0` | `()` |

The first compared layer already differs in the solved equilibrium flux at sample 0: maximum absolute difference `9.681144774731365e-14` (`436` local ulp) at coefficient index `342`. Exact dense-lattice evaluation raises the maximum only to `1.0658141036401503e-13` (`480` local ulp) at map cell `(34,25)`; this is float64 code-generation noise, not the historical residual shift.

## First discrete decision

The first discrete difference is `extremum_selection` / `z_upper` at level `0.20874999999999999`, flattened cell `1416` (`row=29`, `column=24`) on both backends: CPU selected sample slot `39`, H200 selected adjacent slot `38`.

This is a code-generation-scale tie between adjacent samples of the same cell. It changes a reported geometric extremum, not the toroidal-flux integral, so it does not explain the historical boundary-coordinate residual gap. If bitwise shape-selection repeatability is required, the repair is an order-independent tie rule; widening the window tolerance is rejected.

## Downstream toroidal-flux amplification

| sample | CPU Phi_b | H200 Phi_b | absolute difference | relative difference |
|---:|---:|---:|---:|---:|
| 0 | `0.75453222110088702` | `0.75453222110088314` | `3.8857805861880479e-15` | `5.149920013380698e-15` |
| 1 | `0.7735177443586172` | `0.77351774435860654` | `1.0658141036401503e-14` | `1.3778793200457197e-14` |

| sample | Phi path layer | bitwise | max absolute | max relative | location |
|---:|---|---|---:|---:|---|
| 0 | `volume_derivative` | `False` | `4.3520742565306136e-14` | `6.3493223124319502e-14` | `(0,)` |
| 0 | `inverse_radius_squared` | `False` | `1.3322676295501878e-15` | `1.2463466168039417e-15` | `(13,)` |
| 0 | `field_function_surface` | `False` | `8.8817841970012523e-16` | `1.7678817598956161e-16` | `(2,)` |
| 0 | `phi_integrand` | `False` | `3.3750779948604759e-14` | `6.2727590816313119e-14` | `(0,)` |
| 0 | `phi_integrand_edge` | `False` | `9.3258734068513149e-15` | `7.5034131335210921e-15` | `()` |
| 0 | `phi_boundary` | `False` | `3.8857805861880479e-15` | `5.149920013380698e-15` | `()` |
| 1 | `volume_derivative` | `False` | `5.3512749786932545e-14` | `7.6699414160660657e-14` | `(0,)` |
| 1 | `inverse_radius_squared` | `False` | `6.6613381477509392e-16` | `6.6071368625398145e-16` | `(5,)` |
| 1 | `field_function_surface` | `True` | `0` | `0` | `(0,)` |
| 1 | `phi_integrand` | `False` | `4.2077452633293433e-14` | `7.6272481506018136e-14` | `(0,)` |
| 1 | `phi_integrand_edge` | `False` | `2.886579864025407e-15` | `2.280898416151738e-15` | `()` |
| 1 | `phi_boundary` | `False` | `1.0658141036401503e-14` | `1.3778793200457197e-14` | `()` |

The first non-bitwise Phi-path quantity is `volume_derivative`, differing by at most `4.3520742565306136e-14`. Successive surface weights and integrand values remain at float64 code-generation noise scale through Phi_b (`3.8857805861880479e-15` at sample 0 and `1.0658141036401503e-14` at sample 1); there is no three-parts-in-ten-thousand amplification.

The machine-readable TSV carries every per-level topology population, band count, membership flip, clip-vertex flip, extremum selection, surface integrand comparison and the cell/level payload needed to trace the first changed decision into Phi_b.

## Residual provenance resolves the apparent 3.43e-4 gap

The quantity previously called `geometry.phi_boundary` was the iteration-one relative waveform residual, not raw Phi_b. Reconstructing that residual from the same serialized coordinate input gives:

| comparison | left | right | absolute gap |
|---|---:|---:|---:|
| same-tree CPU vs H200 | `0.024544392673843753` | `0.024544392673827374` | `1.6379259060173013e-14` |
| pre-band CPU receipt vs current CPU | `0.024887102508608275` | `0.024544392673843753` | `0.00034270983476452245` |

The historical comparison joined a CPU receipt produced before boundary-band sparsification with an H200 receipt produced after commit `32942ac3861af0b95dd19cf279e4b3ab211fa705`. Its full reported CPU–H200 gap (`0.00034270983476212061`) is reproduced by that cross-revision CPU comparison to within `2.4018e-15`. On one current tree, CPU and H200 agree at float64 noise scale.

Therefore no backend-dependent discrete decision moves the boundary coordinate by 3.43e-4. The confounding algorithmic change packs only level-bracketing cells for clipping and separately accumulates fully included cells, changing floating-point accumulation order while preserving the mathematical integral. The first actual same-code discrete flip is the harmless extrema tie above; it is not on the Phi_b integration path.

## Runtime and window-policy context

CPU sweep plus extraction: `44.476793549023569` s; H200: `67.010981366038322` s.
The landed contraction evidence remains pre-band CPU `0.5371039633417938` versus current-tree H200 `0.6406286426029119` at cap `10`, tolerance `0.005` and damping `0.5`. Because those values also cross the extraction revision, whether the cap should become contraction-aware remains a design question requiring a same-tree window comparison; this probe does not change the cap or tolerance.
