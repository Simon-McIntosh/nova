# First-order coupling term: exact-Biot discriminator

Node `cca-first-order-coupling-discriminator`, run r-20260910T225638998875.
One `all_debug` sbatch (job 1269154, CPU float64, revision 2751ae4d) ran
`benchmarks/first_order_coupling_discriminator.py --run-all`; receipts committed
at 2751ae4d + 6db0b468 under `docs/figures/cut-cell-current-attribution/first-order-coupling/`.

## Cause, one sentence

Not a sign or unit error in the kernel: the single-cell control reproduces the exact
flux of a prescribed linear density to machine precision (2.2e-13 rms outside the
cell, vs 6.5e-4 zeroth-only and 1.3e-3 with the first moments sign-flipped, exactly
2.000x), and the frozen-image degradation is entirely the cut cells' first-order
term (cut-cells-only rms-over-span 0.0101 vs interior-cells-only 0.0061 on a 0.0057
zeroth baseline) — a dipole applied against the full cell while the current occupies
only the clipped plasma region, which measures like a sign error (flipping the
physical first moments removes the whole added error, 0.0106 -> 0.0051, below
zeroth-only) while re-referencing the moments and kernel blocks to the region
centroid does not (0.0106).

## Part one — single cell (interior hex of the weak -110 mesh)

Cell 30 (centroid R,Z = 4.469, -1.252 m; M0 = 1.452e5 A, MR = 194.2 A m,
MZ = 0.306 A m) carries j = g0 + gR(R-Rc) + gZ(Z-Zc) (least-squares linearisation
of the exact density). Its exact total poloidal flux at every mesh node is
computed by tensor-Duffy quadrature of the ring mutual inductance
(M/(Wb/A) filament, same total-flux convention as the analytic oracle and kernel,
validated against both shipped kernels). Variants vs exact flux, nodes outside the
source cell:

| variant | rms (Wb) | max (Wb) |
|---|---|---|
| zeroth only | 6.51e-4 | 2.37e-3 |
| zeroth + first | **2.23e-13** | 8.2e-13 |
| first sign-flipped | 1.30e-3 | 4.74e-3 |
| expansion-point-consistent | 2.23e-13 | 8.2e-13 |

Zeroth-plain-first is exact to second order in the cell size (the linear density
is described exactly by its moments); sign-flipping exactly doubles the dipole
error. Kernel expansion point == moment point (both the cell's section centroid;
polygonanalytic.py `polygon_analytic_flux_moments` default `_section_centroid`,
build_machine expansion_points=atomic-mesh centroids, stencil_mesh.py:321), so
variant (iv) coincides with (ii). A missing factor of R or of the pitch is refuted
by the machine-precision agreement.

## Part two — weak -110 frozen currents (rms/max over the 58.09 Wb span)

| variant | rms/span | max/span | note |
|---|---|---|---|
| zeroth only | **0.00571** | 0.0144 | reproduces gate B's 0.0057 |
| zeroth + first (as built) | **0.01056** | 0.0168 | reproduces gate B's 0.0106 |
| first sign-flipped | **0.00513** | 0.0132 | removes the whole added error |
| expansion point moved to region centroid | 0.01064 | 0.0169 | no recovery |

Cell-class decomposition of the added first-order error: first-order on interior
cells only 0.00611, on cut cells only 0.01009 — the cut cells carry ~all of it
(the interior first-order term is the verified-correct one).

## Units and reference (path:line)

- physical first moments A m = int j*(R-Rc) dA about the centroid
  (`frozen_current_flux_image_gate._frozen_moments`; production
  `stencil_mesh._direct_profile_current_moments` moment_centre=support.centroids).
- `coupling_current_moments` (forward_operator.py:2092-2123) converts A m moments
  to A/m coefficients using the area-normalised m^2 second moments
  (`greens.second_moments`); the kernel's radial/vertical blocks are the Wb m/A
  area means of (R-Rc)K and (Z-Zc)K (polygonanalytic.py), so the product is Wb.

## Artifacts

- docs/figures/cut-cell-current-attribution/first-order-coupling/{receipt.json,
  single-cell.svg, weak-110-errors.svg} (committed); parts/*.json (gitignored on disk).
- single-cell.svg: exact linear-density flux + four variants' error on shared levels.
- weak-110-errors.svg: zeroth-only, zeroth+first, best-of(sign-flip, region-centred)
  error fields on shared levels; wall drawn, line contours, no axes.
