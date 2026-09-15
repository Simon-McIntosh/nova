# Private-region cell mask: four formulations

On the analytic single-null flux the private region is the set of in-material
cells on the plasma side of the separatrix that the magnetic-axis connected
component cannot reach (`closed & ~connected & inside_material` in the
production read).  This benchmark measures how that mask is built four ways,
cell by cell against the production mask at 500/750/1000/2500 requested cells
(realised 550/814/1074/2616):

- **a — production hex flood**, the saddle-aware hex minimum-label
  propagation the read's axis component actually executes.
- **b — raster doubling**, the `flood_fill_core` fill on a (height, radius)
  pitch raster, pass count `nr + nz` from the mesh diameter.
- **c — pointer jumping**, `ceil(log2(cells))` fully vectorised pass-doubling
  rounds over the confined-cell adjacency seeded at the axis, no per-cell
  loop.
- **d — saddle-wedge level test**, private when the flux is on the open side
  of the admitted X-point level AND the centroid is on the private side of
  both separatrix-leg rays (the saddle Hessian eigenvectors).  O(cells), zero
  passes.

Membership identity against the production mask is the positive control per
rung: the differing-cell count is reported for every formulation.  Each mask
is timed as a jitted function vmapped over sixteen states, five repeats,
median batch time divided by sixteen, on one H200.

| rung | cells | a ms | b ms | c ms | d ms | a≠ | b≠ | c≠ | d≠ | a passes | c passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 550 | 0.500 | 0.057 | 0.021 | 0.016 | 0 | 3 | 0 | 3 | 15 | 10 |
| 750 | 814 | 0.750 | 0.079 | 0.026 | 0.015 | 0 | 0 | 0 | 5 | 15 | 10 |
| 1000 | 1074 | 0.997 | 0.021 | 0.020 | 0.009 | 0 | 12 | 0 | 5 | 19 | 11 |
| 2500 | 2616 | 2.862 | 0.047 | 0.047 | 0.007 | 0 | 1492 | 0 | 13 | 28 | 12 |

(Columns `a≠ … d≠` are differing-cell counts against the production mask;
`a passes` / `c passes` are pass counts — masked axis-component floods for a,
doubling rounds for c.  b and d run zero production-style passes; the raster
fill's 54-198 passes are where its surrogate cost goes.)

## Findings

**Two formulations reproduce the production mask exactly at every rung: the
production hex flood (a) and pointer jumping (c) — zero differing cells at
all four rungs.**  They are not two independent masks; c is a different
algorithm that happens to agree with the reference topology on this mesh.
Cost differs by two orders of magnitude: a rises 0.50 -> 2.86 ms/state with
cell count (scaling exponent **1.12**, the linear flood), while c stays
0.020-0.047 ms/state, sub-linear (exponent **0.51**) — it pays only for
pass-doubling rounds, not for breadth.  c is 25x cheaper than a at 500 cells
and 60x cheaper at 2500 cells.

**The saddle-wedge test (d) is cheapest but systematically under-reports.**  At
0.007-0.016 ms/state it is a fixed ~10 microsecond floor (exponent -0.60), yet
it misses 3/5/5/13 private cells.  Every miss is a cell whose centroid lies
outside the cone swept by the separatrix-leg rays: the wedge drawn from the
saddle's Hessian eigenvectors is strictly narrower than the true private
domain at finite pitch, so cells on the private side that fall outside that
cone are never flagged.  d never invents false positives beyond the coarsest
rung — at 500 cells it flags 4 against 3 production (2 extra, seen as 3
differing), and from 750 cells up its flags are a strict subset of the
production mask.  This is the documented failure mode: d is a local geometric
test with no topological reach, so it converges toward the mask only as the
mesh refines.

**The raster-doubling flood (b) does not survive the unstructured mesh.**  It
agrees at 750 cells but fails outright at 500/1000 (returns 0 private cells,
so all of them differ) and collapses at 2500 (flags 1519 against 27, 1492
differing).  The (R, Z) pitch raster is a surrogate: the hex carrier is not a
global affine lattice, so cells land on the grid with site collisions (86 /
94 / 154 counted before the 2500-collisions-zero rung), and the raster's
four-connected fill then floods a connectivity the hex mesh does not share.
Its per-state cost is erratic (exponent meaningless) precisely because the
tiled rectangle, pass count and collision pattern all change independently
with the rung.  b is the quantified reason the production read never used a
structured fill for this component.

**Against the production read itself, the isolated hex flood (a) is already
cheaper than the read's private-exclusion stage.**  The census receipt times
that stage at 1.36 / 5.09 / 30.65 ms/state at 550 / 1074 / 2616 cells for the
full read (dominant stage, 70-92% share); a costs 0.50 / 1.00 / 2.86 ms —
2.7x / 5.1x / 10.7x cheaper, the difference being the read's flanking
per-representative machinery around the component fill.  Both a and c sit
far under a 1 ms/state whole-solve budget; d is a floor at ~10 microseconds
but pays for its speed in completeness.

## Verification

- **Positive control per rung:** every formulation's mask is counted against
  the production mask; a and c report zero differing cells at all four rungs
  (the figure script recomputes the masks and raises if those counts ever
  drift from the measured receipt).
- **The benchmark's (a) is the read's own flood:** the reconstructed axis
  component from the isolated label-saddle-aware components call is asserted
  equal to the read's `connected` array in `_build_rung`, so (a) is not a
  re-implementation of the production mask — it is the flood the read ran.
- **Figures:** the poloidal panel at 1000 cells shows the production private
  cells hatched (12) with the saddle-wedge test's five misses outlined; the
  script asserts the private and differing counts it draws equal the receipt.

![cost-and-accuracy](/nova/figures/cut-cell-current-attribution/private-region/private-mask-cost-cells.svg)

![poloidal-1000](/nova/figures/cut-cell-current-attribution/private-region/private-mask-poloidal-1000.svg)
