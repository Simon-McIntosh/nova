# Cut-cell allocation root cause — three-row report

Measured with the committed driver
`benchmarks/xpoint_cell_allocation_rca.py` at revision `fc1b040f`
(weak-rotation-110 and diverted-300 rows) and `413c6c4b` (diverted-110 row).
Every row ran on `all_debug`, 8 CPUs, 64 GiB, CPU float64, `TMPDIR=/tmp`.
The driver selftest passes at `fc1b040f` (see [Evidence](#evidence)):
exact density agrees with the production density to `1.32e-10` (both
families) and the three banked cut-cell exact zeroth moments reproduce to
`≤ 8.8e-10` relative.

| row | realised cells | focus | boundary-cut | interior | exterior | multi-crossing | elapsed (s) | job | class read |
|-----|---------------|-------|-------------|----------|----------|----------------|-------------|-----|-----------|
| diverted-single-null −110 | 132 | X-point cell 9 | 36 | 50 | 46 | 9 (4 crossings) | 1826 | 1268345 | limited |
| weak-rotation-reactor-static −110 | 135 | cells 101, 102 + 104, 106, 108 | 43 | 78 | 14 | 101, 102 (6) | 72.7 | 1268485 | limited |
| diverted-single-null −300 | 340 | X-point cell 5 | 66 | 146 | 128 | 5, 15 (4 each) | 2386.5 | 1268486 | limited |

**The committed behaviour, named once:** the production support partition
books each participating cell at its **whole atomic hexagon**, selected only
by which side of the read boundary the cell centroid falls on — it never
geometrically clips a cell against the separatrix.  That single rule makes
both families' rows wrong in mirror-image ways: a boundary cell whose
centroid sits on the plasma side is booked whole (chord **over**-attributes),
a boundary cell whose centroid sits outside is booked at zero (chord
**under**-attributes), and the analytic exactly-partial region is never what
is attributed.

## Per row: cells dominating the attribution error

**Diverted −110** — whole-cell booking **over**-attributes the cut cells:
the centroid of each dominated cell falls plasma-side, so it is booked at its
full-cell current against a partial analytic core-side region.

| rank | cell | crossing count | chord current (A) | analytic (A) | fraction of squared error |
|------|------|----------------|-------------------|--------------|---------------------------|
| 1 | 96 | 2 | 840.6 | 85.6 | 0.109 |
| 2 | 75 | 2 | 840.6 | 131.1 | 0.096 |
| 3 | 112 | 2 | 717.5 | 34.9 | 0.089 |
| 4 | 18 | 2 | 654.9 | 21.2 | 0.077 |
| 5 | 70 | 2 | 779.4 | 169.1 | 0.071 |
| 6 | 93 | 2 | 840.6 | 246.9 | 0.067 |

Chords of 654–841 A are booked against 21–247 A of analytic core-side
current — a 4×–40× over-attribution.  The X-point cell itself is not in the
ranking: its 449 A error is below the dominated set's squared threshold.

**Weak −110** — whole-cell booking **under**-attributes: 43 cut cells but only
16 chord-participating.  The dominated cells are the boundary cells whose
centroid-side label excludes them even though the analytic core region
carries ~190 kA through each:

| rank | cell | crossing count | chord current (A) | analytic (A) | fraction of squared error |
|------|------|----------------|-------------------|--------------|---------------------------|
| 1 | 104 | 2 | 0.0 | 192266.4 | 0.089 |
| 2 | 108 | 2 | 0.0 | 192266.4 | 0.089 |
| 3 | 87 | 2 | 0.0 | 188227.9 | 0.086 |
| 4 | 110 | 2 | 0.0 | 188227.9 | 0.086 |
| 5 | 113 | 2 | 0.0 | 165366.7 | 0.066 |
| 6 | 85 | 2 | 0.0 | 165366.7 | 0.066 |

The exact path attributes every one of the 43 cut cells (its 6-crossing
focus cells exact ≈ analytic to twelve significant figures, below).

**Diverted −300** — the same over-attribution as −110, sharpened by the
finer mesh: the two-crossing boundary cells are thinner slivers, so the
booked whole-cell currents (259–314 A) sit against near-zero analytic
core-side current:

| rank | cell | crossing count | chord current (A) | analytic (A) | fraction of squared error |
|------|------|----------------|-------------------|--------------|---------------------------|
| 1 | 115 | 2 | 313.5 | 0.009 | 0.082 |
| 2 | 113 | 2 | 313.5 | 0.681 | 0.081 |
| 3 | 330 | 2 | 286.4 | 0.652 | 0.068 |
| 4 | 84 | 2 | 272.7 | 0.997 | 0.061 |
| 5 | 232 | 2 | 258.9 | 1.826 | 0.055 |
| 6 | 210 | 2 | 300.0 | 74.5 | 0.042 |

As at the −110 row, none of the dominated cells is the X-point cell.

## The X-point cell: four-crossing, whole by centroid-side booking, empty by the exact chain

In both diverted rows the analytic separatrix self-crosses at the saddle, so
the X-point cell sees four boundary crossings.  The −110 row has exactly one
multi-crossing cell (cell 9); the −300 row has two (cell 5, the X-point cell,
and cell 15, its saddle neighbour, both 4-crossing).

| quantity | diverted −110 (cell 9) | diverted −300 (cell 5) |
|----------|------------------------|------------------------|
| analytic separatrix crossing count | **4** | **4** |
| analytic X-point position (m) | (1.50253, −1.01728) | (1.50253, −1.01728) |
| cell centroid (m) | (1.41801, −1.02298) | (1.52921, −1.05662) |
| whole-cell booking current (A) | **459.7** | **187.9** |
| exact chain current (A) | **0.0** | **0.0** |
| exact chain support vertex count | 0 | 0 |
| analytic core-side current (A) | **10.5** | **1.15** |

Whole-cell booking takes the saddle cell at its full hexagon (459.7 A and
187.9 A against analytic slivers of 10.5 A and 1.15 A — a 44× and 160×
over-attribution); the exact chain's traced curve self-crosses inside the
cell, the clip degenerates, and the cell is qualified out with a zero-vertex
support, attributing nothing through the non-empty analytic region.

The two rules (`nova/equilibrium/forward_operator.py`):

- **Whole-cell booking (production chord mode)** — `_profile_support`,
  `forward_operator.py:2126-2131`: `atomic_mesh.traced_clip(...)` over the
  whole mesh qualified by the profile-participation label alone; every
  participating cell keeps its full atomic hexagon, so the X-point cell is
  booked whole regardless of where the separatrix cuts it.
- **Exact chain** — `_profile_support`, `forward_operator.py:2173-2183`:
  vertex-level participation from the curved level at the cell vertices,
  then `traced_support = atomic_mesh.traced_clip(..., curve_evaluator=...,
  participating_cell=participation)`.  At the self-crossing saddle this
  produces a degenerate support that is qualified out (`vertex_count 0`).

## The six-crossing pair 101 and 102 (weak −110)

The weak row's only multi-crossing cells are the focus pair 101 and 102, six
boundary crossings each — the analytic static separatrix wiggles through each
cell several times, netting a region that covers the whole cell.  Both cells
are the sharpest case of centroid-side under-booking: their analytic
core-side region is ~100% of the cell (region area 0.03731 m² ≈ cell area
0.03734 m²) and carries 33795 A, yet the centroid-side label excludes them,
so the chord path books **zero** through them while the exact chain
attributes 33795.34 A (analytic 33795.385 A — agreement to twelve significant
figures).  Chord participation false, exact participation true for both.
Their immediate edge-sharing neighbours 104 and 106 are the row's dominant
error cells (rank 1/2), booked at zero against ~192 kA, so the whole
double-crossing-north cluster is dark under the chord path.

## Why the read admits no X-point at −110 and −300

Both diverted rows' production reads return `class: limited` with
`x_candidate_count: 0` (`o_candidate_count: 1`, `x_point_rz_m: None`).  The
cause is the sign-change census ring detector, established by the
census-assertion node (`~/.config/reckon/crew/reports/nova/s19-review/census-assertion/receipt.json`, stages `reduced` = −110/132 cells and `cells-300` = −300/340 cells): the hex carrier's `_compatibility_census`
(`forward_operator.py:661`, ring crossing count at `:664`, saddle ring mask
at `:665`) reads the per-ring cyclic sign changes of the flux; the saddle
needs a ring with four cyclic sign changes to be seen.  At 132 cells
(pitch 0.151 m) the histogram is {0-crossing: 1, 2-crossing: 54} with
`raw_ring_saddle_count: 0`; at 340 cells (pitch 0.0917 m) it is
{0: 1, 2: 197} with `saddle_crossing_ring_count: 0`.  No X candidate ever
enters containment, polish or dedupe, so the read classifies the state
limited — `defect_stage: 1_sign_change_census`, dropped_reason "the
sign-change census finds no saddle anywhere on the grid (no ring reads four
cyclic sign changes)".  The analytic X-point at (1.5025, −1.0173) is lost two
cells off the wall at both resolutions the RCA measures.  At 550 cells
(pitch 0.071 m, `cells-500`) the census does find one 4-crossing saddle ring
at 1.2 mm from the analytic X-point and the read admits `class: diverted` —
so the loss is purely the ring detector's mesh resolution, not containment,
polish or dedupe.  The weak family is limited by construction (`x_candidate_count: 0`, only O candidates).

The analytic separatrix is diverted regardless — core lobe R ∈ [1.156,
2.244] m, X-point (1.5025, −1.0173), two open legs — so the discrepancy is
between the exact GSP solution's topology and what the coarse-grid census can
admit, not a property of the equilibrium itself.

## What an explicit X-point cell treatment changes in each tabulated number

An explicit treatment clips the X-point cell to its analytic core-side region
(and, at −300, its saddle neighbour 15) instead of the two extremes.  The
measured deltas:

| number | row | chord now | exact now | with explicit X-point clip |
|--------|-----|-----------|-----------|---------------------------|
| X-point cell current (A) | −110 cell 9 | 459.7 | 0.0 | 10.5 |
| X-point cell current (A) | −300 cell 5 | 187.9 | 0.0 | 1.15 |
| X-point cell abs error (A) | −110 | 449.2 over | 10.5 under | 0 |
| X-point cell abs error (A) | −300 | 186.7 over | 1.15 under | 0 |
| largest single-cell relative error | −110 | 44× over (cell 9) | — | removed |
| largest single-cell relative error | −300 | 160× over (cell 5) | — | removed |
| X-point cell in dominated top-6 | both | no | — | remains off the top-6 |
| exact-attribution sum through the saddle | both | — | missing the analytic sliver | adds it |

The explicit clip removes each row's largest single-cell **relative** error
and restores the exact chain's missing contribution through the saddle, but
it does **not** change the dominated ranking: in every row the top squared
error is carried by ordinary 2-crossing boundary cells whose centroid-side
booking is 4–40× over (diverted) or 100% under (weak), and those are
untouched.  Its material effect on the tabulated numbers is confined to the
X-point cell row itself, the exact-attribution sum, and the per-cell error
map near the saddle.  The centroid-side booking rule would need to become a
true per-cell clip for the dominated cells to move at all.

## Evidence

- Figure 1 (focus-cell clip allocation), figure 2 (whole-domain
  |chord − analytic| contours with the analytic cut cells outlined and the
  analytic separatrix drawn dashed), figure 3 (ranked bars) per row under
  `docs/figures/uniform-cell-clip-and-coupling/xpoint-cell-rca/<row>/`
  (project src
  `/nova/figures/uniform-cell-clip-and-coupling/xpoint-cell-rca/<row>/`).
- Receipt JSON per row under `.../xpoint-cell-rca/parts/`: analytic section,
  production read record, cell census, per-cut-cell analytic/chord/exact
  moments, dominated list, `elapsed_seconds`.
- Driver selftest `.../parts/selftest.json`: density agreement 1.32e-10
  (both families), banked zeroth reproduction ≤ 8.8e-10.
- Ring-detector census receipt: [census-assertion receipt](file:///home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/census-assertion/receipt.json)
  (rows reduced/cells-300/cells-500), figures under
  `docs/figures/null-identification-authority/census-assertion/`.

Figure repairs on this run: the focus-cell caption is wrapped so the legend
fits inside the panel (previously clipped at both edges), and the whole-domain
panel now draws the analytic separatrix as a dashed reference and captions the
grey outlines explicitly as "the cells the analytic separatrix crosses" —
the outlined set is the analytic cut cells (verified against the −110 part
data: polygon/centre ordering consistent, cut cells sit inside the error
band), never the read's limited boundary.
