# Referee internal-inductance instrument survey

## Outcome

The `0.6020235566` ratio in the banked Nova receipt is **not a Nova
field-energy deficit**. It is an invalid cross-convention ratio created by
back-calculating field energy from the published `l_i` with a normalizer that
is 1.5484--1.7050 times the challenge referee normalizer on the six frozen
MAST observations.

Using the challenge referee formula on the native 65 x 65 source maps and
each banked stored LCFS gives `l_i` values within 0.975% of the published EFM
values on all six observations. Against those same referee-formula field
energies, Nova's hex-33 field-energy instrument is 0.95653--1.01017, with a
median ratio of 0.98505. Grid/operator differences are therefore at most
4.35% in this comparison, not the approximately 40% implied by 0.60202.

The result is qualified because Ambix commit
`2f3dcd0c87be908f3f7912c50168c6fc0dae2b79` does **not** contain the challenge
Consistency derivation code. Ambix only pins and loads the challenge corpus.
The exact definition below was read from the official scoring repository that
the Ambix-pinned corpus documentation links, at scorer commit
`a67429165b09eb81c311d44db6ff11743f108b0e`. Its exact LCFS extraction could
not be executed in either existing project environment because the scorer's
pinned `scikit-image==0.25.2` dependency is absent. No dependency or
referee-convention code was added to Nova.

## Authority chain and seam status

Ambix pins `Sophelio/fusion-equilibrium-challenge` at dataset revision
`1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`
(`/home/ITER/mcintos/Code/imas-ambix/imas_ambix/challenge/download.py:28-37`).
Its loader declares `efit_li` as a raw scalar and reads it alongside `efit_psirz`
(`/home/ITER/mcintos/Code/imas-ambix/imas_ambix/challenge/loader.py:29-36`,
`/home/ITER/mcintos/Code/imas-ambix/imas_ambix/challenge/loader.py:120-143`),
but does not derive a Consistency scalar. Ambix's existing referee is explicitly
a geometry-only firewall and judge
(`/home/ITER/mcintos/Code/imas-ambix/imas_ambix/eval/efit_referee.py:1-38`,
`/home/ITER/mcintos/Code/imas-ambix/imas_ambix/eval/efit_referee.py:120-165`,
`/home/ITER/mcintos/Code/imas-ambix/imas_ambix/eval/efit_referee.py:272-354`).

The pinned corpus README says that Consistency derives `li` from submitted
`psi`, defines `psi` in V s/rad, and scores the submitted native 65 x 65 map.
It links the official starter/scoring repository. The concrete definition is
in
[`fusion_scoring/derive.py:151-211`](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/blob/a67429165b09eb81c311d44db6ff11743f108b0e/fusion_scoring/derive.py#L151-L211),
while
[`fusion_scoring/lcfs.py:137-146`](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/blob/a67429165b09eb81c311d44db6ff11743f108b0e/fusion_scoring/lcfs.py#L137-L146)
defines the 512-point LCFS extraction. This authority chain is reproducible,
but it is not yet an Ambix executable seam.

Capability verdicts:

| Capability | Verdict | Evidence |
|---|---|---|
| Locate exact referee `l_i` definition | **READ, external authority** | Official scorer commit and source lines above, reached from Ambix's pinned corpus |
| Derive Consistency `l_i` inside Ambix | **ABSENT** | Ambix loads `efit_li`; its only EFIT referee judges geometry |
| Evaluate exact formula on all six maps | **TESTED, stored-LCFS qualification** | Native-65 formula results in `comparison.json` and `normalizer-comparison.json` |
| Run the scorer's exact 512-point LCFS extraction | **UNTESTED** | Observable import failure in `referee-measurement-exact-contour-attempt.log` |
| Compare Nova's instrument on all six observations | **TESTED** | Six banked Nova energies compared below |

## Exact referee definition

The challenge calls this `li(2)` and computes

\[
l_i(2) =
\frac{\langle B_p^2 \rangle_V}
     {\left(\mu_0 I_p/l_p\right)^2}
=
\frac{\int_V B_p^2\,dV}
     {V\left[\left(\oint_C B_p\,dl\right)/l_p\right]^2}.
\]

Here `C` is the extracted LCFS, `l_p` is its poloidal perimeter, and the
scorer obtains `mu0 * Ip` from the boundary circulation
`G = integral_C Bp dl`. Consequently the field-energy normalizer needed to
invert a published referee `l_i` is exactly

\[
D_{\mathrm{ref}} = V(G/l_p)^2,
\qquad
\int_V B_p^2\,dV = l_i D_{\mathrm{ref}}.
\]

This uses a **toroidal-volume average**, not an area average and not a
major-radius-only surface convention.

The grid and quadrature are also part of the definition:

- `Bp = abs(gradient(psi)) / R`, with `numpy.gradient(psi, Z, R)` on the
  submitted native 65 x 65 grid.
- All native grid nodes whose `(R, Z)` coordinates ray-cast inside the LCFS
  contribute. Each node has the full weight
  `dV = 2*pi*R*mean(diff(R))*mean(diff(Z))`; there are no half-weights at the
  mask edge.
- `Bp` is bilinearly sampled on the 512-point LCFS. Boundary circulation and
  perimeter are trapezoidal sums over successive contour segments.
- The official LCFS is extracted from `psi` using the scorer's corpus-wide MAST
  envelope/mask. This survey substituted the banked stored LCFS only at this
  contour-extraction step.

The corpus convention is `psi` in Wb/rad. Nova's equilibrium state stores total
Wb, defines the conversion as `2*pi`
([`nova/equilibrium/convention.py:64-65`](../../nova/equilibrium/convention.py)),
and divides its total-flux gradient by `2*pi*R` when producing `Bp`
([`nova/equilibrium/conservation.py:267-271`](../../nova/equilibrium/conservation.py)).
Both paths therefore produce the same physical field. More generally, any
constant scaling of `psi`, including `2*pi`, multiplies both the numerator and
the squared boundary term by the same factor and cancels from `l_i`. The
rad-vs-total-weber convention cannot cause the 0.60202 ratio.

## Six-shot comparison

The required anchor receipt is
[`field-energy-instrument-control.json`](../../docs/figures/efit-forward-parity/field-energy-instrument-control.json).
It contains shot 22086/slice 43 and the ratio `0.6020235565543425`. The other
five Nova values come from its banked six-observation extension,
[`tared-plasma-support-solve.json`](../../docs/figures/efit-forward-parity/tared-plasma-support-solve.json).
The source maps, published EFM `l_i`, and stored LCFS polygons are the frozen
MAST corpus values for those receipts.

| Shot/slice | Published EFM `l_i` | Referee formula `l_i` | Relative `l_i` error | Nova energy | Referee-formula energy | Prior backcalculation | Nova/referee | Referee/prior |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 21978/35 | 0.706804 | 0.704929 | -0.265% | 0.205592 | 0.207322 | 0.348039 | 0.99166 | 0.59569 |
| 21983/35 | 0.783221 | 0.780090 | -0.400% | 0.229704 | 0.227393 | 0.389262 | 1.01017 | 0.58416 |
| 21985/51 | 0.924814 | 0.920190 | -0.500% | 0.352102 | 0.368103 | 0.599009 | 0.95653 | 0.61452 |
| 21986/46 | 0.810423 | 0.809258 | -0.144% | 0.316121 | 0.322490 | 0.500533 | 0.98025 | 0.64429 |
| 21989/55 | 0.867311 | 0.858857 | -0.975% | 0.352167 | 0.358648 | 0.560779 | 0.98193 | 0.63955 |
| 22086/43 | 0.769380 | 0.767125 | -0.293% | 0.298978 | 0.302559 | 0.496621 | 0.98816 | 0.60923 |

Field energies and normalizers are in T-squared cubic metres. The corresponding
figure is
[`referee-inductance-comparison.png`](../../docs/figures/equilibrium-metric-parity/referee-inductance-comparison.png).

The six-shot aggregate is:

- referee-formula `l_i` versus published `l_i`: maximum absolute relative
  difference 0.975%, median signed difference -0.346%;
- Nova energy / referee-formula energy: minimum 0.95653, median 0.98505,
  maximum 1.01017;
- prior backcalculation normalizer / referee normalizer: minimum 1.54835,
  median 1.62787, maximum 1.70500;
- referee-formula energy / prior backcalculation: minimum 0.58416, median
  0.61188, maximum 0.64429.

Nova's banked instrument downsamples the source map from 65 x 65 to 33 x 33
([`benchmarks/efit_parity_field_instrument.py:261-268`](../../benchmarks/efit_parity_field_instrument.py))
and integrates `Bp^2 * 2*pi*R*cell_area` over 243 stored-LCFS centroids
([`benchmarks/efit_parity_inductance_partition.py:161-194`](../../benchmarks/efit_parity_inductance_partition.py),
[`efit_parity_field_instrument.py:371-382`](../../benchmarks/efit_parity_field_instrument.py)).
That operator/grid difference accounts for the observed -4.35% to +1.02%
range. It cannot explain a 40% loss.

## Attribution of 0.60202

On the anchor observation, the old receipt records

```text
Nova field energy                    = 0.2989777277
prior published-li backcalculation  = 0.4966213107
Nova / prior                        = 0.6020235566
```

The referee formula instead gives

```text
boundary circulation G              = 1.1717234726 T m
boundary perimeter lp               = 5.5276562002 m
native-node toroidal volume V       = 8.7776031327 m^3
referee normalizer V*(G/lp)^2       = 0.3944065817 T^2 m^3
published-li backcalc normalizer    = 0.6454825026 T^2 m^3
normalizer ratio prior/referee      = 1.6365916103
referee-formula field energy        = 0.3025589947 T^2 m^3
Nova / referee-formula energy       = 0.9881634093
```

Thus almost the entire old ratio is the reciprocal normalizer mismatch. The
referee-formula energy divided by the old backcalculation is 0.60923 on the
anchor and 0.58416--0.64429 over all six observations, bracketing 0.60202.
The remaining 1.18% Nova/referee difference on the anchor lies inside the
observed grid/operator band.

The legacy Nova calculation confirms why the ratio was constructed: it reads a
banked `denominator_t2_m3` and divides its measured field integral by that
denominator, without computing the referee `V*(G/lp)^2` normalization
([`benchmarks/efit_parity_inductance_partition.py:196-213`](../../benchmarks/efit_parity_inductance_partition.py)).
Its prose calls this a published mean-squared boundary-field denominator
([`efit_parity_inductance_partition.py:225-233`](../../benchmarks/efit_parity_inductance_partition.py)),
but the six numerical normalizers prove that it is not the challenge scorer's
normalizer.

## Qualification and required follow-on

This is a **normalizer attribution with a stored-LCFS bound**, not a bit-exact
end-to-end scorer reproduction. The exact-contour attempt imported the
official scorer but stopped at its explicit missing-`scikit-image` guard. The
successful comparison then used official `derive.py` logic on each stored
LCFS. This was sufficient to reproduce all six published `l_i` values to
better than 0.975%. For the anchor, the pre-existing receipt independently
bounds stored-LCFS Nova cell-quadrature volume error against exact contour
geometry at 0.4476%; that bound does not prove equality between the stored LCFS
and the scorer-extracted 512-point LCFS or bound a highly nonuniform `Bp^2`
edge contribution.

The Ambix-side follow-on is concrete: vendor or pin the official scorer source
and its runtime dependencies, expose an evaluator-only Consistency adapter for
`l_i`, and test a perfect self-score plus these six frozen observations. The
adapter must pin a scorer commit or content digest because Ambix currently pins
only the corpus revision; the current official scorer commit is otherwise a
drift-prone external input. No such code belongs in Nova.

## Artifacts and reproducibility record

- `comparison.json`: official-source provenance, exact formula details,
  per-shot raw measures, execution qualification, and the six-shot ratios.
- `normalizer-comparison.json`: published `l_i`, both normalizers, three energy
  estimates, and aggregate bounds.
- `referee-measurement.log`: successful six-shot measurement output and exit
  marker.
- `referee-measurement-exact-contour-attempt.log`: preserved exact-contour
  dependency failure and exit marker.
- `figure-generation.log`: successful figure-generation receipt.
- `../../docs/figures/equilibrium-metric-parity/referee-inductance-comparison.png`:
  visually inspected two-panel comparison of energies and ratios.

No measurement code is committed here: the definition belongs at the Ambix
Consistency seam, and this Nova node is fenced to evidence and figures.
