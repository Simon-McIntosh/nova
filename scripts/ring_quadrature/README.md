# Boundary-local ring quadrature measurement

`measure_ring_quadrature.py` evaluates two boundary-local nodal constructions
without changing Nova's production source operator. It resumes from the banked
coarse analytic-seed fixture in `inputs/`, preserves the established
stencil-available moments byte-for-byte, and emits candidate fields, per-cell
errors, a machine-readable scorecard, and the R-Z comparison figure.

Run from the repository root:

```bash
uv run --no-sync python scripts/ring_quadrature/measure_ring_quadrature.py
```

The own-geometry construction fits the cubic to the cell centroid and full
polygon vertices.  Its two first-moment constraints come from the quadratic
least-squares gradient on those same points.  The one-sided construction uses
the identical cubic constraints but obtains the gradient from the cell centre
and the available Delaunay neighbour ring.  Neither construction smooths,
blends, tunes individual cells, or changes an established interior stencil.
