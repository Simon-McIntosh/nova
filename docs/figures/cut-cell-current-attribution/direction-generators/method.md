# Candidate direction measurement

The production pinned map, current pin, eighth-norm merit, sufficient-decrease
slope, strict own-mask residual decrease and model-trust predicate are unchanged.
Inputs are the hash-verified 135-cell and 342-cell continuous-support terminal
receipts. Analytic and production qualified GMRES(30) ladders are controls.
No cold solve or production implementation is changed.

The branch-frozen generator freezes the actual confined polygons, material and
private exclusions, flux-normalisation anchors and residual shadows at the
origin. The density is evaluated with the variable flux on those polygons.
Its Newton equation uses the same qualified GMRES(30) routine. Freezing only
Boolean centroid labels would not freeze the moving subcell current support.
The two-input map is checked against the production map when both inputs agree.

The clip-geometry continuation first takes the unaccepted production Newton
endpoint as a non-oracle support proposal. It reads that endpoint's confined
polygons while holding the origin flux fixed, then solves one Newton equation
for flux on that support. The requested fractions relax that flux displacement.
The geometry jump and any residual-shadow changes are reported separately.

The multi-branch predictor surveys the confined/open classifications along the
origin-to-production-Newton segment. Every encountered classification gets a
representative state, and a fresh qualified Newton correction is computed there.
Predicted endpoints are reclassified by their own state. Branch matching is
reported rather than inferred from the origin. Every predictor's full requested
fraction ladder is evaluated by the production map. The selected predictor is
the one with the lowest accepted merit, if any, otherwise the lowest finite
actual merit. Both matching and nonmatching endpoint classes remain in the raw
census; a nonmatching endpoint cannot establish a self-consistent branch solve.
Adaptive transition bisection and a doubled sampling mesh cross-check the
encountered classification set. This is an empirical finite-path census, not a
proof that arbitrarily narrow unobserved branch excursions do not exist.

All reported production decisions use the origin's unchanged live-map tangent.
The branch model's separate trust decision is also recorded using the same
production predicate, including its own predicted incumbent merit. This keeps
a changed direction separate from changing the model used to accept it.
The six-factor production selector is called as a cross-check; requested
fractions 0.01 and 0.001 are diagnostics, while production additionally tests
0.03125. The map-defect fallback is measured separately on the requested ladder.
No hidden globalization history or recovery radius is replayed.

One all_debug allocation runs each row in its own process with the root Python
interpreter, JAX_PLATFORMS=cpu and TMPDIR=/tmp in submit and payload. Each row
writes checkpoints after its controls and after each generator. Finite rejected
candidates and failed qualification are retained. Figures compare all five
ladders; the best accepted candidate, or a clearly labelled best rejected
candidate when none is accepted, is shown with analytic contours on shared
levels, both null sets and the wall.
