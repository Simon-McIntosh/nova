# Coupling-moment route adjudication

This measurement holds the closed-form oracle carrier, traced support geometry,
and production own-node quadratic flux interpolant fixed. It changes only the
way `j(psi_interp)` becomes the three physical coupling moments about each fixed
cell centroid:

- the production degree-nine weighted-QR density fit with exact polynomial
  moments;
- fixed degree-fifteen Duffy quadrature of the functional itself; and
- a per-cell, order-refined tensor-Duffy integral used only as the truth oracle.

The smooth arm uses the moderate-rotation closed-form source. The stress arm is
a finite-edge pedestal with a narrow shoulder in normalized flux. Results are
split into full interior supports, clipped boundary supports, and the boundary
ring (the clipped cells plus their stencil neighbours).

Run on the CPU lane with the shared repository environment:

```bash
JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv \
  PYTHONPATH="$PWD" uv run --no-sync python scripts/moment_route_adjudication/measure.py
JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv \
  PYTHONPATH="$PWD" uv run --no-sync python scripts/moment_route_adjudication/verify.py
```

`results.json` contains the pre-registered mechanical decision and explicitly
qualifies the H200 number as a projection from the banked phase profile.
