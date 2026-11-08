# Stationary field orientation check

`stationary-field.npz` is an unchanged copy of the frame 146, trip 1 archive
written by `scripts/prototypes/diiid_vertical_loss/trip_trajectory.py`.
Its source is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T134744819124-diiid-frame-146-vertical-loss/trip-trajectory/frame-146-trip-1.npz`.
SHA-256: `340fd7cfa87cd53c618e99ebd6944dbe4a60733994d8eea12f6774398debe26e`.

The writer reshapes the flux vector using `FluxLattice.shape`, whose order is
`(radius, height)`. The tensor spline accepts `(height, radius)`, so its native
input is `stored.T`. Shape inference cannot distinguish the orders on this
33 by 33 archive.

Run from the repository root on a compute node:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/diiid_axis_selection/check_field_orientation.py
```

The diagnostic reads both orientations, uses the repository spline and
stationary polish, and prints gradients, Hessian eigenvalues, convergence,
positions and displacement for both the reported axis and the proposed target.
The native reported axis has zero displacement and two negative eigenvalues;
the transposed field reproduces the quoted 0.328 m displacement. This checks
provenance of that claim; it does not re-run a forward solve or establish the
cause of the support-mask collapse.
