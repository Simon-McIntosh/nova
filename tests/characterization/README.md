# Assembly characterization harness

This harness pins the observable behaviour of the coil-fitting and metrology
code (GP interpolation, CCL/fiducial deltas, gap asymmetry and its Fourier
proxy, vault geometry, the nominal fiducial frame) so it stays provably
unchanged across refactors, branch moves and a future repository extraction.
It **observes**; it does not refactor the assembly source.

## The gate

Acceptance is **tolerance-based, not byte-identity**. Positional, gap and
deviation outputs in millimetres must agree to **1 micron** (three decimal
places on a millimetre — below the metrology noise floor). Other quantities
carry their own class, at least two orders below their physical noise:

| class | atol | rtol | applies to |
|---|---|---|---|
| `length_mm` | 1e-3 | 0 | positions, gaps, deviations in mm |
| `length_m` | 1e-6 | 0 | lengths in metres |
| `angle_rad` | 1e-6 | 1e-9 | angles in radians |
| `coefficient` | 1e-9 | 1e-9 | Fourier/spectral coeffs, normalized geometry |
| `hyperparameter` | 1e-6 | 1e-6 | GP lengthscales, nuggets |

A **sha256 fingerprint** of every canonical output is recorded in
`goldens/manifest.json` as a cheap change detector. A fingerprint mismatch
*triggers* the tolerance comparison; it does not by itself fail the gate — so
the harness survives BLAS and dependency bumps without re-baselining ceremony.

## Layout

- `_canonical.py` — flatten any result (ndarray, xarray, pandas, mapping) into a
  sorted `float64` mapping; serialize to sorted-key `.npz`; sha256.
- `_tolerance.py` — the tolerance classes and the comparison.
- `_environment.py` — env-lock fingerprint and the single-threaded-BLAS pin.
- `_manifest.py` — read/write `goldens/manifest.json`.
- `_registry.py` / `_entrypoints.py` — the characterized entry points.
- `_skips.py` — runtime probes deciding whether a blocked entry can run here.
- `test_kernel.py` — synthetic primitives (GP, transforms, Fourier identity,
  error vectors), millisecond, no data files.
- `test_invariance.py` — physical symmetries (offset/relabelling invariance,
  cylindrical radius under clocking).
- `test_component.py` — every registered entry point re-run and compared to its
  golden at tolerance.
- `test_harness_support.py` — self-tests of the machinery above.
- `generate_goldens.py` — (re)generate goldens + manifest.

## Running

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONHASHSEED=0
python -m pytest tests/characterization/ -q          # compare against goldens
python -m tests.characterization.generate_goldens    # regenerate goldens
```

Goldens **must** be generated under the pinned single-threaded environment
(the generator asserts it). The comparison lanes are tolerance-based and run
anywhere.

## Runnable vs blocked scope

Green in-repo today (goldens committed): the gap Fourier proxy, the IDM /
reverse-engineering / as-built CCL fiducial deltas, the nominal fiducial frame,
the uniform and base-assembly vault geometry, plus the synthetic kernel and
invariance lanes.

Registered as **visible skips** — the fixtures live off-repo and are staged via
the assembly data registry, not silently omitted:

- `sector.fit.ssat`, `pit.gaps` — the SSAT / in-pit sector fits need the
  per-sector IDM workbooks (`Sector_Module_#*.xlsx`) from the IO share, plus the
  nominal ILIS point cloud from `//io-ws-ccstore1`. Stage those (or a cached
  `fiducial_data.nc`) and the entries produce goldens with no code change.
- `vault.fourier_proxy` — the ANSYS-backed spectral extraction needs an ANSYS
  install (`AWP_ROOT*`). The vault *geometry* it sits on is characterized above.
- `windingpack.uniform` — the pyvista mesh build is heavy and its multi-MB vtk
  caches are not in-repo; opt in with `NOVA_CHARACTERIZATION_HEAVY=1` when the
  caches are present.

## Known assembly-source issues (for the port, not fixed here)

These block the full sector/pit lane and are recorded for the assembly port
(the harness stays read-only toward the source):

- `PolyData.cell_points(...)` was removed in recent pyvista; the winding-pack /
  centerline path needs a `get_cell(...).points` shim (sites in
  `centerline.py`, `ccl.py`, `coilcage.py`).
- `FiducialData` fans a `private=` keyword to sources that do not accept it.
- The sector-coupling branch is effectively always taken, so even non-sector
  builds try to read sector-module workbooks.

## Determinism notes

Single-threaded BLAS/OpenMP pins SVD/PCA/griddata reduction order. The GP
regressor pins `random_state=2025`. PCA plane normals can flip sign run to run
(compare magnitudes, not signed normals). The DC-mode Fourier phase is
`np.angle` of a ~1e-16 residual and is meaningless — no golden gates on it.
Filesystem glob order is unstable; sort before comparing.
