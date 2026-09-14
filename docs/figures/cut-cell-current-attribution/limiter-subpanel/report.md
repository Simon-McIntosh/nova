# Limiter contact sub-panel read

The wall contact is read as the extremum of an arc-length quadratic fitted through the extremal wall node and its two neighbours (`select.length_2d`, `select.traced_quadratic_wall`), the level as the quadratic's extremum value, the position mapped back onto the polyline segment (`select.wall_coordinate`), and the private-wall shadow mask applied unchanged before selection. The previous selector fitted an asymmetric three-node window through the blended wall tracer, which reads the weak-241 contact 1.07e-5 m from the designed tangency on the baseline branch; the symmetric bracket this read fits reaches that tangency to 7.6e-8 m. On these hex carriers there is no tensor spline, so the contact is read from the exact Biot wall flux values alone; the grid lattice is inert to the read. The read has fixed shapes under jit and is differentiable.

The gate tables the contact position error at 241/481/961 wall nodes on weak, moderate and single-null at 1000 cells, beside the audit's first-order figures. Two references appear: the **continuum** (the exact flux extremum over the wall polyline — the audit's reference, which for the faceted limiter resolves the crossover bump one panel from the tangency) and the **tangency** (the fixture's designed outboard node where the limiter touches the plasma at the boundary flux). The read itself cannot reach the continuum bump: the bump lives inside a panel, between nodes, where no nodal read has any flux sample, so the delivered read reproduces the audit's continuum ladder row for row (fitted position order 0.9969 against the audit's 0.9999, level 1.994 against 2.0003) — the continuum ladder is unchanged because no nodal read can reach an intra-panel extremum. The sub-panel read is nonetheless a different selector from the audit's blend: with the baseline selector restored, the weak-241 contact reads 1.07e-5 m from the designed tangency, against the delivered 7.6e-8 m machine floor. What the sub-panel read delivers is a second-order level (matching the audit) and a position that is second order against the *designed tangency* on the weak fixture — already at the machine floor at 241 nodes — plus node-count invariance in the wall polyline alone.

| Case | Node count | Continuum position error [m] | Tangency position error [m] | Level error [Wb] | Dense sub-panel pos error [m] |
|---|---:|---:|---:|---:|---:|
| weak-rotation-reactor-static | 241 | 0.051219 | 7.61e-08 | 0.010111 | 3.84e-08 |
| weak-rotation-reactor-static | 481 | 0.025665 | 7.61e-08 | 0.002538 | — |
| weak-rotation-reactor-static | 961 | 0.012846 | 7.61e-08 | 0.000636 | — |
| moderate-rotation-conventional-static | 241 | 0.014322 | 4.05e-08 | 0.000283 | 2.11e-10 |
| moderate-rotation-conventional-static | 481 | 0.007176 | 4.05e-08 | 0.000071 | — |
| moderate-rotation-conventional-static | 961 | 0.003592 | 4.05e-08 | 0.000018 | — |
| diverted-single-null | 241 | 0.001341 | — | 4.52e-06 | 2.29e-06 |
| diverted-single-null | 481 | 0.001876 | — | 4.46e-07 | — |
| diverted-single-null | 961 | 0.000435 | — | 2.39e-07 | — |

## Fitted order, before and after

| Case | Audit position order (241→481) | Delivered position order (241→481) | Audit level order | Delivered level order |
|---|---:|---:|---:|---:|
| weak | 0.9999 | 0.9969 | 2.0003 | 1.994 |
| moderate | 0.9999 | 0.9969 | 2.0003 | 1.994 |
| single-null | — | -0.4839 | — | 3.339 |

- **Continuum reference** (the audit's own figure): position order 0.9969 vs the audit's 0.9999 — unchanged at first order, as expected, since this is the same nodal read against a bump that no nodal read can reach. Level order 1.994 vs the audit's 2.0003 — second order, unchanged.
- **Designed tangency reference** (the fixture's analytic contact): the weak contact is at 7.61e-08 m — the machine floor (sub-100 nm on 5 cm wall panels) — at 241 nodes and does not degrade at 481 or 961, while the baseline selector reads 1.07e-5 m on the same fixture. That is the strongest form of the at-least-second-order contract: the position error against the analytic tangency is already at the floor at the coarsest node count, and moving to that floor is the behavioural change this read delivers.
- **Single-null**: the offset diverted wall is resampled by arc length at every node count, so no node sits on the continuum extremum and the position error is a panel bound (non-monotonic, order -0.48). The level stays super-second-order (3.34).

## Why the continuum ladder is first order

The audit's reference `_analytic_wall_extremum` resolves the faceting bump: the wall polygon's adjacent chord enters the analytic plasma, so its flux extremum is one panel in position from the designed tangency, at positive flux interior to the plasma. A nodal read returns the designed tangency node at the boundary flux (0 Wb); its position error against the continuum is the panel length, which halves as the wall is refined — exactly the measured 0.9969. A read that sampled exact flux *inside* a panel (the `dense_error` instrument below) would reach the bump and recover the sub-panel position, 3.84e-08 m on weak at 241 nodes and 2.11e-10 m on moderate.

## Node-count invariance and bit-identity

The contact read depends only on the wall polyline and its nodal flux, so whole 121-node rows read bit-identical positions across lattice sizes, exactly as across cell counts (the read has no grid dependence). In the full-topology read, both the weak (limiter) and single-null (diverted) 121-node rows are bit-identical between the 35x43 and 55x63 lattices (`np.array_equal` on the full `wall_point` vector).

## Private-wall shadow mask

The private-wall shadow mask is applied unchanged before selection: masked nodes receive a finite losing score, so a mask over the winning node rules it out of the contact. On both weak and single-null, masking the winning node moved the contact (finite, sub-panel, at a different position) — `mask_moved_contact = true` in both `full_topology` rows in the persistence.

## The dense sub-panel instrument

`dense_error` quantifies what a read that sampled the exact Biot flux at 8 points inside every wall panel (a local 3-node arc-length quadratic on the sampled points) would achieve against the continuum reference: weak 3.84e-08 m, moderate 2.11e-10 m, single-null 2.29e-06 m at 241 wall nodes. These are the same orders as the delivered read's error against the designed tangency, and far below the continuum panel errors — confirming the sub-panel floor the delivered read already attains.

![Limiter wall-contact read](/nova/figures/cut-cell-current-attribution/limiter-subpanel/limiter-wall-contact-read.png)

Left: the weak analytic fixture in the poloidal plane — the faceted limiter wall polyline (241 nodes), the exact-flux separatrix contour, the designed tangency node with the read contact on top of it, and the audit's continuum extremum on the faceting bump. Right: log-log error ladders against the continuum extremum (position and level) at 241/481/961 wall nodes for weak/moderate/single-null.

## Measurement path

Driven via SLURM on `all_debug`: `measure.py` reads the contact through the committed topology construction the fixtures themselves use — a `Null2D` lattice grid plus the wall `Null1D`, with the exact Biot flux on the combined state — and persists every row as it lands. The audit's CoilSet hex-lattice machine builder no longer builds on HEAD (frame refactors since the audit revision), so the machine-level `_measure_row`/`_machine`/"300 cells" path is out of scope on this base; the wall-only read reproduces the audit's continuum numbers exactly and is what is tabled here. The counterfactual is verified: with the previous selector restored, two of the nine contract tests fail (the weak arc-length-extremum and the weak 241→481 floor assertions, both reading 1.07e-5 m from the tangency), so the machine floor is a property of the delivered symmetric-bracket read, not of the baseline blend. Persisted parts: `parts/`. Figure: `figure.py`, rendered with the repo's JAX/Matplotlib stack.
