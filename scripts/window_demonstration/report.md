NEEDS-HELP: the extraction service refuses the fixture's 126-cell core and no public exact-evaluation handoff can produce its required denser structured map

tried: Composed the public `solve_window`, `transport_sweep`, `forward_source_from_receipt`, `equilibrium_sweep`, and `extract_flux_surface_geometry` seams in `run_window.py`, pinned JAX to CPU before import, and started the one measurement process. The process stopped before entering `solve_window` because `TransportGeometry` rejected the extraction service's baseline record as invalid. A single diagnostic fixture solve then exercised four service configurations: normalized-flux ranges 0.04–0.985, 0.10–0.95, and 0.10–0.90 with 14 bins, plus 0.10–0.90 with 28 bins. All four returned `valid=false` despite valid arcs, zero invalid arcs, positive volume, positive toroidal flux, nonzero edge current, and positive diffusion metrics. The axis-connected core count was 126; the service contract requires at least 200.

options: (1) Add a public exact grid-free evaluation method that evaluates a converged `ForwardEquilibrium` on a caller-supplied structured extraction lattice, then use at least enough points to put 200 cells inside the connected core. (2) Publish a cached higher-resolution version of the same free-boundary fixture whose solve and extraction maps share the required structured lattice. (3) Change the service's 200-cell validity floor only if independent accuracy evidence justifies a lower floor; the present diagnostic does not justify that change.

leaning: Option 1, because it implements the live architecture decision that non-structured or coarse forward solves enter the extraction service by exact forward evaluation, never interpolation, and it lets the service own its extraction resolution without changing the equilibrium solve grid.

cost-if-wrong: If the service's 200-cell floor is the wrong constraint rather than the missing exact-evaluation seam, the evaluator and its Green-operator rows would be unnecessary and the demonstration would need to be rerun after revalidating the lower-resolution service record. If a cached higher-resolution fixture is chosen instead, the demonstration driver must be rebound to that fixture and all baseline preparation timing must be remeasured.

# Evidence

No coupled-window iteration ran, so there are no convergence, contraction, ledger, or exchange-sweep timing receipts to report. Reporting fabricated placeholders for those fields would misstate the solver. The failed process and the diagnostic are preserved in the run logs named by the worker manifest.

The diagnostic isolated the service validity floor:

| ψ_N range | surface bins | core cells | arc valid | invalid arcs | record valid |
|---|---:|---:|---|---:|---|
| 0.040–0.985 | 14 | 126 | true | 0 | false |
| 0.100–0.950 | 14 | 126 | true | 0 | false |
| 0.100–0.900 | 14 | 126 | true | 0 | false |
| 0.100–0.900 | 28 | 126 | true | 0 | false |

For the broad default-range record, volume was 0.9253114669703377 m³, boundary toroidal flux was 0.759020720636671 Wb, edge current was 553961.7346746189 A, and the minimum positive diffusion metric was 19.243218155941587. These finite values, together with `surface_arc_valid=true`, rule out the geometry and arc diagnostics exposed by the record. The remaining explicit validity condition that fails is `sum(core) >= 200`.
