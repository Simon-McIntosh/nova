NEEDS-HELP: a later exchanged equilibrium loses the entire axis-connected core, so exact extraction fails closed before the window can return typed receipts

tried: Ran the repaired tree exactly once with the declared 10 ms window, equilibrium and transport samples at 0 and 10 ms, iteration cap 10, convergence tolerance 0.005, damping 0.5, 49 x 49 exact extraction, 8 transport cells, and 14 surface bins. The baseline fixture equilibrium evaluated exactly and produced a valid transport geometry. The TORAX multi-channel rung advanced and the equilibrium callback solved an exchanged source. A subsequent exact extraction returned zero axis-connected core cells, `surface_arc_valid=false`, zero invalid-arc entries, first diagnostic cell 0, and first diagnostic level 0.04. The driver refused that record before it could enter the returned waveform, and `solve_window` therefore returned no convergence or conservation receipt. No retry, tolerance adjustment, damping change, window shortening, or source clipping was attempted.

options: (1) Instrument and repair the coupled source-to-equilibrium path so the exchanged source either preserves a closed axis-connected core or raises a typed topology/admissibility receipt before extraction. (2) Define a validated admissibility policy for `forward_source_from_receipt`, including the physical bounds that distinguish a meaningful transported source from one that destroys the fixture equilibrium. (3) Change the demonstration regime only after the live plan decides which window length or source bound remains representative; that would be a different measurement and must start from a fresh run.

leaning: Option 2 followed by option 1, because the extractor is correctly refusing a map with no connected core and reports no invalid arc to repair. The missing contract is now upstream: the coupled return channel can produce a source for which the free-boundary solve has no extractable closed-flux region, but it supplies neither an admissibility receipt nor a policy boundary explaining whether that state is physical. Authoring that policy in the demonstration directory would hide a product-level scientific decision.

cost-if-wrong: If the zero-core map is caused by an equilibrium branch-selection defect rather than an inadmissible transported source, bounding the source would tune away a solver bug and invalidate the eventual coupling measurement. The source policy would need to be removed, branch/topology handling repaired, and the entire fixed window rerun. If the demonstration knobs are changed instead, none of the resulting convergence, ledger, or timing figures can be compared as the requested run.

# Run contract and observed refusal

| quantity | value |
|---|---:|
| backend | CPU |
| window length | 0.01 s |
| equilibrium sample grid | [0.0, 0.01] s |
| transport sample grid | [0.0, 0.01] s |
| iteration cap | 10 |
| convergence tolerance | 0.005 |
| damping | 0.5 |
| exact extraction lattice | 49 x 49 |
| transport radial cells | 8 |
| surface bins | 14 |
| baseline extraction valid | true |
| failing map connected-core cells | 0 |
| service validity floor | 200 |
| surface-arc validity | false |
| invalid-arc diagnostic entries | 0 |
| first diagnostic cell | 0 |
| first diagnostic level | 0.04 |

The first cell and level are reported verbatim because the fail-closed contract requires them. With zero connected-core cells and zero invalid-arc entries, they identify the service's first unsupported surface diagnostic rather than a failing arc interval.

# Unavailable receipts

`solve_window` did not return, and the raised error is not a `WindowConvergenceError` carrying a diagnostic convergence receipt. Consequently iterations used, measured contraction, per-field exit residuals, damping applied by the returned receipt, flux-consumption closure, plasma-current closure, and completed per-exchange wall times are unavailable. Fabricating those fields from partial callback state would misrepresent the public window contract; the machine-readable TSV marks every one unavailable.

The complete single-run traceback is preserved in the worker's named `window-run.log`.
