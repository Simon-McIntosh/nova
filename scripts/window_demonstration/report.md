NEEDS-HELP: the exact transported equilibrium clears the extraction core floor but the service rejects four surface arcs before a window receipt can be returned

tried: Updated the demonstration driver to retain the fixture's exact coil and plasma source sections, evaluate every converged equilibrium through `evaluate_forward_equilibrium`, and feed the resulting structured map to the extraction service. The first attempt used 33 x 33, which was valid for the baseline but lost the service validity margin after the transported source changed the equilibrium. The one permitted correction increased only the extraction target to 49 x 49; it left the 10 ms window, two-point sample grids, iteration cap 10, tolerance 0.005, and damping 0.5 unchanged. That attempt reached `solve_window`, ran the TORAX transport update and equilibrium update, and produced an exact transported map with 559 axis-connected core cells. Extraction then returned `surface_arc_valid=false` with four invalid arcs, so `TransportGeometry` correctly refused the degraded record. No convergence or conservation receipt was returned.

options: (1) Repair or extend the flux-surface extraction service so its arc clipping can represent this exact transported fixture map while retaining the existing validity checks. (2) Define and validate a transport-to-forward-source admissibility policy that prevents TORAX updates from producing surface geometry outside the extractor's supported class. (3) Declare a different demonstration regime, such as a shorter window or bounded source update, but only after the live plan decides that this remains representative of the intended coupling rather than tuning the run into submission.

leaning: Option 1, because the exact handoff now works, the transported map exceeds the 200-cell floor by 359 cells, and the remaining refusal is isolated to four surface arcs. The extraction service owns that representation and validity mechanism; implementing it in this demonstration directory would duplicate product machinery.

cost-if-wrong: If the transported source is physically inadmissible rather than merely outside the extractor's numerical arc support, changing the clipping machinery would admit a state that should instead be rejected. The service change would need to be reverted and the admissibility policy implemented before this same fixed window could be measured. If the demonstration regime changes, all convergence, ledger, and wall-time evidence must be generated afresh because none was returned here.

# Quantitative evidence

The baseline exact map on the 49 x 49 lattice returned a valid extraction record; otherwise the driver would have stopped before constructing the initial geometry. The coupled attempt then advanced through TORAX and the equilibrium sweep. Its first unsupported transported record reported:

| quantity | observed |
|---|---:|
| extraction lattice | 49 x 49 |
| axis-connected core cells | 559 |
| extraction floor | 200 |
| margin above floor | 359 |
| surface arcs valid | false |
| invalid surface arcs | 4 |
| window length | 0.01 s |
| iteration cap | 10 |
| convergence tolerance | 0.005 |
| damping | 0.5 |

The earlier 33 x 33 attempt and the final 49 x 49 attempt are preserved in `window-run.log` and `window-run-final.log`, respectively. Both fail closed before `solve_window` returns its typed receipt. Consequently there are no honest values for iterations used, measured contraction, per-field exit residuals, conservation closures, or completed exchange-sweep wall times. Those fields are marked unavailable in the TSV rather than fabricated or inferred from partial internal state.
