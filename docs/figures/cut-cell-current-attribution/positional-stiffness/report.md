# Fixture positional stiffness

The analytic-clipped fixture **does not hold the plasma position** under the registered criterion: every nonzero displacement must restore after four production trips and the fitted one-map residual at 36 mm must exceed 8.6e-2 of span in every row and direction. Restoring=False; steeper=False.

| case | cells | direction | posing | rms curvature [span/m2] | sup curvature [span/m2] | first rms > 10x floor [mm] | rms at 36 mm |
|---|---:|---|---|---:|---:|---:|---:|
| weak-rotation-reactor-static | 110 | outboard | analytic_clipped | 0.140325 | 1.44136 | 2 | 0.0429825 |
| weak-rotation-reactor-static | 110 | outboard | whole_cell | 3.53458 | -0.916643 |  | 0.18794 |
| weak-rotation-reactor-static | 110 | vertical | analytic_clipped | 0.683911 | 0.664038 | 5 | 0.00412752 |
| weak-rotation-reactor-static | 110 | vertical | whole_cell | -1.1584 | 1.8338 |  | 0.221004 |
| weak-rotation-reactor-static | 300 | outboard | analytic_clipped | 0.0767613 | 1.53498 | 2 | 0.0441831 |
| weak-rotation-reactor-static | 300 | outboard | whole_cell | 3.65592 | 1.57676 |  | 0.186125 |
| weak-rotation-reactor-static | 300 | vertical | analytic_clipped | 0.709481 | 0.846199 | 5 | 0.00415109 |
| weak-rotation-reactor-static | 300 | vertical | whole_cell | -1.23268 | 0.710644 |  | 0.220491 |
| moderate-rotation-conventional-static | 300 | outboard | analytic_clipped | 0.306717 | 21.9279 | 2 | 0.155701 |
| moderate-rotation-conventional-static | 300 | outboard | whole_cell | 118.593 | 34.3095 |  | 0.124581 |
| moderate-rotation-conventional-static | 300 | vertical | analytic_clipped | 13.9438 | 22.2518 | 2 | 0.0199025 |
| moderate-rotation-conventional-static | 300 | vertical | whole_cell | -9.26148 | 12.3297 |  | 0.200812 |

## Figures

![Residual stiffness for weak-rotation-reactor-static](/nova/figures/cut-cell-current-attribution/positional-stiffness/figures/weak-rotation-reactor-static-cells-110-residual.png)

![Residual stiffness for weak-rotation-reactor-static](/nova/figures/cut-cell-current-attribution/positional-stiffness/figures/weak-rotation-reactor-static-cells-300-residual.png)

![Residual stiffness for moderate-rotation-conventional-static](/nova/figures/cut-cell-current-attribution/positional-stiffness/figures/moderate-rotation-conventional-static-cells-300-residual.png)

![Forty-millimetre translated analytic states](/nova/figures/cut-cell-current-attribution/positional-stiffness/figures/translated-40mm.png)

Blue contours and solid triangles are the analytic state; ochre contours and solid triangles are the rigidly translated state. Every panel uses shared analytic Wb levels and draws the machine wall.

## Interpretation

The analytic-clipped exterior does not supply the registered restoring basin. At least one measured start fails to move toward the analytic axis by the fourth trip or its fitted residual at 36 mm remains below the low state's 8.6e-2-of-span separation. The two fixed points are therefore consistent with weak positional holding, not a trusted certificate basin.

Retain the analytic oracle and pose a non-vacuum exterior family from the analytic total minus analytically clipped plasma images at nearby rigid displacements; its displacement derivatives supply an explicit restoring Taylor term. A fitted vacuum-coil exterior is not admissible because the Solovev exterior is not a vacuum field.

No Nova source file was changed by this measurement.
