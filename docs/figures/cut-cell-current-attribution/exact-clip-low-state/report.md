# Exact-clip low-state discriminator

The arms jointly support a **seed-selected second self-consistent state**, not a
wrong discrete map, a simple booking shortfall, or a uniquely displaced analytic
equilibrium. The 110-cell row is the positive control: analytic and repaired-seed
starts reach the same oracle-near state. At 300 cells the analytic start stays at
the oracle while the repaired seed reaches a distinct converged low state.

| requested cells | arm A residual | arm A max difference / span | arm A axis flux [Wb] | arm B residual | arm B axis flux [Wb] | arm C boundary offset explained | arm D changed classes | supported reading |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 110 | 3.666e-13 | 1.135e-4 | 49.942887 | 3.420e-13 | 49.942887 | not material: starts coincide | 0 | oracle-near positive control |
| 300 | 9.305e-14 | 1.080e-4 | 49.952060 | 2.079e-13 | 46.664612 | 85.3% | 25 | seed-selected second fixed point |

## Trip snapshots

### 110 requested cells

| state | residual | axis flux [Wb] | boundary level [Wb] | max difference / span |
|---|---:|---:|---:|---:|
| after 1 trip(s) | 3.666e-13 | 49.942887 | -0.0018776984 | 0.000113478 |
| after 2 trip(s) | 3.666e-13 | 49.942887 | -0.0018776984 | 0.000113478 |
| after 4 trip(s) | 3.666e-13 | 49.942887 | -0.0018776984 | 0.000113478 |
| terminal (1 trips) | 3.666e-13 | 49.942887 | -0.0018776984 | 0.000113478 |

![Arm A and B terminals](/nova/figures/cut-cell-current-attribution/exact-clip-low-state/panels/weak-static-110-terminals.png)

Blue contours and solid triangles are analytic; ochre contours and solid triangles are solved. Both difference panels use the same fixed signed fractions of the analytic flux span. Arm A residual 3.666e-13, converged True; arm B residual 3.420e-13, converged True.

### 300 requested cells

| state | residual | axis flux [Wb] | boundary level [Wb] | max difference / span |
|---|---:|---:|---:|---:|
| after 1 trip(s) | 9.305e-14 | 49.95206 | -0.0018674919 | 0.000107956 |
| after 2 trip(s) | 9.305e-14 | 49.95206 | -0.0018674919 | 0.000107956 |
| after 4 trip(s) | 9.305e-14 | 49.95206 | -0.0018674919 | 0.000107956 |
| terminal (1 trips) | 9.305e-14 | 49.95206 | -0.0018674919 | 0.000107956 |

![Arm A and B terminals](/nova/figures/cut-cell-current-attribution/exact-clip-low-state/panels/weak-static-300-terminals.png)

Blue contours and solid triangles are analytic; ochre contours and solid triangles are solved. Both difference panels use the same fixed signed fractions of the analytic flux span. Arm A residual 9.305e-14, converged True; arm B residual 2.079e-13, converged True.

## Interpretation

Arm A rules out a wrong discrete map at the analytic state. At both cell counts it
converges after one production trip and its snapshots after 1, 2 and 4 trips are
unchanged. At 300 cells the terminal residual is 9.305e-14 and the maximum field
difference is 1.080e-4 of span, while the repaired-seed arm separately converges at
2.079e-13 with axis flux 46.664612 Wb. The production map therefore has two
converged fixed points on that mesh, and the seed selects the low one.

Arm C shows what distinguishes the low state geometrically. Translating the analytic
flux rigidly by the low state's measured 33.2 mm axis displacement moves the
production limiter level by 2.319 Wb, against the low state's 2.022 Wb offset. The
0.297 Wb mismatch is 14.7% of the low-state offset: a rigid displacement explains
85.3% of it. This is strong displacement evidence, but it does not make the low
state a unique displaced equilibrium because arm A demonstrates a second stable
oracle-near state under the same exterior.

Arm D rules out a net-current booking error as the discriminator. Twenty-five cells
change participation class between the analytic and low states. Evaluated against
the same analytically clipped fixture current by terminal class, cut cells book
427,983 A more and whole cells book 427,632 A less; inactive cells differ by -143 A,
leaving only 207 A across all classes. Current is redistributed at the moving
support boundary rather than missing by the 428 kA class exchange.

The consequence is explicit: two fixed points with the same target current and the
same exterior mean this fixture exterior holds plasma position weakly. The
operational defect is seed-basin selection. The next measurement is positional
stiffness: rigidly translate the analytic state from 0 to 60 mm, apply one production
map at each displacement, and plot residual against displacement. That measurement
will show the width and curvature of the weakly restoring basin without conflating it
with iteration history.

Job 1271758 completed both rows in one H200 allocation in 58 minutes with exit zero
and `LOW_STATE_EXIT rows=2 completed=True`. The receipt retains every changed
participation cell with its centre, analytic and terminal class, and both normalized
flux values. This measurement changed no Nova source file.
