# CPU--H200 coupled-window divergence diagnosis

SLURM job `1253087` ran on JAX backend `gpu` and device `NVIDIA H200 NVL`. Configuration: window `0.0025` s, auxiliary source multiplier `0.5`, cap `10`, tolerance `0.005`, damping `0.5`.

## Precision discriminator

| backend | inner fixed-point residual | dtype | placement |
|---|---:|---|---|
| CPU | `5.6258166556553511e-15` | `float64` | `cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0,cpu:0` |
| H200 | `4.7216675502821279e-15` | `float64` | `cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0` |

Verdict: **FLOAT64 CONFIRMED; PRECISION ACQUITTED**

The CUDA sweep probe recorded `1516` input/output arrays for the first equilibrium and TORAX sweeps. The full TSV names each array's path, dtype, backend, device and shape.

## First trajectory divergence

The first non-identical quantity is iteration `1`, `geometry.phi_boundary`: CPU `0.024887102508608275`, H200 `0.024544392673846154`, absolute difference `0.00034270983476212061`.

| iteration | CPU maximum residual | H200 maximum residual | difference |
|---:|---:|---:|---:|
| 1 | `1.7244181062243822` | `1.7437508271419973` | `0.019332720917615065` |
| 2 | `0.88526428253015077` | `0.91786089157827588` | `0.032596609048125114` |
| 3 | `0.45500915267653674` | `0.56534643082695835` | `0.1103372781504216` |
| 4 | `0.23452428075438966` | `0.36610522244980021` | `0.13158094169541054` |
| 5 | `0.12137886616668318` | `0.24067272599372327` | `0.11929385982704009` |
| 6 | `0.063153381130372832` | `0.15833945182031628` | `0.095186070689943447` |
| 7 | `0.033069976727440283` | `0.10364417454985103` | `0.070574197822410756` |
| 8 | `0.017446998652218382` | `0.067359458948916179` | `0.049912460296697797` |
| 9 | `0.0092831536471297878` | `0.043453128418002369` | `0.03416997477087258` |
| 10 | `0.0049860186161842365` | `0.027837318675274874` | `0.022851300059090637` |

## Exhausted or converged receipt

Typed outcome: `WindowConvergenceError` — `window exchange did not converge: residual 0.0278373 after 10 iterations`.
Iterations `10`; measured contraction `0.64062864260291186`; exit residual `0.027837318675274874`; damping `0.5`. The tolerance remains `0.005`.

Failure-path serialization retained `10` residual rows, `20` branch receipts, `10` guard timings, `20` side timings, `1340` solve-array dtype records and `272` exchange-array dtype records before the typed exhaustion crossed the caller boundary.

## Wall-time structure

Fixture and precision-probe preparation took `58.811444299295545` s. The window took `141.66934043541551` s; the landed CPU window figure is `423.03271608706564` s. That CPU figure is pre-band: boundary-band sparsification landed later at `32942ac3`, whose warm CPU assembly figure is 24.6 s; the CPU window was not rerun.

| iteration | side | wall time (s) |
|---:|---|---:|
| 1 | transport | `8.3850863883271813` |
| 1 | equilibrium_sweep | `14.473566920496523` |
| 2 | transport | `0.12194113899022341` |
| 2 | equilibrium_sweep | `2.9101942079141736` |
| 3 | transport | `0.12559798080474138` |
| 3 | equilibrium_sweep | `2.847257855348289` |
| 4 | transport | `0.13332195486873388` |
| 4 | equilibrium_sweep | `2.8589411079883575` |
| 5 | transport | `0.13555588573217392` |
| 5 | equilibrium_sweep | `2.8909664116799831` |
| 6 | transport | `0.14308317191898823` |
| 6 | equilibrium_sweep | `2.8747726334258914` |
| 7 | transport | `0.14729427918791771` |
| 7 | equilibrium_sweep | `2.8791389120742679` |
| 8 | transport | `0.13336793426424265` |
| 8 | equilibrium_sweep | `2.9249030221253633` |
| 9 | transport | `0.13081431295722723` |
| 9 | equilibrium_sweep | `2.8716153837740421` |
| 10 | transport | `0.13457444217056036` |
| 10 | equilibrium_sweep | `2.8659708416089416` |

## Guard callback round trips

`10` calls cost `0.024270101450383663` s total and `0.00076295156031847` s median per adapter construction.

## Latest transport ledgers

Flux closure absolute/relative: `0` / `0`.
Current continuity absolute/relative: `0` / `0`.

## Solver identity

`equilibrium_sweep` at `/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s-20260821-geometry/gpu-divergence-diagnosis/nova/transport/coupled_window.py:955` routes through the plasma-current-bearing cold portfolio and `ForwardProfile.solve_portfolio` at `/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s-20260821-geometry/gpu-divergence-diagnosis/nova/equilibrium/forward.py:978`.
