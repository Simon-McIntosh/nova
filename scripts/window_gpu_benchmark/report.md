# Current-tree H200 coupled-window measurement

Tree: `143250540524dabcdd4eb0515a9e8de982498f8f`. SLURM job: `1253190`.

One job measured the identical gentle window on the CPU and one H200. The first run on each backend supplies the typed receipt and total window wall. Three deterministic repetitions supply nine warm iteration-pair samples by omitting each repetition's first pair.

## Declared window

- Length: `0.0025 s`
- Auxiliary-source multiplier: `0.5`
- Ordinary iteration cap: `10`; hard ceiling: `20`
- Convergence tolerance: `0.005`; contraction threshold: `0.8`
- Initial damping: `1.0`; damping floor: `0.125`
- Platforms: `cuda,cpu`; temporary directory: `/tmp`

## Direct same-tree result

| backend | device | iterations | first window wall (s) | warm equilibrium + FSA median (s) | warm TORAX median (s) | warm pair median (s), n=9 | gating exit norm | all-field exit norm | flux closure relative | current closure relative |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | cpu:0 | 4 | 80.045707 | 14.191111 | 0.038787 | 14.230813 | 0.00129137499 | 0.0325971191 | 0 | 2.08833542e-16 |
| gpu | cuda:0 | 4 | 65.634592 | 11.455618 | 0.167608 | 11.623376 | 0.00129137498 | 0.0325971191 | 0 | 2.08833542e-16 |

The measured warm iteration-pair speedup is `1.224327x`.

The prior cross-tree wall projection is retired by these direct, tree-identical measurements. Every TSV row carries the full tree SHA.

## Receipt comparison

CPU contraction: `0.23619506586175637`; H200: `0.23619506414001976`.
CPU gating/all-field exit: `0.0012913749879029162` / `0.03259711908303891`.
H200 gating/all-field exit: `0.0012913749800045624` / `0.032597119133942108`.
CPU flux closure absolute/relative: `0` / `0`.
H200 flux closure absolute/relative: `0` / `0`.
CPU current closure absolute/relative: `1.1641532182693481e-10` / `2.0883354249038394e-16`.
H200 current closure absolute/relative: `1.1641532182693481e-10` / `2.0883354249038429e-16`.

## Solver and placement checks

The equilibrium leg routes through `ForwardProfile.solve_portfolio` at `/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/47247cc9-be75-4e83-a22e-ed27792dda52/h200-window-measurement/nova/equilibrium/forward.py:1123`. The run inspected `1903` CPU and `1903` H200 solve/state array observations and failed closed if any JAX array occupied the wrong backend.
