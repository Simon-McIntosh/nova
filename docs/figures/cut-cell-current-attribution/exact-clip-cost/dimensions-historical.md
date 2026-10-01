# The swept support capacity in these receipts is historical

Each `parts/production-<cells>.json` receipt records an H200 warm-cost run
(jobs 1271886 / 1271899 / 1271922, 2026-09-16) compiled against the traced
support capacity as it was then: `atomic_support_capacity *
spline_chain_samples_per_chord`, the swept product. It declared

| requested cells | `atomic_support_capacity` | recorded `exact_support_capacity` |
|---|---|---|
| 110 | 30 | 3840 |
| 300 | 24 | 3072 |
| 1000 | 22 | 2816 |

The swept product was retired by revision `f7fefbd7`, which derives the bound
from the realised layout instead — one traced arc plus the straight edge chain,
`spline_chain_samples_per_chord + atomic_support_capacity` — giving 158, 152 and
150 for the same three realised layouts. The retirement was code-only at the
time of the repair; these committed receipts still carried the swept value
while the code no longer computed it.

Each receipt's own `dimensions_provenance` block now states that
`dimensions.exact_support_capacity` is the historical swept value, names the
retiring revision, gives the bound the current code derives for the same
realised layout, and does the same for
`dimensions.exact_quadrature_points_per_cell`, which is a property of the swept
capacity rather than of any live route (the exact route's closed-form polygon
moments evaluate no per-cell quadrature points).

The recorded value is kept rather than rewritten because these receipts record
runs compiled before the retirement: the capacity is part of the program those
runs compiled, and the compile wall, solve timings, stage fraction and allocator
statistics beside it were measured with it. Rewriting the field would describe a
program that was never compiled there. Nothing downstream reads the committed
value — `benchmarks/program_shape_census.py` rebuilds capacities from
`solovev_certificate._certificate_compile_problem` — so no consumer inherits the
historical number.

Annotated by `annotate_dimensions_provenance.py` in this directory (login node,
root interpreter, exit 0); it touches only `dimensions_provenance` and leaves
every measurement field byte-identical.