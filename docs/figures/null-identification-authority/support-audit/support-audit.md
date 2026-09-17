# Support-row audit of the bank drift

Question put to this node: do the producer tree (`fae50f15`) and the current
tree carry different *support in the solve* — the constraint rows and their
compensating unknowns, net current, vertical position and any other support —
such that the active-set cycle on the drifted arms is a missing or misrouted
row, rather than something downstream of the operands?

**Answer: no.** The support is identical on both trees, on all twelve paired
arms. What differs is *where the same support enters the computation*: the
current tree traces the three declared-anchor rows as data through
`_dynamic_extra_names` (`nova/equilibrium/forward_operator.py:2121-2131`),
while the producer tree held
the same three values as host constants. The partition is the first stage that
differs, and it differs for exactly that reason. No solver code was changed by
this audit.

Measured at the producer tree archived under
`…/nova/s19-local/bank-drift/faefloor-tree` and the current tree at
`299e4b64`, from the banked emissions and
receipts. Audit program: `benchmarks/bank_drift_support_audit.py`. Run log:
`…/nova/s19-local/bank-drift-support/audit-run4.log` (an earlier run of the same program, `audit-run3.log`, reports
the identical digests and summary). Machine-readable table:
`…/bank-drift-support/support-table.json`.

## What was read, and where each column comes from

Both families of column are reported, and the source is stated rather than
implied, because the two live in different files and merging them silently is
how a blank column passes for a measured absence:

| Column family | Fields | Read from |
|---|---|---|
| receipt | converged, termination reason, terminal residual, stage exception | the `arms[…]` block on each row of an emission or receipt file |
| stage | `profile_support`, `partition_stage`, `partition_values` digests, leaf count, pytree flag | the `stages` block **on the row** |

The `stages` block is on the row, not on the payload. Reading a sizeable
`stages` key at payload level, as a first pass of this audit did, returns
nothing for every file and turns the whole comparison into an empty one that
prints like a pass. The audit refuses that case explicitly: a pair with stage
evidence on only one side is reported as `NO STAGE EVIDENCE — not a pass` and
counted separately, and a receipt field that is present but empty is reported as
`EMPTY@<source file>` rather than a blank.

## The paired arms

Twelve arms: six shots (21978/35, 21983/35, 21985/51, 21986/46, 21989/55,
22086/43) × {pure, mixed}. All twelve paired; none unpaired.

### Receipts

| Identity | Arm | producer `fae50f15` | current tree |
|---|---|---|---|
| 21978/35 | pure | converged, 3.85734e-16 | active_set_cycle_detected, 2.4506488e-03 |
| 21978/35 | mixed | converged, 3.09965e-16 | converged, 2.2042e-16 |
| 21983/35 | pure | active_set_settled, 2.39868e-04 | active_set_settled, 4.36866e-03 |
| 21983/35 | mixed | active_set_settled, 2.46096e-04 | active_set_cycle_detected, 3.78715e-04 |
| 21985/51 | pure | active_set_settled, 6.68082e-04 | active_set_settled, 6.70245e-04 |
| 21985/51 | mixed | active_set_settled, 6.53523e-04 | active_set_settled, 6.56942e-04 |
| 21986/46 | pure | active_set_settled, 1.35305e-03 | active_set_iteration_budget_exhausted, 5.88986e-03 |
| 21986/46 | mixed | active_set_settled, 1.34655e-03 | active_set_settled, 1.34642e-03 |
| 21989/55 | pure | active_set_settled, 7.34998e-04 | active_set_cycle_detected, 3.64673e-03 |
| 21989/55 | mixed | active_set_settled, 7.37635e-04 | active_set_settled, 7.32123e-04 |
| 22086/43 | pure | converged, 3.58498e-16 | active_set_settled, 4.77494e-04 |
| 22086/43 | mixed | converged, 4.92935e-16 | converged, 2.68874e-16 |

### Stage digests

`profile_support` is `5683f88ca97c076e` on **every row of the table** — all
twelve arms, both trees, and both arms of every identity. Its parts are
identical everywhere too: `inside_material 7406355ef5eb56a2`,
`physical_node_number e706f97a7d7d60e5`. The operands did not move.

`partition_structure` splits the two trees cleanly, and does so in a way that
names the mechanism:

| Identity | producer `fae50f15` | current tree |
|---|---|---|
| 21978/35 | 236cbb2e1af2dded, values `empty` | 0fe0dcdf920a9b66, values 35c0be67ae2abdd8 |
| 21983/35 | 236cbb2e1af2dded, values `empty` | c6153172715a1fbf, values 15188b6a218bec95 |
| 21985/51 | 236cbb2e1af2dded, values `empty` | 1aa36966cece689e, values d1e66e165e3f6289 |
| 21986/46 | 236cbb2e1af2dded, values `empty` | 7ce799b69367d32d, values f7f8a469f04c1a05 |
| 21989/55 | 236cbb2e1af2dded, values `empty` | 2bffef758c015b38, values 4999adc10ee82597 |
| 22086/43 | 236cbb2e1af2dded, values `empty` | b4214b763f3aed67, values 8d95bb38b1c218c |

The producer tree's structure digest is **one constant across six different
shots**, with the partition *values* digest literally `empty` and a leaf count
of 1. A partition that is byte-identical for six different equilibria and
carries no values is a partition with nothing shot-specific in it: the declared
anchors were host constants. The current tree gives a distinct structure digest
per identity — with the same digest for the two arms of one identity, since the
partition depends on the declared rows and not on the arm's policy overrides —
a leaf count of 76 and a `leaf_count=76 / tree_node_count=90` pytree.

## The mechanism

Source inventory of `nova/equilibrium/forward_operator.py` in each tree:

| Tree | pytree registered | `_dynamic_extra_names` | rows returned |
|---|---|---|---|
| `fae50f15` | yes | **absent** | — |
| current | yes | present | `declared_axis_flux`, `declared_boundary_flux`, `declared_support` |

At the current tree, `__init__` casts each of those members to a `jnp` array
(`forward_operator.py:2117-2118`) and `_dynamic_extra_names` returns them so they
enter the trace **as data**; the same three members are declared and written
identically in both trees, and are also consumed by
`_current_moments_on_partition`, which is present in both trees at the same
count. No hook exists at `fae50f15`, so the identical values were captured as
host constants at trace time. That is the whole of the difference: identical
support content, rerouted into the traced partition.

### Corroboration that the builder did not change

The bank builder writes those three rows, and between `fae50f15` and `299e4b64`
it is byte-identical: the builder is the only consumer of the anchor names in
`benchmarks/efit_forward_parity_slice.py`, and its occurrences of
`declared_axis_flux` (18), `declared_boundary_flux` (15) and `declared_support`
(27) are the same in both trees, as is every expression that derives them
(`declared_support` remains the LCFS-and-material mask, the axis and boundary
fluxes remain `TOTAL_FLUX_FACTOR`-scaled bank values). The only difference
between the two trees in that file is the solve-seam conversion (`solve_portfolio`
/ `solve_branch` giving way to `ForwardSolveRequest` plus `profile.solve`), and
nothing in it touches the anchors.

## The caveat: these are not augmented rows

Stated plainly, because "constraint rows and compensating unknowns" invites a
reading this solve does not support. On this corroboration solve **neither tree
carries `constraint_pairs`** — the request field defaults to empty on both
sides. There is therefore no `CircuitCurrentUnknown`, no
`BoundedExteriorFieldUnknown` and no `ProfileAmplitudeUnknown` row present on
one tree and absent on the other. The support that *does* exist is:

* **net current** — the operator-internal normalisation
  (`current_normalisation_amplitude`, present in both trees'
  `forward_operator.py` at identical count 4), not an augmented circuit row;
* **vertical position / axis** — the declared `declared_axis_flux` anchor,
  consumed through the same partitioned operator, not an independent row;
* **participation support** — `declared_support`, the LCFS-and-material mask,
  digest-identical on both trees.

So the honest form of "the rows are present on both sides" is: *every piece of
support this solve carries is present on both trees; none is missing, none is
reordered, and none is a compensating unknown that one tree omits.*

## What this establishes, and what it does not

**Established.** Support is not the explanation for the active-set cycle. Any
repair that begins by adding, restoring or reordering a support row begins from
a defect that is not there. The difference to work on is trace-time
specialisation of the declared anchors: identical host values, carried as data
at the current tree.

**Not established.** This audit does not show that the traced-row routing
*causes* the non-convergence. It removes support as the cause and localises the
drift to the partition, which is where the paired probe (`nia-drift-probe-*`)
already placed it. Establishing causation needs the two trees compared at equal
compile cost with the anchor routing held fixed.

## Reproduce

```bash
R=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local
/home/ITER/mcintos/Code/nova/.venv/bin/python benchmarks/bank_drift_support_audit.py \
  --reports-root "$R" \
  --old-tree "$R/bank-drift/faefloor-tree" \
  --new-tree "$PWD" \
  --json-out "$R/bank-drift-support/support-table.json"
```

Source-level and receipt-level reads only: no geometry is built, no IMAS data is
opened, nothing is compiled. No figure accompanies this record: the result is
two digest columns and a partition summary, which is a table, and a plot of
digest hex would add no spatial or sequential information the table withholds.

## Instrument note

One defect found and fixed in this audit's own instrument is worth recording.
The first version of the per-arm table printed a blank residual column for every
arm while the same field was present in the merged record and in the JSON
output: the printer looked up `termination_residual` against a field named
`terminal_residual`, and `dict.get` on a misspelled key returns `None`, which
the cell formatter rendered as a bare `-`. Nothing raised, and the column read
as "the emission does not report a residual". The lookup now goes through a
helper that names the source file for an empty value and prints `NO-FIELD` for a
key that was never populated — the guard was made to speak. The audit's payload-level
`stages` read and a `"self.%s" in source` membership test that never substituted
its placeholder were the same defect in a different costume: a lookup whose
failure mode is an empty result that reads as a finding about the data.