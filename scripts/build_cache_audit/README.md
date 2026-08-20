# Forward build cache audit

## Outcome

Production machine construction has a persistent identity cache, but the two
reference fixtures do not use it. `Machine` stores its frame, subframe and
solved Biot groups in a versioned Zarr store below the platform user-data
directory. Its root group is `xxh64(canonical_key(Machine.group_attrs))`; the
identity includes geometry-source identities, data-dictionary version,
discretisation attributes, precision, and the four explicit source/target route
policies. Solved coupling products are child groups such as `plasmagrid` and
`plasmawall` beneath that root identity. The store is user-global, not
worktree-local. This workstation has a current-format component-cache example at
`~/.local/share/nova/0.11.0/iter_md_111001_203.zarr`, with root group
`e70e259654342847`; its frame, subframe and supply groups show that the
persistent production storage path is live, but it is neither a complete
`Machine` entry nor evidence for either reference fixture.

Focused synthetic round trips confirm the production mechanism rather than
merely finding its source code: a second `Datafile` construction reuses the
stored group without a second build, a Zarr write/read is xarray-identical, a
warm `Machine` has the same group and bit-identical stored data as its cold
construction, and changing only the data-dictionary version causes a new build
while repeating the same version does not.

The reference fixture constructs a plain `CoilSet`, calls
`plasmagrid.solve()` and `plasmawall.solve()`, and returns an in-memory
`HexMachine`. Its `_machine(cells, passive)` memo is process-local only. More
importantly, all three standalone consumers audited here call
`reference.build_machine(...)` directly, so even that memo is bypassed. There
is no persistent store and no persistent key for the coarse or fine fixture.
The measured miss is therefore a **bypassed persistent cache architecture**,
not a disabled cache, churning key, worktree-local store, or genuinely
uncacheable input.

## Build inventory

| Expensive product | Current reuse | Store | Key |
|---|---|---|---|
| Production machine description | Cached across processes | `${user_data_dir}/nova/${nova_version}/${machine_filename}.zarr` | `xxh64(canonical_key(Machine.group_attrs))` |
| Production coilset coupling | Cached with the machine | `<machine-store>/<machine-key>/<method-name>` | Machine semantic identity plus method group name |
| Production plasma-grid operators | Cached with the machine | `<machine-store>/<machine-key>/plasmagrid` and `plasmawall` | Machine semantic identity plus method group name |
| Coarse reference operator set, requested cells `-500`, wall nodes `3` | Rebuilt by every standalone run | None | None |
| Fine reference operator set, requested cells `-1000`, wall nodes `6` | Rebuilt by every standalone run | None | None |

The fixture operator set includes source-to-centroid, plasma-to-centroid,
source-to-direct-sample, plasma-to-direct-sample, source-to-wall and
plasma-to-wall flux/derivative blocks. Those arrays are deterministic once the
semantic fixture identity is fixed and are therefore cacheable.

## Timing evidence

The banked progress logs expose each completed `plasmagrid` and `plasmawall`
assembly. Summing those assembly durations gives three completed cold samples
per fixture:

| Fixture | Cold samples | Median | Range | Persisted warm |
|---|---:|---:|---:|---:|
| Coarse | 785 s, 794 s, 1,157 s | 794 s (13 min 14 s) | 13 min 05 s–19 min 17 s | Not measurable: no entry exists to load |
| Fine | 2,278 s, 2,614 s, 3,260 s | 2,614 s (43 min 34 s) | 37 min 58 s–54 min 20 s | Not measurable: no entry exists to load |

These are operator-build times, not whole-script wall times. The exact logs and
parsed figures are recorded in `results.json`. A cold-versus-warm identity test
cannot honestly be run until the fixture path persists and reloads the operator
carrier. The required change is outside this audit's write scope: the reference
fixture module must serialize/restore `HexMachine`, and the standalone
consumers must request that loader instead of calling `build_machine` directly.

## Required semantic identity

The fixture key must be a semantic descriptor, never a digest of an HDF5 file.
HDF5 write metadata can change a file digest while leaving the equilibrium
identical. At minimum the descriptor must include the pulse, run,
data-dictionary version and time-slice selector; normalized content hashes for
the wall, active and passive conductor geometry, turns and currents; the
requested plasma-cell count, coil filament count, wall-node count and passive
decomposition; numerical precision; every source/target route policy; and a
cache-schema identity. The coarse and fine descriptors differ in requested
cell count and wall-node count. After loading, every stored array must compare
bitwise at stored precision with the cold construction before the warm entry is
accepted.

## Agent-facing cache doctrine

Name a reusable machine by its complete semantic inputs, resolve that identity
to Nova's versioned user-data Zarr store, and ask the cache for the machine
before constructing any geometry or Green operator. A miss may build once and
atomically publish the fixed-shape frame and operator groups; a hit must restore
those groups at their stored dtype and validate the stored semantic descriptor.
Never use a worktree path, source-file modification time, or raw HDF5 digest as
identity, and never call the expensive builder directly from a run harness—the
harness names the cached instance, while only the cache-owned miss path is
allowed to build it.

## Reproduction

Run `audit.py` with the repository root plus the four banked log files. It
asserts the production identity contract, verifies each standalone consumer
still calls the direct builder, parses completed operator-build timings, and
writes `results.json`.
