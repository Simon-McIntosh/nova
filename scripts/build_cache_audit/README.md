# Forward build cache audit

## Outcome

Every reusable operator carrier in the audited forward path now has a
persistent semantic identity. Production `Machine` construction already stored
its frame, subframe, coil coupling and solved plasma operators in a versioned
Zarr store below the platform user-data directory. The Solovev reference lane
was the exception: each of its three standalone measurement programs called
`build_machine(...)` directly, rebuilding deterministic coarse and fine
`HexMachine` arrays on every process invocation. Those consumers now request a
shared semantic cache. Its only cold path calls the direct builder; a warm
request restores the native array carrier without constructing a `CoilSet` or
evaluating a Green operator.

The shared fixture store on this host is
`/home/ITER/mcintos/.local/share/nova/0.11.0/solovev_hex_machine.zarr`. The
coarse identity is `746fbe1553c4b242`; the fine identity is
`f0f96aa214aa9459`. Both groups were confirmed absent immediately before their
measurement jobs, built exactly once, then requested a second time in the same
job. The coarse request fell from 1,246.880145 s cold to 0.265939 s warm, and
the fine request fell from 3,335.325393 s cold to 0.817662 s warm. In both cases
the cold carrier and warm restore agreed in dtype, shape and bytes for all 31
arrays (`51,411,470` bytes coarse and `169,311,277` bytes fine).

Publication is serialized with an advisory lock beside the shared store. The
writer publishes the named Zarr group in place rather than relying on a
no-clobber rename, which GPFS rejects. Readers use the same lock and accept a
group only after its schema, exact semantic descriptor, payload digest, array
inventory and key validate. An interrupted or invalid group is deleted and
rebuilt while the lock remains held, so in-place publication does not turn a
partial group into a warm hit.

## Build inventory

| Expensive product | Reuse | Store | Key |
|---|---|---|---|
| Production machine description | Cached across processes | `${user_data_dir}/nova/${nova_version}/${machine_filename}.zarr` | `xxh64(canonical_key(Machine.group_attrs))` |
| Production coilset coupling | Cached with the machine | `<machine-store>/<machine-key>/<method-name>` | Machine semantic identity plus method group name |
| Production plasma-grid operators | Cached with the machine | `<machine-store>/<machine-key>/plasmagrid` and `plasmawall` | Machine semantic identity plus method group name |
| Coarse Solovev operator carrier, requested cells `-500`, wall nodes `3` | Cached across processes | Shared `solovev_hex_machine.zarr` | `746fbe1553c4b242` |
| Fine Solovev operator carrier, requested cells `-1000`, wall nodes `6` | Cached across processes | Shared `solovev_hex_machine.zarr` | `f0f96aa214aa9459` |

Production's root machine identity includes geometry-source identities,
data-dictionary version, discretisation, numerical precision and the explicit
source/target route policies. Its solved coupling products are child method
groups below that root. Focused synthetic tests confirm this is an active
mechanism: repeated `Datafile` and `Machine` construction select the same group
without a second build, the warm machine data are bit-identical, and changing
only the data-dictionary version selects a distinct group.

The reference carrier includes source current, mesh nodes and areas, packed
cell polygons, stencil and sampling geometry, every source-to-grid,
plasma-to-grid, source-to-sample, plasma-to-sample, source-to-wall and
plasma-to-wall flux/derivative block, plus the radial and vertical field
blocks. Each array is stored at the dtype produced by the builder. The packed
polygon vertices and offsets preserve the one variable-length part of the
carrier without object arrays. The in-module `_machine` memo remains a direct,
process-local build because a few plotting tests require the live `CoilSet`;
the three long-running standalone consumers use only the persisted numerical
carrier and are the repeated-run path this cache removes.

## Why the fixtures rebuilt

The miss mechanism was measured as a **bypassed persistent cache
architecture**. Source inspection showed all three standalone consumers
calling `reference.build_machine(...)`, and completed logs showed the full
operator progress on every invocation. The reference module had only an
`lru_cache`, which cannot survive process exit and was bypassed by those calls
anyway. This rules out a disabled cache, a churning key, a worktree-local store
and genuinely uncacheable input. HDF5 file digests were deliberately excluded:
write metadata changes those bytes between equivalent writes, while the
equilibrium's semantic content remains the same.

The fixture identity instead contains the pulse, run, data-dictionary version
and time-slice locator; normalized content hashes for every reference array;
all reference scalars; active and passive conductor geometry, names, turns and
currents; unplaced conductors; requested cell count, plasma shape, coil
filaments, wall nodes, passive decomposition and passive inclusion; numerical
precision; all route policies; and the cache schema. The coarse and fine keys
differ through their discretisation rather than through a filename or a raw
input-file digest.

## Timing and identity evidence

The historic progress logs contain three complete cold operator builds per
fixture. Their summed assembly durations establish the repeated-run baseline:

| Fixture | Banked cold samples | Median | Range |
|---|---:|---:|---:|
| Coarse | 785 s, 794 s, 1,157 s | 794 s (13 min 14 s) | 13 min 05 s–19 min 17 s |
| Fine | 2,278 s, 2,614 s, 3,260 s | 2,614 s (43 min 34 s) | 37 min 58 s–54 min 20 s |

The activation measurement times the complete cache request, including its
miss build/store/validation or hit load/validation:

| Fixture | Measured cold | Warm reload | Warm/cold | Reduction | Stored identity |
|---|---:|---:|---:|---:|---|
| Coarse | 1,246.880145 s | 0.265939 s | 0.021328% | 4,688.59× | 31 arrays / 51,411,470 bytes, bitwise |
| Fine | 3,335.325393 s | 0.817662 s | 0.024515% | 4,079.10× | 31 arrays / 169,311,277 bytes, bitwise |

`measurements-coarse.json` and `measurements-fine.json` retain the cache
receipt components, exact keys, realised cell counts and byte totals.
`results.json` combines those records with the historic cold distributions and
the source-contract audit.

## Agent-facing cache doctrine

Name a reusable machine by all semantic inputs that can change its fixed
geometry or operators, including discretisation, precision, route policy and a
bumpable schema, then resolve that name in Nova's versioned shared user-data
Zarr store before constructing geometry or evaluating a Green kernel. A run
requests the named instance and works with the returned carrier; only the
cache-owned, locked miss path may invoke the expensive builder. Store numerical
inputs at their native dtype, validate descriptor and bytes on reload, recover
an interrupted in-place publication under the same lock, and never put a
worktree path, modification time, or raw HDF5 digest into the identity.

## Reproduction

`measure_cache.py --fixture coarse|fine --require-cold` performs the cold then
warm request and writes one measurement artifact per fixture. `audit.py` takes
those two artifacts plus the four banked run logs, asserts the source and
measurement contracts, and writes `results.json`. The long measurement jobs
run on CPU compute nodes; their complete logs are named in the worker manifest.
