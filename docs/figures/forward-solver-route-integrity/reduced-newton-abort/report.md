# The SIGABRT at test_threshold_one_reproduces_the_refusal_only_policy

Status: **partial.** The abort's class is established from the symbolised C stack.
The whole-file reproduction at main HEAD is still running in all_debug job
1276136, so the smallest reproducing subset is not yet closed.

## Verdict so far: a compile-memory abort, not a numerical abort or an assertion

The prior abort (`baseline-reduced-newton.log`, revision 7c74bf18b) died inside
`jax/_src/compilation_cache.py:352`. That is `backend.deserialize_executable`,
which loads an executable read back from the persistent compilation cache. The
native frames, symbolised against jaxlib 0.11.0 `libjax_common.so`
(`prior-abort-symbolised.txt`), are:

```
0x76a2178  llvm::report_fatal_error(llvm::Twine const&, bool)
0xc77fc2d  llvm::RuntimeDyldImpl::emitSection(...)
0xc77ea5a  llvm::RuntimeDyldImpl::findOrEmitSection(...)
0xc77ce0a  llvm::RuntimeDyldImpl::loadObjectImpl(...)
0xc788c76  llvm::RuntimeDyldELF::loadObject(...)
0xc7823e2  llvm::jitLinkForORC(...)
0xc7726ca  llvm::orc::RTDyldObjectLinkingLayer::emit(...)
```

`emitSection` calls `report_fatal_error` in one place: when the memory manager
returns no memory for a code or data section ("Unable to allocate section
memory!"). This is the same failure as the centroid-sweep abort at
`LLVM ERROR: Unable to allocate section memory` recorded on this plan. The abort
is the linker running out of mapped memory while loading object code. It is not
a Python assertion, which would be a pytest failure rather than an abort, and it
is not a numerical abort, since no NaN check or XLA runtime frame is on the stack.

**Why earlier logs carry no `LLVM ERROR` line.** pytest captures output at the
file-descriptor level by default, so LLVM's direct write to fd 2 goes into
pytest's capture file. The process aborts before pytest can replay that file.
`faulthandler` writes to the saved original stderr, so only its dump reaches the
log. Job 1276137 runs the same processes with `--capture=no` so the message can
land if the abort fires.

## Fresh-process arms at f5af729a9 (main HEAD), all_debug

Memory maps come from `/proc/<pid>/maps`, sampled every 5 s. The kernel ceiling
on this node is `vm.max_map_count = 65530` (`allocation.txt`).

| Arm | Result | Peak RSS (GiB) | Peak map count |
|---|---|---|---|
| `abort-alone` (the aborting test alone) | 1 passed, 149.78 s, EXIT=0 | 2.73 | 28,987 |
| `uncaptured-abort-alone` (same, `--capture=no`) | 1 passed, 147.66 s, EXIT=0 | 2.61 | 31,106 |
| `certificate-then-abort` (first test, then the aborting test) | certificate FAILED, aborting test PASSED, EXIT=1 | 6.24 | 48,327 |
| `uncaptured-certificate-then-abort` | same outcome, EXIT=1 | 6.47 | 48,036 |
| `tests-before` (the eight tests before it) | running | — | 45,957 so far |
| `whole-file` (the reproduction) | running, 4 tests reported | — | 48,325 so far |
| `whole-file-no-persistent-cache` (`JAX_ENABLE_COMPILATION_CACHE=false`) | running, 4 tests reported | — | 48,323 so far |

The certificate test fails the same way in both arms that run it
(`assert 2.340823366528062 == 2.248351582217136`), as recorded at base on this
plan. That failure is outside this node's scope.

What the arms show so far:

- The aborting test does not abort on its own. It is not a defect of that test's
  program.
- Running the persistent-cache-enabling certificate test first is not enough
  either, even though the mapping count peaks at 48k during that test. The count
  falls back afterwards (to 12,259 at the end of `certificate-then-abort`).
  Mappings are released as executables are freed, so the count reached depends on
  what the intervening tests keep alive, not only on the certificate test.
- The mapping count is the quantity that approaches the kernel ceiling. Resident
  memory stays under 7 GiB against a 64 GiB allocation. A `mmap` refusal at the
  map-count ceiling is therefore the leading candidate over resident exhaustion.
  It is not yet confirmed at the moment of the abort.

## Still open, and what closes it

1. Whether `whole-file` aborts at the same test at main HEAD, with its map count
   and RSS in the samples just before the abort.
2. Whether `whole-file-no-persistent-cache` aborts. If it passes the aborting
   test, the cache-read path (`deserialize_executable`) is necessary. If it
   aborts in `compile` instead, the mapping growth is program count in general.
3. The smallest ordered subset: the prefix after the certificate test whose
   retained executables push the aborting test's section allocation over the
   ceiling. Bisect the six middle tests once (1) is in.

Each arm's pytest log (`*.log`, first line names revision, tree, host and
command) and its memory samples (`*.memory.txt`) sit beside this report.
`abort-arms.sh` and `abort-uncaptured.sh` are the job payloads.
