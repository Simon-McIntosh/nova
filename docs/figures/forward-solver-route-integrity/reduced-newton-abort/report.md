# The SIGABRT exit 134 in tests/test_reduced_newton.py

## Verdict: a compile-memory abort, caused by the memory-map count reaching the kernel ceiling

The abort is LLVM's `report_fatal_error` in the CPU JIT linker. It fires when
the process has used up its memory-map entries: `/proc/<pid>/maps` sits at the
kernel limit, `vm.max_map_count = 65530`, when the next executable's code
sections are mapped. Resident memory is not the constraint: it peaks at 8.6 GiB
against a 64 GiB allocation. The abort is not a numerical abort and not a
runtime assertion; no NaN check, XLA runtime frame or Python `assert` is on the
aborting stack.

It is a whole-process effect of the programs this file compiles, not a defect of
the test it lands in. At main HEAD (f5af729a9) the abort has moved. The test
this node was dispatched on, `test_threshold_one_reproduces_the_refusal_only_policy`,
now **passes** inside the whole file (test 9 of 21). The same process aborts
eleven tests later, in test 20,
`test_compiled_slice_replays_the_host_step_and_trip_counters`. The abort needs
enough programs to have run first, not one particular test, so where it lands
moves whenever a test is added or a program's size changes.

![Memory-map count against elapsed time, fresh processes](/nova/figures/forward-solver-route-integrity/reduced-newton-abort/map-count.png)

*Entries in `/proc/<pid>/maps`, sampled every 5 s, for each fresh-process arm at
main HEAD. The dashed line is `vm.max_map_count`. Within each test the count
rises as programs are compiled and loaded, and falls at the test boundary, but
the level it falls back to ratchets up: about 15k, 21k, 34k, then 42k. Both
whole-file arms end at the ceiling: 62,814 five seconds before the abort with the
persistent cache on, and 65,078 (452 below the ceiling) with it off.*

## The log lines that name it

Both whole-file processes print `Fatal Python error: Aborted` and `EXIT=134` in
test 20. The Python frames end at `nova/equilibrium/reduced_newton.py:428 in
_timed`, called from `_plain_newton_trip` (line 1179), `_drive_trips`
(line 1374) and `solve_reduced_newton` (line 1939). The frame beneath differs by
arm:

| Arm | Frame under `compile_or_get_cached` | Maps, last sample | Peak RSS |
|---|---|---|---|
| `whole-file` (persistent cache on) | `compilation_cache.py:352 get_executable_and_time`, i.e. loading a cached executable | 62,814 | 8.63 GiB |
| `whole-file-no-persistent-cache` (`JAX_ENABLE_COMPILATION_CACHE=false`) | `compiler.py:350 backend_compile_and_load`, i.e. compiling and loading a new one | 65,078 | 8.27 GiB |

The native frames of both aborts symbolise, against jaxlib 0.11.0
`libjax_common.so`, to the same chain (`prior-abort-symbolised.txt` for the
earlier run, and the `whole-file` log's own addresses re-symbolised identically):

```
llvm::report_fatal_error(llvm::Twine const&, bool)
llvm::RuntimeDyldImpl::emitSection(...)
llvm::RuntimeDyldImpl::findOrEmitSection(...)
llvm::RuntimeDyldImpl::loadObjectImpl(...)
llvm::RuntimeDyldELF::loadObject(...)
llvm::jitLinkForORC(...)
llvm::orc::RTDyldObjectLinkingLayer::emit(...)
```

`emitSection` raises `report_fatal_error` only when the memory manager returns
no memory for a section: "Unable to allocate section memory!", the same message
the centroid sweep printed on this plan. LLVM's section memory manager takes
each section from its own `mmap`, so a process at the map-count ceiling cannot
place the next section, whatever its free RAM. The persistent cache is not
necessary: with it off, the fresh compile's load aborts at the same site, one
arm-length (35 min against 22 min) later, because compiling instead of reading
takes longer, not because the ceiling moved.

The `LLVM ERROR` text itself is not in these logs. pytest's default fd-level
capture swallows LLVM's direct write to file descriptor 2, and the abort comes
before pytest can replay it; only `faulthandler`, which writes to the saved
original stderr, reaches the log. The two `--capture=no` arms did not abort, so
they could not recover the message either.

## Fresh-process arms at f5af729a9, all_debug job 1276136 (and 1276137 for the uncaptured pair)

| Arm | Result | Peak RSS | Peak maps |
|---|---|---|---|
| `whole-file` | 2 failed, 15 passed, then SIGABRT in test 20, EXIT=134 | 8.63 GiB | 62,814 |
| `whole-file-no-persistent-cache` | same outcomes, SIGABRT in test 20, EXIT=134 | 8.27 GiB | 65,078 |
| `tests-before` (tests 1 to 8) | 2 failed, 6 passed, 736.58 s, EXIT=1 | 6.90 GiB | 54,653 |
| `abort-alone` (test 9 alone) | 1 passed, 149.78 s, EXIT=0 | 2.73 GiB | 28,987 |
| `uncaptured-abort-alone` | 1 passed, 147.66 s, EXIT=0 | 2.61 GiB | 31,106 |
| `certificate-then-abort` (tests 1 and 9) | test 1 failed, test 9 passed, EXIT=1 | 6.24 GiB | 48,327 |
| `uncaptured-certificate-then-abort` | same, EXIT=1 | 6.47 GiB | 48,036 |

Before the abort, both whole-file arms also fail tests 1, 5, 18 and 19. The
failure text of tests 18 and 19 is lost, because the abort prevents pytest's
summary. Test 1 fails with `assert 2.340823366528062 == 2.248351582217136` and
test 5 with `assert False`, as the tests-before summary shows. None of these is
an abort.

## Smallest ordered subset

BISECT-PENDING

Each arm's pytest log (`*.log`; its first line names revision, tree, host and
command), its 5 s memory samples (`*.memory.txt`) and, for the bisect arms, its
per-test boundary rows (`*.timeline.txt`, written by `abort_timeline.py`) sit
beside this report. `abort-arms.sh`, `abort-uncaptured.sh` and `abort-bisect.sh`
are the job payloads; `map_count_figure.py` draws the figures.
