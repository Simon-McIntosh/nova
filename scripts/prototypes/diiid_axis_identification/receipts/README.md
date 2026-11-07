# Receipts — frame 144 limited-branch axis identification

Committed copies of the JSON receipts the evidence fragment cites. They were
written under the crew run directory, which is not part of the repository, so
they are copied here to keep the evidence readable after the run is collected.
Absolute worktree paths inside them are rewritten to `<repository>`.

| file | produced by | what it holds |
|---|---|---|
| `replay-frame-144-limited.json` | `run_probe_frames.py --frames 144 --branches limited` on base `8130772bb`, debug job `1308081` | the frame read from scratch with no cached seed: `status: refused`, `NoQualifiedAxisError`, empty `arms`, seed L2 from label 5.57988848578467, and the traceback placing the raise in the current-moment integral |
| `conjunct-144-limited-c.json` | `conjunct_probe.py` on the cached limited seed, job `1308039` | the three qualification conjuncts per candidate, mapped over the candidate array |
| `conjunct-captured-field.json` | `conjunct_probe.py --capture`, job `1308066` | the same conjuncts on the field the solve hands to `read_qualification` |
| `refusal-144-limited-b.json` | `capture_refusal.py`, job `1308062` | the read calls made during the solve, and the trace that ends the capture |

Reproduce, on any debug partition with the shared environment:

```bash
W=$(git rev-parse --show-toplevel)
PYTHONPATH="$W" /home/ITER/mcintos/Code/nova/.venv/bin/python \
  scripts/prototypes/diiid_forward_convergence/run_probe_frames.py --frames 144 \
  --branches limited --probe PROBE.json --part-dir PARTS --artifact-dir ARTIFACTS \
  --repository-root "$W"
```

`PROBE.json` must exist and carry the frames to merge into; a fresh empty object
works for a single frame.
