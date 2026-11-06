"""Run paired per-file suites and the declared conic-channel omission."""

from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_booking_primal_buckets.suite import run_file
from scripts.prototypes.bank_wall_events.suite import archive

FILES = (
    "test_bank_step_booking_batch",
    "test_bank_step_booking",
    "test_book_primal_buckets",
    "test_conic_endpoint_consistency",
    "test_bank_newton_step",
    "test_bank_solve_driver",
)


def receipt(run, label, revision):
    logs = [run / f"{label}-{name}.log" for name in FILES]
    contents = [path.read_text() for path in logs]
    failures = [
        failure
        for text in contents
        for failure in re.findall(r"^FAILED (\S+)", text, re.M)
    ]
    complete = all(re.search(r"^=+ .* in .* =+$", text, re.M) for text in contents)
    exits = [int(text.strip().splitlines()[-1].split("=", 1)[1]) for text in contents]
    return dict(
        revision=revision,
        completed=complete,
        failure_count=len(failures),
        failure_ids=failures,
        exit_status=max(exits),
        log_paths=list(map(str, logs)),
    )


def main():
    root, run = map(Path, sys.argv[1:3])
    base_revision = sys.argv[3]
    head_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    run.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="jacobian-suite-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base, head, negative = (scratch / name for name in ("base", "head", "negative"))
        archive(root, base_revision, base)
        archive(root, head_revision, head)
        archive(root, head_revision, negative)

        def arm(label, tree, revision):
            for name in FILES:
                run_file(tree, revision, name, label, run, scratch)
            result = receipt(run, label, revision)
            (run / f"{label}-suite.json").write_text(json.dumps(result, indent=2))
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            before = pool.submit(arm, "base", base, base_revision)
            after = pool.submit(arm, "head", head, head_revision)
            baseline, candidate = before.result(), after.result()
        path = negative / "nova/equilibrium/current.py"
        source = path.read_text()
        needle = (
            "            with xla_metadata.set_xla_metadata("
            'tangent_site="interior:conic"):'
        )
        assert source.count(needle) == 1
        changed = source.replace(
            needle, "            velocity = velocity.at[0].set(0.0)\n" + needle
        )
        path.write_text(changed)
        (run / "negative-mutation.diff").write_text(
            "".join(
                difflib.unified_diff(
                    source.splitlines(True),
                    changed.splitlines(True),
                    fromfile="current.py",
                    tofile="dropped-channel/current.py",
                )
            )
        )
        negative_exit = run_file(
            negative,
            head_revision + "+dropped-channel",
            "test_bank_step_booking_batch",
            "negative",
            run,
            scratch,
            "conic_channel_jacobian",
        )
        added = set(candidate["failure_ids"]) - set(baseline["failure_ids"])
        print(
            "SUITES",
            json.dumps(
                dict(
                    base=baseline,
                    head=candidate,
                    added=sorted(added),
                    negative_exit=negative_exit,
                )
            ),
            flush=True,
        )
        print("Removing scratch", scratch, flush=True)
        return int(
            bool(added)
            or not baseline["completed"]
            or not candidate["completed"]
            or negative_exit == 0
        )


if __name__ == "__main__":
    raise SystemExit(main())
