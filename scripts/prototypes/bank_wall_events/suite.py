"""Run isolated per-file CPU arms and retain complete, attributable logs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import difflib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

BASE_FILES = (
    "test_support_read",
    "test_tracked_read",
    "test_bank_solve_driver",
    "test_bank_solve_cycle",
)


def archive(root, revision, target):
    target.mkdir()
    stream = subprocess.Popen(
        [
            "git",
            "-C",
            str(root),
            "archive",
            revision,
            "nova",
            "scripts",
            "tests",
            "benchmarks",
            "pyproject.toml",
            "uv.lock",
        ],
        stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-x", "-C", str(target)], stdin=stream.stdout, check=True)
    stream.stdout.close()
    assert stream.wait() == 0


def run_file(tree, revision, name, label, run, scratch, selection=None):
    command = [
        sys.executable,
        "-P",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        f"--basetemp={scratch / (label + '-' + name)}",
        str(tree / "tests" / (name + ".py")),
        "-m",
        "slow or not slow",
        "-vv",
        "-s",
    ]
    if selection:
        command += ["-k", selection]
    environment = dict(
        os.environ,
        PYTHONPATH=str(tree),
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
    )
    path = run / f"{label}-{name}.log"
    with path.open("w") as log:
        log.write(
            f"# revision: {revision} tree: {tree} command: {shlex.join(command)}\n"
        )
        log.flush()
        subprocess.run(
            [
                sys.executable,
                "-P",
                "-c",
                "import nova, nova.equilibrium.topology as m; "
                "from pathlib import Path; print('nova',nova.__file__,'module',m.__file__,"
                "'cwd',Path.cwd().resolve(),flush=True)",
            ],
            cwd=tree,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        result = subprocess.run(
            command, cwd=tree, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        log.write(f"EXIT={result.returncode}\n")
    print("FILE_COMPLETE", label, name, result.returncode, flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="wall-suite-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base = scratch / "base"
        archive(args.root, args.base, base)
        negative = scratch / "negative"
        archive(args.root, head, negative)
        path = negative / "nova/equilibrium/topology.py"
        source = path.read_text()
        target = "concave, jnp.clip(parameter - update, lower, upper), parameter"
        assert source.count(target) == 1
        path.write_text(
            source.replace(
                target,
                "concave, jnp.where(jnp.all(first == start[0]), parameter - update, "
                "jnp.clip(parameter - update, lower, upper)), parameter",
            )
        )
        (args.run / "negative-mutation.diff").write_text(
            "".join(
                difflib.unified_diff(
                    source.splitlines(True),
                    path.read_text().splitlines(True),
                    fromfile="original",
                    tofile="unclipped-first-edge",
                )
            )
        )
        run_file(
            negative,
            head + "+unclipped-first-edge",
            "test_wall_events_batched",
            "negative",
            args.run,
            scratch,
            "seed_brackets_and_candidate_order",
        )

        def arm(label, tree, revision, names):
            return [
                run_file(tree, revision, name, label, args.run, scratch)
                for name in names
            ]

        with ThreadPoolExecutor(max_workers=2) as pool:
            before = pool.submit(arm, "base", base, args.base, BASE_FILES)
            after = pool.submit(
                arm, "head", args.root, head, ("test_wall_events_batched", *BASE_FILES)
            )
            results = {"base": before.result(), "head": after.result()}
        (args.run / "suite-status.json").write_text(json.dumps(results))
        print("SUITES", results, flush=True)
        print("Removing scratch", scratch, flush=True)
        return int(any(results["base"] + results["head"]))


if __name__ == "__main__":
    raise SystemExit(main())
