"""Run paired suites or measurements in one allocation with isolated processes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
BASE = "519d378f9e6f8d2786f69af64569625971651a73"


def execute(tree, command, log, revision, cpus):
    env = dict(
        os.environ, TMPDIR=os.environ["TMPDIR"], PYTHONPATH=".", JAX_PLATFORMS="cpu"
    )
    with log.open("w") as stream:
        stream.write(f"REVISION={revision} TREE={tree} CWD={tree}\n")
        stream.write(
            "# command: PYTHONPATH=. JAX_PLATFORMS=cpu " + " ".join(command) + "\n"
        )
        stream.flush()
        probe = subprocess.run(
            [
                "taskset",
                "-c",
                cpus,
                PYTHON,
                "-c",
                "import nova.equilibrium.support_read as m; import os; "
                "print('MODULE='+m.__file__); print('CWD='+os.getcwd())",
            ],
            cwd=tree,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        assert probe.returncode == 0
        result = subprocess.run(
            ["taskset", "-c", cpus, *command],
            cwd=tree,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        stream.write(f"EXIT={result.returncode}\n")
    return result.returncode


def extract(root, revision, destination):
    paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", revision], text=True
    ).splitlines()
    paths = [
        path
        for path in paths
        if path.endswith((".py", ".toml", ".json", ".mplstyle"))
        and path.split("/")[0]
        in ("nova", "tests", "scripts", "benchmarks", "pyproject.toml")
    ]
    archive = subprocess.Popen(
        ["git", "-C", str(root), "archive", revision, *paths], stdout=subprocess.PIPE
    )
    subprocess.run(
        ["tar", "-x", "-C", str(destination)], stdin=archive.stdout, check=True
    )
    archive.stdout.close()
    assert archive.wait() == 0


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "mode", choices=("suite", "measure", "reproduce", "wall-controls")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    scratch = Path(tempfile.mkdtemp(prefix="wall-read-", dir=os.environ["TMPDIR"]))
    print("SCRATCH", scratch, flush=True)
    try:
        os.environ["TMPDIR"] = str(scratch)
        baseline = scratch / "base"
        baseline.mkdir()
        extract(root, BASE, baseline)
        cpus = sorted(os.sched_getaffinity(0))
        groups = [",".join(map(str, cpus[i::2])) for i in range(2)]
        tasks = []
        if args.mode == "wall-controls":
            command = [
                PYTHON,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                "-s",
                "tests/test_read_wall_skip.py",
                "-k",
                "wall_execution",
            ]
            tasks.append((root, command, run / "wall-controls.log", revision))
        elif args.mode == "reproduce":
            shutil.copyfile(
                root / "tests/test_read_wall_skip.py",
                baseline / "tests/test_read_wall_skip.py",
            )
            command = [
                PYTHON,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                "-s",
                "tests/test_read_wall_skip.py",
                "-k",
                "wall_execution and classes0 and False",
            ]
            tasks.append((baseline, command, run / "reproduce.log", BASE))
        elif args.mode == "suite":
            files = [
                "test_tracked_read.py",
                "test_support_read.py",
                "test_read_wall_skip.py",
            ]
            for arm, tree, sha in (("base", baseline, BASE), ("head", root, revision)):
                for filename in files:
                    if not (tree / "tests" / filename).exists():
                        continue
                    command = [
                        PYTHON,
                        "-m",
                        "pytest",
                        "-p",
                        "no:cacheprovider",
                        "-q",
                        "tests/" + filename,
                    ]
                    tasks.append((tree, command, run / f"{arm}-{filename}.log", sha))
            mutant = scratch / "mutant"
            mutant.mkdir()
            extract(root, revision, mutant)
            source = mutant / "nova/equilibrium/support_read.py"
            text = source.read_text()
            anchor = "all_diverted = jnp.all("
            assert text.count(anchor) == 1
            source.write_text(text.replace(anchor, "all_diverted = jnp.any("))
            command = [
                PYTHON,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                "tests/test_read_wall_skip.py",
                "-k",
                "batch_matches_lane_wall_decision",
            ]
            tasks.append((mutant, command, run / "negative-control.log", revision))
        else:
            for arm, tree, sha in (("base", baseline, BASE), ("head", root, revision)):
                for process in (1, 2):
                    command = [
                        PYTHON,
                        str(
                            root
                            / "scripts/prototypes/read_diverted_wall_skip/measure.py"
                        ),
                        "--helper",
                        str(root / "tests/test_read_wall_skip.py"),
                        "--output",
                        str(run / f"{arm}-{process}.json"),
                        "--revision",
                        sha,
                    ]
                    tasks.append(
                        (tree, command, run / f"measure-{arm}-{process}.log", sha)
                    )

        # Two processes own disjoint CPU sets; each arm uses fresh interpreters.
        def lane(index):
            return [execute(*task, groups[index]) for task in tasks[index::2]]

        if args.serial:
            all_cpus = ",".join(map(str, cpus))
            if args.mode == "measure":
                tasks = [tasks[index] for index in (0, 2, 1, 3)]
            outcomes = [execute(*task, all_cpus) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(lane, range(2)))
        print("OUTCOMES", outcomes, flush=True)
    finally:
        print("REMOVE_SCRATCH", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
