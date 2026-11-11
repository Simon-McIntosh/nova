"""Run isolated candidate checks and the committed attribution instrument."""

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

BASE = "597c44f4e9a568c007b1f01e6a8c79545c333925"
INSTRUMENT = "d3d17c064bd10c209aa4adec67c0ce9e5d1a3785"
PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def snapshot(root, revision, target):
    target.mkdir()
    content = git(
        root, "archive", revision, "nova", "tests", "pyproject.toml", "conftest.py"
    )
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        archive.extractall(target, filter="data")
    (target / "scripts").symlink_to(root / "scripts", target_is_directory=True)
    metadata = target / ".git"
    metadata.mkdir()
    (metadata / "HEAD").write_text(revision + "\n")
    (metadata / "refs").mkdir()
    common = Path(git(root, "rev-parse", "--git-common-dir").decode().strip())
    if not common.is_absolute():
        common = root / common
    (metadata / "objects").symlink_to(
        common.resolve() / "objects", target_is_directory=True
    )


def run(tree, revision, command, log, env):
    display = "PYTHONPATH=. JAX_PLATFORMS=cpu " + " ".join(command)
    with log.open("w") as stream:
        stream.write(f"# revision: {revision} tree: {tree} command: {display}\n")
        stream.write(f"# measurement_cwd: {tree.resolve()}\n")
        stream.write(f"# measurement_module: {tree / 'nova/equilibrium/topology.py'}\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=tree,
            env=dict(env, PYTHONPATH="."),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        stream.write(f"EXIT={result.returncode}\n")
    print(log.name, "EXIT", result.returncode, flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("mode", choices=["reproduce", "suites", "measure", "armed"])
    args = parser.parse_args()
    root, output = args.root.resolve(), args.run.resolve()
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    env = dict(os.environ, TMPDIR="/tmp", JAX_PLATFORMS="cpu")
    with tempfile.TemporaryDirectory(
        prefix="live-candidates-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        env["TMPDIR"] = str(scratch)
        if args.mode == "armed":
            command = [
                PYTHON,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                "tests/test_read_live_candidates.py",
            ]
            code = run(root, revision, command, output / "final-head.log", env)
            assert code == 0, "final committed head must pass its armed file"
            return
        if args.mode == "reproduce":
            test = (
                "tests/test_read_live_candidates.py::"
                "test_support_inspects_only_live_saddles[1]"
            )
            code = run(
                root,
                revision,
                [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-s", test],
                output / "reproduction.log",
                env,
            )
            assert code == 1, (
                "the unoptimized support inspection must fail its count guard"
            )
            return
        trees = {}
        for arm, sha in (("base", BASE), ("head", revision)):
            tree = scratch / arm
            snapshot(root, sha, tree)
            trees[arm] = tree
        if args.mode == "suites":
            receipts = {}
            for arm, tree in trees.items():
                files = ["tests/test_tracked_read.py", "tests/test_support_read.py"]
                if arm == "head":
                    files.append("tests/test_read_live_candidates.py")
                for file in files:
                    log = output / f"{arm}-{Path(file).stem}.log"
                    command = [
                        PYTHON,
                        "-m",
                        "pytest",
                        "-p",
                        "no:cacheprovider",
                        "-q",
                        file,
                    ]
                    code = run(
                        tree, BASE if arm == "base" else revision, command, log, env
                    )
                    receipts[f"{arm}:{file}"] = code
            mutant = trees["head"] / "nova/equilibrium/topology.py"
            source = mutant.read_text()
            anchor = "saddles, size=compact_capacity"
            assert source.count(anchor) == 1
            mutant.write_text(
                source.replace(anchor, "jnp.zeros_like(saddles), size=compact_capacity")
            )
            test = (
                "tests/test_read_live_candidates.py::"
                "test_read_and_book_matches_full_candidates[diverted-1]"
            )
            code = run(
                trees["head"],
                revision,
                [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", test],
                output / "negative-control.log",
                env,
            )
            assert code == 1, "dropping the real saddle must fail parity"
            (output / "suite-exits.json").write_text(json.dumps(receipts, indent=2))
            assert not any(receipts.values()), receipts
        else:
            instrument = scratch / "measure.py"
            instrument.write_bytes(
                git(
                    root,
                    "show",
                    INSTRUMENT + ":scripts/prototypes/bank_read_attribution/measure.py",
                )
            )
            for arm, tree in trees.items():
                for kind in ("full", "carried"):
                    for process in (1, 2):
                        name = f"{arm}-{kind}-{process}"
                        command = [
                            PYTHON,
                            str(instrument),
                            str(tree),
                            "--kind",
                            kind,
                            "--output",
                            str(output / f"{name}.json"),
                        ]
                        code = run(
                            tree,
                            BASE if arm == "base" else revision,
                            command,
                            output / f"{name}.log",
                            env,
                        )
                        assert code == 0, name


if __name__ == "__main__":
    main()
