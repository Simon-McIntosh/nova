"""Run revision-pinned correctness arms and the stale-reuse mutation."""

import argparse
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
TEST = "tests/test_carried_read_refresh.py"
EXISTING = [
    "tests/test_tracked_read.py",
    "tests/test_read_tie_break.py",
    "tests/test_read_wall_skip.py",
    "tests/test_read_live_candidates.py",
    "tests/test_tracked_read_steady_batch.py",
    "tests/test_tracked_read_batch.py",
]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("base")
    parser.add_argument("head")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    receipts = {}
    with tempfile.TemporaryDirectory(
        prefix="refresh-gate-", dir=os.environ["TMPDIR"]
    ) as temporary:
        for arm, revision in (
            ("baseline", args.base),
            ("after", args.head),
            ("negative", args.head),
        ):
            tree = Path(temporary) / arm
            tree.mkdir()
            data = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(args.root),
                    "archive",
                    revision,
                    "nova",
                    "scripts",
                    "tests",
                    "benchmarks",
                    "conftest.py",
                    "pyproject.toml",
                ]
            )
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                archive.extractall(tree, filter="data")
            if arm == "negative":
                path = tree / "nova/equilibrium/topology.py"
                source = path.read_text()
                anchor = (
                    "reuse = _fragment_reuse_mask(geometry, signed, "
                    "form, decision, sigma)"
                )
                assert source.count(anchor) == 1
                path.write_text(
                    source.replace(
                        anchor, "reuse = jnp.ones(len(coefficient), dtype=bool)"
                    )
                )
            env = dict(
                os.environ,
                TMPDIR=temporary,
                PYTHONPATH=".",
                JAX_PLATFORMS="cpu",
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                PYTHONDONTWRITEBYTECODE="1",
            )
            env.pop("PYTEST_ADDOPTS", None)
            files = [TEST] if (tree / TEST).exists() else []
            groups = {arm: files}
            if arm != "negative":
                groups[arm + "-existing"] = EXISTING
            for label, files in groups.items():
                log = args.output / (label + ".log")
                if not files:
                    command = "git ls-tree --name-only " + revision + " -- " + TEST
                    with log.open("w") as stream:
                        stream.write(
                            f"# revision: {revision} tree: {tree} command: {command}\n"
                        )
                        stream.flush()
                        result = subprocess.run(
                            shlex.split(command),
                            cwd=args.root,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                        )
                        stream.write(
                            "BASELINE_UNARMED scoped test absent at this revision\n"
                        )
                        stream.write(f"EXIT={result.returncode}\n")
                    receipts[label] = dict(
                        revision=revision,
                        command=command,
                        completed=True,
                        exit_status=result.returncode,
                        failure_count=0,
                        failure_ids=[],
                        log_path=str(log),
                        unarmed=True,
                    )
                    continue
                suffix = " -rfP" if label.endswith("existing") else ""
                command = (
                    f"PYTHONPATH=. JAX_PLATFORMS=cpu {PYTHON} "
                    f"-m pytest -p no:cacheprovider -q{suffix} "
                    + " ".join(files)
                )
                with log.open("w") as stream:
                    stream.write(
                        f"# revision: {revision} tree: {tree} command: {command}\n"
                    )
                    stream.flush()
                    subprocess.run(
                        [
                            PYTHON,
                            "-c",
                            "from nova.equilibrium import topology; "
                            "from pathlib import Path; "
                            "print('MODULE='+topology.__file__); "
                            "print('CWD='+str(Path.cwd().resolve()))",
                        ],
                        cwd=tree,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    result = subprocess.run(
                        command,
                        shell=True,
                        executable="/bin/bash",
                        cwd=tree,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                    stream.write(f"EXIT={result.returncode}\n")
                text = log.read_text()
                failures = re.findall(r"^FAILED (\S+)", text, re.M)
                receipts[label] = dict(
                    revision=revision,
                    command=command,
                    completed=bool(
                        re.search(r"\d+ (?:passed|failed).* in [\d.]+s", text)
                    ),
                    exit_status=result.returncode,
                    failure_count=len(failures),
                    failure_ids=failures,
                    log_path=str(log),
                    tree=str(tree),
                )
                (args.output / "receipts.json").write_text(
                    json.dumps(receipts, indent=2)
                )
                print("ARM", label, json.dumps(receipts[label]), flush=True)
                if arm == "negative":
                    assert result.returncode and failures, "stale reuse was not refused"
                elif result.returncode:
                    raise RuntimeError(f"correctness arm failed: {log}")
    (args.output / "receipts.json").write_text(json.dumps(receipts, indent=2))
    print("SUMMARY correctness arms and stale-reuse refusal complete", flush=True)


if __name__ == "__main__":
    main()
