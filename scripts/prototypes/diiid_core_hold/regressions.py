"""Run immutable CPU regression files in fresh, bounded pytest processes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_warmup_admission"))
from measure import extract  # noqa: E402


def gate(tree, revision, paths, log, scratch):
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *paths]
    literal = "PYTHONPATH=. JAX_PLATFORMS=cpu " + shlex.join(command)
    env = dict(
        os.environ,
        PYTHONPATH=".",
        JAX_PLATFORMS="cpu",
        PYTEST_TIMEOUT="3600",
        TMPDIR=str(scratch),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
    )
    with log.open("w") as stream:
        stream.write(f"# revision: {revision}; tree: {tree}; command: {literal}\n")
        stream.flush()
        probe = [
            sys.executable,
            "-c",
            "from nova.equilibrium import solve_request; import os; "
            "print('IMPORTED_MODULE=' + solve_request.__file__); "
            "print('RESOLVED_CWD=' + os.getcwd())",
        ]
        check = subprocess.run(
            probe,
            cwd=tree,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if check.returncode:
            result = check
        else:
            result = subprocess.run(
                command,
                cwd=tree,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        stream.write(f"EXIT={result.returncode}\n")
    text = log.read_text()
    failures = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)
    complete = bool(
        re.search(r"\d+ (?:passed|failed|skipped|errors?).* in [\d.]+s", text)
    )
    if result.returncode and not failures:
        failures = [paths[-1] + "::unattributed_process_abort"]
    return dict(
        revision=revision,
        command=literal,
        exit_status=result.returncode,
        log_path=str(log),
        completed=complete,
        failure_count=len(failures),
        failure_ids=failures,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--repaired-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(
        prefix="nova-core-regression-", dir=os.environ["TMPDIR"]
    ) as scratch:
        trees = {}
        for arm, revision in (
            ("base", args.base),
            ("head", args.head),
            ("repaired", args.repaired_head),
        ):
            tree = Path(scratch) / arm
            extract(tree, revision, ROOT)
            trees[arm] = (tree, revision)
        scoped = (
            "tests/test_sampled_flux_batching.py",
            "tests/test_sampled_profile_core_hold.py",
        )
        for arm, (tree, revision) in trees.items():
            paths = [path for path in scoped if (tree / path).is_file()]
            record = gate(
                tree, revision, paths, args.output / (arm + "-suite.log"), Path(scratch)
            )
            results[arm + ":suite"] = record
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
        tree, revision = trees["repaired"]
        source = tree / "nova/equilibrium/solve_request.py"
        text = source.read_text()
        marker = "    return jnp.where(below, samples[0], result)\n"
        assert text.count(marker) == 1
        source.write_text(text.replace(marker, "    return result\n"))
        results["negative"] = gate(
            tree,
            revision,
            list(scoped),
            args.output / "negative-control.log",
            Path(scratch),
        )
        assert results["negative"]["exit_status"] != 0
        source.write_text(text)
        files = json.loads(args.files.read_text())
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            for arm in ("base", "repaired"):
                tree, revision = trees[arm]
                for path in files:
                    if path in scoped or not (tree / path).is_file():
                        continue
                    name = arm + ":" + path
                    folder = Path(scratch) / name.replace("/", "-").replace(":", "-")
                    folder.mkdir()
                    log = args.output / (
                        name.replace("/", "-").replace(":", "-") + ".log"
                    )
                    future = pool.submit(gate, tree, revision, [path], log, folder)
                    futures[future] = name
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
                (args.output / "results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                print("REGRESSION", name, json.dumps(results[name]), flush=True)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
