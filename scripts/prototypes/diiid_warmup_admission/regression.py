"""Run isolated regression processes against an archived solver revision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from measure import PYTHON, extract

FILES = (
    "tests/test_active_set_telemetry.py",
    "tests/test_active_set_stagnation.py",
    "tests/test_amplification_observation.py",
    "tests/test_equilibrium_solve.py",
    "tests/test_diverted_300_convergence.py",
)
NODE_TEST = "tests/test_warmup_finite_admission.py"


def gate(
    tree, revision, path, output, quiet=False, platforms="cpu", extra_pythonpath=""
):
    command = [
        PYTHON,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q" if quiet else "-vv",
        path,
    ]
    pythonpath = "." + (":" + extra_pythonpath if extra_pythonpath else "")
    literal = f"PYTHONPATH={pythonpath} JAX_PLATFORMS={platforms} " + shlex.join(
        command
    )
    env = dict(
        os.environ,
        PYTHONPATH=pythonpath,
        JAX_PLATFORMS=platforms,
        TMPDIR=os.environ["TMPDIR"],
    )
    module = tree / "nova/equilibrium/fixed_point.py"
    with output.open("w") as stream:
        stream.write(f"# revision: {revision}; tree: {tree}; command: {literal}\n")
        stream.write(f"# module: {module}; cwd: {tree.resolve()}\n")
        stream.flush()
        check = subprocess.run(
            [
                PYTHON,
                "-c",
                "from nova.equilibrium import fixed_point; import os; "
                "print('IMPORTED_MODULE=' + fixed_point.__file__); "
                "print('RESOLVED_CWD=' + os.getcwd())",
            ],
            cwd=tree,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert check.returncode == 0
        result = subprocess.run(
            command,
            cwd=tree,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(f"EXIT={result.returncode}\n")
    text = output.read_text()
    failures = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.MULTILINE)
    completed = bool(
        re.search(r"\d+ (?:passed|failed|errors?|skipped).* in [\d.]+s", text)
    )
    if result.returncode and not failures:
        executed = re.findall(r"^(tests/\S+::\S+)", text, re.MULTILINE)
        failures = [executed[-1] if executed else path + "::unattributed_process_abort"]
    record = dict(
        revision=revision,
        command=literal,
        exit_status=result.returncode,
        log_path=str(output),
        completed=completed,
        failure_count=len(failures),
        failure_ids=failures,
    )
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope-only", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="warmup-regression-", dir=os.environ["TMPDIR"]
    ) as temporary:
        tree = Path(temporary) / "tree"
        extract(tree, args.revision, repository)
        records = {}
        if (tree / NODE_TEST).exists():
            records["suite"] = gate(
                tree, args.revision, NODE_TEST, args.output / "suite.log", quiet=True
            )
        else:
            assert (tree / "nova/equilibrium/fixed_point.py").is_file()
            log = args.output / "suite.log"
            code = (
                "from pathlib import Path; "
                "from nova.equilibrium import fixed_point; "
                "import os; "
                "print('IMPORTED_MODULE=' + fixed_point.__file__); "
                "print('RESOLVED_CWD=' + os.getcwd()); "
                "assert Path('nova/equilibrium/fixed_point.py').is_file(); "
                f"assert not Path({NODE_TEST!r}).exists(); "
                f"print('BASE_SCOPE_EMPTY={NODE_TEST}'); "
                "print('KNOWN_PRESENT=nova/equilibrium/fixed_point.py')"
            )
            command = [PYTHON, "-c", code]
            literal = "PYTHONPATH=. JAX_PLATFORMS=cpu " + shlex.join(command)
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {args.revision}; tree: {tree}; command: {literal}\n"
                )
                stream.flush()
                result = subprocess.run(
                    command,
                    cwd=tree,
                    env=dict(os.environ, PYTHONPATH=".", JAX_PLATFORMS="cpu"),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                stream.write(f"EXIT={result.returncode}\n")
            assert result.returncode == 0
            records["suite"] = dict(
                revision=args.revision,
                command=literal,
                exit_status=0,
                log_path=str(log),
                completed=True,
                failure_count=0,
                failure_ids=[],
                note="Empty scoped suite: the node creates its only test file.",
            )
        (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        if args.scope_only:
            return
        for path in FILES:
            records[path] = gate(
                tree, args.revision, path, args.output / (Path(path).stem + ".log")
            )
            (args.output / "results.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
