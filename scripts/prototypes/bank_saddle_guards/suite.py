"""Run isolated file suites and the production-wiring mutation with receipts."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
FILES = (
    "test_saddle_derivatives",
    "test_support_read",
    "test_bank_newton_step",
    "test_bank_step_booking_batch",
)


def archive(repo, revision, tree):
    tree.mkdir()
    with tempfile.TemporaryFile() as stream:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                revision,
                "nova",
                "scripts",
                "tests",
                "benchmarks",
                "pyproject.toml",
                "conftest.py",
            ],
            stdout=stream,
            check=True,
        )
        stream.seek(0)
        subprocess.run(["tar", "-x", "-C", str(tree)], stdin=stream, check=True)


def mutate(tree):
    path = tree / "nova/equilibrium/support_read.py"
    source = path.read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "read"
    )
    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "field_from_jets"
    )
    names = {"bank_rows", "bank_moments", "bank_exterior"}
    keywords = [node for node in call.keywords if node.arg in names]
    assert {node.arg for node in keywords} == names
    removed = {
        line for node in keywords for line in range(node.lineno, node.end_lineno + 1)
    }
    path.write_text(
        "".join(
            line
            for number, line in enumerate(source.splitlines(True), 1)
            if number not in removed
        )
    )


def run_file(run, scratch, label, tree, revision, name, extra=(), overrides=None):
    env = dict(
        os.environ,
        PYTHONPATH=str(tree),
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(scratch),
    )
    env.pop("XLA_FLAGS", None)
    env.update(overrides or {})
    command = [
        PYTHON,
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
        *extra,
    ]
    log = run / f"{label}-{name}.log"
    settings = dict(PYTHONPATH=str(tree), JAX_PLATFORMS="cpu", **(overrides or {}))
    literal = " ".join(f"{key}={shlex.quote(value)}" for key, value in settings.items())
    literal += " " + shlex.join(command)
    with log.open("w") as stream:
        stream.write(f"# revision: {revision} tree: {tree} command: {literal}\n")
        stream.flush()
        subprocess.run(
            [
                PYTHON,
                "-P",
                "-c",
                "import nova, nova.equilibrium.fixed_points as m; "
                "from pathlib import Path; print('nova', nova.__file__, 'module', "
                "m.__file__, 'cwd', Path.cwd().resolve(), flush=True)",
            ],
            cwd=tree,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        result = subprocess.run(
            command, cwd=tree, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        stream.write(f"EXIT={result.returncode}\n")
    content = log.read_text()
    failures = [
        line.split(" - ")[0][7:]
        for line in content.splitlines()
        if line.startswith("FAILED ")
    ]
    return dict(
        revision=revision,
        command=literal,
        exit_status=result.returncode,
        log_path=str(log),
        completed="short test summary info" in content or " passed" in content,
        failure_count=len(failures),
        failure_ids=failures,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--states", type=Path)
    args = parser.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="saddle-guards-", dir=os.environ["TMPDIR"]))
    print(f"scratch={scratch}", flush=True)
    try:
        if args.focused:
            receipt = run_file(
                args.run,
                scratch,
                "focused",
                args.repo,
                args.base + "+working",
                "test_saddle_derivatives",
            )
            print(json.dumps(receipt), flush=True)
            return receipt["exit_status"]
        if args.reproduce:
            tree = scratch / "mutation"
            archive(args.repo, args.base, tree)
            mutate(tree)
            receipt = run_file(
                args.run,
                scratch,
                "reproduction",
                tree,
                args.base,
                "test_saddle_derivatives",
            )
            (args.run / "reproduction.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt), flush=True)
            return receipt["exit_status"]

        def arm(label, revision):
            tree = scratch / label
            archive(args.repo, revision, tree)
            rows = [
                run_file(args.run, scratch, label, tree, revision, name)
                for name in FILES
            ]
            (args.run / f"suite-{label}.json").write_text(json.dumps(rows, indent=2))
            return rows

        with ThreadPoolExecutor(max_workers=2) as executor:
            base = executor.submit(arm, "base", args.base)
            head = executor.submit(arm, "head", args.head)
            rows = base.result() + head.result()
        for label, revision in (("base", args.base), ("head", args.head)):
            tree = scratch / label
            command = [
                PYTHON,
                "-P",
                str(args.repo / "scripts/prototypes/bank_saddle_guards/derivatives.py"),
                "--run",
                str(args.run),
                "--states",
                str(args.states),
                "--revision",
                revision,
                "--arm",
                label,
            ]
            log = args.run / f"derivatives-{label}.log"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {revision} tree: {tree} command: "
                    f"PYTHONPATH={tree} JAX_PLATFORMS=cpu {shlex.join(command)}\n"
                )
                stream.flush()
                env = dict(
                    os.environ,
                    PYTHONPATH=str(tree),
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="2",
                    PYTHONDONTWRITEBYTECODE="1",
                    TMPDIR=str(scratch),
                )
                env.pop("XLA_FLAGS", None)
                result = subprocess.run(
                    command, cwd=tree, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                stream.write(f"EXIT={result.returncode}\n")
            if result.returncode:
                return result.returncode
        certificate = run_file(
            args.run,
            scratch,
            "certificate",
            scratch / "head",
            args.head,
            "test_saddle_derivatives",
            overrides={
                "NOVA_SADDLE_CERTIFICATE": "1",
                "NOVA_SADDLE_STATES": str(args.states),
            },
        )
        (args.run / "certificate.json").write_text(json.dumps(certificate, indent=2))
        negative = scratch / "negative"
        archive(args.repo, args.head, negative)
        mutate(negative)
        control = run_file(
            args.run,
            scratch,
            "negative",
            negative,
            args.head,
            "test_saddle_derivatives",
            ("-k", "production"),
        )
        (args.run / "negative.json").write_text(json.dumps(control, indent=2))
        assert control["exit_status"] != 0 and control["failure_ids"], control
        return int(any(row["exit_status"] for row in [*rows, certificate]))
    finally:
        print(f"Removing job scratch {scratch}", flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    raise SystemExit(main())
