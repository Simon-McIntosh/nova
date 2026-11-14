"""Run archived bank-input gates in fresh processes on a compute allocation."""

import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
FILES = (
    "test_support_read.py",
    "test_bank_drive_continuation.py",
    "test_bank_lane_drive.py",
)


def archive(source, revision, destination):
    payload = destination.parent / (destination.name + ".tar")
    with payload.open("wb") as stream:
        subprocess.run(
            ["git", "-C", str(source), "archive", revision], stdout=stream, check=True
        )
    destination.mkdir()
    with tarfile.open(payload) as stream:
        stream.extractall(destination, filter="data")
    payload.unlink()


def run_file(tree, revision, file, output, *, mutation=False):
    target = "tests/" + file
    command = (
        f"PYTHONPATH=. JAX_PLATFORMS=cpu {PYTHON} -m pytest "
        f"-p no:cacheprovider -q {target}"
    )
    temporary = tree.parent / "temporary"
    temporary.mkdir(exist_ok=True)
    environment = dict(
        os.environ,
        TMPDIR=str(temporary),
        PYTHONPATH=".",
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="7",
        PYTEST_ADDOPTS="-rfP",
    )
    if mutation:
        environment["NOVA_CAPTURE_SLICE_CURRENTS"] = "1"
        command = "NOVA_CAPTURE_SLICE_CURRENTS=1 " + command
    with output.open("w") as log:
        log.write(f"# revision: {revision} tree: {tree}\n# command: {command}\n")
        log.write("# output: PYTEST_ADDOPTS=-rfP\n")
        log.flush()
        subprocess.run(
            [
                PYTHON,
                "-c",
                "from pathlib import Path; "
                "import nova.equilibrium.support_read as m; "
                "print('MODULE=' + m.__file__); "
                "print('CWD=' + str(Path.cwd().resolve()))",
            ],
            cwd=tree,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        args = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", target]
        if mutation:
            args.extend(["-k", "slice_inputs_cause_zero_new_compilations", "-s"])
            command += " -k slice_inputs_cause_zero_new_compilations -s"
            log.write("# mutation command: " + command + "\n")
            log.flush()
        process = subprocess.run(
            args, cwd=tree, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        log.write(f"EXIT={process.returncode}\n")
    text = output.read_text()
    failures = [
        line.split(" - ", 1)[0][7:]
        for line in text.splitlines()
        if line.startswith("FAILED ")
    ]
    completed = any(
        " passed" in line or " failed" in line for line in text.splitlines()[-10:]
    )
    return dict(
        revision=revision,
        command=command,
        exit_status=process.returncode,
        log_path=str(output),
        completed=completed,
        failure_count=len(failures),
        failure_ids=failures,
        tree=str(tree),
    )


def reproduce():
    from nova.equilibrium import support_read

    try:
        inspect.signature(support_read.solve_batch).bind_partial(coil_currents=[0.0])
    except TypeError as error:
        print("REPRODUCTION", type(error).__name__, str(error), flush=True)
        return
    raise AssertionError("base already accepts traced coil inputs")


def run_graph(tree, revision, output):
    command = [
        PYTHON,
        str(Path(__file__).with_name("trace_graph.py")),
        "--revision",
        revision,
        "--output",
        str(output.with_suffix(".json")),
    ]
    with output.open("w") as log:
        log.write(
            f"# revision: {revision} tree: {tree}\n# command: {' '.join(command)}\n"
        )
        log.flush()
        result = subprocess.run(
            command,
            cwd=tree,
            env=dict(
                os.environ,
                TMPDIR=str(tree.parent),
                PYTHONPATH=str(tree),
                JAX_PLATFORMS="cpu",
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="7",
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(f"EXIT={result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"graph construction measurement failed: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--graph-base")
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()
    if args.reproduce:
        reproduce()
        return
    for name in ("base", "head"):
        setattr(
            args,
            name,
            subprocess.check_output(
                ["git", "-C", str(args.source), "rev-parse", getattr(args, name)],
                text=True,
            ).strip(),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = {}
    with tempfile.TemporaryDirectory(
        prefix="bank-inputs-", dir=os.environ["TMPDIR"]
    ) as scratch:
        root = Path(scratch)
        if not args.focused:
            base = root / "base"
            archive(args.source, args.base, base)
            with (args.output / "reproduction.log").open("w") as log:
                process = subprocess.run(
                    [PYTHON, str(Path(__file__).resolve()), "--reproduce"],
                    cwd=base,
                    env=dict(os.environ, PYTHONPATH=str(base), JAX_PLATFORMS="cpu"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                log.write(f"EXIT={process.returncode}\n")
            for file in FILES:
                name = "base-" + file.removesuffix(".py")
                receipts[name] = run_file(
                    base, args.base, file, args.output / (name + ".log")
                )
                (args.output / "receipts.json").write_text(
                    json.dumps(receipts, indent=2)
                )
        head = root / "head"
        archive(args.source, args.head, head)
        if args.graph_base:
            graph_base = root / "graph-base"
            archive(args.source, args.graph_base, graph_base)
            run_graph(graph_base, args.graph_base, args.output / "graph-before.log")
            run_graph(head, args.head, args.output / "graph-after.log")
        files = (
            ("test_bank_traced_inputs.py",)
            if args.focused
            else (*FILES, "test_bank_traced_inputs.py")
        )
        for file in files:
            name = "head-" + file.removesuffix(".py")
            receipts[name] = run_file(
                head, args.head, file, args.output / (name + ".log")
            )
            (args.output / "receipts.json").write_text(json.dumps(receipts, indent=2))
        if not args.skip_control:
            name = "closure-mutant"
            receipts[name] = run_file(
                head,
                args.head,
                "test_bank_traced_inputs.py",
                args.output / (name + ".log"),
                mutation=True,
            )
        (args.output / "receipts.json").write_text(json.dumps(receipts, indent=2))
    print(json.dumps(receipts, indent=2), flush=True)


if __name__ == "__main__":
    main()
