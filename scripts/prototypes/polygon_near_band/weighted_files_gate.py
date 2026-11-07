"""Run the graded-residual test files at both arms inside one allocation.

Every file of every arm runs as its own process with its own basetemp, and the
truncation mutation of the vertex primal-JVP row runs beside them, so the whole
comparison is one scheduler job of six processes rather than one job per row.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import identity_gate

FILES = (
    "tests/biot/test_vertex_edge_tangent_recurrences.py",
    "tests/biot/test_graded_layer_near_corner.py",
)
MUTATION_ROW = (
    "tests/biot/test_vertex_term_tangents.py::test_tangent_matches_primal_jvp"
)
PYTHON = identity_gate.PYTHON


def _run(name, command, env, root, log_path, scratch):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name} command: {shlex.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"EXIT={result.returncode}\n")
    rows = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith(("PASSED ", "FAILED ", "ERROR ")):
            status, rest = stripped.split(" ", 1)
            rows[rest.split(" ")[0]] = status.lower()
    return name, {
        "command": shlex.join(command),
        "exit_status": result.returncode,
        "log_path": str(log_path),
        "basetemp": str(scratch / f"pytest-{name}"),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.log_root.mkdir(parents=True, exist_ok=True)
    head_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt = {"base_revision": args.base_revision, "head_revision": head_revision}
    jobs = []
    with tempfile.TemporaryDirectory(
        prefix="polygon-weighted-", dir=os.environ.get("TMPDIR", "/tmp")
    ) as directory:
        scratch = Path(directory)
        overlay = identity_gate._base_overlay(root, args.base_revision, scratch)
        for arm, revision in (
            ("base", args.base_revision),
            ("head", head_revision),
        ):
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [str(overlay), str(root)] if arm == "base" else [str(root)]
            )
            for target in FILES:
                stem = Path(target).stem
                name = f"{arm}-{stem}"
                command = [
                    PYTHON,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "-v",
                    f"--timeout={args.timeout}",
                    "--basetemp",
                    str(scratch / f"pytest-{name}"),
                    target,
                ]
                jobs.append(
                    (name, command, env, args.log_root / f"{name}.log", scratch)
                )
        mutation_env = os.environ.copy()
        mutation_env["JAX_PLATFORMS"] = "cpu"
        mutation_env["PYTHONUNBUFFERED"] = "1"
        mutation_env["PYTHONPATH"] = str(root)
        mutation_env["NOVA_TANGENT_TRUNCATION"] = "1"
        jobs.append(
            (
                "head-mutation-vertex-primal-jvp",
                [
                    PYTHON,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "-v",
                    f"--timeout={args.timeout}",
                    "--basetemp",
                    str(scratch / "pytest-mutation"),
                    MUTATION_ROW,
                ],
                mutation_env,
                args.log_root / "head-mutation-vertex-primal-jvp.log",
                scratch,
            )
        )
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = [
                pool.submit(_run, name, command, env, root, log_path, scratch)
                for name, command, env, log_path, scratch in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                name, record = future.result()
                receipt[name] = record
                print(
                    f"{name} EXIT={record['exit_status']} "
                    f"ROWS={json.dumps(record['rows'])}",
                    flush=True,
                )
    (args.log_root / "weighted-files-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
