"""Run each JVP identity row as its own process inside one allocation.

The rows exceed the suite's per-test bound on a compile-bound host, so the bound
is raised on this command line only.  One process per row per arm, each with its
own log and basetemp, all inside the allocation the caller holds: the row is the
unit of isolation, the job is not.
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

PYTHON = identity_gate.PYTHON


def _run(name, command, env, root, log_path):
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
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.log_root.mkdir(parents=True, exist_ok=True)
    head_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt = {"base_revision": args.base_revision, "head_revision": head_revision}
    jobs = []
    with tempfile.TemporaryDirectory(
        prefix="polygon-identity-", dir=os.environ.get("TMPDIR", "/tmp")
    ) as directory:
        scratch = Path(directory)
        overlay = identity_gate._base_overlay(root, args.base_revision, scratch)
        for arm, revision in (("base", args.base_revision), ("head", head_revision)):
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [str(overlay), str(root)] if arm == "base" else [str(root)]
            )
            for row in identity_gate.IDS:
                short = row.replace("tests/biot/test_edge_term_tangents.py::", "")
                name = f"{arm}-{short}"
                tag = "".join(c if c.isalnum() else "-" for c in short)
                command = [
                    PYTHON,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "-v",
                    "--durations=0",
                    f"--timeout={args.timeout}",
                    "--basetemp",
                    str(scratch / f"pytest-{arm}-{tag}"),
                    row,
                ]
                jobs.append((name, command, env, args.log_root / f"{arm}-{tag}.log"))
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = [
                pool.submit(_run, name, command, env, root, log_path)
                for name, command, env, log_path in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                name, record = future.result()
                receipt[name] = record
                print(
                    f"{name} EXIT={record['exit_status']} {json.dumps(record['rows'])}",
                    flush=True,
                )
        (args.log_root / "identity-rows-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
