"""Run each tangent test file against the base and head in a fresh process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

TEST_FILES = (
    "tests/biot/test_edge_term_tangents.py",
    "tests/biot/test_vertex_term_tangents.py",
)


def _summary_reached(log_path: Path) -> bool:
    contents = log_path.read_text(encoding="utf-8", errors="replace")
    return bool(
        re.search(
            r"(?:\d+ (?:passed|failed|error|errors|skipped|xfailed|xpassed|deselected)|no tests ran)",
            contents,
        )
    )


def _failure_ids(log_path: Path) -> list[str]:
    contents = log_path.read_text(encoding="utf-8", errors="replace")
    return sorted(
        set(re.findall(r"^(?:FAILED|ERROR)\s+([^\s]+)", contents, re.MULTILINE))
    )


def _run_file(
    root: Path,
    log_root: Path,
    revision: str,
    arm: str,
    test_file: str,
    env: dict[str, str],
    scratch: Path,
) -> dict:
    python = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
    name = Path(test_file).stem
    log_path = log_root / f"{arm}-{name}.log"
    basetemp = scratch / f"pytest-{arm}-{name}"
    step_name = f"{arm}-{name}"
    command = [
        "srun",
        "--overlap",
        "--ntasks=1",
        "--cpus-per-task=8",
        f"--job-name={step_name}",
        python,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(basetemp),
        str(root / test_file),
    ]
    probe = (
        "import os; from pathlib import Path; "
        "import nova.biot.polygonanalytic as polygon; "
        "print('# module_under_test:', Path(polygon.__file__).resolve()); "
        "print('# resolved_cwd:', Path.cwd().resolve())"
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            f"# revision: {revision} tree: {root} command: {shlex.join(command)}\n"
        )
        log.flush()
        probe_result = subprocess.run(
            [python, "-c", probe],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if probe_result.returncode:
            log.write(f"MODULE_PROBE_EXIT={probe_result.returncode}\n")
            log.write(f"EXIT={probe_result.returncode}\n")
            return {
                "test_file": test_file,
                "command": shlex.join(command),
                "exit_status": probe_result.returncode,
                "log_path": str(log_path),
                "completed": False,
                "failure_count": None,
                "failure_ids": [],
            }
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"EXIT={result.returncode}\n")
    failure_ids = _failure_ids(log_path)
    return {
        "test_file": test_file,
        "command": shlex.join(command),
        "exit_status": result.returncode,
        "log_path": str(log_path),
        "completed": _summary_reached(log_path),
        "failure_count": len(failure_ids),
        "failure_ids": failure_ids,
    }


def _base_overlay(root: Path, base_revision: str, directory: Path) -> Path:
    overlay = directory / "overlay"
    package = overlay / "nova" / "biot"
    package.mkdir(parents=True)
    for name in ("gradedresidual.py", "polygonanalytic.py"):
        source = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{base_revision}:nova/biot/{name}"]
        )
        (package / name).write_bytes(source)
    sitecustomize = directory / "sitecustomize.py"
    sitecustomize.write_text(
        "import nova\n"
        f"nova.__path__.insert(0, {str(overlay / 'nova')!r})\n"
        "import nova.biot\n"
        f"nova.biot.__path__.insert(0, {str(package)!r})\n",
        encoding="utf-8",
    )
    return directory


def _arm_receipt(revision: str, runs: list[dict]) -> dict:
    failure_ids = sorted({item for run in runs for item in run["failure_ids"]})
    status_codes = [run["exit_status"] for run in runs]
    return {
        "revision": revision,
        "command": [run["command"] for run in runs],
        "exit_status": next((code for code in status_codes if code != 0), 0),
        "log_path": [run["log_path"] for run in runs],
        "completed": len(runs) == len(TEST_FILES)
        and all(run["completed"] for run in runs),
        "failure_count": len(failure_ids),
        "failure_ids": failure_ids,
        "files": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.log_root.mkdir(parents=True, exist_ok=True)
    head_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    python = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
    run_results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(
        prefix="polygon-gate-", dir=os.environ["TMPDIR"]
    ) as directory:
        scratch = Path(directory)
        base_overlay = _base_overlay(root, args.base_revision, scratch)
        for arm, revision in (("base", args.base_revision), ("head", head_revision)):
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [str(base_overlay), str(root)]
                if arm == "base"
                else [str(root)]
            )
            runs = []
            for test_file in TEST_FILES:
                result = _run_file(
                    root,
                    args.log_root,
                    revision,
                    arm,
                    test_file,
                    env,
                    scratch,
                )
                runs.append(result)
                receipt = _arm_receipt(revision, runs)
                (args.log_root / f"{arm}-receipts.json").write_text(
                    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"{arm.upper()}_FILE={test_file} EXIT={result['exit_status']} "
                    f"COMPLETED={result['completed']} LOG={result['log_path']}",
                    flush=True,
                )
            run_results[arm] = _arm_receipt(revision, runs)
    print(json.dumps(run_results, indent=2), flush=True)
    return int(any(item["exit_status"] != 0 for item in run_results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
