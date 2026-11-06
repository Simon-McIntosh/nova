"""Run the focused Biot test gate against the base and current revisions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

BASE_TESTS = (
    "tests/biot/test_edge_term_tangents.py",
    "tests/biot/test_vertex_term_tangents.py",
)


def _run_arm(root: Path, log_root: Path, revision: str, *, overlay: Path | None):
    label = "base" if overlay is not None else "head"
    log_path = log_root / f"{label}-suite.log"
    python = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(overlay), str(root)] if overlay is not None else [str(root)]
    )
    probe = (
        "import nova.biot.gradedresidual as module; "
        "from pathlib import Path; "
        "print('MEASUREMENT_MODULE=' + str(Path(module.__file__).resolve())); "
        "print('MEASUREMENT_CWD=' + str(Path.cwd().resolve()))"
    )
    command = [
        python,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        *(str(root / path) for path in BASE_TESTS),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"REVISION={revision}\nTREE={root}\nCOMMAND={' '.join(command)}\n")
        log.flush()
        probe_result = subprocess.run(
            [python, "-c", probe],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if probe_result.returncode:
            log.write(
                f"MODULE_PROBE_EXIT={probe_result.returncode}\nEXIT={probe_result.returncode}\n"
            )
            return probe_result.returncode, log_path
        result = subprocess.run(
            command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        log.write(f"EXIT={result.returncode}\n")
    return result.returncode, log_path


def _base_overlay(root: Path, base_revision: str, directory: Path) -> Path:
    overlay = directory / "overlay"
    package = overlay / "nova" / "biot"
    package.mkdir(parents=True)
    for name in ("gradedresidual.py", "polygonanalytic.py"):
        source = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{base_revision}:nova/biot/{name}"]
        )
        (package / name).write_bytes(source)
    # sitecustomize inserts the base source overlay before pytest imports nova.biot.
    sitecustomize = directory / "sitecustomize.py"
    sitecustomize.write_text(
        "import nova\n"
        f"nova.__path__.insert(0, {str(overlay / 'nova')!r})\n"
        "import nova.biot\n"
        f"nova.biot.__path__.insert(0, {str(package)!r})\n",
        encoding="utf-8",
    )
    return directory


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
    with tempfile.TemporaryDirectory(
        prefix="polygon-gate-", dir=os.environ["TMPDIR"]
    ) as directory:
        overlay = _base_overlay(root, args.base_revision, Path(directory))
        base_status, base_log = _run_arm(
            root, args.log_root, args.base_revision, overlay=overlay
        )
        head_status, head_log = _run_arm(
            root, args.log_root, head_revision, overlay=None
        )
    print(f"BASE_LOG={base_log} EXIT={base_status}", flush=True)
    print(f"HEAD_LOG={head_log} EXIT={head_status}", flush=True)
    return int(base_status != 0 or head_status != 0)


if __name__ == "__main__":
    raise SystemExit(main())
