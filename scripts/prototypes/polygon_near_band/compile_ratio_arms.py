"""Measure one compile ratio repeatedly at the base and at the head revision.

The compile bound on the tangent rows asserts a median tangent-over-primal CPU
ratio, so deciding whether a head arm moved the bound needs the ratio's own
spread rather than one sample.  Each call is a fresh process that itself spawns
fresh compile probes, and the base arm resolves its source through the same
overlay the gate uses, so both arms differ only in the code under test.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

BASE_FILES = ("gradedresidual.py", "polygonanalytic.py")
PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"

_CALL = r"""
import sys
sys.path.insert(0, sys.argv[1])
from tangent_identity import compile_ratio
print("CALL_RATIO", compile_ratio(sys.argv[2], sys.argv[3], repeats=int(sys.argv[4])))
"""


def _base_overlay(root: Path, revision: str, directory: Path) -> Path:
    overlay = directory / "overlay"
    package = overlay / "nova" / "biot"
    package.mkdir(parents=True)
    for name in BASE_FILES:
        source = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{revision}:nova/biot/{name}"]
        )
        (package / name).write_bytes(source)
    (directory / "sitecustomize.py").write_text(
        "import nova\n"
        f"nova.__path__.insert(0, {str(overlay / 'nova')!r})\n"
        "import nova.biot\n"
        f"nova.biot.__path__.insert(0, {str(package)!r})\n",
        encoding="utf-8",
    )
    return directory


def _call(root: Path, env: dict[str, str], log, module: str, name: str, repeats: int):
    tests = root / "tests" / "biot"
    result = subprocess.run(
        [PYTHON, "-c", _CALL, str(tests), module, name, str(repeats)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log.write(result.stdout)
    log.write(result.stderr)
    log.flush()
    if result.returncode != 0:
        return {"ratio": None, "exit": result.returncode}
    ratios = [
        float(line.split()[-1])
        for line in result.stdout.splitlines()
        if line.startswith("CALL_RATIO ")
    ]
    return {"ratio": ratios[-1] if ratios else None, "exit": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--module", default="test_edge_term_tangents")
    parser.add_argument("--name", default="second_residual")
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--log-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.log_root.mkdir(parents=True, exist_ok=True)
    head_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(
        prefix="polygon-compile-", dir=os.environ["TMPDIR"]
    ) as directory:
        scratch = Path(directory)
        overlay = _base_overlay(root, args.base_revision, scratch)
        for arm, revision in (("base", args.base_revision), ("head", head_revision)):
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"
            env["PYTHONUNBUFFERED"] = "1"
            env.pop("NOVA_TANGENT_COMPILE_HEAVY", None)
            env.pop("NOVA_TANGENT_TRUNCATION", None)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(overlay), str(root)] if arm == "base" else [str(root)]
            )
            log_path = args.log_root / f"compile-ratio-{arm}.log"
            runs = []
            with log_path.open("w", encoding="utf-8") as log:
                for index in range(1, args.calls + 1):
                    run = _call(root, env, log, args.module, args.name, args.repeats)
                    runs.append(run)
                    log.write(
                        f"CALL {arm} {index} ratio={run['ratio']} exit={run['exit']}\n"
                    )
                    log.flush()
                log.write(f"EXIT={int(any(run['exit'] for run in runs))}\n")
            receipt[arm] = {
                "revision": revision,
                "module": args.module,
                "name": args.name,
                "repeats": args.repeats,
                "ratios": [run["ratio"] for run in runs],
                "log_path": str(log_path),
            }
            print(
                f"{arm.upper()} revision={revision} ratios={receipt[arm]['ratios']}",
                flush=True,
            )
    (args.log_root / "compile-ratio-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    missing = [value for arm in receipt.values() for value in arm["ratios"]]
    return int(any(value is None for value in missing))


if __name__ == "__main__":
    raise SystemExit(main())
