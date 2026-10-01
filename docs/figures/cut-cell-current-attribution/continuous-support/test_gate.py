"""Run each affected test module in a fresh interpreter with a pinned source."""

import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import runpy

ROOT = Path(__file__).resolve().parent
MUTATION = (
    "Replace moving confined-support integration "
    "with fixed atomic-cell pointwise selection."
)
MODULES = (
    "tests/test_xpoint_cell_wedge_clip.py",
    "tests/test_clipped_support_quadrature.py",
    "tests/test_solovev_certificate_builder.py",
    "tests/test_exact_clip_moments.py",
    "tests/test_moment_path_separatrix_test.py",
    "tests/test_exact_clip_memory.py",
)


class BaselineLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = ROOT / "baseline-quadrature.txt"
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)


class BaselineFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "nova.equilibrium.clip_quadrature":
            return importlib.util.spec_from_loader(fullname, BaselineLoader())
        return None


if len(sys.argv) > 2 and sys.argv[1] == "--baseline-child":
    sys.meta_path.insert(0, BaselineFinder())
    sys.argv = sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")
    raise SystemExit(0)


if len(sys.argv) == 3:
    variant, target = sys.argv[1:]
    if variant in ("baseline", "negative"):
        sys.meta_path.insert(0, BaselineFinder())
        original_run = subprocess.run

        def pinned_run(command, *args, **kwargs):
            if (
                isinstance(command, (list, tuple))
                and len(command) > 1
                and command[0] == sys.executable
                and str(command[1]).endswith("terminal_driver.py")
            ):
                command = [
                    command[0],
                    str(Path(__file__).resolve()),
                    "--baseline-child",
                    *command[1:],
                ]
            return original_run(command, *args, **kwargs)

        subprocess.run = pinned_run
    if variant == "negative":
        print(MUTATION, flush=True)
    import pytest

    raise SystemExit(pytest.main(["-p", "no:cacheprovider", target]))

variant = sys.argv[1]
if variant == "paired":
    rows = []
    for target in MODULES:
        for lane in ("baseline", "after"):
            log = ROOT / ("paired-" + lane + "-" + Path(target).stem + ".log")
            command = [sys.executable, str(Path(__file__).resolve()), lane, target]
            with log.open("w") as stream:
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
            rows.append(
                {
                    "variant": lane,
                    "target": target,
                    "command": command,
                    "exit_status": result.returncode,
                    "log_path": str(log.resolve()),
                    "completed": True,
                }
            )
            (ROOT / "paired-suite.json").write_text(json.dumps(rows, indent=2))
            print(lane, target, result.returncode, flush=True)
    raise SystemExit(any(row["exit_status"] for row in rows))

modules = (
    ("tests/test_continuous_confined_moments.py",) if variant == "negative" else MODULES
)
rows = []
for target in modules:
    log = ROOT / (variant + "-" + Path(target).stem + ".log")
    command = [sys.executable, str(Path(__file__).resolve()), variant, target]
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    rows.append(
        {
            "command": command,
            "exit_status": result.returncode,
            "log_path": str(log.resolve()),
            "completed": True,
        }
    )
    (ROOT / (variant + "-suite.json")).write_text(json.dumps(rows, indent=2))
    print(variant, target, result.returncode, flush=True)
raise SystemExit(any(row["exit_status"] for row in rows))
