"""Run the step gate against a scratch copy whose read decision sits in the map.

The mutation replaces :func:`support_read.step_batch`'s batch-uniform read with
a ``vmap`` of the single-lane read, so the rebuild decision is formed inside
the map and each lane decides for itself. That is the defect the branch clause
refuses: a batch carrying one stale track must rebuild every lane and report
the rebuild on every lane.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BATCHED = """    support = read_batch(
        operands,
        cache,
        moments,
        seeds,
        previous,
        convention,
        policy,
        axis_sign,
        iterations=iterations,
        tracking=tracking,
    )"""

MUTANT = """    support = jax.vmap(
        lambda state, seed, warm, track: read(
            operands,
            cache,
            state,
            seed,
            warm,
            convention,
            policy,
            axis_sign,
            iterations=iterations,
            tracking=track,
        )
    )(moments, seeds, previous, tracking)"""


def transplant(root: Path, scratch: Path):
    for name in ("nova", "tests", "scripts", "benchmarks"):
        source = root / name
        if source.is_dir():
            shutil.copytree(
                source, scratch / name, copy_function=os.link, symlinks=True
            )
    target = scratch / "nova" / "equilibrium" / "support_read.py"
    text = target.read_text()
    if BATCHED not in text:
        raise SystemExit(
            "the batch-uniform read call was not found in the scratch copy"
        )
    target.unlink()
    target.write_text(text.replace(BATCHED, MUTANT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--basetemp", type=Path, required=True)
    arguments = parser.parse_args()

    scratch = arguments.scratch
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    transplant(arguments.root, scratch)

    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{scratch}:{arguments.root}"
    print(f"SCRATCH={scratch}", flush=True)
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import nova, nova.equilibrium.support_read as m;"
            "print('NOVA='+nova.__file__+' MODULE='+m.__file__)",
        ],
        cwd=arguments.root,
        env=environment,
        text=True,
        capture_output=True,
    )
    print(check.stdout, flush=True)
    if str(scratch) not in check.stdout or "support_read.py" not in check.stdout:
        print("the scratch copy did not shadow the shipped module", flush=True)
        print("EXIT=2", flush=True)
        return 2
    if check.stdout.count(str(scratch)) != 2:
        print("the scratch copy did not provide both nova and support_read", flush=True)
        print("EXIT=2", flush=True)
        return 2

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(scratch / "tests" / "test_bank_step_batch.py"),
            f"--basetemp={arguments.basetemp}",
        ],
        cwd=arguments.root,
        env=environment,
    )
    print(f"EXIT={result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
