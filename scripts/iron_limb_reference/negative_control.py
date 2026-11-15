"""Ablate the landed surface solver's neutrality projection in a scratch copy."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    target = "tests/test_iron_limb_reference.py::test_each_body_charge_is_deflated"
    command = [
        "/home/ITER/mcintos/Code/nova/.venv/bin/python",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        target,
    ]
    with tempfile.TemporaryDirectory(
        prefix="surface-neutrality-", dir=args.scratch_root
    ) as folder:
        scratch = Path(folder).resolve()
        for name in [
            "scripts/__init__.py",
            "tests/__init__.py",
            "scripts/iron_limb_reference/surface.py",
            "scripts/prototypes/iron_limb_fill/bricks.py",
            "tests/test_iron_limb_reference.py",
        ]:
            path = scratch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, path)
        (scratch / "pytest.ini").write_text("[pytest]\n")
        module = scratch / "scripts/iron_limb_reference/surface.py"
        env = dict(
            os.environ,
            PYTHONPATH=str(scratch),
            JAX_PLATFORMS="cpu",
            NUMBA_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        probe = (
            "from pathlib import Path; "
            "from scripts.iron_limb_reference import surface; "
            "print('IMPORTED_MODULE=' + surface.__file__); "
            "assert Path(surface.__file__).resolve() == "
            "Path('scripts/iron_limb_reference/surface.py').resolve(); "
            "print('RESOLVED_CWD=' + str(Path.cwd()))"
        )
        for mutated in (False, True):
            if mutated:
                original = (
                    "result[members] -= "
                    "np.dot(weights, result[members]) / weights.sum()"
                )
                source = module.read_text()
                if source.count(original) != 1:
                    raise ValueError("neutrality mutation target is not unique")
                module.write_text(source.replace(original, "result[members] -= 0.0"))
            log_path = args.log if mutated else Path(str(args.log) + ".base")
            with log_path.open("w") as log:
                log.write(
                    f"# revision: {revision} tree: {scratch} "
                    f"command: {' '.join(command)}\n"
                )
                log.write(f"# module: {module}\n# cwd: {scratch}\n")
                mutation = (
                    "disable area-weighted body-neutrality projection"
                    if mutated
                    else "none: positive control"
                )
                log.write(f"# mutation: {mutation}\n")
                log.flush()
                subprocess.run(
                    [command[0], "-c", probe],
                    cwd=scratch,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                result = subprocess.run(
                    command, cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                log.write(f"EXIT={result.returncode}\n")
            text = log_path.read_text()
            if mutated:
                if result.returncode == 0 or f"FAILED {target}" not in text:
                    raise RuntimeError(
                        "the landed-solver mutation did not fail the named test"
                    )
            elif result.returncode:
                raise RuntimeError("the unmutated scratch control failed")
        print(f"NEGATIVE_CONTROL_OBSERVED test={target} log={args.log}")
        print(f"Removing mutation scratch: {scratch}")


if __name__ == "__main__":
    main()
