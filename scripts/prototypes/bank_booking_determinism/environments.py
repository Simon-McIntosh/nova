"""Run fresh device processes under controlled compiler environments."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
SETTINGS = {
    "unset": None,
    "deterministic": "--xla_gpu_deterministic_ops=true",
    "autotune-disabled": "--xla_gpu_autotune_level=0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tree", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    cache = args.run / "compilation-cache"
    with tempfile.TemporaryDirectory(prefix="booking-environments-") as temporary:
        scratch = Path(temporary)
        base = scratch / "base"
        base.mkdir()
        archive = subprocess.Popen(
            [
                "git",
                "-C",
                str(args.tree),
                "archive",
                args.base,
                "nova",
                "scripts",
                "tests",
                "benchmarks",
                "pyproject.toml",
                "conftest.py",
            ],
            stdout=subprocess.PIPE,
        )
        subprocess.run(["tar", "-x", "-C", str(base)], stdin=archive.stdout, check=True)
        assert archive.wait() == 0
        outcomes = []
        for name, flags in SETTINGS.items():
            for repeat in range(5):
                output = args.run / f"{name}-{repeat}"
                command = [
                    PYTHON,
                    "-P",
                    str(
                        args.tree
                        / "scripts/prototypes/bank_booking_determinism/measure.py"
                    ),
                    "--output",
                    str(output),
                    "--revision",
                    args.base,
                ]
                env = dict(
                    os.environ,
                    TMPDIR=str(scratch),
                    PYTHONPATH=str(base),
                    JAX_PLATFORMS="cuda,cpu",
                    OMP_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="2",
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                    JAX_COMPILATION_CACHE_DIR=str(cache),
                    JAX_ENABLE_COMPILATION_CACHE="true",
                    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
                    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
                )
                env.pop("XLA_FLAGS", None)
                if flags is not None:
                    env["XLA_FLAGS"] = flags
                prefix = [
                    "env",
                    "-u",
                    "XLA_FLAGS",
                    f"PYTHONPATH={base}",
                    f"JAX_COMPILATION_CACHE_DIR={cache}",
                    "JAX_ENABLE_COMPILATION_CACHE=true",
                    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1",
                    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0",
                ]
                if flags is not None:
                    prefix.append(f"XLA_FLAGS={flags}")
                literal = shlex.join(prefix + command)
                with output.with_suffix(".log").open("w") as log:
                    log.write(
                        f"# revision: {args.base} tree: {base} command: {literal}\n"
                    )
                    log.write(f"EFFECTIVE_XLA_FLAGS={flags!r}\nCACHE_ROOT={cache}\n")
                    log.flush()
                    result = subprocess.run(
                        command, cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    log.write(f"EXIT={result.returncode}\n")
                row = dict(
                    setting=name,
                    process=repeat,
                    flags=flags,
                    command=literal,
                    exit_status=result.returncode,
                    log_path=str(output.with_suffix(".log")),
                )
                outcomes.append(row)
                (args.run / "processes.json").write_text(json.dumps(outcomes, indent=2))
                print("PROCESS", json.dumps(row), flush=True)
        print("REMOVING_SCRATCH", scratch, flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
