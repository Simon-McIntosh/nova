"""Run a fresh-certificate regression against a tracked-read scratch mutation."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("revision")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(
        prefix="polish-control-", dir=os.environ["TMPDIR"]
    ) as temporary:
        tree = Path(temporary)
        archive = subprocess.Popen(
            [
                "git",
                "-C",
                str(args.repo),
                "archive",
                args.revision,
                "nova",
                "scripts",
                "tests",
                "benchmarks",
                "pyproject.toml",
                "conftest.py",
            ],
            stdout=subprocess.PIPE,
        )
        subprocess.run(["tar", "-x", "-C", str(tree)], stdin=archive.stdout, check=True)
        assert archive.wait() == 0
        source = tree / "nova/equilibrium/support_read.py"
        original = "return magnitude(trial, trial_read), trial_read.valid, trial_read"
        mutant = (
            "return jnp.where(certifying, magnitude(trial, fresh), "
            "magnitude(trial, trial_read)), trial_read.valid, trial_read"
        )
        text = source.read_text()
        assert text.count(original) == 1
        source.write_text(text.replace(original, mutant))
        command = [
            PYTHON,
            "-P",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            f"--basetemp={tree / 'pytest'}",
            str(tree / "tests/test_bank_solve_driver.py")
            + "::test_uncertified_polish_resumes_adaptive_search[False]",
            "-vv",
            "-s",
        ]
        env = dict(
            os.environ,
            PYTHONPATH=str(tree),
            JAX_PLATFORMS="cpu",
            PYTHONDONTWRITEBYTECODE="1",
        )
        literal = shlex.join(
            ["env", f"PYTHONPATH={tree}", "JAX_PLATFORMS=cpu", *command]
        )
        with (args.run / "negative-control.log").open("w") as log:
            log.write(f"# revision: {args.revision} tree: {tree} command: {literal}\n")
            log.write(f"MUTATION: {original} -> {mutant}\n")
            log.flush()
            subprocess.run(
                [
                    PYTHON,
                    "-P",
                    "-c",
                    "import nova, nova.equilibrium.support_read as m; "
                    "from pathlib import Path; print('nova', nova.__file__, "
                    "'module', m.__file__, 'cwd', Path.cwd().resolve())",
                ],
                cwd=tree,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            result = subprocess.run(
                command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            log.write(f"EXIT={result.returncode}\n")
        print("NEGATIVE_CONTROL_EXIT", result.returncode)
        assert result.returncode == 1
        print("REMOVING_SCRATCH", tree)


if __name__ == "__main__":
    main()
