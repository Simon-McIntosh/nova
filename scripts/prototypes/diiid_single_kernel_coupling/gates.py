"""Run focused base/head tests and the hybrid substitution control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

TESTS = (
    "tests/equilibrium/test_finite_edge_booking.py",
    "tests/imas/test_diiid_forward_gs_match.py",
    "tests/imas/test_diiid_forward_gs_match_helpers.py",
    "tests/imas/test_diiid_current_pinned_forward.py",
    "tests/test_default_wiring.py",
)
REGRESSION = "tests/equilibrium/test_diiid_single_kernel_coupling.py"


def run(tree, revision, targets, log):
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *targets]
    environment = dict(os.environ, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu")
    with log.open("w") as stream:
        stream.write(
            f"# revision: {revision}; tree: {tree}; "
            f"command: PYTHONPATH=. JAX_PLATFORMS=cpu {shlex.join(command)}\n"
        )
        stream.flush()
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from benchmarks import diiid_forward_gs_match as m; "
                "from pathlib import Path; print('MODULE='+m.__file__); "
                "print('CWD='+str(Path.cwd().resolve()))",
            ],
            cwd=tree,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        status = subprocess.run(
            command, cwd=tree, env=environment, stdout=stream, stderr=subprocess.STDOUT
        ).returncode
        stream.write(f"EXIT={status}\n")
    if status > 1:
        raise RuntimeError(f"test command did not complete: {log}, exit {status}")
    return dict(
        revision=revision,
        command="PYTHONPATH=. JAX_PLATFORMS=cpu " + shlex.join(command),
        exit_status=status,
        log_path=str(log),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focused-only", action="store_true")
    parser.add_argument("--targets", nargs="+", default=TESTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(
        prefix="coupling-tests-", dir=os.environ["TMPDIR"]
    ) as temporary:
        for arm, revision in (("base", args.base), ("head", args.head)):
            tree = Path(temporary) / arm
            tree.mkdir()
            paths = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(args.repository),
                    "ls-tree",
                    "-r",
                    "--name-only",
                    revision,
                    "nova",
                    "benchmarks",
                    "scripts",
                    "tests",
                    "pyproject.toml",
                    "docs/figures/diiid-forward-onboarding/current-pinned",
                    "docs/figures/current-constrained-forward-solve/pinned-cohort",
                    "docs/figures/primary-xpoint-evidence/real_equilibria_reachability.py",
                ],
                text=True,
            ).splitlines()
            paths = [
                path
                for path in paths
                if path.endswith((".py", ".toml", ".json", ".jsonl", ".mplstyle"))
            ]
            with subprocess.Popen(
                ["git", "-C", str(args.repository), "archive", revision, *paths],
                stdout=subprocess.PIPE,
            ) as archive:
                subprocess.run(
                    ["tar", "-x", "-C", str(tree)], stdin=archive.stdout, check=True
                )
                assert archive.wait() == 0
            results[arm] = run(
                tree, revision, args.targets, args.output / f"focused-{arm}.log"
            )
            if arm == "head" and not args.focused_only:
                results["scoped-head"] = run(
                    tree, revision, [REGRESSION], args.output / "scoped-head.log"
                )
                path = tree / "benchmarks/diiid_forward_gs_match.py"
                source = path.read_text()
                for name, target in (
                    ("grid", "coordinate"),
                    ("wall", "wall"),
                    ("sample", "sample_coordinate"),
                ):
                    old = f"plasma_to_{name} = {name}_moments[0]"
                    assert source.count(old) == 1
                    source = source.replace(
                        old,
                        f"plasma_to_{name} = _green_block({target}, coordinate, "
                        "float(np.diff(radius).mean()), float(np.diff(height).mean()))",
                    )
                path.write_text(source)
                results["negative"] = run(
                    tree,
                    revision + "+hybrid-uniform-mutation",
                    [REGRESSION],
                    args.output / "negative-control.log",
                )
            (args.output / "gates.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print("GATES " + json.dumps(results), flush=True)
    if not args.focused_only and (
        results["scoped-head"]["exit_status"] != 0
        or results["negative"]["exit_status"] == 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
