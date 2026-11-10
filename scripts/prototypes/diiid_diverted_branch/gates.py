"""Run immutable solver arms and remove branch qualification for a red control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_warmup_admission"))
from measure import extract  # noqa: E402
from regression import FILES, gate  # noqa: E402

NODE_TEST = "tests/test_diverted_branch_continuation.py"


def main():
    os.environ["PYTEST_TIMEOUT"] = "3600"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("reproduce", "negative", "verify", "suite", "regression"),
        required=True,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="diverted-branch-", dir=os.environ["TMPDIR"]
    ) as temporary:
        tree = Path(temporary) / "tree"
        extract(tree, args.revision, ROOT)
        if args.mode == "reproduce":
            shutil.copyfile(ROOT / NODE_TEST, tree / NODE_TEST)
            result = gate(
                tree,
                args.revision,
                NODE_TEST
                + "::test_newton_refuses_boundary_loss_and_upper_saddle_switch",
                args.output / "reproduction.log",
                quiet=True,
            )
            assert result["exit_status"] != 0 and result["failure_ids"]
        elif args.mode in ("negative", "verify"):
            if args.mode == "verify":
                after = gate(
                    tree,
                    args.revision,
                    NODE_TEST,
                    args.output / "suite.log",
                    quiet=True,
                )
                (args.output / "after-suite.json").write_text(
                    json.dumps(after, indent=2) + "\n"
                )
                assert after["exit_status"] == 0 and after["completed"]
            module = tree / "nova/equilibrium/fixed_point.py"
            text = module.read_text()
            marker = "    seed = partition_read(initial, None).topology"
            assert text.count(marker) == 1
            module.write_text(text.replace(marker, "    return None\n" + marker))
            result = gate(
                tree,
                args.revision,
                NODE_TEST,
                args.output / "negative-control.log",
                quiet=True,
            )
            assert result["exit_status"] != 0 and result["failure_ids"]
        elif args.mode == "suite":
            if (tree / NODE_TEST).is_file():
                result = gate(
                    tree,
                    args.revision,
                    NODE_TEST,
                    args.output / "suite.log",
                    quiet=True,
                )
            else:
                # An empty scoped arm must not fall through to unscoped pytest.
                command = [
                    sys.executable,
                    "-c",
                    "import os; from pathlib import Path; "
                    "from nova.equilibrium import fixed_point; "
                    "print('IMPORTED_MODULE=' + fixed_point.__file__); "
                    "print('RESOLVED_CWD=' + os.getcwd()); "
                    "assert Path('nova/equilibrium/fixed_point.py').is_file(); "
                    "assert not Path('tests/test_diverted_branch_continuation.py')"
                    ".exists(); print('BASE_SCOPE_EMPTY; "
                    "KNOWN_PRESENT=nova/equilibrium/fixed_point.py')",
                ]
                log = args.output / "suite.log"
                literal = "PYTHONPATH=. JAX_PLATFORMS=cpu " + shlex.join(command)
                with log.open("w") as stream:
                    stream.write(
                        f"# revision: {args.revision}; tree: {tree}; "
                        f"command: {literal}\n"
                    )
                    stream.flush()
                    run = subprocess.run(
                        command,
                        cwd=tree,
                        env=dict(os.environ, PYTHONPATH=".", JAX_PLATFORMS="cpu"),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    stream.write(f"EXIT={run.returncode}\n")
                assert run.returncode == 0
                result = dict(
                    revision=args.revision,
                    command=literal,
                    exit_status=0,
                    log_path=str(log),
                    completed=True,
                    failure_count=0,
                    failure_ids=[],
                    note=(
                        "No scoped tests exist at base; positive module-presence "
                        "and scoped-test-absence probe substituted."
                    ),
                )
        else:
            result = {}
            for path in FILES:
                result[path] = gate(
                    tree, args.revision, path, args.output / (Path(path).stem + ".log")
                )
                (args.output / "results.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
