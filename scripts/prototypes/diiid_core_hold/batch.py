"""Run paired immutable source arms, persisting every completed row."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_warmup_admission"))
from measure import extract  # noqa: E402
from regression import gate  # noqa: E402

CASES = (
    (137, "diverted"),
    (146, "diverted"),
    (146, "limited"),
    (179, "limited"),
    (35, "mast"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    parser.add_argument("--cpu-files", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="nova-core-hold-", dir=os.environ["TMPDIR"]
    ) as scratch:
        summary = {}
        for arm, revision in (("base", args.base), ("head", args.head)):
            tree = Path(scratch) / arm
            extract(tree, revision, ROOT)
            if args.cpu_files:
                files = json.loads(args.cpu_files.read_text())
                for file in files:
                    if not (tree / file).is_file():
                        continue
                    result = gate(
                        tree,
                        revision,
                        file,
                        args.output / (arm + "-" + Path(file).stem + ".log"),
                        quiet=True,
                    )
                    summary[arm + ":" + file] = result
                    (args.output / "results.json").write_text(
                        json.dumps(summary, indent=2) + "\n"
                    )
                continue
            for frame, branch in CASES:
                case = (
                    "mast-21978-35" if branch == "mast" else f"frame-{frame}-{branch}"
                )
                output = args.output / arm / case
                output.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("measure.py")),
                    "--tree",
                    str(tree),
                    "--revision",
                    revision,
                    "--frame",
                    str(frame),
                    "--branch",
                    branch,
                    "--output",
                    str(output),
                    "--trace-helper",
                    str(args.trace_helper),
                ]
                log = output / "row.log"
                with log.open("w") as stream:
                    stream.write(
                        f"# revision: {revision}; tree: {tree}; "
                        f"command: {shlex.join(command)}\n"
                    )
                    stream.flush()
                    result = subprocess.run(
                        command,
                        cwd=tree,
                        env=dict(os.environ, PYTHONPATH=str(tree)),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    stream.write(f"EXIT={result.returncode}\n")
                summary[arm + ":" + case] = dict(
                    exit_status=result.returncode, log_path=str(log)
                )
                (args.output / "results.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                print("ROW", arm, case, result.returncode, flush=True)
                if result.returncode:
                    raise SystemExit(result.returncode)
            result = gate(
                tree,
                revision,
                "tests/test_equilibrium_forward_solve.py",
                args.output / (arm + "-forward.log"),
                platforms=os.environ["JAX_PLATFORMS"],
            )
            summary[arm + ":forward"] = result
            (args.output / "results.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
