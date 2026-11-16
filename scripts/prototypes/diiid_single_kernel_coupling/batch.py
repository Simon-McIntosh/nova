"""Run paired committed DIII-D revisions sequentially inside one GPU job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROWS = ((137, "diverted"), (146, "diverted"), (146, "limited"), (179, "limited"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).with_name("measure.py").resolve()
    records = []
    with tempfile.TemporaryDirectory(
        prefix="coupling-arms-", dir=os.environ["TMPDIR"]
    ) as temporary:
        trees = {}
        for arm, revision in (("base", args.base), ("head", args.head)):
            tree = Path(temporary) / arm
            tree.mkdir()
            with subprocess.Popen(
                ["git", "-C", str(args.repository), "archive", revision],
                stdout=subprocess.PIPE,
            ) as archive:
                subprocess.run(
                    ["tar", "-x", "-C", str(tree)], stdin=archive.stdout, check=True
                )
                assert archive.wait() == 0
            trees[arm] = tree
        for frame, branch in ROWS:
            for arm, revision in (("base", args.base), ("head", args.head)):
                tree = trees[arm]
                folder = args.output / arm / f"frame-{frame}-{branch}"
                folder.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(driver),
                    "--tree",
                    str(tree),
                    "--revision",
                    revision,
                    "--frame",
                    str(frame),
                    "--branch",
                    branch,
                    "--output",
                    str(folder),
                ]
                environment = dict(
                    os.environ,
                    PYTHONPATH=str(tree),
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                )
                with (folder / "row.log").open("w") as stream:
                    stream.write(
                        f"# revision: {revision}; tree: {tree}; "
                        f"command: {shlex.join(command)}\n"
                    )
                    stream.flush()
                    status = subprocess.run(
                        command,
                        cwd=tree,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    ).returncode
                    stream.write(f"EXIT={status}\n")
                item = dict(
                    arm=arm,
                    frame=frame,
                    branch=branch,
                    revision=revision,
                    exit_status=status,
                    log_path=str(folder / "row.log"),
                )
                if status == 0:
                    receipt = json.loads((folder / "receipt.json").read_text())
                    solved = json.loads((folder / "solve-receipt.json").read_text())
                    item.update(
                        converged=solved["converged"],
                        residual=solved["residual"],
                        terminal_reason=solved["terminal_reason"],
                        topology=receipt["terminal_topology"],
                        efit_x_point_rz_m=receipt["efit_x_point_rz_m"],
                    )
                records.append(item)
                (args.output / "comparison.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                print("ROW " + json.dumps(item), flush=True)
                if status:
                    raise SystemExit(status)
    print(f"PAIRED_ROWS={len(records)}", flush=True)


if __name__ == "__main__":
    main()
