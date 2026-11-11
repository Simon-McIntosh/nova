"""Run one GPU allocation's paired rows, regressions and render outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--recover-from", type=Path)
    source.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    os.environ["PYTEST_TIMEOUT"] = "3600"
    with tempfile.TemporaryDirectory(
        prefix="nova-branch-", dir=os.environ["TMPDIR"]
    ) as scratch:
        os.environ["TMPDIR"] = scratch
        summary = {}

        def run(name, script, arguments, platforms):
            command = [sys.executable, str(scripts / script), *arguments]
            log = args.output / (name + ".log")
            print(
                "JOB_STAGE "
                + json.dumps(dict(name=name, command=command, log=str(log))),
                flush=True,
            )
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {args.head}; tree: {scripts.parents[2]}; "
                    f"command: {command!r}\n"
                )
                stream.flush()
                result = subprocess.run(
                    command,
                    env=dict(os.environ, JAX_PLATFORMS=platforms),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                stream.write(f"EXIT={result.returncode}\n")
            summary[name] = dict(exit_status=result.returncode, log_path=str(log))
            (args.output / "job-results.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            return result.returncode

        retained = ["--resume-from", str(args.resume_from)] if args.resume_from else []
        if args.recover_from is not None:
            previous = args.output / "recovered"
            status = run(
                "recover-base137",
                "recover.py",
                [
                    "--source",
                    str(args.recover_from),
                    "--revision",
                    args.base,
                    "--output",
                    str(previous / "base/frame-137-diverted"),
                ],
                "cuda,cpu",
            )
            if status:
                raise SystemExit(status)
            retained = ["--resume-from", str(previous)]
        status = run(
            "measurement",
            "measure.py",
            [
                "--base",
                args.base,
                "--head-revision",
                args.head,
                "--output",
                str(args.output / "measurement"),
                "--forward-regressions",
                *retained,
            ],
            "cuda,cpu",
        )
        if status:
            raise SystemExit(status)
        for arm, revision in (("base", args.base), ("head", args.head)):
            status = run(
                "regression-" + arm,
                "gates.py",
                [
                    "--revision",
                    revision,
                    "--mode",
                    "regression",
                    "--output",
                    str(args.output / ("regression-" + arm)),
                ],
                "cpu",
            )
            if status:
                raise SystemExit(status)
        status = run(
            "render",
            "render.py",
            [
                "--input",
                str(args.output / "measurement"),
                "--output",
                str(args.output / "rendered-svg"),
                "--previews",
                str(args.output / "previews"),
            ],
            "cpu",
        )
        raise SystemExit(status)


if __name__ == "__main__":
    main()
