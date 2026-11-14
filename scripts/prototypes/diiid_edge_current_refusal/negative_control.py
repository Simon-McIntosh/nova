"""Require the receipt verifier to reject a fabricated zero edge profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    tree = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="edge-receipt-control-") as temporary:
        root = Path(temporary) / "cohort"
        folder = root / "edge/frame-137-diverted"
        folder.mkdir(parents=True)
        for name in ("row.log", "receipt.json", "spatial.json", "source-inputs.json"):
            shutil.copy2(args.input / "edge/frame-137-diverted" / name, folder / name)
        path = folder / "source-inputs.json"
        record = json.loads(path.read_text())
        record["p_prime"]["values"][-1] = 0.0
        path.write_text(json.dumps(record) + "\n")
        command = [
            sys.executable,
            str(Path(__file__).with_name("verify.py")),
            "--input",
            str(root),
            "--output",
            str(root / "summary.json"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        args.log.write_text(
            f"# revision: {revision}; tree: {tree}; command: "
            + shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
            + f"\n# child command: {shlex.join(command)}\n"
            + f"# module: {Path(__file__).with_name('verify.py')}; cwd: {Path.cwd()}\n"
            + "# mutation: first row p_prime(1) changed to zero in an isolated copy\n"
            + output
            + f"EXIT={result.returncode}\n"
        )
        assert result.returncode == 1
        assert 'assert table["values"][-1] != 0' in output
        print(f"CONTROL: zero boundary profile refused; log={args.log}")
        last = root / "edge/frame-179-limited"
        last.mkdir(parents=True)
        for name in ("row.log", "spatial.json"):
            shutil.copy2(args.input / "edge/frame-179-limited" / name, last / name)
        failed = root / "inactive-field-import-failure"
        failed.mkdir()
        shutil.copy2(
            args.input / "inactive-field-import-failure/control.log",
            failed / "control.log",
        )
        terminal = json.loads((last / "spatial.json").read_text())["terminal"]
        lines = (args.input.parent / "titan-trace.log").read_text().splitlines()
        kept = [
            line
            for line in lines
            if not (
                line.startswith("SPATIAL_SUMMARY ")
                and json.loads(line.removeprefix("SPATIAL_SUMMARY "))["terminal"]
                == terminal
            )
        ]
        assert len(lines) - len(kept) == 1
        (root.parent / "titan-trace.log").write_text("\n".join(kept) + "\n")
        child = (
            "import json,sys;from pathlib import Path;"
            f"sys.path.insert(0,{str(Path(__file__).parent)!r});"
            "from verify import completion;"
            f"p=Path({str(root)!r});q=p/'edge/frame-179-limited';"
            "completion(p,q,json.loads((q/'spatial.json').read_text()))"
        )
        command = [sys.executable, "-c", child]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        completion_log = args.log.with_name("completion-negative-control.log")
        completion_log.write_text(
            f"# revision: {revision}; tree: {tree}; command: {shlex.join(command)}\n"
            "# mutation: remove exactly the completed last-row spatial summary\n"
            + output
            + f"EXIT={result.returncode}\n"
        )
        assert result.returncode == 1
        assert "assert len(matched) == 1" in output
        print(f"CONTROL: incomplete failed-tail trace refused; log={completion_log}")
        print(f"Removing only isolated copied receipts and logs under {temporary}")
    print(f"Removed {temporary}; original receipts preserved")


if __name__ == "__main__":
    main()
