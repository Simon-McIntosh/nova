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
        root = Path(temporary)
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
        print(
            f"Removing only the four copied receipt files and their directory: {root}"
        )
    print(f"Removed {temporary}; original receipts preserved")


if __name__ == "__main__":
    main()
