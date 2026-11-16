"""Run a diagnostic arm from an immutable source archive in job-local scratch."""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--point-budget", type=int, default=0)
    args = parser.parse_args()
    revision = "ff92ac8f63e3088b4950104a0d146f37dcdc1842"
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="polygon-trace-", dir=os.environ["TMPDIR"]
    ) as temporary:
        root = Path(temporary)
        archive = root / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "-C", str(args.repo), "archive", revision, "nova", "scripts"],
                stdout=stream,
                check=True,
            )
        with tarfile.open(archive) as stream:
            stream.extractall(root, filter="data")
        target = root / "scripts/prototypes/polygon_trace_miss"
        shutil.copytree(Path(__file__).parent, target, dirs_exist_ok=True)
        for name in ("measure.py", "report.py"):
            data = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(args.repo),
                    "show",
                    "bf9964e4caab435a75e03265aa6af94f0a88cd8f:"
                    "scripts/prototypes/bank_hessian_diagnosis/" + name,
                ]
            )
            (root / "scripts/prototypes/bank_hessian_diagnosis" / name).write_bytes(
                data
            )
        module = root / "nova/biot/polygonanalytic.py"
        if args.mode == "stable-corner":
            source = module.read_text()
            start = source.index("def _packed_moments_height_tangent(")
            end = source.index("\ndef packed_analytic_moments(", start)
            block = source[start:end]
            assert block.count("return jax.jvp(evaluate, primals, direction)") == 1
            block = block.replace(
                "return jax.jvp(evaluate, primals, direction)",
                "return jax.jvp(tangent_expression, primals, direction)",
            )
            block = block.replace(
                "_, other = jax.jvp(evaluate, primals, without_height)",
                "_, other = jax.jvp(tangent_expression, primals, without_height)",
            )
            block = block.replace(
                "lambda shift: evaluate(", "lambda shift: tangent_expression("
            )
            block = block.replace(
                "            evaluate(primals[0], target_z + shift, *primals[2:])",
                "            tangent_expression(primals[0], target_z + shift, "
                "*primals[2:])",
            )
            block = block.replace(
                "    return value, result", "    return evaluate(*primals), result"
            )
            assert "return evaluate(*primals), result" in block
            module.write_text(source[:start] + block + source[end:])
            (args.output / "scratch-kernel.patch").write_text(
                "".join(
                    difflib.unified_diff(
                        source.splitlines(True),
                        module.read_text().splitlines(True),
                        fromfile="candidate",
                        tofile="stable-corner",
                    )
                )
            )
        print("# revision:", revision, "scratch_variant:", args.mode, flush=True)
        print(
            "# module:",
            module,
            "sha256:",
            hashlib.file_digest(module.open("rb"), "sha256").hexdigest(),
            flush=True,
        )
        environment = dict(
            os.environ, PYTHONPATH=str(root), JAX_PLATFORMS="cpu", TMPDIR=str(root)
        )
        command = [
            "/home/ITER/mcintos/Code/nova/.venv/bin/python",
            "-m",
            "scripts.prototypes.polygon_trace_miss.diagnose",
            "--mode",
            args.mode,
            "--output",
            str(args.output),
        ]
        if args.point_budget:
            command += ["--point-budget", str(args.point_budget)]
        print("# tree:", root, flush=True)
        print("# command:", " ".join(command), flush=True)
        batches = 16 if args.mode == "stable-corner" else 1
        for batch in range(batches):
            print("PROCESS_BATCH", batch, flush=True)
            result = subprocess.run(command, cwd=root, env=environment)
            if result.returncode:
                break
            receipt = args.output / "receipt.json"
            if receipt.exists() and json.loads(receipt.read_text()).get(
                "completed"
            ):
                break
        print("EXIT=" + str(result.returncode), flush=True)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
