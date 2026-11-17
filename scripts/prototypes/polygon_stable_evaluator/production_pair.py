"""Profile paired production rows using the existing cold H200 harness."""

import argparse
import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

from scripts.prototypes.compile_pass_profile.analyse import (
    INSTRUCTION,
    SKIP,
    describe,
)


def function_counts(path):
    """Count each HLO computation with the profile's instruction predicate."""
    counts = {}
    function = "module"
    with path.open() as stream:
        for line in stream:
            if line and not line[0].isspace() and line.rstrip().endswith("{"):
                function = line.removeprefix("ENTRY ").split("(", 1)[0].strip(" %{\n")
            if line.rstrip().endswith(") {") or any(token in line for token in SKIP):
                continue
            if INSTRUCTION.match(line):
                counts[function] = counts.get(function, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def dump_breakdown(dumps, attribution, output):
    """Retain the first and post-splitter HLO, including per-function counts."""
    modules = {}
    for path in dumps.iterdir():
        if path.name.endswith((".txt", ".ll")):
            module, label, _ = describe(path)
            modules.setdefault(module, []).append(
                (path.stat().st_mtime_ns, path, label)
            )
    chosen = sorted(
        max(modules.values(), key=len), key=lambda row: (row[0], row[1].name)
    )
    sequence = attribution["sequence"]
    assert [row[2] for row in chosen] == [row["pass"] for row in sequence]
    splitter = [
        index
        for index, row in enumerate(sequence)
        if row["pass"].endswith("/variadic-op-splitter")
    ]
    assert len(splitter) == 1
    following = next(
        index
        for index in range(splitter[0] + 1, len(sequence))
        if not sequence[index]["pass"].endswith(("priority_fusion_dump.txt", ".ll"))
    )
    result = {}
    for label, index in (("first_hlo", 0), ("post_splitter", following)):
        source = chosen[index][1]
        counts = function_counts(source)
        total = sum(counts.values())
        assert total == sequence[index]["instructions"] and total > 0
        archive = output / (label + ".hlo.gz")
        with source.open("rb") as incoming, gzip.open(archive, "wb") as outgoing:
            while block := incoming.read(1024 * 1024):
                outgoing.write(block)
        result[label] = dict(
            instructions=total,
            functions=counts,
            pass_name=sequence[index]["pass"],
            archive=str(archive),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.repository),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        text=True,
    ).strip()
    result = dict(
        base=args.base,
        head=args.head,
        program="production_point_batch",
        size=132,
        instruction_limit=2.3,
        cold_compile_limit=2.3,
        completed=False,
        arms={},
    )
    summary = args.output / "production-budget.json"
    summary.write_text(json.dumps(result, indent=2))
    for name, revision in (("base", args.base), ("head", args.head)):
        output = args.output / name
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=os.environ["TMPDIR"], prefix="polygon-production-"
        ) as scratch:
            scratch = Path(scratch)
            tree = scratch / "tree"
            tree.mkdir()
            archive = scratch / "source.tar"
            with archive.open("wb") as stream:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(args.repository),
                        "archive",
                        revision,
                        "nova",
                        "scripts",
                        "tests",
                        "pyproject.toml",
                    ],
                    stdout=stream,
                    check=True,
                )
            with tarfile.open(archive) as stream:
                stream.extractall(tree, filter="fully_trusted")
            gitdir = scratch / "gitdir"
            gitdir.mkdir()
            (gitdir / "HEAD").write_text(revision + "\n")
            (gitdir / "commondir").write_text(common + "\n")
            (gitdir / "gitdir").write_text(str(tree / ".git") + "\n")
            (tree / ".git").write_text("gitdir: " + str(gitdir) + "\n")
            assert (
                subprocess.check_output(
                    ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
                ).strip()
                == revision
            )
            dumps = scratch / "dumps"
            command = [
                "bash",
                str(tree / "scripts/prototypes/compile_pass_profile/run.sh"),
                str(tree),
                "production_point_batch",
                "132",
                str(output),
                str(dumps),
            ]
            environment = dict(
                os.environ,
                TMPDIR=str(scratch),
                PYTHONPATH=str(tree),
                JAX_PLATFORMS="cuda",
                JAX_ENABLE_COMPILATION_CACHE="false",
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
            )
            log_path = output / "profile.log"
            with log_path.open("w") as log:
                log.write(
                    f"# revision: {revision}; tree: {tree}; "
                    f"command: {shlex.join(command)}\n"
                )
                log.write(
                    f"MEASUREMENT_MODULE {tree / 'nova/biot/polygonanalytic.py'}\n"
                    f"MEASUREMENT_CWD {tree}\n"
                )
                log.flush()
                code = subprocess.run(
                    command,
                    cwd=tree,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ).returncode
                log.write(f"EXIT={code}\n")
            assert code == 0, (name, code)
            profile = json.loads((output / "profile.json").read_text())
            attribution = json.loads((output / "attribution.json").read_text())
            assert profile["phase"] == "done" and profile["revision"] == revision
            assert "H200" in profile["device"] and not profile["cache_enabled"]
            assert (
                profile["cache_guard_hits"] == 0 and profile["cache_guard_misses"] > 0
            )
            breakdown = dump_breakdown(dumps, attribution, output)
            (output / "function-instructions.json").write_text(
                json.dumps(breakdown, indent=2)
            )
            result["arms"][name] = dict(
                revision=revision,
                instructions=breakdown["post_splitter"]["instructions"],
                compile_seconds=profile["compile_seconds"],
                log_path=str(log_path),
                breakdown=str(output / "function-instructions.json"),
                measurement_module=str(tree / "nova/biot/polygonanalytic.py"),
                measurement_cwd=str(tree),
            )
            identities = (
                "edge_construction",
                "second_residual",
                "channel_split",
                "flux_line_moments",
            )
            command = [
                "/home/ITER/mcintos/Code/nova/.venv/bin/python",
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-vv",
                "-s",
                *[
                    "tests/biot/test_edge_term_tangents.py::"
                    f"test_tangent_matches_base_jvp[{identity}]"
                    for identity in identities
                ],
            ]
            environment["PYTHONPATH"] += ":" + str(tree / "tests/biot")
            rows_log = output / "pinned-rows.log"
            with rows_log.open("w") as log:
                log.write(
                    f"# revision: {revision}; tree: {tree}; "
                    f"command: {shlex.join(command)}\n"
                )
                log.write(
                    f"MEASUREMENT_MODULE {tree / 'nova/biot/polygonanalytic.py'}\n"
                    f"MEASUREMENT_CWD {tree}\n"
                )
                log.flush()
                code = subprocess.run(
                    command,
                    cwd=tree,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ).returncode
                log.write(f"EXIT={code}\n")
            result["arms"][name]["pinned_rows"] = dict(
                command=shlex.join(command), exit_status=code, log_path=str(rows_log)
            )
            summary.write_text(json.dumps(result, indent=2))
            print("PRODUCTION_ARM", name, json.dumps(result["arms"][name]), flush=True)
    base, head = (result["arms"][name] for name in ("base", "head"))
    result.update(
        completed=True,
        instruction_ratio=head["instructions"] / base["instructions"],
        cold_compile_ratio=head["compile_seconds"] / base["compile_seconds"],
    )
    result["passed"] = (
        result["instruction_ratio"] <= 2.3 and result["cold_compile_ratio"] <= 2.3
    )
    summary.write_text(json.dumps(result, indent=2))
    print("PRODUCTION_BUDGET", json.dumps(result), flush=True)
    assert result["passed"]


if __name__ == "__main__":
    main()
