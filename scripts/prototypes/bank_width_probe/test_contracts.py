"""Regression checks for width admission, repeat agreement and launcher gates."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.prototypes.bank_width_probe import report, run


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    directory = tmp_path / "receipts"
    source = directory / "measured-source/nova/equilibrium/support_read.py"
    source.parent.mkdir(parents=True)
    source.write_text("source identity\n")
    inputs = tmp_path / "inputs.npz"
    inputs.write_bytes(b"common population")
    summary = dict(
        revision="fixture-revision",
        complete=True,
        inert_control="inert lever: identical lowered programs and execution counts",
        input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        measurement_module="/scratch/nova/equilibrium/support_read.py",
        measurement_cwd="/scratch",
        largest_admitted_width=78,
        first_refused_width=79,
        arms=[],
    )
    monkeypatch.setattr(
        report.subprocess, "check_output", lambda command: b"source identity\n"
    )
    for allocator, width, process in (
        [("default", w, p) for w in (64, 78) for p in (0, 1, 2)]
        + [("default", w, 0) for w in (79, 80, 96, 256)]
        + [("async", 96, 0)]
    ):
        admitted = width in (64, 78) or allocator == "async"
        outcome = (
            "executed"
            if admitted
            else "compiled-memory-refusal"
            if width == 256
            else "runtime-oom"
        )
        exit_status = 0 if admitted else 77 if width == 256 else 1
        path = directory / f"{allocator}-{width}-{process}"
        path.mkdir()
        memory = 2000 if width == 256 else width
        record = dict(
            revision=summary["revision"],
            complete=outcome != "runtime-oom",
            module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            module=summary["measurement_module"],
            cwd=summary["measurement_cwd"],
            inputs_sha256=summary["input_sha256"],
            inputs_path=str(inputs),
            device="NVIDIA H200 NVL",
            x64=True,
            programs={"solve": {"working_bytes": memory}},
            fresh_residual=[1e-12] * width,
            precision={k: [1e-6] * width for k in ("axis", "saddle", "boundary")},
        )
        if width == 256:
            record.update(memory_limited=True, limiting_program="solve")
        (path / "receipt.json").write_text(json.dumps(record))
        message = (
            "jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: "
            "Out of memory while trying to allocate 8GiB.\n"
            if outcome == "runtime-oom"
            else ""
        )
        (path / "gate.log").write_text(message + f"EXIT={exit_status}\n")
        summary["arms"].append(
            dict(
                allocator=allocator,
                width=width,
                process=process,
                allocator_setting="unset"
                if allocator == "default"
                else "cuda_malloc_async",
                admitted=admitted,
                outcome=outcome,
                exit_status=exit_status,
                working_bytes=memory,
                device_limit_bytes=1000,
                receipt=str(path / "receipt.json"),
                log=str(path / "gate.log"),
                batch_ms=float(width) if admitted else None,
                compile_seconds=1.0,
            )
        )
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory, summary


def audit(receipts):
    directory, summary = receipts
    (directory / "summary.json").write_text(json.dumps(summary))
    return report.audit(directory, directory.parent)


def test_complete_family_accepts_runtime_refusals_and_separate_allocator(receipts):
    result = audit(receipts)
    assert result["largest_admitted_width"] == 78
    assert result["first_refused_width"] == 79
    rows = {(r["allocator"], r["width"]): r for r in result["widths"]}
    assert rows["default", 78]["spread"] == 0.0
    assert rows["async", 96]["admitted"]
    assert rows["async", 96]["spread"] is None
    assert rows["default", 96]["batch_ms"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_repeat",
        "disagreement",
        "nonadjacent",
        "missing_async",
        "false_execution",
        "incomplete",
    ],
)
def test_report_refuses_incomplete_or_inconsistent_families(receipts, mutation):
    _, summary = receipts
    if mutation == "missing_repeat":
        summary["arms"] = [
            a for a in summary["arms"] if not (a["width"] == 78 and a["process"] == 2)
        ]
    elif mutation == "disagreement":
        next(a for a in summary["arms"] if a["width"] == 78 and a["process"] == 2)[
            "batch_ms"
        ] *= 1.1
    elif mutation == "nonadjacent":
        summary["first_refused_width"] = 80
    elif mutation == "missing_async":
        summary["arms"] = [a for a in summary["arms"] if a["allocator"] != "async"]
    elif mutation == "false_execution":
        next(a for a in summary["arms"] if a["width"] == 79)["outcome"] = "executed"
    else:
        summary["complete"] = False
    with pytest.raises(AssertionError):
        audit(receipts)


def test_report_refuses_changed_input_bytes(receipts):
    directory, _ = receipts
    (directory.parent / "inputs.npz").write_bytes(b"different population")
    with pytest.raises(AssertionError):
        audit(receipts)


def simulated_processes(fault=None):
    calls = []

    def execute(width, process, allocator="default"):
        calls.append((width, process, allocator))
        admitted = width <= 78 or allocator == "async"
        if fault == "repeat_refused" and width == 78 and process == 2:
            admitted = False
        outcome = (
            "executed"
            if admitted
            else "compiled-memory-refusal"
            if width == 256
            else "runtime-oom"
        )
        if fault == "oversized_executed" and width == 256:
            admitted, outcome = True, "executed"
        wall = float(width)
        if fault == "disagreement" and width == 78 and process == 2:
            wall *= 1.1
        return dict(
            admitted=admitted,
            outcome=outcome,
            batch_ms=wall if admitted else None,
            hlo_sha256="same" if fault == "inert" else f"width-{width}",
        )

    return execute, calls


def test_bisection_reaches_adjacent_boundary_and_repeats_only_endpoints():
    execute, calls = simulated_processes()
    summary = {"complete": False}
    run.measure_widths(execute, summary, lambda: None)
    assert summary["complete"]
    assert (summary["largest_admitted_width"], summary["first_refused_width"]) == (
        78,
        79,
    )
    assert [w for w, p, a in calls if p == 0 and a == "default"] == [
        64,
        96,
        256,
        80,
        72,
        76,
        78,
        79,
    ]
    assert {(w, p) for w, p, _ in calls if p} == {(64, 1), (64, 2), (78, 1), (78, 2)}
    assert calls[-1] == (96, 0, "async")


@pytest.mark.parametrize(
    "fault", ["repeat_refused", "disagreement", "oversized_executed", "inert"]
)
def test_measurement_refuses_invalid_repeats_or_controls(fault):
    execute, _ = simulated_processes(fault)
    summary = {"complete": False}
    with pytest.raises((AssertionError, ValueError)):
        run.measure_widths(execute, summary, lambda: None)
    assert not summary["complete"]


@pytest.mark.parametrize(
    "message",
    ["out of memory", "jax.errors.JaxRuntimeError: INVALID_ARGUMENT: bad operand"],
)
def test_unwitnessed_runtime_failures_are_not_memory_refusals(message):
    with pytest.raises(RuntimeError, match="unclassified"):
        run.classify_outcome(1, {"complete": False, "programs": {"solve": {}}}, message)


@pytest.mark.parametrize("launcher", ["job.sh", "report_job.sh"])
def test_launchers_stop_when_contract_tests_fail(tmp_path, launcher):
    package = tmp_path
    for name in ("scripts", "prototypes", "bank_width_probe"):
        package = package / name
        package.mkdir()
        (package / "__init__.py").write_text("")
    for name in ("run.py", "report.py"):
        (package / name).write_text(
            "print('MEASUREMENT_MUST_NOT_START')\nraise SystemExit(99)\n"
        )
    (tmp_path / "audited.json").write_text("{}")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\nif [ \"$1 $2\" = '-m pytest' ]; then "
        "echo CONTRACT_CONTROL_REFUSED; exit 23; fi\n"
        "echo MEASUREMENT_MUST_NOT_START; exit 99\n"
    )
    fake_python.chmod(0o755)
    root = Path(__file__).resolve().parents[3]
    command = [
        "bash",
        str(root / "scripts/prototypes/bank_width_probe" / launcher),
        str(tmp_path),
        str(tmp_path),
        "fixture-revision",
        str(tmp_path / "input"),
    ]
    if launcher == "job.sh":
        command.append(str(tmp_path / "figure.svg"))
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=dict(
            os.environ,
            NOVA_MEASUREMENT_PYTHON=str(fake_python),
            SLURM_JOB_ID="contract-check",
        ),
        capture_output=True,
        text=True,
    )
    assert "MEASUREMENT_MUST_NOT_START" not in result.stdout
    assert result.returncode == 23, result.stdout + result.stderr
    assert "CONTRACT_CONTROL_REFUSED" in result.stdout
    assert result.stdout.splitlines()[-1] == "EXIT=23"


@pytest.mark.parametrize("driver", ["width", "slice"])
@pytest.mark.parametrize("helper", ["archive_revision", "execute_arm"])
def test_drivers_delegate_to_shared_archive_and_executor(
    tmp_path, monkeypatch, driver, helper
):
    import io
    import tarfile
    from types import SimpleNamespace
    from scripts.prototypes.bank_h200_slice_profile import run as profile

    def archived(command):
        payload = io.BytesIO()
        names = ["nova/equilibrium/support_read.py", "nova/equilibrium/topology.py"]
        if command[5:] != ["nova"]:
            names.append("scripts/prototypes/bank_h200_slice_profile/measure.py")
        with tarfile.open(fileobj=payload, mode="w") as archive:
            for name in names:
                item = tarfile.TarInfo(name)
                item.size = 1
                archive.addfile(item, io.BytesIO(b"x"))
        return payload.getvalue()

    monkeypatch.setattr(profile.subprocess, "check_output", archived)

    def duplicated_execution(*args, **kwargs):
        raise AssertionError("duplicated subprocess execution reached")

    def shared_helper(*args, **kwargs):
        raise RuntimeError("shared owner reached")

    monkeypatch.setattr(profile, helper, shared_helper, raising=False)
    monkeypatch.setattr(profile.subprocess, "run", duplicated_execution)
    monkeypatch.setattr(
        run.hashlib,
        "sha256",
        lambda data: SimpleNamespace(
            hexdigest=lambda: (
                "52f8d4fb9e60dbdbe1530e8b64a171d3bb6470fb5cf73a1ddcf760895d09a0e9"
            )
        ),
    )
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    inputs = tmp_path / "inputs.npz"
    inputs.write_bytes(b"fixture")
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            str(tmp_path),
            str(tmp_path / "output"),
            "fixture-revision",
            "--inputs",
            str(inputs),
        ],
    )
    with pytest.raises(RuntimeError, match="shared owner reached"):
        (run if driver == "width" else profile).main()


@pytest.mark.parametrize("exit_status", [0, 77, 1])
def test_shared_executor_retains_receipt_arguments_and_status(tmp_path, exit_status):
    from scripts.prototypes.bank_h200_slice_profile import run as profile

    tree = tmp_path / "tree with spaces"
    tree.mkdir()
    instrument = tree / "measure.py"
    instrument.write_text(
        "import json, os, pathlib, sys\n"
        "out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])\n"
        "(out/'receipt.json').write_text(json.dumps(dict(argv=sys.argv[1:], "
        "cwd=os.getcwd(), allocator=os.environ.get('TF_GPU_ALLOCATOR'))))\n"
        "raise SystemExit(int(os.environ['CONTROL_EXIT']))\n"
    )
    env = dict(
        os.environ,
        CONTROL_EXIT=str(exit_status),
        TF_GPU_ALLOCATOR="cuda_malloc_async",
        JAX_PLATFORMS="cpu",
    )
    before = env.copy()
    destination = tmp_path / "arm"
    inputs = tmp_path / "inputs"
    arm = profile.execute_arm(
        tree,
        instrument,
        destination,
        inputs=inputs,
        width=16,
        revision="source-pin",
        instrument_revision="instrument-pin",
        environment=env,
        smoke=True,
        read_only=True,
    )
    assert env == before
    assert arm["exit_status"] == exit_status
    record = json.loads(Path(arm["receipt"]).read_text())
    assert record == dict(
        argv=[
            "--output",
            str(destination),
            "--inputs",
            str(inputs),
            "--width",
            "16",
            "--revision",
            "source-pin",
            "--instrument",
            "instrument-pin",
            "--smoke",
            "--read-only",
        ],
        cwd=str(tree),
        allocator="cuda_malloc_async",
    )
    lines = Path(arm["log"]).read_text().splitlines()
    assert lines[-1] == f"EXIT={exit_status}"
    assert "TF_GPU_ALLOCATOR=cuda_malloc_async" in lines[0]
    assert f"PYTHONPATH='{tree}'" in lines[0]


def test_shared_archive_reads_the_pin_instead_of_the_working_copy(tmp_path):
    from scripts.prototypes.bank_h200_slice_profile import run as profile

    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    (repository / "payload.txt").write_text("pinned content")
    subprocess.run(["git", "-C", str(repository), "add", "payload.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "test: store archive fixture",
            "-m",
            "Pin content for an archive identity check.",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    (repository / "payload.txt").write_text("uncommitted content")
    tree = tmp_path / "tree"
    tree.mkdir()
    profile.archive_revision(repository, revision, tree, ("payload.txt",))
    assert (tree / "payload.txt").read_text() == "pinned content"


@pytest.mark.parametrize("smoke", [False, True])
def test_slice_profile_preserves_arm_schedule_and_receipt_schema(
    tmp_path, monkeypatch, smoke
):
    from scripts.prototypes.bank_h200_slice_profile import run as profile

    calls, archives = [], []

    def archive(root, revision, tree, paths):
        archives.append((revision, tuple(paths)))

    def execute(tree, instrument, destination, **kwargs):
        destination.mkdir()
        calls.append((destination.name, kwargs))
        limited = kwargs["width"] == 256
        receipt = destination / "receipt.json"
        receipt.write_text(json.dumps(dict(memory_limited=limited, end_to_end_ms=10.0)))
        return dict(
            exit_status=77 if limited else 0,
            receipt=str(receipt),
            log=str(destination / "gate.log"),
        )

    monkeypatch.setattr(profile, "archive_revision", archive)
    monkeypatch.setattr(profile, "execute_arm", execute)
    inputs = tmp_path / "inputs.npz"
    inputs.write_bytes(b"fixture")
    output = tmp_path / "output"
    argv = [
        "profile",
        str(tmp_path),
        str(output),
        "source-pin",
        "--inputs",
        str(inputs),
    ]
    if smoke:
        argv.append("--smoke")
    monkeypatch.setattr("sys.argv", argv)
    profile.main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] and summary["instrument_revision"] == "source-pin"
    processes = [1] if smoke else [0, 1, 2]
    widths = [1] if smoke else [1, 16, 64, 256]
    expected = [f"main-slice-{w}-{p}" for w in widths for p in processes]
    expected += [
        f"{label}-read-{1 if smoke else 16}-{p}"
        for label in ("historical", "main")
        for p in processes
    ]
    assert [name for name, _ in calls] == expected
    assert archives == [
        ("source-pin", profile.PROFILE_TREE_PATHS),
        (profile.HISTORICAL, ("nova",)),
    ]
    for arm, (name, arguments) in zip(summary["arms"], calls, strict=True):
        assert set(arm) == {
            "label",
            "width",
            "process",
            "read_only",
            "exit_status",
            "receipt",
            "log",
        }
        assert arguments["smoke"] == smoke
        assert arguments["read_only"] == ("-read-" in name)
        assert arguments["revision"] == (
            profile.HISTORICAL if name.startswith("historical") else "source-pin"
        )
        assert arguments["environment"]["JAX_PLATFORMS"] == (
            "cpu" if smoke else "cuda,cpu"
        )
    if not smoke:
        assert summary["memory_limit_width"] == 256
        assert summary["agreement"] == {"1": 0.0, "16": 0.0, "64": 0.0}


def test_measurement_environment_preserves_parent_and_profile_defaults(
    tmp_path, monkeypatch
):
    from scripts.prototypes.bank_h200_slice_profile import run as profile

    monkeypatch.setenv("UV_NO_SYNC", "1")
    monkeypatch.setenv("XLA_FLAGS", "sentinel")
    monkeypatch.setenv("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    env = profile.measurement_environment(tmp_path, smoke=True)
    assert "UV_NO_SYNC" not in env and "XLA_FLAGS" not in env
    assert os.environ["UV_NO_SYNC"] == "1" and os.environ["XLA_FLAGS"] == "sentinel"
    assert env["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "compilation-cache")
