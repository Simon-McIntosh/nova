"""Node-local persistent compilation cache selection for public solves."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.solve_request import default_forward_compilation_cache_root


_CACHE_WRITER = textwrap.dedent(
    """
    from pathlib import Path
    import sys
    import time

    import jax
    import jax.numpy as jnp

    from nova.equilibrium.solve_request import default_forward_compilation_cache_root
    from nova.jax.config import configure_persistent_compilation_cache

    ready_path = Path(sys.argv[1])
    start_path = Path(sys.argv[2])
    cache = configure_persistent_compilation_cache(
        default_forward_compilation_cache_root(),
        minimum_compile_seconds=0.0,
    )
    ready_path.touch()
    deadline = time.monotonic() + 30.0
    while not start_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("cache-writer start signal was not received")
        time.sleep(0.01)

    @jax.jit
    def compiled(value):
        return jnp.sin(value) + jnp.cos(value * 0.5)

    compiled(jnp.arange(4096, dtype=jnp.float64)).block_until_ready()
    print(cache.directory, flush=True)
    """
)


def _run_concurrent_cache_writers(
    cache_base: Path,
    control_directory: Path,
) -> tuple[subprocess.CompletedProcess[str], ...]:
    """Run two CPU compilers against one default cache namespace."""

    cache_base.mkdir(parents=True, exist_ok=True)
    control_directory.mkdir(parents=True, exist_ok=True)
    start_path = control_directory / "start"
    environment = os.environ.copy()
    environment.update(
        JAX_ENABLE_X64="true",
        JAX_PLATFORMS="cpu",
        TMPDIR=str(cache_base),
    )
    processes = tuple(
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CACHE_WRITER,
                str(control_directory / f"ready-{index}"),
                str(start_path),
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(2)
    )
    deadline = time.monotonic() + 60.0
    while not all(
        (control_directory / f"ready-{index}").exists() for index in range(2)
    ):
        exited = [
            process.returncode for process in processes if process.poll() is not None
        ]
        if exited:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    start_path.touch()

    results: list[subprocess.CompletedProcess[str]] = []
    for process in processes:
        try:
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            stderr = f"{stderr}\ncache writer exceeded 120 seconds"
        results.append(
            subprocess.CompletedProcess(
                process.args,
                process.returncode,
                stdout,
                stderr,
            )
        )
    return tuple(results)


def test_default_cache_root_is_scoped_to_user_and_host(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(socket, "gethostname", lambda: "cpu-node.example")

    assert default_forward_compilation_cache_root() == (
        tmp_path
        / "nova-forward-cache"
        / f"user-{os.getuid()}"
        / "host-cpu-node.example"
    )


def test_public_solve_selects_the_node_local_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    selected_directory = tmp_path / "root/nova/jax-compilation/runtime-cpu"
    selected_roots: list[Path] = []

    def configure(root: Path) -> SimpleNamespace:
        selected_roots.append(root)
        return SimpleNamespace(directory=selected_directory)

    monkeypatch.setattr(
        "nova.equilibrium.forward.jax",
        SimpleNamespace(config=SimpleNamespace(jax_compilation_cache_dir=None)),
    )
    monkeypatch.setattr(
        "nova.equilibrium.forward.default_forward_compilation_cache_root",
        lambda: tmp_path / "root",
    )
    monkeypatch.setattr(
        "nova.equilibrium.forward.configure_persistent_compilation_cache",
        configure,
    )

    assert ForwardProfile._configure_solve_compilation_cache(True) == str(
        selected_directory
    )
    assert selected_roots == [tmp_path / "root"]


def test_public_solve_preserves_an_explicit_launcher_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    shared_directory = tmp_path / "shared-runtime"
    monkeypatch.setattr(
        "nova.equilibrium.forward.jax",
        SimpleNamespace(
            config=SimpleNamespace(jax_compilation_cache_dir=str(shared_directory))
        ),
    )

    def unexpected_configuration(_root: Path) -> None:
        raise AssertionError("the public solve replaced the launcher cache")

    monkeypatch.setattr(
        "nova.equilibrium.forward.configure_persistent_compilation_cache",
        unexpected_configuration,
    )

    assert ForwardProfile._configure_solve_compilation_cache(True) == str(
        shared_directory.resolve()
    )


def test_two_processes_compile_into_one_node_local_default(tmp_path: Path) -> None:
    cache_base = tmp_path / "temporary-runtime"
    results = _run_concurrent_cache_writers(cache_base, tmp_path / "coordination")

    assert [result.returncode for result in results] == [0, 0], [
        result.stderr for result in results
    ]
    directories = {result.stdout.strip() for result in results}
    assert len(directories) == 1
    directory = Path(directories.pop())
    assert directory.is_relative_to(cache_base)
    assert list(directory.glob("*-cache"))
