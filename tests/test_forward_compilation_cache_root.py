"""Shared, host-keyed forward compilation cache root selection.

The default root must persist across SLURM allocations on the shared home
filesystem (never the node-local temporary filesystem), keep builds of one
node out of every other node's cache, and let concurrent writers on one node
share a subtree without aborting.  These tests pin that contract and the
explicit base-directory override launches use to name their own location.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import time

from nova.equilibrium.solve_request import default_forward_compilation_cache_root


_CACHE_WRITER = textwrap.dedent(
    """
    from pathlib import Path
    import sys

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
    import time
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

    control_directory.mkdir(parents=True, exist_ok=True)
    start_path = control_directory / "start"
    environment = os.environ.copy()
    environment.update(
        JAX_ENABLE_X64="true",
        JAX_PLATFORMS="cpu",
        NOVA_FORWARD_COMPILATION_CACHE_ROOT=str(cache_base),
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


def test_default_cache_root_persists_on_the_shared_home_filesystem(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(socket, "gethostname", lambda: "cpu-node.example")

    assert default_forward_compilation_cache_root() == (
        tmp_path
        / ".local"
        / "share"
        / "nova"
        / "forward-compilation-cache"
        / f"user-{os.getuid()}"
        / "host-cpu-node.example"
    )


def test_default_cache_root_ignores_tmpdir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(socket, "gethostname", lambda: "cpu-node.example")
    monkeypatch.setenv("TMPDIR", "/node-local-tmp")

    root = default_forward_compilation_cache_root()

    assert not root.is_relative_to(Path("/node-local-tmp"))
    assert root == (
        tmp_path
        / ".local"
        / "share"
        / "nova"
        / "forward-compilation-cache"
        / f"user-{os.getuid()}"
        / "host-cpu-node.example"
    )


def test_explicit_base_override_selects_its_own_shard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "cpu-node.example")
    launch_owned_base = tmp_path / "launch-scratch"
    monkeypatch.setenv("NOVA_FORWARD_COMPILATION_CACHE_ROOT", str(launch_owned_base))

    assert default_forward_compilation_cache_root() == (
        launch_owned_base / f"user-{os.getuid()}" / "host-cpu-node.example"
    )


def test_two_processes_compile_into_one_shared_default(tmp_path: Path) -> None:
    cache_base = tmp_path / "shared-runtime"
    results = _run_concurrent_cache_writers(cache_base, tmp_path / "coordination")

    assert [result.returncode for result in results] == [0, 0], [
        result.stderr for result in results
    ]
    directories = {result.stdout.strip() for result in results}
    assert len(directories) == 1
    directory = Path(directories.pop())
    assert directory.is_relative_to(cache_base)
    assert list(directory.glob("*-cache"))
