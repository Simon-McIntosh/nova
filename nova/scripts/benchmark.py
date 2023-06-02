"""Manage asv benchmark script parameters."""
from contextlib import contextmanager
import subprocess

import click


@contextmanager
def pyperf():
    """Run asv benchmarks with pyperf system tune and reset."""
    subprocess.run(["sudo", "pyperf", "system", "tune"])
    yield
    subprocess.run(["sudo", "pyperf", "system", "reset"])


@click.command()
def benchmark():
    """Run asv benchmarks with pyperf tune."""
    with pyperf():
        subprocess.run(
            [
                "asv",
                "run",
                "2022.0.0..main",
                "--cpu-affinity",
                "0",
                "--record-samples",
                "--append-samples",
                "-a",
                "rounds=4",
                "-a",
                "repeat=(2, 10, 20.0)",
                "-a",
                "warmup_time=1.0",
            ]
        )


if __name__ == "__main__":
    benchmark()
