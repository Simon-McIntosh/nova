"""Manage asv benchmark script parameters."""
from contextlib import contextmanager
import subprocess

import click


@contextmanager
def pyperf():
    """Run asv benchmarks with pyperf system tune and reset."""
    subprocess.run(['sudo', 'pyperf', 'system', 'tune'])
    yield
    subprocess.run(['sudo', 'pyperf', 'system', 'reset'])


@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', default='Your name',
              help='The person to greet.')
def benchmark(count, name):
    """Run asv benchmarks with pyperf tune."""
    with pyperf():
        subprocess.run(['asv', 'run', 'EXISTING', '--cpu-affinity', '0-3'])


if __name__ == '__main__':
    benchmark()
