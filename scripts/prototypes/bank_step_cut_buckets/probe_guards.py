"""Exercise the overflow test against invalid output and a wider live batch."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current


class PopulationObserved(Exception):
    """The test selected more normal cells than the live capacity."""


def load(path: Path):
    spec = importlib.util.spec_from_file_location("overflow_probe_target", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(path: Path, check: str) -> None:
    target = load(path)
    if check == "capacity":
        current._NORMAL_CUT_BATCH_WIDTH = 16

        def count_selected(psi, direction, read, *args):
            count = int(np.count_nonzero(np.asarray(read.normal_form_cells)))
            print(f"selected={count} capacity=16", flush=True)
            if count <= current._NORMAL_CUT_BATCH_WIDTH:
                raise AssertionError("test did not select an overflow population")
            raise PopulationObserved

        current.book_tangent = count_selected
        try:
            target.test_normal_cut_count_over_batch_capacity()
        except PopulationObserved:
            print("capacity: PASS\nEXIT=0", flush=True)
            return
        except AssertionError as error:
            print(f"capacity: FAIL ({error})\nEXIT=1", flush=True)
            raise SystemExit(1) from error
        print("capacity: FAIL\nEXIT=1", flush=True)
        raise SystemExit(1)

    def invalid_scalar(psi, *args):
        return tuple(jnp.full((psi.shape[0],), jnp.nan) for _ in range(3))

    def invalid_channels(psi, *args):
        return jnp.full((psi.shape[0], 3, 1), jnp.nan)

    current.book_tangent = invalid_scalar
    current.book_tangent_channels = invalid_channels
    try:
        target.test_normal_cut_count_over_batch_capacity()
    except AssertionError:
        print("finite: PASS (invalid response refused)\nEXIT=0", flush=True)
        return
    print("finite: FAIL (invalid response accepted)\nEXIT=1", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", required=True, choices=("capacity", "finite"))
    main(**vars(parser.parse_args()))
