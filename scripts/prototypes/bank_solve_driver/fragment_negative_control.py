"""Prove that a same-cell fragment switch must reject the frozen step."""

# ruff: noqa: E402
import importlib.util
import os
from pathlib import Path
import sys

import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()

import nova.equilibrium as equilibrium
from nova.equilibrium import support_read


SOURCE = """        drift = lock & selection_changed(
            next_read.reading, cut_set, normal_set, locked_fragments
        )"""
MUTATION = """        next_cut = (next_read.reading.membership > 0) & (
            next_read.reading.membership < 1 - 1e-14
        )
        drift = lock & (
            jnp.any(next_cut != cut_set, axis=-1)
            | jnp.any(next_read.reading.normal_form_cells != normal_set, axis=-1)
        )"""


def main():
    source = Path(support_read.__file__).read_text()
    assert source.count(SOURCE) == 1
    scratch = Path(os.environ["TMPDIR"])
    path = scratch / "support_read.py"
    path.write_text(source.replace(SOURCE, MUTATION))
    name = "nova.equilibrium.support_read"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    equilibrium.support_read = module
    print(f"MUTANT_MODULE={module.__file__} CWD={Path.cwd().resolve()}", flush=True)
    print("MUTATION=fragment_selected removed from next-read drift", flush=True)
    return pytest.main(
        [
            "-p",
            "no:cacheprovider",
            "tests/test_bank_solve_driver.py::test_fragment_switch_forces_fresh_extra_step",
            "-q",
            "-s",
            f"--basetemp={scratch / 'pytest'}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
