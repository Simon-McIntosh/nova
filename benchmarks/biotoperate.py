"""Benchmark Biot operators and true fresh-process Zarr reloads."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import timeit

import numpy as np

from nova.frame.coilset import CoilSet

# Fixed absolute cache directory: asv re-imports this module in a separate
# process for setup_cache and for each benchmark, so the path must resolve
# identically in every process (a randomised mkdtemp would differ per process).
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "nova_asv_biotoperate")
os.makedirs(_CACHE_DIR, exist_ok=True)


class PlasmaGrid:
    """Benchmark biotoperate methods - plasmagrid base class."""

    timer = timeit.default_timer
    dirname = _CACHE_DIR

    @property
    def filename(self):
        """Return coilset filename."""
        return "plasmagrid_coilset"

    @property
    def filepath(self):
        """Return coilset filepath."""
        return CoilSet(filename=self.filename, dirname=self.dirname).filepath

    def setup_cache(self):
        """Build reference coilset."""
        self.remove()
        coilset = CoilSet(dplasma=-500, filename=self.filename, dirname=self.dirname)
        coilset.firstwall.insert({"ellip": [4.2, -0.4, 1.25, 4.2]}, turn="hex")
        coilset.plasmagrid.solve()
        coilset.plasmagrid.svd_rank = 75
        coilset.store()

    def remove(self):
        """Remove the exact Zarr cache directory and no sibling entries."""
        if self.filepath.is_dir():
            shutil.rmtree(self.filepath)

    def setup(self):
        """Load coilset from file."""
        self.coilset = CoilSet(filename=self.filename, dirname=self.dirname).load()


class PlasmaTurns(PlasmaGrid):
    """Benchmark biotoperate methods."""

    number = 5000
    params = [10, 75, 200, 500, -1]
    param_names = ["svd_rank"]

    def setup(self, svd_rank):
        """Load coilset from file and set svd rank."""
        self.coilset = CoilSet(filename=self.filename, dirname=self.dirname).load()
        self.coilset.plasmagrid.svd_rank = svd_rank

    def time_update_turns(self, svd_rank):
        """Time generation of plasma grid."""
        self.coilset.plasmagrid.update_turns("Psi", svd_rank != -1)


class PlasmaEvaluate(PlasmaGrid):
    """Time evaluation of plasma operators."""

    number = 5000

    def time_flux_function_ev_only(self):
        """Time forced evaluation of flux function."""
        self.coilset.plasmagrid.operator["Psi"].evaluate()

    def time_flux_function(self):
        """Time computation of radial field."""
        return self.coilset.plasmagrid.psi

    def time_radial_field(self):
        """Time computation of radial field."""
        return self.coilset.plasmagrid.br

    def time_field_magnitude(self):
        """Time computation of poloidal field magnitude."""
        return self.coilset.plasmagrid.bp


class PlasmaOperate(PlasmaGrid):
    """Time plasma grid operations."""

    def time_solve(self):
        """Time plasma grid biot solution."""
        self.coilset.plasmagrid.solve()

    def time_fresh_process_reload(self):
        """Time a distinct process loading and touching the stored operator."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reader.json"
            _run_child("read", self.filename, self.dirname, output)


PlasmaOperate.time_solve.number = 10
PlasmaOperate.time_fresh_process_reload.number = 1


def _operator_record(coilset: CoilSet) -> dict:
    """Return semantic evidence that a loaded operator was materialized."""
    matrix = np.asarray(coilset.plasmagrid.data["Psi"].data)
    return {
        "pid": os.getpid(),
        "frame_count": len(coilset.frame),
        "subframe_count": len(coilset.subframe),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "checksum": float(np.sum(matrix, dtype=np.float64)),
    }


def _write_reload_fixture(filename: str, dirname: str, output: Path, dplasma: int):
    """Prepare one stored operator in a writer process, then report its meaning."""
    coilset = CoilSet(
        dplasma=dplasma,
        filename=filename,
        dirname=dirname,
    )
    coilset.firstwall.insert({"circle": [3.0, 0.0, 0.5]}, turn="hex")
    coilset.plasmagrid.solve()
    record = _operator_record(coilset)
    coilset.store()
    output.write_text(json.dumps(record))


def _read_reload_fixture(filename: str, dirname: str, output: Path):
    """Time construction, Zarr loading, and a concrete operator checksum."""
    start = time.perf_counter()
    coilset = CoilSet(filename=filename, dirname=dirname).load()
    record = _operator_record(coilset)
    record["load_seconds"] = time.perf_counter() - start
    output.write_text(json.dumps(record))


def _run_child(
    operation: str,
    filename: str,
    dirname: str,
    output: Path,
    dplasma: int | None = None,
) -> tuple[dict, float]:
    """Run one writer or reader process and return its JSON plus process time."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        operation,
        filename,
        dirname,
        str(output),
    ]
    if dplasma is not None:
        command.append(str(dplasma))
    start = time.perf_counter()
    subprocess.run(command, check=True, capture_output=True, text=True)
    process_seconds = time.perf_counter() - start
    return json.loads(output.read_text()), process_seconds


def measure_fresh_process_reload(
    dirname: str,
    *,
    filename: str = "fresh_process_coilset",
    dplasma: int = -4,
    readers: int = 2,
) -> dict:
    """Prepare once, then time distinct reader processes against one Zarr store.

    The internal interval begins immediately before ``CoilSet(...).load()`` and
    ends after the loaded ``Psi`` matrix has been materialized and checksummed.
    Parent-side process time is reported separately. Operating-system page cache
    state is deliberately uncontrolled, so this is a process-cold Python/Zarr
    reload measurement rather than a claim about cold storage media.
    """
    cache = CoilSet(filename=filename, dirname=dirname).filepath
    if cache.is_dir():
        shutil.rmtree(cache)
    try:
        with tempfile.TemporaryDirectory() as output_directory:
            output_directory = Path(output_directory)
            writer, writer_seconds = _run_child(
                "write",
                filename,
                dirname,
                output_directory / "writer.json",
                dplasma,
            )
            writer["process_seconds"] = writer_seconds
            samples = []
            for index in range(readers):
                sample, process_seconds = _run_child(
                    "read",
                    filename,
                    dirname,
                    output_directory / f"reader_{index}.json",
                )
                sample["process_seconds"] = process_seconds
                samples.append(sample)
        return {"writer": writer, "readers": samples}
    finally:
        if cache.is_dir():
            shutil.rmtree(cache)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in {"write", "read"}:
    operation, filename, dirname, output = sys.argv[1:5]
    if operation == "write":
        _write_reload_fixture(filename, dirname, Path(output), int(sys.argv[5]))
    else:
        _read_reload_fixture(filename, dirname, Path(output))
elif __name__ == "__main__":
    biot = PlasmaTurns()
    biot.setup_cache()
    biot.setup(75)
    biot.time_update_turns(75)
    biot.remove()
