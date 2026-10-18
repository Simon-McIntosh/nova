"""Compare the imported field-swap spline against an independent scipy rebuild."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import nova  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField  # noqa: E402
from scripts.prototypes.efit_field_swap.trace_record import read_trace_record  # noqa: E402
from scripts.prototypes.efit_maximin_certificate.certificate import (  # noqa: E402
    load_record,
    spline_of,
)
from scripts.prototypes.efit_maximin_certificate.run import ROWS  # noqa: E402


def main(output: pathlib.Path):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    print(f"nova.__file__={nova.__file__}", flush=True)
    rows = []
    for spec in ROWS:
        path = pathlib.Path(spec["source"])
        record = load_record(path)
        spline = spline_of(record)
        field = TraceSplineField.from_record(read_trace_record(path))
        rs = np.linspace(record.rgrid[0], record.rgrid[-1], 31)
        zs = np.linspace(record.zgrid[0], record.zgrid[-1], 31)
        rr, zz = np.meshgrid(rs, zs, indexing="ij")
        points = np.column_stack((rr.ravel(), zz.ravel()))
        points = np.vstack((points, spec["axis"], spec["target"]))
        nova_values = np.asarray(jax.vmap(field.value)(jnp.asarray(points)))
        scipy_values = spline.ev(points[:, 0], points[:, 1])
        discrepancy = np.abs(nova_values - scipy_values)
        row = {
            "case": spec["case"],
            "sample_count": len(points),
            "max_abs_field_difference": float(discrepancy.max()),
            "exact_axis_field_difference": float(discrepancy[-2]),
            "exact_target_field_difference": float(discrepancy[-1]),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
