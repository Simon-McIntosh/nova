"""Locate the sampled point-jet discrepancy without compiling a full map."""

# ruff: noqa: E402
import json

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

from nova.biot import gradedresidual
from nova.equilibrium.topology import BiotMomentCoupling
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT
from scripts.prototypes.read_route.jets import _source_jets


def main():
    assert jax.config.jax_enable_x64
    geometry = dict(np.load(ROOT / "h200/geometry.npz"))
    polygons = tuple(
        p[:n]
        for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
    )
    coupling = BiotMomentCoupling.from_polygons(polygons)
    data = dict(np.load(ROOT / "diagnosis/accuracy-220.npz"))
    point = jnp.array(data["points"][0])
    moments = jnp.array(data["moments"])
    source = jax.jit(_source_jets)
    jets = np.asarray(source(point, coupling))
    total = np.einsum("cmq,cm->q", jets, moments)
    result = dict(node=0, fresh_bank=total.tolist())
    (ROOT / "localize.json").write_text(json.dumps(result, indent=2))
    print("FRESH_BANK", result, flush=True)
    saved = gradedresidual._fixed_nodes
    try:
        gradedresidual._fixed_nodes = lambda value, xp: value
        moving = jax.jit(lambda p, c: _source_jets(p, c))
        altered = np.asarray(moving(point, coupling))
    finally:
        gradedresidual._fixed_nodes = saved
    result["moving_nodes"] = np.einsum("cmq,cm->q", altered, moments).tolist()
    result["source_difference"] = np.einsum(
        "cmq,cm->cq", jets - altered, moments
    ).tolist()
    (ROOT / "localize.json").write_text(json.dumps(result, indent=2))
    print("LOCALIZE_COMPLETE", result["moving_nodes"], flush=True)


if __name__ == "__main__":
    main()
