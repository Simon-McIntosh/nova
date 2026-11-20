"""Re-reference a published bank onto the certificate's booking without rebuilding.

A bank's exterior is ``analytic jets - rows contracted with the reference
moments``.  The rows themselves do not depend on the reference, so moving the
reference is one dense contraction on the stored operands:

    exterior' = exterior + rows · (m_old - m_new)

which is exactly the exterior a fresh build around ``m_new`` would produce.
The stored bank must have been built around ``m_old``; both bookings are
recomputed from the fixture so the shift carries the difference honestly.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp

from dataclasses import replace

from nova.equilibrium import support_read
from scripts.prototypes.smfp_autodiff_step.fixture import (
    Certificate,
    book_analytic,
    emit,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-store", type=Path, required=True)
    parser.add_argument("--new-store", type=Path, required=True)
    args = parser.parse_args()
    certificate = Certificate(args.old_store)
    graph = certificate.graph
    _, oracle, analytic, _, geometry, profile, convention, policy, quadrature, _ = graph
    fixture = certificate.fixture
    old = fixture._kernel_backed_field("diverted", oracle, analytic, geometry)
    old_reference = jnp.stack(tuple(old.moments), axis=-1)
    booked = book_analytic(analytic, geometry, profile, convention, policy, quadrature)
    new_reference = jnp.stack(tuple(booked[index] for index in range(3)), axis=-1)
    bank = certificate.bank
    difference = old_reference - new_reference
    emit(
        "SHIFT",
        max_difference=float(jnp.max(jnp.abs(difference))),
        max_rows=float(jnp.max(jnp.abs(bank.rows))),
    )
    shift = jnp.einsum("ncmq,cm->nq", bank.rows, difference)
    exterior = bank.exterior + shift
    jax.block_until_ready(exterior)
    emit(
        "EXTERIOR",
        max_shift=float(jnp.max(jnp.abs(shift))),
        max_exterior=float(jnp.max(jnp.abs(bank.exterior))),
    )
    moved = replace(bank, exterior=exterior)
    digest = support_read.write_bank(moved, args.new_store)
    args.new_store.mkdir(parents=True, exist_ok=True)
    (args.new_store / "receipt.json").write_text(
        json.dumps(
            {
                "kind": "nova-support-operator-bank",
                "digest": digest,
                "bucket": int(moved.bucket),
                "active_nodes": int(moved.active_nodes),
                "cells": int(certificate.seed_moments[0].size),
                "shifted_from": str(args.old_store),
            },
            sort_keys=True,
        )
    )
    emit("BANK_WRITTEN", digest=digest, store=str(args.new_store))


if __name__ == "__main__":
    main()
