"""Assemble the diverted certificate fixture and publish its bank if absent.

This is the first process of a device job: the bank build is the expensive
step, so it runs once and lands in the store the later arms read.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from pathlib import Path

import jax

from scripts.prototypes.smfp_autodiff_step.fixture import Certificate, emit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    certificate = Certificate(args.store)
    jax.block_until_ready(certificate.bank.rows)
    support = certificate.read(certificate.seed_moments)
    jax.block_until_ready(support)
    emit(
        "BANK_READY",
        bucket=int(certificate.bank.bucket),
        active_nodes=int(certificate.bank.active_nodes),
        nodes=int(len(certificate.bank.nodes)),
        cells=int(certificate.seed_moments[0].size),
        support_valid=bool(support.valid),
        reading_valid=bool(support.reading.valid),
        qualified=bool(support.reading.qualified),
        cut_cells=int(
            ((support.reading.membership > 0) & (support.reading.membership < 1)).sum()
        ),
    )


if __name__ == "__main__":
    main()
