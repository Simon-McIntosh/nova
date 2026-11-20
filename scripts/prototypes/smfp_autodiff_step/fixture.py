"""Load the diverted 550-cell certificate state, its bank and its oracle.

The published operator bank is reused when the store holds it and rebuilt on
the device when it does not, so every arm of the section gate sees one bank
identity.  Nothing here differentiates anything: this module only assembles
the fixture and the analytic reference.
"""

# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.database import content_store
from nova.equilibrium import current, fixed_points, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy


CELLS = 550
NORMAL_FORM_RADIUS = 0.054806712567833996


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def analytic_fixture():
    import tests.equilibrium.test_topology_read as fixture
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures.measure import analytic_profile

    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs("diverted")
    geometry = fixture._realised_hex_geometry(wall, CELLS)
    _, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    profile = analytic_profile(source)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(normal_form_radius=NORMAL_FORM_RADIUS)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    return (
        fixture,
        oracle,
        analytic,
        wall,
        geometry,
        profile,
        convention,
        policy,
        quadrature,
        jnp.asarray([axis, saddle]),
    )


def book_analytic(analytic, geometry, profile, convention, policy, quadrature):
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    saturate = jax.jit(topology.read)
    reading = saturate(analytic, geometry, convention, policy)
    return current.book(psi, reading, geometry, profile, quadrature)


def load_bank(store: Path, geometry, graph):
    """Return the bank and the moments the certificate books at its analytic ψ."""
    mesh_of = fixed_points.build_mesh
    receipt_path = store / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
    else:
        receipt = None
    if receipt is not None:
        object_root = content_store.verified_object_root(store, create=False)
        destination = content_store.verified_destination(
            object_root, content_store.digest_hex(receipt["digest"])
        )
        if destination is not None:
            with np.load(destination / "bank.npz", allow_pickle=False) as archive:
                nodes = archive["nodes"]
                sample_index = archive["sample_index"]
            mesh = mesh_of(nodes, sample_index, geometry)
            bank = support_read.read_bank(store, receipt["digest"], geometry, mesh)
            emit("BANK_LOAD", digest=receipt["digest"], built=False)
            return bank, mesh

    from scripts.prototypes.read_route.jets import build_jet_matrix

    fixture, oracle, analytic, _, _, _, _, _, _, _ = graph
    reference = fixture._kernel_backed_field("diverted", oracle, analytic, geometry)
    started = perf_counter()
    matrix, exterior, nodes, sample_index, build_seconds = build_jet_matrix(
        reference.coupling, geometry, analytic, reference.moments, persist=False
    )
    bank = support_read.SupportOperands.from_rows(
        geometry, matrix, exterior, nodes, sample_index, bucket=len(nodes)
    )
    jax.block_until_ready(bank.rows)
    mesh = mesh_of(nodes, sample_index, geometry)
    digest = support_read.write_bank(bank, store)
    store.mkdir(parents=True, exist_ok=True)
    (store / "receipt.json").write_text(
        json.dumps(
            {
                "kind": "nova-support-operator-bank",
                "digest": digest,
                "bucket": int(bank.bucket),
                "active_nodes": int(bank.active_nodes),
                "cells": CELLS,
                "build_seconds": build_seconds,
            },
            sort_keys=True,
        )
    )
    emit(
        "BANK_BUILD",
        digest=digest,
        seconds=build_seconds,
        elapsed=perf_counter() - started,
        nodes=len(nodes),
        bucket=int(bank.bucket),
    )
    return bank, mesh


class Certificate:
    """The diverted 550-cell state: geometry, bank, oracle and seed read."""

    def __init__(self, store, kind="diverted", seed=None):
        if kind != "diverted":
            raise ValueError("the fast-route gate measures the diverted certificate")
        self.graph = analytic_fixture()
        (
            self.fixture,
            self.oracle,
            self.analytic,
            self.wall,
            self.geometry,
            self.profile,
            self.convention,
            self.policy,
            self.quadrature,
            self.roots,
        ) = self.graph
        self.bank, self.mesh = load_bank(store, self.geometry, self.graph)
        booked = book_analytic(
            self.analytic,
            self.geometry,
            self.profile,
            self.convention,
            self.policy,
            self.quadrature,
        )
        self.seed_moments = tuple(booked[index] for index in range(3))
        self.cache = fixed_points.RowCache(
            self.bank.mesh,
            self.bank.nodes,
            jnp.arange(len(self.bank.nodes)),
            self.bank.rows,
            self.bank.exterior,
        )
        pitch = jnp.median(self.geometry.pitch)
        self.seed = self.roots + 0.03 * pitch
        if seed is not None:
            self.seed = seed

    def read(self, moments):
        return jax.jit(
            lambda b, c, m, s, r: support_read.read(
                b, c, m, s, r, self.convention, self.policy
            )
        )(self.bank, self.cache, moments, self.seed, self.roots)

    def step_residual(self, moments, support):
        return jax.jit(
            lambda b, m, s: support_read.step_residual(
                b, m, m, s, self.profile, self.quadrature
            )
        )(self.bank, moments, support)

    def gradient(self, radius, flux):
        return jnp.zeros_like(radius)
