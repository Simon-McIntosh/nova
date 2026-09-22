"""The null-direction benchmark hands the response carrier to the state request.

The benchmark rebuilds each banked terminal through the persisted response
carrier, so the MAST state request it issues must name that carrier.  A request
that omits the identity resolves nothing: the solve entry point declares
``carrier_identity`` as a required keyword, so the omission fails before any
solve runs.  These cases drive the benchmark's own row loop against synthesised
collaborators and observe what the state request was handed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from benchmarks import jacobian_null_direction as driver


CARRIER_IDENTITY = "carrier-fixture-semantic-response-identity"
TARGET = (21985, 51)
BANKED_IDENTITY = "fixture-banked-terminal"


class _PureBranch:
    """Leaf standing in for the pure-branch handle the row loop slices out.

    ``jax.tree.map`` treats an unregistered object as a leaf, so the mapping
    applies the index to this object itself; returning ``self`` keeps the
    ``equilibrium`` handle reachable downstream.
    """

    def __init__(self) -> None:
        self.equilibrium = SimpleNamespace(fixed_point=jnp.zeros((2,)))

    def __getitem__(self, index: int) -> "_PureBranch":
        return self


class _Observed:
    """Observable branch portfolio returned in place of a production solve."""

    def __init__(self) -> None:
        self.portfolio = SimpleNamespace(branches=_PureBranch())


def _synthesised_measure(
    monkeypatch: pytest.MonkeyPatch, operands: Path, tmp_path: Path
):
    """Run the benchmark row loop with every collaborator synthesised."""
    requests: list[dict[str, object]] = []

    def _record_state_request(observed, state, target_current, *, carrier_identity):
        requests.append(
            {
                "observed": observed,
                "target_current": target_current,
                "carrier_identity": carrier_identity,
            }
        )
        return {"pure": SimpleNamespace(state=jnp.zeros((2,)))}

    row = {
        "identity": BANKED_IDENTITY,
        "arm": "pure",
        "verdict": {
            "smallest_to_largest_ratio": 0.25,
            "vertical_projection_fraction": 0.75,
            "near_null_is_vertical_mode": True,
            "near_null_direction": True,
        },
        "bank_validation": {"passes": True},
    }

    monkeypatch.setattr(driver, "STALLED_TARGETS", (TARGET,))
    monkeypatch.setattr(
        driver,
        "_load_terminals",
        lambda path: {
            (*TARGET, "pure"): SimpleNamespace(identity=BANKED_IDENTITY),
        },
    )
    monkeypatch.setattr(driver, "_measure_state", lambda *args, **kwargs: row)
    monkeypatch.setattr(driver, "_source_revision", lambda: "fixture-revision")
    monkeypatch.setattr(driver, "_draw_figure", lambda receipt, path: None)
    monkeypatch.setattr(driver, "_write_report", lambda receipt, path: None)
    monkeypatch.setattr(driver, "configure_dtypes", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        driver,
        "configure_persistent_compilation_cache",
        lambda root: SimpleNamespace(receipt=lambda: {"fixture": True}),
    )
    monkeypatch.setattr(
        driver,
        "default_persistent_compilation_cache_root",
        lambda: Path("/fixture/compilation-cache"),
    )
    monkeypatch.setattr(
        driver.settled,
        "_persisted_response_cache",
        lambda carrier, receipt: (
            SimpleNamespace(),
            {"carrier": {"semantic_response_identity": CARRIER_IDENTITY}},
        ),
    )
    monkeypatch.setattr(
        driver.settled,
        "select_slices_by_shot",
        lambda bank: [
            ({"shot": TARGET[0], "slice_index": TARGET[1]}, {"fixture": True})
        ],
    )
    monkeypatch.setattr(
        driver.settled,
        "_mast_case_from_selection",
        lambda store, selected_row, qualification: (
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        driver.settled,
        "_passive_inclusive_case",
        lambda case, context, response_cache: (
            {
                "state": jnp.zeros((4,)),
                "reference": {"plasma_current_a": 1.0e6},
            },
            SimpleNamespace(),
            {"section_kernel_evaluations_this_shot": 0},
        ),
    )
    monkeypatch.setattr(
        driver.settled.bank_producer,
        "_ObservedProfile",
        lambda profile: _Observed(),
    )
    monkeypatch.setattr(
        driver.settled.reachability, "_mast_states", _record_state_request
    )

    receipt = driver.measure(
        operands=operands,
        output=tmp_path / "jacobian-null-direction.json",
        figure=tmp_path / "jacobian-null-direction.png",
        report=tmp_path / "jacobian-null-direction.md",
        matrix_block_size=4,
    )
    return requests, receipt


@pytest.fixture(name="operands")
def fixture_operands(tmp_path: Path) -> Path:
    """A path standing in for the banked operands the benchmark hashes."""
    path = tmp_path / "operands.npz"
    path.write_bytes(b"fixture-operands")
    return path


def test_state_request_carries_the_case_carrier_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operands: Path
) -> None:
    """The row loop's state request names the carrier it rebuilt the case from."""
    requests, receipt = _synthesised_measure(monkeypatch, operands, tmp_path)

    assert len(requests) == 1
    assert requests[0]["carrier_identity"] == CARRIER_IDENTITY
    assert requests[0]["target_current"] == pytest.approx(1.0e6)
    assert (
        receipt["evidence_inputs"]["response_carrier"]["carrier"][
            "semantic_response_identity"
        ]
        == CARRIER_IDENTITY
    )
