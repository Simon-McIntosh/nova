"""Exact endpoints and strict spline levels for the comparison records."""

from __future__ import annotations

import pathlib

import pytest

from scripts.prototypes.efit_maximin_certificate.bounds import enclosing_target_cut
from scripts.prototypes.efit_maximin_certificate.certificate import (
    certify,
    load_record,
    spline_of,
    vessel_ring,
)
from scripts.prototypes.efit_maximin_certificate.cuts import saddle_cut
from scripts.prototypes.efit_maximin_certificate.run import ROWS


@pytest.mark.parametrize("spec", ROWS, ids=lambda spec: spec["case"])
def test_exact_targets_and_claimed_levels(spec):
    record = load_record(pathlib.Path(spec["source"]))
    spline = spline_of(record)
    target = spec["target"]
    value = float(record.polarity * spline.ev(*target))
    if spec is ROWS[0]:
        assert target == ROWS[0]["target"]
        assert abs(value - spec["nova_level"]) < 1e-12
        cut = saddle_cut(
            spline, vessel_ring(record), spec["axis"], target, spec["nova_level"]
        )
        assert cut["separates"]
        assert cut["bound"]["maximum_upper"] < spec["nova_level"]
        assert abs(cut["saddle_value"] - spec["efit_level"]) < 1e-12
        assert cut["bound"]["maximum_upper"] > spec["efit_level"]
    else:
        assert target == (0.85, 0.2)
        assert spec["nova_level"] < value < spec["efit_level"]
        assert 1e-11 < spec["efit_level"] - value < 2e-11
        cut = enclosing_target_cut(
            spline, record.polarity, target, spec["axis"], spec["efit_level"]
        )
        assert cut is not None
        assert cut["bound"]["maximum_upper"] < spec["efit_level"]
        assert cut["radius"] > 0


@pytest.mark.parametrize("spec", ROWS, ids=lambda spec: spec["case"])
def test_sampled_path_checks_exact_endpoints_and_continuous_minimum(spec):
    row = certify(**spec, path=pathlib.Path(spec["source"]), n=400)
    assert row.target == list(spec["target"])
    assert row.attaining_path[0][:2] == pytest.approx(spec["axis"], abs=1e-14)
    assert row.attaining_path[-1][:2] == pytest.approx(spec["target"], abs=1e-14)
    assert row.path_bound["lower"] <= row.path_bound["upper"]
    assert row.path_bound["midpoint_residual"] < row.path_bound["arithmetic_pad"]
    if spec is ROWS[0]:
        assert row.path_bound["upper"] < spec["nova_level"]
    else:
        assert row.path_bound["upper"] < spec["efit_level"]
