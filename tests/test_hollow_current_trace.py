"""Known closures and a measured bank pin the signed source decomposition."""

import numpy as np
import pytest

from scripts.prototypes.diiid_hollow_current.trace import profile_shape, split_current


class SyntheticClosure:
    """A pressure-only source with a declared central gradient shape."""

    def __init__(self, inverted=False):
        self.inverted = inverted

    def pressure_gradient(self, radius, coordinate):
        return -(0.1 + coordinate if self.inverted else 1 - coordinate)

    def ff_prime(self, coordinate):
        return np.zeros_like(coordinate)

    def current_density(self, radius, coordinate):
        return -2 * np.pi * radius * self.pressure_gradient(radius, coordinate)


@pytest.mark.parametrize(("inverted", "verdict"), [(False, "peaked"), (True, "hollow")])
def test_known_profile_shape(inverted, verdict):
    assert profile_shape(1.7, SyntheticClosure(inverted))["verdict"] == verdict


def test_term_split_and_common_amplitude_preserve_shape():
    coordinate = np.linspace(0, 1, 21)
    raw = split_current(1.7, coordinate, SyntheticClosure())
    scaled = split_current(1.7, coordinate, SyntheticClosure(), amplitude=3.5)
    np.testing.assert_allclose(
        raw["pressure_term"] + raw["field_term"], raw["unscaled_density"]
    )
    np.testing.assert_allclose(scaled["density"], 3.5 * raw["unscaled_density"])
    assert scaled["density"][0] / scaled["density"][7] == pytest.approx(
        raw["density"][0] / raw["density"][7]
    )


def test_support_clipping_is_explicit():
    read = split_current(
        1.7, np.array([0.0, 0.2]), SyntheticClosure(), support=[False, True]
    )
    assert read["unscaled_density"][0] > 0
    assert read["density"][0] == 0
    assert read["density"][1] > 0


def test_pressure_and_field_contributions_are_both_observed():
    from nova.equilibrium.source import DomainProfile
    from scipy.constants import mu_0

    closure = DomainProfile(
        p_prime=lambda flux: -2e4 * (1 - flux),
        ff_prime=lambda flux: -0.4 * (1 - flux),
    )
    radius, coordinate = np.array([1.4, 1.8]), np.array([0.0, 0.3])
    terms = split_current(radius, coordinate, closure)
    np.testing.assert_allclose(
        terms["pressure_term"], 2 * np.pi * radius * 2e4 * (1 - coordinate)
    )
    np.testing.assert_allclose(
        terms["field_term"], 2 * np.pi * 0.4 * (1 - coordinate) / (mu_0 * radius)
    )
    assert np.all(terms["pressure_term"] > 0)
    assert np.all(terms["field_term"] > terms["pressure_term"])


def test_source_recovery_refuses_a_forward_solve(monkeypatch):
    from nova.equilibrium.forward import ForwardProfile
    from scripts.prototypes.diiid_hollow_current.recover import SolveGuard

    guard = SolveGuard()
    monkeypatch.setattr(ForwardProfile, "solve", guard.refuse)
    with pytest.raises(AssertionError, match="forward solve forbidden"):
        ForwardProfile.solve(None)
    assert guard.calls == 1


def test_committed_closure_and_saved_terminal_pin_both_density_ratios():
    import hashlib
    import json
    from pathlib import Path
    from nova.equilibrium.map_extraction import apply_delta_star
    from nova.equilibrium.solve_request import SampledFluxFunction
    from nova.equilibrium.source import DomainProfile

    root = Path(__file__).resolve().parents[1]
    bank = json.loads(
        (root / "scripts/prototypes/diiid_hollow_current/closure-bank.json").read_text()
    )
    record = next(row for row in bank if row["case"] == "frame-137-diverted")
    path = root / record["terminal_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["terminal_sha256"]
    data = dict(np.load(path))
    closure = DomainProfile(
        **{
            name: SampledFluxFunction(value["coordinate"], value["values"])
            for name, value in record["profiles"].items()
        }
    )
    rr, zz = np.meshgrid(data["radius"], data["height"], indexing="ij")
    coordinate = (data["flux"] - record["axis_flux_wb"]) / (
        record["boundary_flux_wb"] - record["axis_flux_wb"]
    )
    source = split_current(rr, coordinate, closure, record["amplitude"])["density"]
    radial = (
        np.abs(rr - record["core_radius_m"]) <= 1.6 * np.diff(data["radius"]).mean()
    )
    central = radial & (np.abs(zz - record["centroid_z_m"]) <= 0.1)
    shoulder = (
        radial
        & (np.abs(zz - record["centroid_z_m"]) >= 0.15)
        & (np.abs(zz - record["centroid_z_m"]) <= 0.35)
    )
    ratio = np.median(source[central]) / np.median(source[shoulder])
    assert ratio == pytest.approx(record["source_central_to_shoulder"], abs=1e-10)
    native = apply_delta_star(data["radius"], data["height"], data["flux"])
    native_ratio = np.median(native.toroidal_current_density[central]) / np.median(
        native.toroidal_current_density[shoulder]
    )
    assert native_ratio == pytest.approx(0.25318762743400713, abs=1e-10)
    assert profile_shape(1.7231301069259644, closure)["ratio"] == pytest.approx(
        0, abs=1e-14
    )
    assert (
        split_current(1.7231301069259644, np.array([0.05]), closure)["density"][0] > 1e6
    )
