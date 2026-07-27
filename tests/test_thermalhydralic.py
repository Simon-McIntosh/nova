"""Import guards for the thermal-hydraulic cluster.

Two properties are checked here and nowhere else: every module in the subtree
imports, and none of them pulls a plotting stack into the import graph. The
second guard runs in a subprocess with matplotlib and seaborn blocked at the
finder, because once this process has imported them an in-process check can
no longer tell a module-scope import from a deferred one.
"""

import importlib
import pkgutil
import subprocess
import sys

import numpy as np
import pytest

import nova.thermalhydralic as thermalhydralic
from nova.thermalhydralic.frequencyresponse import FrequencyResponse
from nova.thermalhydralic.lumpedparameter import LumpedCapacitance

# Distributions the cluster reaches only behind an extra. A missing one is an
# environment gate; a missing nova module is a stale import and must fail.
OPTIONAL_DISTRIBUTION = frozenset(
    {
        "bs4",
        "CoolProp",
        "fitz",
        "ftputil",
        "matplotlib",
        "mechanize",
        "nlopt",
        "openpyxl",
        "regex",
        "seaborn",
        "tables",
        "xlrd",
    }
)

DEFERRED_PLOTTING = ("matplotlib", "seaborn", "pylab")

LEAN_CORE_PROBE = """
import pkgutil
import sys

BLOCKED = frozenset({blocked!r})


class BlockRoots:
    \"\"\"Raise on any import below a blocked root package.\"\"\"

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"{{name}} blocked by the lean-core probe")
        return None


sys.meta_path.insert(0, BlockRoots())

import nova.thermalhydralic as thermalhydralic

for module in pkgutil.walk_packages(
    thermalhydralic.__path__, prefix="nova.thermalhydralic."
):
    __import__(module.name)

leaked = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
if leaked:
    raise AssertionError(f"plotting stack reached at import: {{leaked}}")
print("LEAN_CORE_OK")
"""


def module_names():
    """Return every module name in the thermal-hydraulic subtree."""
    return sorted(
        module.name
        for module in pkgutil.walk_packages(
            thermalhydralic.__path__, prefix="nova.thermalhydralic."
        )
    )


@pytest.mark.parametrize("name", module_names())
def test_module_imports(name):
    """Import each module, skipping only on an absent optional distribution."""
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        missing = (error.name or "").split(".")[0]
        if missing in OPTIONAL_DISTRIBUTION:
            pytest.skip(f"{name} requires the optional {missing} distribution")
        raise


@pytest.mark.slow
def test_lean_core_defers_plotting():
    """No module in the subtree imports a plotting stack at module scope.

    Marked slow because the probe pays the whole cluster's cold import cost --
    CoolProp, nlopt, ftputil, mechanize and the data dictionary -- in a fresh
    interpreter, which the in-process tests have already amortised.
    """
    probe = subprocess.run(
        [sys.executable, "-c", LEAN_CORE_PROBE.format(blocked=DEFERRED_PLOTTING)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert "LEAN_CORE_OK" in probe.stdout


class FirstOrderLag:
    """Closed-form magnitude response of K / (1 + s tau).

    The corner magnitude and the high-frequency roll-off are exact, so they
    pin the dB conversion without reference to any fitted model.
    """

    def __init__(self, gain=4.0, time_constant=0.25):
        self.gain = gain
        self.time_constant = time_constant

    def amplitude(self, omega):
        """Return |H| at the angular frequencies omega."""
        return self.gain / np.sqrt(1 + (np.asarray(omega) * self.time_constant) ** 2)

    def decibel(self, omega):
        """Return 20 log10 |H| at the angular frequencies omega."""
        return 20 * np.log10(self.amplitude(omega))

    @property
    def corner(self):
        """Return the angular frequency at which |H| falls by 3 dB."""
        return 1 / self.time_constant


def test_frequency_response_row_per_sample():
    """Each appended sample is one row, not one column."""
    lag = FirstOrderLag()
    omega = np.logspace(-1, 2, 17)
    response = FrequencyResponse()
    response.append_data(omega, lag.amplitude(omega))
    data = response.data["dataset0"]
    assert len(data) == len(omega)
    assert list(data.columns) == ["frequency", "rms_power", "magnitude"]


def test_frequency_response_sorts_on_frequency():
    """An unordered input is stored in ascending frequency."""
    lag = FirstOrderLag()
    omega = np.array([30.0, 0.3, 4.0, 100.0, 1.0])
    response = FrequencyResponse()
    response.append_data(omega, lag.amplitude(omega))
    data = response.data["dataset0"]
    assert np.allclose(data["frequency"].to_numpy(), np.sort(omega))
    assert np.allclose(data["rms_power"].to_numpy(), lag.amplitude(np.sort(omega)))


def test_frequency_response_magnitude_matches_analytic_lag():
    """Stored magnitude is the analytic 20 log10 |H| of the first-order lag."""
    lag = FirstOrderLag()
    omega = np.logspace(-1, 2, 33)
    response = FrequencyResponse()
    response.append_data(omega, lag.amplitude(omega))
    magnitude = response.data["dataset0"]["magnitude"].to_numpy()
    assert np.allclose(magnitude, lag.decibel(omega))


def test_frequency_response_corner_falls_three_decibel():
    """At 1/tau the magnitude sits 10 log10 2 dB below the dc gain."""
    lag = FirstOrderLag()
    omega = np.array([1e-6 * lag.corner, lag.corner])
    response = FrequencyResponse()
    response.append_data(omega, lag.amplitude(omega))
    magnitude = response.data["dataset0"]["magnitude"].to_numpy()
    assert np.isclose(magnitude[0], 20 * np.log10(lag.gain), atol=1e-9)
    assert np.isclose(magnitude[1] - magnitude[0], -10 * np.log10(2))


def test_frequency_response_rolls_off_twenty_decibel_per_decade():
    """Well above the corner a single pole loses 20 dB per decade."""
    lag = FirstOrderLag()
    omega = np.array([1e4, 1e5]) * lag.corner
    response = FrequencyResponse()
    response.append_data(omega, lag.amplitude(omega))
    magnitude = response.data["dataset0"]["magnitude"].to_numpy()
    assert np.isclose(np.diff(magnitude)[0], -20, atol=1e-3)


def test_frequency_response_labels_successive_datasets():
    """Unlabelled datasets take successive prefixed labels."""
    lag = FirstOrderLag()
    omega = np.logspace(-1, 1, 5)
    response = FrequencyResponse()
    response.append_data(omega, lag.amplitude(omega))
    response.append_data(omega, lag.amplitude(omega))
    assert sorted(response.data) == ["dataset0", "dataset1"]


def test_frequency_response_rejects_duplicate_label():
    """A repeated label is refused rather than silently overwriting."""
    lag = FirstOrderLag()
    omega = np.logspace(-1, 1, 5)
    response = FrequencyResponse()
    response.append_data(omega, lag.amplitude(omega), label="run")
    with pytest.raises(IndexError):
        response.append_data(omega, lag.amplitude(omega), label="run")


def test_lumped_capacitance_solves_analytic_exponential():
    """With no external swing the response is Q0 exp(-t hA / C)."""
    heat_transfer, capacitance = 0.4, 1.6
    time_constant = capacitance / heat_transfer
    time = np.linspace(0, 8 * time_constant, 401)
    external = np.zeros_like(time)
    heat = 3.0 * np.exp(-time / time_constant)
    lumped = LumpedCapacitance(time, external, heat)
    modelled = lumped.solve(heat_transfer, capacitance)
    assert modelled.shape == time.shape
    assert np.allclose(modelled, heat, rtol=1e-5, atol=1e-6)


def test_lumped_capacitance_recovers_time_constant():
    """The fit identifies C / hA; hA alone is not identifiable from a decay."""
    time_constant = 3.2
    time = np.linspace(0, 6 * time_constant, 301)
    external = np.zeros_like(time)
    heat = 2.5 * np.exp(-time / time_constant)
    lumped = LumpedCapacitance(time, external, heat)
    heat_transfer, capacitance, error = lumped.fit_hA()
    assert np.isclose(capacitance / heat_transfer, time_constant, rtol=1e-2)
    assert error < 1e-3
