"""Import guards for the thermal-hydraulic cluster.

Two properties are checked here and nowhere else: every module in the subtree
imports, and none of them pulls a plotting stack into the import graph. The
second guard runs in a subprocess with matplotlib and seaborn blocked at the
finder, because once this process has imported them an in-process check can
no longer tell a module-scope import from a deferred one.
"""

import importlib
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas
import pytest

import nova.thermalhydralic as thermalhydralic
import nova.thermalhydralic.cold_test as cold_test_module
from nova.thermalhydralic.frequencyresponse import FrequencyResponse
from nova.thermalhydralic.lumpedparameter import LumpedCapacitance

COLDTEST_FIXTURE = Path(__file__).parent / "data" / "coldtest"

#: Seconds between samples in the committed cold-test export.
COLDTEST_TIMESTEP = 10.0

#: Displacement law the export was generated from, mm per kA squared. The peak
#: current sits below the 45 kA validity ceiling the conditioning enforces.
COLDTEST_COEFFICENT = 18.0 / 44.0**2

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
import importlib.util
import pkgutil
import sys
import types
from unittest.mock import MagicMock

BLOCKED = frozenset({blocked!r})
OPTIONAL = frozenset({optional!r})


def missing(root):
    \"\"\"Return whether an optional root is absent from this environment.\"\"\"
    try:
        return importlib.util.find_spec(root) is None
    except (ImportError, ValueError):
        return True


# Stand in for the extras this environment does not carry, so every module in the
# subtree still executes its body and is inspected. Skipping those modules instead
# would make the guard weaker the leaner the environment is, which is backwards --
# the lean environment is the one it exists for, and here ten of the twelve extras
# are absent, covering most of the sultan, naka and twente packages.
#
# A blocked root is never stood in for. Each of them is also an extra, so ordering
# these two tests is the whole correctness of the arrangement: excusing an absent
# plotting stack would excuse exactly the property under test.
STUBBED = frozenset(root for root in OPTIONAL - BLOCKED if missing(root))


class StubLoader:
    \"\"\"Execute a stand-in as an empty package answering any attribute.\"\"\"

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = lambda attribute: MagicMock()
        return module

    def exec_module(self, module):
        pass


class Finder:
    \"\"\"Refuse a blocked root outright; stand in for an absent extra.\"\"\"

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(
                f"{{name}} blocked by the lean-core probe", name=name
            )
        if root in STUBBED:
            return importlib.util.spec_from_loader(name, StubLoader(), is_package=True)
        return None


sys.meta_path.insert(0, Finder())

import nova.thermalhydralic as thermalhydralic

for module in pkgutil.walk_packages(
    thermalhydralic.__path__, prefix="nova.thermalhydralic."
):
    __import__(module.name)

leaked = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
if leaked:
    raise AssertionError(f"plotting stack reached at import: {{leaked}}")
print(f"LEAN_CORE_OK stood in for {{sorted(STUBBED)}}")
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

    Every module is inspected whether or not its own extras are installed: an
    absent extra is stood in for, so the guard does not quietly shrink to the
    modules a given environment happens to be able to import. A blocked plotting
    root is never stood in for, which is what keeps the assertion live.

    Marked slow because the probe pays the whole cluster's cold import cost --
    CoolProp, nlopt, ftputil, mechanize and the data dictionary -- in a fresh
    interpreter, which the in-process tests have already amortised.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            LEAN_CORE_PROBE.format(
                blocked=DEFERRED_PLOTTING,
                optional=tuple(sorted(OPTIONAL_DISTRIBUTION)),
            ),
        ],
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


@pytest.fixture
def coldtest(tmp_path, monkeypatch):
    """Point the cold-test data root at a copy of the committed fixture.

    cold_test resolves its project directory from the package root at
    construction and pickles conditioned groups back beside the source, so the
    root is redirected rather than the instance patched.
    """
    shutil.copytree(COLDTEST_FIXTURE, tmp_path / "data" / "CSM")
    monkeypatch.setattr(cold_test_module, "root_dir", str(tmp_path))
    return cold_test_module.cold_test(project_dir="CSM_SYNTH", read_txt=True)


def test_coldtest_splits_channels_by_sensor_and_unit(coldtest):
    """Each group holds the channels its prefix or unit selects, and no others."""
    for group, channels in (
        ("current", [("IBus", "kA"), ("PSIOut", "kA")]),
        ("displace", [("DS001", "mm"), ("DS002", "mm")]),
        ("temperature", [("TT001", "K"), ("TT002", "K")]),
    ):
        coldtest.load_coldtest(group, read_txt=True)
        assert list(getattr(coldtest, group).columns) == channels
    assert coldtest.channels["PSIOut"] == "current"
    assert coldtest.channels["DS001"] == "displace"


def test_coldtest_indexes_on_the_timestamp(coldtest):
    """The export's timestamp column becomes a datetime index."""
    coldtest.load_coldtest("current", read_txt=True)
    index = coldtest.current.index
    assert index.name == "timestamp"
    assert isinstance(index, pandas.DatetimeIndex)
    assert np.allclose(np.diff(coldtest.t(index)), COLDTEST_TIMESTEP)


def test_coldtest_caches_the_conditioned_group(coldtest):
    """A conditioned group is pickled and served from the pickle after."""
    coldtest.load_coldtest("displace", read_txt=True)
    conditioned = coldtest.displace.copy()
    del coldtest.displace
    coldtest.load_coldtest("displace")
    assert coldtest.displace.equals(conditioned)


def test_coldtest_zeroes_the_channel_on_the_first_samples(coldtest):
    """The returned frame is offset so the unloaded coil reads zero."""
    coldtest.load_coldtest("displace", read_txt=True)
    current, frame = coldtest.get_current("displace", "test")
    assert np.isclose(current.iloc[0], 0)
    assert np.allclose(frame.iloc[0].to_numpy(), 0)


def test_coldtest_recovers_the_current_squared_law(coldtest):
    """The fit returns the coefficient the fixture was generated from.

    The fixture's displacement is exactly -k I**2 above a standoff, and the
    offset removes the standoff, so the fitted coefficient is -k with nothing
    left over. The second channel carries half the coefficient of the first.
    """
    coldtest.load_coldtest("displace", read_txt=True)
    coefficent = coldtest.fit("displace", index="test", plot=False)
    assert np.allclose(
        coefficent, [-COLDTEST_COEFFICENT, -0.5 * COLDTEST_COEFFICENT], rtol=1e-9
    )
