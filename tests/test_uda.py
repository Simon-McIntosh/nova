import asyncio
import logging
import pytest

from nova.database.filepath import Connect
from nova.datachain.uda import Diagnostic, UdaBase, UdaInfo, mark_uda

mark_connect = Connect(("10.153.0.254", 3090), "codac_uda").mark

logger = logging.getLogger()
logger.setLevel(logging.ERROR)


@pytest.mark.parametrize("name", ["55.AI.00-MSA-9007", "D1-H1-AI00:MSA-9007"])
def test_diagnostic_split(name):
    diagnostic = Diagnostic(name)
    assert diagnostic.group == "AI"
    assert diagnostic.category == "MSA"
    assert diagnostic.index == 9007


@pytest.mark.parametrize(
    "attr,name", [("ids", "55.AI.00-MSA-9007"), ("uda", "D1-H1-AI00:MSA-9007")]
)
def test_diagnostic_name(attr, name):
    diagnostic = Diagnostic(name)
    assert getattr(diagnostic, "name") == name
    assert getattr(diagnostic, f"{attr}_name") == name


@pytest.mark.parametrize("intergral", [True, False])
def test_diagnostic_variable(intergral):
    field = "adcI" if intergral else "adcP"
    diagnostic = Diagnostic("55.AI.00-MSA-9007", intergral)
    assert diagnostic.variable == f"D1-H1-AI00:MSA-9007/{field}"


@mark_uda
@mark_connect
@pytest.mark.asyncio
async def test_UdaBase():
    async with UdaBase() as uda:
        assert uda.handle == -1


@mark_uda
@mark_connect
def test_info_path():
    info = UdaInfo(machine="ITER", pulse_root="PCS", pulse_id=3)
    assert info.pulse == "PCS/3"
    assert info.uri == "ITER:PCS/3"
    info.pulse = "PDS/7"
    assert info.uri == "ITER:PDS/7"
    with pytest.raises(ValueError):
        info.pulse = "ITER/PDS:3"
    with pytest.raises(ValueError):
        info.uri = "ITER/PDS:3"


@mark_uda
@mark_connect
def test_info():
    info = UdaInfo(machine="ITER", pulse_root="PCS", pulse_id="*")
    assert info.pulse_id == "*"
    info.update_uri()
    assert isinstance(info.pulse_id, int)
    assert info.pulse_id != "*"


@mark_uda
@mark_connect
@pytest.mark.asyncio
async def test_async_info():
    info = UdaInfo(machine="ITER", pulse_root="PCS", pulse_id="*")
    await info.aupdate_uri()
    assert info.pulse_id != "*"
    variables = await info.avariables
    assert isinstance(variables, tuple)
    assert len(variables) > 0


@mark_uda
@mark_connect
def test_pulse_info():
    info = UdaInfo(machine="ITER", pulse_root="PCS", pulse_id=62)
    assert isinstance(info.timestamp_from, int)
    assert isinstance(info.timestamp_to, int)
    assert info.timestamp_from < info.timestamp_to


@mark_uda
@mark_connect
@pytest.mark.asyncio
async def test_async_pulse_info():
    info = UdaInfo(machine="ITER", pulse_root="PCS", pulse_id=63)
    pulse_info = await info.apulse_info
    assert isinstance(pulse_info.timeFrom, int)
    assert isinstance(pulse_info.timeTo, int)
    assert pulse_info.timeFrom < pulse_info.timeTo
    assert pulse_info.timeFrom == info.timestamp_from
    assert pulse_info.timeTo == info.timestamp_to


if __name__ == "__main__":
    pytest.main([__file__])
