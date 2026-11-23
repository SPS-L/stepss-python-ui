"""AC/DC converter getters and DC buses, on a two-area system joined by a DC link.

In ``acdc_two_areas.dat`` converter C1 holds the DC voltage next to the slack
bus A, and C2 is the reference of the island of bus B (50 MW, 10 Mvar of load
and no generator), holding its AC voltage with EFF 0.98.
"""

import pytest

from stepss.globals import HeliosError
from stepss.helios import BusType, ConverterMode, HeliosSession, SolverStatus
from conftest import LIB_DIR


@pytest.fixture
def two_areas(session, data_dir):
    session.load_file(data_dir / "acdc_two_areas.dat")
    assert session.solve()
    return session


@pytest.fixture
def with_converters(two_areas):
    if not two_areas.has_converter_api:
        pytest.skip("helios library predates the converter getters (C API 1.7)")
    return two_areas


def test_dc_bus_type(two_areas):
    info = two_areas.get_bus_info("D1")
    assert info.bus_type is BusType.DC
    assert info.v_pu == pytest.approx(1.0, abs=1e-9)
    assert two_areas.get_bus_info("A").bus_type is BusType.SLACK


def test_converter_names(with_converters):
    pf = with_converters
    assert pf.converter_count == 2
    assert pf.converter_names() == ["C1", "C2"]
    assert pf.find_converter("C2") == 1
    with pytest.raises(HeliosError):
        pf.find_converter("NOPE")


def test_converter_info(with_converters):
    pf = with_converters
    c2 = pf.get_converter_info("C2")
    assert (c2.ac_bus, c2.dc_bus, c2.in_service) == ("B", "D2", True)
    assert c2.mode is ConverterMode.REF
    assert not c2.rectifying
    assert c2.pac_mw == pytest.approx(-50.0, abs=1e-6)
    assert c2.pdc_mw == pytest.approx(-50.0 / 0.98, abs=1e-6)
    assert c2.qac_mvar == pytest.approx(10.0, abs=1e-6)
    assert c2.vc_pu == pytest.approx(1.0, abs=1e-9)
    # no station: the converter node is the AC bus
    assert c2.station_loss_mw == pytest.approx(0.0, abs=1e-9)
    assert c2.converter_loss_mw == pytest.approx(c2.pac_mw - c2.pdc_mw, abs=1e-9)

    c1 = pf.get_converter_info("C1")
    assert c1.mode is ConverterMode.VDC
    assert c1.rectifying
    assert c1.pac_mw == pytest.approx(c1.pdc_mw, abs=1e-9)   # EFF 1
    # the DC line loss is the gap between what C1 sends and C2 receives
    line_loss = c1.pdc_mw + c2.pdc_mw
    assert 0.0 < line_loss < 1.0


def test_converter_outputs(with_converters):
    pf = with_converters
    pac, qac, pdc, status = pf.get_converter_outputs()
    assert list(status) == [1, 1]
    c2 = pf.get_converter_info("C2")
    assert (pac[1], qac[1], pdc[1]) == pytest.approx((c2.pac_mw, c2.qac_mvar, c2.pdc_mw))


def test_no_converters(solved):
    if not solved.has_converter_api:
        pytest.skip("helios library predates the converter getters (C API 1.7)")
    assert solved.converter_count == 0
    assert solved.converter_names() == []


def test_old_library_raises(two_areas):
    if two_areas.has_converter_api:
        pytest.skip("library has the converter getters")
    with pytest.raises(HeliosError, match="1.7"):
        two_areas.get_converter_info("C1")


def test_no_dc_regulation_status(session, data_dir):
    # C1 is a droop converter capped at PMAX = 20 MW, against about 51 MW that
    # the island of B needs through C2, so nothing holds the DC voltage
    session.load_file(data_dir / "acdc_no_dc_regulation.dat")
    assert not session.solve()
    assert session.solver_status is SolverStatus.NO_DC_REGULATION
    assert "BUSDC 'D1'" in session.last_error
